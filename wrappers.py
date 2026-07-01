import os

import torch
from torch import nn
from lgatr.interface import embed_vector, extract_scalar

from models.attention_lloca_mup import build_block_diagonal_bias


def _build_particle_encoder(n_features: int, d_hidden: int, encoder_hidden: int = 0):
    """Project the physical property vector (n_features) → d_hidden.

    encoder_hidden == 0 → a single Linear (legacy: the embedding is a *linear*
    function of the quantum numbers, so only one-hot columns get independent
    directions). encoder_hidden > 0 → a 2-layer MLP (Linear→GELU→Linear), which
    lets the encoder form nonlinear *products* of quantum numbers (charged-AND-
    colored, fermion-AND-massive, …) a single Linear cannot. Negligible params
    vs the transformer. Marked standard-parametrization by mup_finalize either way.
    """
    if encoder_hidden and encoder_hidden > 0:
        enc = nn.Sequential(
            nn.Linear(n_features, encoder_hidden, bias=True),
            nn.GELU(),
            nn.Linear(encoder_hidden, d_hidden, bias=True),
        )
        for lin in (enc[0], enc[2]):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        return enc
    enc = nn.Linear(n_features, d_hidden, bias=True)
    nn.init.xavier_uniform_(enc.weight)
    nn.init.zeros_(enc.bias)
    return enc


def _block_diagonal_attn_mask(ptr, device):
    """Boolean (N_total, N_total) mask: True where two particles share an event.

    Built from the CPU ``ptr`` (event boundaries) so each event's particles attend
    only within that event. Passed as ``attn_mask`` to the (geometric) attention.
    """
    counts = (ptr[1:] - ptr[:-1]).tolist()
    blocks = [torch.ones(c, c, dtype=torch.bool, device=device) for c in counts]
    return torch.block_diag(*blocks)


def _segment_mean(values, ptr):
    """Mean of `values` over each event segment defined by `ptr`.

    Vectorised equivalent of
        torch.stack([values[ptr[i]:ptr[i+1]].mean(dim=0) for i in range(len(ptr)-1)])
    but using a single scatter (index_add_) instead of one slice+mean kernel per
    event — so its cost is independent of the number of events in the batch.

    values : (N_total, C)   per-particle outputs
    ptr    : (B+1,)         cumulative event boundaries (ptr[-1] == N_total)
    returns: (B, C)         per-event means
    """
    counts = ptr[1:] - ptr[:-1]                                  # (B,) particles per event
    B = counts.shape[0]
    seg = torch.repeat_interleave(
        torch.arange(B, device=values.device), counts
    )                                                            # (N_total,) event id per particle
    summed = values.new_zeros((B,) + tuple(values.shape[1:]))
    summed.index_add_(0, seg, values)
    denom = counts.clamp(min=1).view((B,) + (1,) * (values.dim() - 1)).to(values.dtype)
    return summed / denom


def _pool_events(values, ptr):
    """Per-event mean pooling.

    Defaults to the vectorised path. Set env var LLOCA_POOL=loop to fall back to
    the original Python-loop implementation (kept only for A/B timing/equivalence
    checks); any other value uses the vectorised path.
    """
    if os.environ.get("LLOCA_POOL", "vectorized") == "loop":
        B = ptr.shape[0] - 1
        return torch.stack([values[ptr[i]:ptr[i + 1]].mean(dim=0) for i in range(B)])
    return _segment_mean(values, ptr)


def _build_block_diagonal_mask(ptr, device):
    """Boolean mask (N_total, N_total): True where two particles share an event."""
    N = int(ptr[-1])
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(len(ptr) - 1):
        s, e = int(ptr[i]), int(ptr[i + 1])
        mask[s:e, s:e] = True
    return mask


class AmplitudeGATrWrapper(nn.Module):
    def __init__(self, net, token_size):
        super().__init__()
        self.net = net
        self.token_size = token_size

    def forward(self, particles, particle_type_indices, ptr, **kwargs):
        """
        particles             : (N_total, 4)   flat sparse (same format as LLoCa)
        particle_type_indices : (N_total,)     long
        ptr                   : (B+1,)         event boundaries
        """
        # one-hot encode particle types as scalars
        scalars = torch.nn.functional.one_hot(
            particle_type_indices, num_classes=self.token_size
        ).to(dtype=particles.dtype)                          # (N_total, token_size)

        # embed 4-momenta into multivectors: (1, N_total, 1, 16)
        mv = embed_vector(particles.unsqueeze(0).unsqueeze(-2))
        s  = scalars.unsqueeze(0)                            # (1, N_total, token_size)

        # block-diagonal attention mask: particles attend only within their event
        attn_mask = _build_block_diagonal_mask(ptr, particles.device)  # (N_total, N_total)

        mv_out, _ = self.net(mv, scalars=s, attention_mask=attn_mask)

        # extract scalar output per particle, then mean-pool per event
        amp_per_particle = extract_scalar(mv_out)[0, :, 0, 0]  # (N_total,)

        B = ptr.shape[0] - 1
        pooled = torch.stack([
            amp_per_particle[ptr[i]:ptr[i + 1]].mean()
            for i in range(B)
        ]).unsqueeze(-1)                                     # (B, 1)
        return pooled


class AmplitudeLLoCaWrapper(nn.Module):
    def __init__(self, net, token_size, d_particle_hidden: int = 16,
                 particle_encoder_hidden: int = 0,
                 use_diagrams: bool = False, d_diag: int = 32,
                 diagrams_dir: str = "data/diagrams", diagram_encoder=None,
                 use_diagram_virtuality: bool = False, virt_log_scale: float = 0.1,
                 virt_standardize: bool = True, virt_clamp: float = 4.0,
                 virt_mode: str = "edge",
                 diagram_scanned_mass: bool = False):   # read in experiment.init_model; accepted here so Hydra can pass it
        super().__init__()
        self.net = net
        self.network_dtype = torch.float32
        self.token_size = token_size
        self.d_particle_hidden = d_particle_hidden
        # 0 → single Linear encoder; >0 → 2-layer MLP (see _build_particle_encoder).
        # Hydra passes this from config/model/lloca.yaml; setup_particle_features
        # receives the same value explicitly from experiment._post_instantiate_model.
        self.particle_encoder_hidden = particle_encoder_hidden
        self.use_pids = True   # overridden by setup_particle_features()
        # particle_encoder and property_matrix are set by setup_particle_features()

        # Data-derived per-particle mass (data.mass_from_momenta); off until the
        # experiment calls setup_mass_from_momenta(). When on, the forward replaces
        # the gathered property matrix's log10_mass_gev column with the on-shell mass
        # m=sqrt(E^2-|p|^2) of each particle (see setup_mass_from_momenta).
        self.mass_from_momenta = False
        self._mom_div = 1.0
        self._mass_spec = None

        # Feynman-diagram conditioning config (Hydra passes these from lloca.yaml).
        # The encoder module + per-process graphs are attached later by the
        # experiment via setup_diagram_conditioning() (they need the loaded
        # diagram registry, which Hydra cannot build). use_diagrams stays False
        # until then, so an un-set-up wrapper behaves exactly like before.
        self.use_diagrams = False
        self.use_diagram_virtuality = False
        self._cfg_use_diagrams = bool(use_diagrams)
        self.d_diag = int(d_diag)
        self.diagrams_dir = diagrams_dir
        self._diag_enc_cfg = diagram_encoder
        self._pd_by_pid = None
        self._pd_device = None
        self._virt_by_pid = None
        self._virt_log_scale = 0.1   # signed-log virtuality scale; see _diagram_features_virtuality

    def setup_particle_features(self, use_pids: bool, property_matrix=None,
                                encoder_hidden: int = 0):
        """
        Configure particle encoding after construction (called from experiment.init_model).

        use_pids=False (default):
            Looks up a fixed property matrix (global_pdg_idx → quantum numbers) then
            projects through a small learned encoder (n_features → d_particle_hidden).
            d_particle_hidden is fixed regardless of n_features, so adding a new quantum
            number only requires extending the projection — all transformer weights
            stay identical and can be reloaded without retraining. ``encoder_hidden>0``
            makes that encoder a 2-layer MLP instead of a single Linear.

        use_pids=True (legacy):
            One-hot encodes particle type indices; no projection layer.
        """
        self.use_pids = use_pids
        if not use_pids:
            assert property_matrix is not None
            t = torch.tensor(property_matrix, dtype=torch.float32)
            self.register_buffer("property_matrix", t)
            n_features = property_matrix.shape[1]
            self.particle_encoder = _build_particle_encoder(
                n_features, self.d_particle_hidden, encoder_hidden
            )

    def setup_mass_from_momenta(self, mom_div: float, mass_spec: dict):
        """Enable data-derived per-particle mass (``data.mass_from_momenta``).

        ``mom_div`` is the run's global momentum scale (the model sees momenta
        already divided by it), so physical GeV are recovered as ``p*mom_div``.
        ``mass_spec`` is :func:`particle_ids.mass_feature_spec` — the column index
        and transform constants that put the on-shell mass on the same scale the
        frozen ``log10_mass_gev`` column would have under the active encoding flags.
        Off (default) ⇒ the forward is unchanged."""
        self.mass_from_momenta = True
        self._mom_div = float(mom_div)
        self._mass_spec = dict(mass_spec)

    def _apply_mass_from_momenta(self, raw_props, fourmomenta):
        """Replace the log10_mass_gev column of ``raw_props`` (N, n_features) with
        each particle's on-shell mass derived from ``fourmomenta`` (N, 4), matching
        the table column's massless-neutralise → standardize pipeline. Returns the
        (cloned) feature tensor. Shared by the LLoCa + GATr wrappers."""
        spec = self._mass_spec
        raw_props = raw_props.clone()
        p = fourmomenta * self._mom_div                       # scaled → physical GeV
        m2 = p[:, 0] ** 2 - (p[:, 1:] ** 2).sum(dim=-1)
        m = torch.sqrt(m2.clamp(min=0.0))
        floor = spec["floor_log10"]
        log10m = torch.log10(m.clamp(min=10.0 ** floor))      # massless → _MASSLESS floor
        massless = log10m < spec["threshold_log10"]
        if spec["neutralize_log10"] is not None:              # is_massless flag active
            log10m = torch.where(
                massless,
                torch.full_like(log10m, spec["neutralize_log10"]),
                log10m,
            )
        if spec["std_mu"] is not None:
            log10m = (log10m - spec["std_mu"]) / spec["std_sd"]
        raw_props[:, spec["mass_col"]] = log10m.to(raw_props.dtype)
        if spec["is_massless_col"] is not None:
            raw_props[:, spec["is_massless_col"]] = massless.to(raw_props.dtype)
        return raw_props

    def setup_diagram_conditioning(self, encoder, pd_by_pid, d_diag):
        """Attach the diagram graph encoder + per-process graphs (from experiment).

        encoder   : models.diagram_encoder.DiagramEncoder (a real submodule → trained,
                    checkpointed, and marked standard-parametrisation by mup_finalize).
        pd_by_pid : list indexed by process_id (0..n_processes-1) of
                    diagram_graphs.ProcessDiagrams | None (None → that process has no
                    sidecar; it gets a zero diagram embedding so the run still works).
        d_diag    : width of the appended embedding (must match in_channels sizing).
        """
        from diagram_graphs import build_diagram_batch
        self.diagram_encoder = encoder
        self._pd_by_pid = list(pd_by_pid)
        # Pre-batch every process's diagrams into one padded tensor bundle so each
        # step is ONE encoder forward (vs one launch-bound forward per process).
        self._diag_batch = build_diagram_batch(pd_by_pid)
        self._pd_device = None
        self.d_diag = int(d_diag)
        self.use_diagrams = True

    def setup_diagram_virtuality(self, virt_by_pid, log_scale=0.1,
                                 standardize=True, clamp=4.0, mode="edge"):
        """Enable Tier B: per-event propagator virtualities (call after
        setup_diagram_conditioning). ``virt_by_pid`` is a list indexed by process_id
        of the precompute dicts from diagram_graphs.build_process_virtuality (or None
        for processes without a usable per-event leg↔slot map — those events get a
        zero virtuality, i.e. the topology embedding through the same encoder).

        The propagator virtuality is fed as signed-log(s). Its raw distribution is
        very wide and bimodal by channel (s-channel s>0 ~ +9..+19, t-channel s<0 ~
        −8..−19), which destabilises training. With ``standardize=True`` we z-score it
        (mean 0 / std 1, calibrated once on the first training batch and stored as
        buffers so eval/reload are consistent) and clamp the tails to ``±clamp``;
        otherwise the legacy ``signed-log·log_scale`` is used.

        Precomputes each process's static graph tensors padded to the GLOBAL max node
        count so the per-step path can concatenate every (event,diagram) graph into
        ONE batched encoder call (see _diagram_features_virtuality)."""
        self._virt_by_pid = list(virt_by_pid)
        self._virt_log_scale = float(log_scale)
        self._virt_standardize = bool(standardize)
        self._virt_clamp = float(clamp)
        self._virt_mode = str(mode)        # "edge" (per-event encode) | "pool" (cached topology + weighted)
        self.use_diagram_virtuality = True
        # Per-process diagram slice in the concatenated static batch (build_diagram_batch
        # iterates pids in order over non-None pds) — used by the "pool" mode to map
        # events to their process's diagram CLS embeddings.
        offs, cur = {}, 0
        for pid, pd in enumerate(self._pd_by_pid):
            if pd is not None:
                offs[pid] = (cur, pd.n_diagrams); cur += pd.n_diagrams
        self._diag_offsets = offs
        # calibration buffers (persisted in the checkpoint); _virt_cal_done mirrors
        # _virt_calibrated as a Python flag to avoid a GPU→CPU sync after calibration.
        self.register_buffer("_virt_mu", torch.zeros(()))
        self.register_buffer("_virt_sigma", torch.ones(()))
        self.register_buffer("_virt_calibrated", torch.zeros((), dtype=torch.bool))
        self._virt_cal_done = False

        Mstar = max(pd.node_feat.shape[1] for pd in self._pd_by_pid if pd is not None)
        self._virt_Mstar = Mstar
        static = []
        for pd in self._pd_by_pid:
            if pd is None:
                static.append(None); continue
            D, N = pd.node_feat.shape[0], pd.node_feat.shape[1]
            nf = pd.node_feat.new_zeros(D, Mstar, pd.node_feat.shape[-1]); nf[:, :N] = pd.node_feat
            lp = pd.lap_pe.new_zeros(D, Mstar, pd.lap_pe.shape[-1]);       lp[:, :N] = pd.lap_pe
            nm = pd.node_mask.new_zeros(D, Mstar);                         nm[:, :N] = pd.node_mask
            ef = pd.edge_feat.new_zeros(D, Mstar, Mstar, pd.edge_feat.shape[-1]); ef[:, :N, :N] = pd.edge_feat
            em = pd.edge_mask.new_zeros(D, Mstar, Mstar);                  em[:, :N, :N] = pd.edge_mask
            static.append({"node_feat": nf, "lap_pe": lp, "node_mask": nm,
                           "edge_feat": ef, "edge_mask": em, "D": D})
        self._virt_static = static
        self._pd_device = None   # force re-move of the new tensors

    def _diagram_features_virtuality(self, fourmomenta, std, process_ids, ptr, device, dtype):
        """Tier B per-particle diagram embedding (N_total, d_diag), batched.

        Builds every (event, diagram) graph in the batch — static topology tensors
        tiled per event, plus the event's per-propagator virtuality s=(Σ sign·p)²
        (all-outgoing; signed-log compressed) as the encoder's edge feature — into ONE
        flat batch, then a SINGLE encode_grouped call segment-pools per event. The
        per-process loop only does cheap tensor ops (gather/einsum/scatter/tile); the
        expensive graph-transformer forward runs once on the whole batch (vs one
        launch-bound forward_per_event per process). A process with no per-event map
        gets zero virtuality (topology embedding through the same encoder); a process
        with no diagrams leaves its events at zero.
        """
        if self._pd_device != device:
            self._virt_by_pid = [
                ({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vt.items()}
                 if vt is not None else None) for vt in self._virt_by_pid]
            self._virt_static = [
                ({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in st.items()}
                 if st is not None else None) for st in self._virt_static]
            self._pd_device = device

        B = process_ids.shape[0]
        M = self._virt_Mstar

        # --- pass 1: per-process raw signed-log virtuality (cheap einsum, no encoder) ---
        jobs, calib = [], []
        for pid in torch.unique(process_ids).tolist():
            st = self._virt_static[pid] if 0 <= pid < len(self._virt_static) else None
            if st is None:
                continue
            ev_idx = (process_ids == pid).nonzero(as_tuple=True)[0]      # (E_p,)
            vt = self._virt_by_pid[pid]
            raw = None
            if vt is not None:
                n_part = vt["n_part"]
                gather = ptr[ev_idx][:, None] + torch.arange(n_part, device=device)[None, :]
                mom = fourmomenta[gather]                               # (E_p, n_part, 4)
                p_prop = torch.einsum("kn,enc->ekc", vt["mask"], mom)   # (E_p, K, 4)
                s = p_prop[..., 0] ** 2 - (p_prop[..., 1:] ** 2).sum(-1)
                raw = torch.sign(s) * torch.log1p(s.abs())             # (E_p, K) signed-log
                calib.append(raw.detach().reshape(-1))
            jobs.append((ev_idx, st, vt, raw))

        # --- calibrate standardization stats once (first batch); persisted as buffers ---
        if self._virt_standardize and not self._virt_cal_done:
            if not bool(self._virt_calibrated) and calib:    # sync only until calibrated
                allraw = torch.cat(calib)
                self._virt_mu.copy_(allraw.mean())
                self._virt_sigma.copy_(allraw.std().clamp_min(1e-3))
                self._virt_calibrated.fill_(True)
            self._virt_cal_done = True

        # --- pass 2: standardize+clamp the virtuality, scatter, tile static graphs ---
        nf_l, lp_l, nm_l, ef_l, em_l, ee_l, seg_l = [], [], [], [], [], [], []
        for (ev_idx, st, vt, raw) in jobs:
            E_p, D_p = ev_idx.shape[0], st["D"]
            ee = torch.zeros(E_p, D_p, M, M, 1, device=device, dtype=dtype)
            if vt is not None:
                if self._virt_standardize:
                    feat = ((raw - self._virt_mu) / self._virt_sigma).clamp(
                        -self._virt_clamp, self._virt_clamp)
                else:
                    feat = raw * self._virt_log_scale
                feat = feat.to(dtype)
                ee[:, vt["diag_idx"], vt["i_idx"], vt["j_idx"], 0] = feat
                ee[:, vt["diag_idx"], vt["j_idx"], vt["i_idx"], 0] = feat   # symmetric

            def tile(t):   # (D_p, ...) -> (E_p*D_p, ...) event-major, sharing static graph
                return t.unsqueeze(0).expand(E_p, *t.shape).reshape(E_p * D_p, *t.shape[1:])
            nf_l.append(tile(st["node_feat"])); lp_l.append(tile(st["lap_pe"]))
            nm_l.append(tile(st["node_mask"])); ef_l.append(tile(st["edge_feat"]))
            em_l.append(tile(st["edge_mask"])); ee_l.append(ee.reshape(E_p * D_p, M, M, 1))
            seg_l.append(ev_idx.repeat_interleave(D_p))                 # graph -> event id

        if not seg_l:
            E_event = torch.zeros(B, self.d_diag, device=device, dtype=dtype)
        else:
            E_event = self.diagram_encoder.encode_grouped(
                torch.cat(nf_l), torch.cat(lp_l), torch.cat(nm_l),
                torch.cat(ef_l), torch.cat(em_l), torch.cat(seg_l), B,
                edge_extra=torch.cat(ee_l)).to(dtype)                   # (B, d_diag)

        counts = ptr[1:] - ptr[:-1]
        return E_event.repeat_interleave(counts, dim=0)

    def _diagram_features_virtuality_pool(self, fourmomenta, std, process_ids, ptr, device, dtype):
        """Tier B, "pool" mode (the structural speedup): encode every diagram's
        TOPOLOGY once (static, ~#diagrams graphs/step, like Tier A), then let the
        per-event virtuality enter ONLY the pooling weights — a learned ``virt_score``
        of each propagator's virtuality up/down-weights its diagram (near-resonant
        s≈m² → larger weight). The expensive graph transformer no longer runs per
        event, so the cost is ~Tier A's. A different model from "edge" mode (the
        virtuality doesn't change a diagram's internal representation), so it is A/B'd
        separately."""
        if self._pd_device != device:
            self._diag_batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                                for k, v in self._diag_batch.items()} if self._diag_batch else None
            self._virt_by_pid = [
                ({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vt.items()}
                 if vt is not None else None) for vt in self._virt_by_pid]
            self._pd_device = device

        enc = self.diagram_encoder
        cls_all = enc.encode_static(self._diag_batch)                   # (TotalD, d) — ONE forward
        base_all = (cls_all @ enc.pool_query) / (enc.d_model ** 0.5)    # (TotalD,)
        B = process_ids.shape[0]

        # pass 1: raw signed-log virtuality per process (for calibration)
        jobs, calib = [], []
        for pid in torch.unique(process_ids).tolist():
            if pid not in self._diag_offsets:
                continue
            ev_idx = (process_ids == pid).nonzero(as_tuple=True)[0]
            vt = self._virt_by_pid[pid]
            raw = None
            if vt is not None:
                n_part = vt["n_part"]
                gather = ptr[ev_idx][:, None] + torch.arange(n_part, device=device)[None, :]
                p_prop = torch.einsum("kn,enc->ekc", vt["mask"], fourmomenta[gather])
                s = p_prop[..., 0] ** 2 - (p_prop[..., 1:] ** 2).sum(-1)
                raw = torch.sign(s) * torch.log1p(s.abs())             # (E_p, K)
                calib.append(raw.detach().reshape(-1))
            jobs.append((pid, ev_idx, vt, raw))

        if self._virt_standardize and not self._virt_cal_done:
            if not bool(self._virt_calibrated) and calib:
                allraw = torch.cat(calib)
                self._virt_mu.copy_(allraw.mean())
                self._virt_sigma.copy_(allraw.std().clamp_min(1e-3))
                self._virt_calibrated.fill_(True)
            self._virt_cal_done = True

        # pass 2: per-event virtuality-weighted pool over the process's diagrams
        E_event = torch.zeros(B, self.d_diag, device=device, dtype=dtype)
        for (pid, ev_idx, vt, raw) in jobs:
            start, D_p = self._diag_offsets[pid]
            cls_p = cls_all[start:start + D_p]                          # (D_p, d)
            base_p = base_all[start:start + D_p]                        # (D_p,)
            E_p = ev_idx.shape[0]
            if vt is not None:
                if self._virt_standardize:
                    feat = ((raw - self._virt_mu) / self._virt_sigma).clamp(
                        -self._virt_clamp, self._virt_clamp)
                else:
                    feat = raw * self._virt_log_scale
                # [signed-log(s)] ++ propagator property (incl. mass) -> resonance-aware
                ep = vt["edge_prop"].to(feat.dtype).unsqueeze(0).expand(E_p, -1, -1)  # (E_p,K,f_edge)
                vin = torch.cat([feat.unsqueeze(-1), ep], dim=-1)       # (E_p, K, 1+f_edge)
                score = enc.virt_score(vin).squeeze(-1)                # (E_p, K)
                contrib = torch.zeros(E_p, D_p, device=device, dtype=score.dtype)
                contrib.index_add_(1, vt["diag_idx"], score)           # (E_p, D_p)
            else:
                contrib = torch.zeros(E_p, D_p, device=device, dtype=base_p.dtype)
            w = torch.softmax(base_p.unsqueeze(0) + contrib, dim=1)     # (E_p, D_p)
            E_event[ev_idx] = enc.out(w @ cls_p).to(dtype)             # (E_p, d_diag)

        counts = ptr[1:] - ptr[:-1]
        return E_event.repeat_interleave(counts, dim=0)

    def _diagram_features(self, process_ids, ptr, device, dtype):
        """Per-particle diagram embedding (N_total, d_diag).

        Encodes ALL processes' (static) diagram sets in a single batched encoder
        forward → E_all (n_proc, d_diag), gathers per event via process_ids, then
        broadcasts to particles — the order_labels broadcast pattern. The static
        topology embedding only depends on the encoder weights (not the batch), so
        encoding all processes once and gathering is both correct and far cheaper
        than a per-process Python loop (no GPU→CPU unique() sync, one big GPU call).
        """
        if self._diag_batch is None:                                       # no sidecars
            return torch.zeros(int(ptr[-1]), self.d_diag, device=device, dtype=dtype)
        # Move the static batch onto the compute device once (regenerable data, not
        # nn buffers, so it doesn't auto-migrate with .to()).
        if self._pd_device != device:
            self._diag_batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                                for k, v in self._diag_batch.items()}
            self._pd_device = device

        E_all = self.diagram_encoder.encode_all(self._diag_batch).to(dtype)  # (n_proc, d_diag)
        E_event = E_all[process_ids]                                       # (B, d_diag)
        counts = ptr[1:] - ptr[:-1]                                        # (B,)
        return E_event.repeat_interleave(counts, dim=0)                    # (N_total, d_diag)

    def forward(self, fourmomenta, particle_type_indices, mean, std, ptr, order_labels=None,
                seq_lens=None, process_ids=None):
        """
        fourmomenta           : (N_total, 4)
        particle_type_indices : (N_total,)              long
        mean, std             : float
        ptr                   : (B+1,)                  event boundaries
        order_labels          : (B, n_order_features)   coupling-order vector per event,
                                e.g. [n_loops, alpha_s_power]. Broadcast to each particle.
        seq_lens              : tuple | None            CPU per-event particle counts;
                                forwarded to the attention-mask builder to skip a
                                GPU→CPU sync (see build_block_diagonal_bias).
        """
        if self.use_pids:
            particle_type = torch.nn.functional.one_hot(
                particle_type_indices, num_classes=self.token_size
            ).to(dtype=self.network_dtype, device=fourmomenta.device)
        else:
            raw_props = self.property_matrix[particle_type_indices]   # (N_total, n_features)
            if self.mass_from_momenta:
                raw_props = self._apply_mass_from_momenta(raw_props, fourmomenta)
            particle_type = self.particle_encoder(raw_props)           # (N_total, d_particle_hidden)

        if order_labels is not None:
            # Broadcast each event's coupling-order vector to all its particles.
            # repeat_interleave is vectorised: no Python loop over events.
            counts = ptr[1:] - ptr[:-1]                                # (B,) particles per event
            order_per_particle = order_labels.repeat_interleave(counts, dim=0)  # (N_total, n_order_features)
            particle_type = torch.cat(
                [particle_type, order_per_particle.to(particle_type.dtype)], dim=-1
            )

        # Feynman-diagram conditioning: append the per-process diagram embedding to
        # every particle's scalar features (broadcast like order_labels). E(process)
        # is a Lorentz scalar, so this preserves equivariance. Appended AFTER the
        # order labels so the feature layout is [property | order | diagram], matching
        # the in_channels/num_scalars sizing in experiment._finalize_data_sizing.
        if self.use_diagrams and process_ids is not None:
            if self.use_diagram_virtuality:
                feat_fn = (self._diagram_features_virtuality_pool
                           if getattr(self, "_virt_mode", "edge") == "pool"
                           else self._diagram_features_virtuality)
                diag_feat = feat_fn(fourmomenta, std, process_ids, ptr,
                                    particle_type.device, particle_type.dtype)
            else:
                diag_feat = self._diagram_features(
                    process_ids, ptr, particle_type.device, particle_type.dtype)
            particle_type = torch.cat([particle_type, diag_feat], dim=-1)

        outputs = self.net(fourmomenta, particle_type, mean, std, ptr=ptr, seq_lens=seq_lens)

        # Per-event mean pool (vectorised; see _pool_events / LLOCA_POOL env toggle)
        return _pool_events(outputs, ptr)


class _AmplitudeGATrMuPBase(nn.Module):
    """Shared scaffolding for the μP L-GATr amplitude wrappers.

    Matches the AmplitudeLLoCaWrapper interface (same forward signature and
    ``setup_particle_features``) so the experiment treats these models exactly like
    the LLoCa transformer: sparse flat 4-momenta + ``ptr`` event boundaries, per-event
    mean pooling, and the fixed-property particle encoding. Subclasses implement
    ``_run_net`` (embed 4-momenta, call the geometric net, return a scalar per particle).
    """

    def __init__(self, net, token_size, d_particle_hidden: int = 16):
        super().__init__()
        self.net = net
        self.network_dtype = torch.float32
        self.token_size = token_size
        self.d_particle_hidden = d_particle_hidden
        self.use_pids = True
        self.mass_from_momenta = False
        self._mom_div = 1.0
        self._mass_spec = None

    # identical to AmplitudeLLoCaWrapper.setup_mass_from_momenta / _apply_mass_from_momenta
    setup_mass_from_momenta = AmplitudeLLoCaWrapper.setup_mass_from_momenta
    _apply_mass_from_momenta = AmplitudeLLoCaWrapper._apply_mass_from_momenta

    # identical to AmplitudeLLoCaWrapper.setup_particle_features
    def setup_particle_features(self, use_pids: bool, property_matrix=None,
                                encoder_hidden: int = 0):
        self.use_pids = use_pids
        if not use_pids:
            assert property_matrix is not None
            t = torch.tensor(property_matrix, dtype=torch.float32)
            self.register_buffer("property_matrix", t)
            n_features = property_matrix.shape[1]
            self.particle_encoder = _build_particle_encoder(
                n_features, self.d_particle_hidden, encoder_hidden
            )

    def _scalars(self, particle_type_indices, fourmomenta, ptr, order_labels):
        if self.use_pids:
            scalars = torch.nn.functional.one_hot(
                particle_type_indices, num_classes=self.token_size
            ).to(dtype=self.network_dtype, device=fourmomenta.device)
        else:
            raw_props = self.property_matrix[particle_type_indices]
            if self.mass_from_momenta:
                raw_props = self._apply_mass_from_momenta(raw_props, fourmomenta)
            scalars = self.particle_encoder(raw_props)
        if order_labels is not None:
            counts = ptr[1:] - ptr[:-1]
            order_per_particle = order_labels.repeat_interleave(counts, dim=0)
            scalars = torch.cat([scalars, order_per_particle.to(scalars.dtype)], dim=-1)
        return scalars

    def forward(self, fourmomenta, particle_type_indices, mean, std, ptr,
                order_labels=None, seq_lens=None):
        scalars = self._scalars(particle_type_indices, fourmomenta, ptr, order_labels)
        # Scale by std only (a Lorentz-invariant rescaling); do NOT subtract the mean,
        # which would break equivariance. (The 4-momenta are already preprocessed.)
        fm = fourmomenta / std
        # Prefer the xformers block-diagonal kernel (memory-efficient, O(sum n_i^2)) —
        # passed as `attn_bias`, which lgatr's backend dispatcher routes to xformers
        # `memory_efficient_attention` (same kernel LLoCa uses). The dense O(N_total^2)
        # `attn_mask` path materialises the full cross-event score matrix and OOMs at
        # batchsize 1024. μP's 1/d query pre-scaling is unaffected: it wraps the
        # `scaled_dot_product_attention` dispatcher *above* backend selection, and
        # xformers defaults to the same 1/sqrt(d) scale as native SDPA.
        # `build_block_diagonal_bias` returns None when xformers is unavailable
        # (e.g. login-node CPU) -> fall back to the dense mask + native attention.
        attn_bias = build_block_diagonal_bias(ptr, seq_lens)
        if attn_bias is not None:
            amp_per_particle = self._run_net(fm, scalars, attn_bias=attn_bias)
        else:
            attn_mask = _block_diagonal_attn_mask(ptr, fourmomenta.device)
            amp_per_particle = self._run_net(fm, scalars, attn_mask=attn_mask)
        return _pool_events(amp_per_particle.unsqueeze(-1), ptr)  # (B, 1)


class AmplitudeLGATrMuPWrapper(_AmplitudeGATrMuPBase):
    """μP full L-GATr: 4-momentum -> one multivector, scalar amplitude via extract_scalar."""

    def _run_net(self, fm, scalars, **attn_kwargs):
        mv = embed_vector(fm.unsqueeze(0).unsqueeze(-2))  # (1, N, 1, 16)
        s = scalars.unsqueeze(0)                          # (1, N, n_scalars)
        out_mv, _ = self.net(mv, scalars=s, **attn_kwargs)
        return extract_scalar(out_mv)[0, :, 0, 0]         # (N,)


class AmplitudeLGATrSlimMuPWrapper(_AmplitudeGATrMuPBase):
    """μP L-GATr-slim: 4-momentum -> one vector, scalar amplitude from the scalar output."""

    def _run_net(self, fm, scalars, **attn_kwargs):
        vectors = fm.unsqueeze(0).unsqueeze(-2)           # (1, N, 1, 4)  (v_channels=1)
        s = scalars.unsqueeze(0)                          # (1, N, n_scalars)
        _, out_s = self.net(vectors, s, **attn_kwargs)
        return out_s[0, :, 0]                             # (N,)
