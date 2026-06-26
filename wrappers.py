import os

import torch
from torch import nn
from lgatr.interface import embed_vector, extract_scalar

from models.attention_lloca_mup import build_block_diagonal_bias


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
    def __init__(self, net, token_size, d_particle_hidden: int = 16):
        super().__init__()
        self.net = net
        self.network_dtype = torch.float32
        self.token_size = token_size
        self.d_particle_hidden = d_particle_hidden
        self.use_pids = True   # overridden by setup_particle_features()
        # particle_encoder and property_matrix are set by setup_particle_features()

    def setup_particle_features(self, use_pids: bool, property_matrix=None):
        """
        Configure particle encoding after construction (called from experiment.init_model).

        use_pids=False (default):
            Looks up a fixed property matrix (global_pdg_idx → 8D quantum numbers)
            then projects through a small learned Linear(n_features → d_particle_hidden).
            d_particle_hidden is fixed regardless of n_features, so adding a new quantum
            number only requires extending the projection matrix — all transformer weights
            stay identical and can be reloaded without retraining.

        use_pids=True (legacy):
            One-hot encodes particle type indices; no projection layer.
        """
        self.use_pids = use_pids
        if not use_pids:
            assert property_matrix is not None
            t = torch.tensor(property_matrix, dtype=torch.float32)
            self.register_buffer("property_matrix", t)
            n_features = property_matrix.shape[1]
            # Learned projection: n_features → d_particle_hidden (fixed dim)
            self.particle_encoder = nn.Linear(n_features, self.d_particle_hidden, bias=True)
            nn.init.xavier_uniform_(self.particle_encoder.weight)
            nn.init.zeros_(self.particle_encoder.bias)

    def forward(self, fourmomenta, particle_type_indices, mean, std, ptr, order_labels=None,
                seq_lens=None):
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
            particle_type = self.particle_encoder(raw_props)           # (N_total, d_particle_hidden)

        if order_labels is not None:
            # Broadcast each event's coupling-order vector to all its particles.
            # repeat_interleave is vectorised: no Python loop over events.
            counts = ptr[1:] - ptr[:-1]                                # (B,) particles per event
            order_per_particle = order_labels.repeat_interleave(counts, dim=0)  # (N_total, n_order_features)
            particle_type = torch.cat(
                [particle_type, order_per_particle.to(particle_type.dtype)], dim=-1
            )

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

    # identical to AmplitudeLLoCaWrapper.setup_particle_features
    def setup_particle_features(self, use_pids: bool, property_matrix=None):
        self.use_pids = use_pids
        if not use_pids:
            assert property_matrix is not None
            t = torch.tensor(property_matrix, dtype=torch.float32)
            self.register_buffer("property_matrix", t)
            n_features = property_matrix.shape[1]
            self.particle_encoder = nn.Linear(n_features, self.d_particle_hidden, bias=True)
            nn.init.xavier_uniform_(self.particle_encoder.weight)
            nn.init.zeros_(self.particle_encoder.bias)

    def _scalars(self, particle_type_indices, fourmomenta, ptr, order_labels):
        if self.use_pids:
            scalars = torch.nn.functional.one_hot(
                particle_type_indices, num_classes=self.token_size
            ).to(dtype=self.network_dtype, device=fourmomenta.device)
        else:
            scalars = self.particle_encoder(self.property_matrix[particle_type_indices])
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
