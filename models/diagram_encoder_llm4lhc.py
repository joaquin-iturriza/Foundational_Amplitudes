"""The diagram encoder of arXiv:2606.23791 (Bahl, Plehn, Schiller, Sivagnanalingam),
ported to this repo as an alternative to ``models/diagram_encoder.py``.

Selected by ``model.diagram_encoder_impl: llm4lhc`` (with ``model.use_diagrams: true``);
hyperparameters in ``model.diagram_encoder_llm4lhc``. The reference is their
``src/networks/diagram_encoder.py`` (``DiagramSet``, ``_compute_laplacian_pe``,
``_GraphTransformerLayer``, ``DiagramEncoder``) and their App. B, Eq. (19). What is the
same and what had to change is listed here once.

The same
--------
* Graph: nodes are the external legs, typed incoming / outgoing, and the interaction
  vertices (node vocabulary {0 pad, 1 ext_in, 2 ext_out, 3 vertex}); edges are the
  external legs and the internal propagators, each labelled by the signed identity of
  its particle (a 30-entry vocabulary: the SM particles and antiparticles, as their
  ``particles.json``). No masses, couplings, coupling orders or momenta enter.
* Node input: ``node_type_embed(type) + pe_proj(LapPE)``, LapPE the top-``k`` non-trivial
  eigenvectors of ``I - D^-1/2 A D^-1/2`` with their sign fix (largest-magnitude entry
  non-negative); pad nodes zeroed.
* A CLS token prepended per diagram; ``num_layers`` pre-LN layers of full self-attention
  over the diagram's nodes with the edge terms of Eq. (19): a per-head scalar
  ``beta_l`` on the logit and, for ``edge_feature_mode="bias_kv"``, vectors ``r^K_l``,
  ``r^V_l`` (zero-init, shared across layers) in the keys and values, all gated by the
  adjacency; FFN 4x with GELU; readout ``out_proj(LayerNorm(CLS))``.
* ``pooled``: attention pooling of the per-diagram readouts with one learnable query,
  ``softmax_d(q . E_d / sqrt(d))``, over the diagrams of a process.
  ``pooled_external``: their in/out set encoder over the external legs, no graph.
* Initialisation: PyTorch defaults for Linear/Embedding/LayerNorm, ``trunc_normal(0.02)``
  for the CLS token and the pooling queries, zero ``r^K``/``r^V`` tables and zero pad row
  of the direction embedding.

What changed, and why
---------------------
1. Output width. Their readout is ``Linear(H, H)`` with ``H`` the LLM's hidden size, so
   the vector can be added to a token embedding. Here ``out_proj`` maps the encoder
   width ``hidden_size`` (256, their value) to the LLoCa transformer's hidden width
   ``out_dim`` (``attn_reps.dim * num_heads``, 128 at 8 heads). With ``out_dim ==
   hidden_size`` it is their layer exactly. The encoder width stays fixed (standard
   parametrisation, off the muP width axis, like their encoder relative to their LLM);
   ``out_proj`` is then an input-type layer of the backbone (finite fan-in), whose muP
   rule under Adam (lr and init independent of width) is what SP gives it.
2. Where it is added. They add the pooled vector with ``index_put(accumulate=True)`` to
   the embedding of the one process token per event (``proc_pos``), replacing the
   one-hot process embedding. Our events have no process token and cannot get one: every
   LLoCa token needs a 4-momentum to build its local frame, and a frameless token would
   carry vector channels that do not transform. So the vector is added, with the same
   ``index_put(accumulate=True)``, to the token embedding (after ``linear_in``, before the
   first block) of every particle of the event (:func:`process_token_add`). It is a
   function of the process only, hence Lorentz invariant, and the backbone's attention
   distributes it as theirs does from the process token. It does not enter the frames
   net (theirs has none; our old encoder did, through the scalar channel). There is no
   one-hot process embedding to replace: particles are encoded by their properties.
3. Batching. They encode the few processes of a batch padded to ``(K, D_max, N_max)``
   and materialise the edge key/value tensors at ``(.., heads, M, M, head_dim)``. Our
   batches hold every process, so the diagrams of all processes are one flat ragged
   batch (padded to ``N_max`` only), pooled with a segmented softmax over each process's
   diagrams (equal to their masked softmax over ``D``), and the key/value edge terms are
   computed through the 30-entry label vocabulary (``q . r^K`` as a per-label table then a
   gather; ``sum_j a_ij r^V_lij`` as attention mass binned per label then a matmul). Both
   are the same numbers as theirs (``tests/test_diagram_encoder_llm4lhc.py`` checks them
   against a direct transcription of Eq. 19) at a fraction of the memory.
4. Inputs from our sidecars (``tools/dump_diagrams.py``). External-leg labels are read
   from the subprocess's ``external`` list by leg number (the process definition's
   physical identity, as their converter un-crosses initial legs); propagator labels
   from the edge PDG, with MadGraph's arbitrary sign on self-conjugate particles
   (``-22``, ``-23``, ...) folded. A flavour-summed process (several subprocesses) pools
   over the diagrams of all of them: a deliberate deviation, since their converter
   (``generate_diagram_jsons.py``, ``amplitude_index=0``) keeps only the first
   subprocess's amplitude. Loop diagrams (one-loop datasets) are graphs like the trees;
   they have no loop case. The sign of a charged propagator's label comes here from the
   drawing's line id, there from MadGraph's vertex output leg; the two conventions were
   not checked to agree (the graph is undirected and the labels are learned embeddings,
   so a consistent flip only relabels). ``per_diagram`` (one prefix token per diagram)
   has no analogue without prefix tokens and is refused by the wrapper.
"""

import json
import math

import numpy as np
import torch
from torch import nn

PAD_TYPE, EXT_IN_TYPE, EXT_OUT_TYPE, VERTEX_TYPE = 0, 1, 2, 3
NODE_TYPE_VOCAB_SIZE = 4
VALID_MODES = ("pooled", "per_diagram", "pooled_external")
VALID_EDGE_FEATURE_MODES = ("bias", "bias_kv")

# MadGraph SM particles that are their own antiparticle; their sign in a diagram's line
# list is an orientation artefact (250: the neutral Goldstone G0).
_SELF_CONJUGATE = {21, 22, 23, 25, 250}

# Internal lines of MadGraph's one-loop diagrams (loop_sm, Feynman gauge) that their tree-level data never has, so their
# 30-entry vocabulary lacks them: the ghosts (82 gluon, 9000001 photon, 9000002 Z, 9000003 W+, 9000004 W-) and the
# Goldstones (250 G0, 251 G+). Each gets its own learned label after the 30, in a FIXED order (not built from the data),
# so the embedding tables have the same size for every recipe and a parent loads into any fine-tune (2026-10-08: rung 9's
# one-loop processes stopped the first GPU run on PDG 82).
_LOOP_LINES = (82, -82, 9000001, -9000001, 9000002, -9000002, 9000003, -9000003, 9000004, -9000004, 250, 251, -251)


def particle_label_vocab():
    """Signed PDG -> edge label: [0, 30) the SM particles and antiparticles of
    ``particle_ids.PARTICLE_PROPERTIES`` in sorted-PDG order (the same 30 entries as
    their ``particles.json``; only the numbering differs, which a learned embedding
    does not see), then the one-loop ghosts and Goldstones of ``_LOOP_LINES``."""
    from particle_ids import GLOBAL_PDG_IDX
    vocab = {int(pdg): int(i) - 1 for pdg, i in GLOBAL_PDG_IDX.items()}
    n = len(vocab)
    for k, pdg in enumerate(_LOOP_LINES):
        vocab[pdg] = n + k
    return vocab


def _label(pdg, vocab):
    pdg = int(pdg)
    if abs(pdg) in _SELF_CONJUGATE:
        pdg = abs(pdg)
    if pdg not in vocab:
        raise KeyError(f"diagram_encoder_llm4lhc: PDG {pdg} not in the particle vocabulary "
                       f"(particle_ids.PARTICLE_PROPERTIES)")
    return vocab[pdg]


def laplacian_pe(adj, k):
    """Top-``k`` non-trivial eigenvectors of ``L = I - D^-1/2 A D^-1/2`` (their
    ``_compute_laplacian_pe``): drop the smallest, zero-pad below ``k``, flip each so its
    largest-magnitude entry is non-negative. ``adj`` (n, n) bool/0-1."""
    n = adj.shape[0]
    if n == 0:
        return np.zeros((0, k), dtype=np.float32)
    a = adj.astype(np.float32)
    deg = a.sum(axis=1)
    with np.errstate(divide="ignore"):
        dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    lap = np.eye(n, dtype=np.float32) - dinv[:, None] * a * dinv[None, :]
    try:
        _, vecs = np.linalg.eigh(lap)
    except np.linalg.LinAlgError:
        return np.zeros((n, k), dtype=np.float32)
    out = vecs[:, 1:k + 1]
    if out.shape[1] < k:
        out = np.concatenate([out, np.zeros((n, k - out.shape[1]), dtype=out.dtype)], axis=1)
    for c in range(out.shape[1]):
        v = out[:, c]
        if v.size and v[np.argmax(np.abs(v))] < 0:
            out[:, c] = -v
    return out.astype(np.float32)


def graph_from_sidecar_diagram(diagram, external, vocab):
    """One sidecar diagram -> ``(node_types, edges)`` in their schema: nodes ordered as
    externals by leg number then vertices in sidecar order; ``edges`` as
    ``[(i, j, label)]``."""
    nodes, edges = diagram["nodes"], diagram["edges"]
    leg_pdg = {int(leg["number"]): int(leg["pdg"]) for leg in (external or [])}
    ext = [(i, n) for i, n in enumerate(nodes) if n["kind"] == "external"]
    ext.sort(key=lambda t: (t[1].get("leg_number") is None, t[1].get("leg_number") or 0, t[0]))
    vtx = [(i, n) for i, n in enumerate(nodes) if n["kind"] != "external"]
    order = [i for i, _ in ext] + [i for i, _ in vtx]
    new = {old: k for k, old in enumerate(order)}
    node_types = [EXT_OUT_TYPE if n.get("state") == "out" else EXT_IN_TYPE for _, n in ext]
    node_types += [VERTEX_TYPE] * len(vtx)
    out_edges = []
    for e in edges:
        pdg = e["pdg"]
        if e.get("external"):
            # the process definition's identity of the leg (un-crossed), by leg number
            ln = e.get("leg_number")
            if ln is not None and int(ln) in leg_pdg:
                pdg = leg_pdg[int(ln)]
        out_edges.append((new[e["u"]], new[e["v"]], _label(pdg, vocab)))
    return node_types, out_edges


class LLM4LHCDiagramSet:
    """Every process's diagrams as one flat padded batch (``T`` diagrams, ``N`` nodes).

    node_types (T, N) long, node_mask (T, N) bool, adj (T, N, N) bool,
    edge_labels (T, N, N) long (0 where no edge; gated by ``adj``), lap_pe (T, N, k),
    seg (T,) long process id per diagram, n_proc; per process the external legs for
    ``pooled_external``: ext_label (P, E) long, ext_dir (P, E) long {0 pad, 1 in, 2 out},
    ext_count (P,) long; n_diagrams (P,) long.
    """

    def __init__(self, **tensors):
        self.__dict__.update(tensors)

    @property
    def n_total(self):
        return int(self.node_types.shape[0])


def load_llm4lhc_diagram_set(payloads, laplacian_pe_dim=8, max_diagrams=None):
    """Build an :class:`LLM4LHCDiagramSet` from sidecar payloads.

    payloads : list indexed by process id of a sidecar path / loaded dict, or None
        (a process without a sidecar: no diagrams, a zero vector; the experiment refuses
        that for this encoder, as theirs does).
    max_diagrams : keep at most this many diagrams per process, in MadGraph order
        (their ``--max-diagrams``); None keeps all, their default.
    """
    vocab = particle_label_vocab()
    graphs, seg, ext_by_pid = [], [], []
    for pid, src in enumerate(payloads):
        if src is None:
            ext_by_pid.append([])
            continue
        if isinstance(src, str):
            with open(src) as f:
                src = json.load(f)
        subs = src["subprocesses"]
        diags = []
        for sub in subs:
            diags.extend((d, sub.get("external")) for d in sub["diagrams"])
        if max_diagrams is not None and len(diags) > int(max_diagrams):
            diags = diags[:int(max_diagrams)]
        if not diags:
            raise ValueError(f"process {pid}: sidecar has no diagrams")
        for d, external in diags:
            graphs.append(graph_from_sidecar_diagram(d, external, vocab))
            seg.append(pid)
        external = subs[0].get("external") or []
        ext_by_pid.append([(_label(leg["pdg"], vocab), 1 if leg["state"] == "in" else 2)
                           for leg in sorted(external, key=lambda l: int(l["number"]))])

    n_proc = len(payloads)
    T = len(graphs)
    N = max((len(nt) for nt, _ in graphs), default=1)
    node_types = torch.zeros(T, N, dtype=torch.long)
    node_mask = torch.zeros(T, N, dtype=torch.bool)
    adj = torch.zeros(T, N, N, dtype=torch.bool)
    edge_labels = torch.zeros(T, N, N, dtype=torch.long)
    lap = torch.zeros(T, N, laplacian_pe_dim, dtype=torch.float32)
    for t, (nt, edges) in enumerate(graphs):
        n = len(nt)
        node_types[t, :n] = torch.tensor(nt, dtype=torch.long)
        node_mask[t, :n] = True
        for i, j, lbl in edges:
            adj[t, i, j] = adj[t, j, i] = True
            edge_labels[t, i, j] = edge_labels[t, j, i] = lbl
        lap[t, :n] = torch.from_numpy(laplacian_pe(adj[t, :n, :n].numpy(), laplacian_pe_dim))
    E = max((len(x) for x in ext_by_pid), default=1) or 1
    ext_label = torch.zeros(n_proc, E, dtype=torch.long)
    ext_dir = torch.zeros(n_proc, E, dtype=torch.long)
    ext_count = torch.zeros(n_proc, dtype=torch.long)
    for pid, legs in enumerate(ext_by_pid):
        ext_count[pid] = len(legs)
        for k, (lbl, dr) in enumerate(legs):
            ext_label[pid, k] = lbl
            ext_dir[pid, k] = dr
    seg_t = torch.tensor(seg, dtype=torch.long)
    n_diagrams = torch.bincount(seg_t, minlength=n_proc) if T else torch.zeros(n_proc, dtype=torch.long)
    return LLM4LHCDiagramSet(
        node_types=node_types, node_mask=node_mask, adj=adj, edge_labels=edge_labels,
        lap_pe=lap, seg=seg_t, n_proc=n_proc, ext_label=ext_label, ext_dir=ext_dir,
        ext_count=ext_count, n_diagrams=n_diagrams, edge_vocab=len(vocab))


class GraphTransformerLayer(nn.Module):
    """Their ``_GraphTransformerLayer``: pre-LN multi-head self-attention over the
    (1 + N) tokens of a diagram with the Eq. (19) edge terms, then a 4x GELU FFN.

    The edge terms are passed as label tables, not per-pair tensors:
    ``edge_bias`` (B, h, M, M) the gated scalar bias (as theirs); ``labels`` (B, N, N)
    long and ``adj`` (B, N, N) bool over the node block; ``key_table``/``val_table``
    (V, h, d) the ``r^K``/``r^V`` vectors per label, or None for ``edge_feature_mode="bias"``.
    """

    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(),
                                 nn.Linear(4 * hidden_size, hidden_size))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_bias, attn_mask, labels=None, adj=None,
                key_table=None, val_table=None):
        B, M, H = x.shape
        scale = 1.0 / math.sqrt(self.head_dim)
        qkv = self.qkv(self.norm1(x)).reshape(B, M, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                          # (B, h, M, d)
        scores = (q @ k.transpose(-2, -1)) * scale + edge_bias
        if key_table is not None or val_table is not None:
            adj_f = adj.unsqueeze(1).to(x.dtype)                       # (B, 1, N, N)
            idx = labels.unsqueeze(1).expand(B, self.num_heads, -1, -1)       # (B, h, N, N)
        if key_table is not None:
            # q_i . r^K_{l_ij} for node pairs (i, j >= 1): a per-label table then a gather
            qk_lab = torch.einsum("bhmd,vhd->bhmv", q[:, :, 1:], key_table)   # (B, h, N, V)
            ek = torch.gather(qk_lab, -1, idx) * adj_f * scale
            scores = scores.clone()
            scores[:, :, 1:, 1:] = scores[:, :, 1:, 1:] + ek
        scores = scores.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        attn = self.drop(scores.softmax(dim=-1))
        out = attn @ v                                                  # (B, h, M, d)
        if val_table is not None:
            # sum_j a_ij r^V_{l_ij}: attention mass per label, then the label table
            w = attn[:, :, 1:, 1:] * adj_f                              # (B, h, N, N)
            V = val_table.shape[0]
            mass = w.new_zeros(B, self.num_heads, w.shape[2], V)
            mass.scatter_add_(-1, idx, w)                               # (B, h, N, V)
            ev = torch.einsum("bhmv,vhd->bhmd", mass, val_table)
            out = out.clone()
            out[:, :, 1:] = out[:, :, 1:] + ev
        out = self.out(out.transpose(1, 2).reshape(B, M, H))
        x = x + self.drop(out)
        return x + self.drop(self.ffn(self.norm2(x)))


class LLM4LHCDiagramEncoder(nn.Module):
    """Their ``DiagramEncoder``, on the flat batch of :class:`LLM4LHCDiagramSet`.

    ``forward()`` returns one vector per process, (P, out_dim) (``pooled`` and
    ``pooled_external``) or one per diagram, (T, out_dim) (``per_diagram``). Processes
    without diagrams get a zero row.
    """

    def __init__(self, diagrams, out_dim, hidden_size=256, num_layers=3, num_heads=4,
                 laplacian_pe_dim=8, dropout=0.0, mode="pooled", edge_feature_mode="bias_kv"):
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"mode={mode!r}; valid: {VALID_MODES}")
        if edge_feature_mode not in VALID_EDGE_FEATURE_MODES:
            raise ValueError(f"edge_feature_mode={edge_feature_mode!r}; valid: {VALID_EDGE_FEATURE_MODES}")
        if int(diagrams.lap_pe.shape[-1]) != int(laplacian_pe_dim):
            raise ValueError("diagram set built with a different laplacian_pe_dim")
        self.mode = mode
        self.edge_feature_mode = edge_feature_mode
        self.hidden_size = int(hidden_size)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.n_proc = int(diagrams.n_proc)
        edge_vocab = int(diagrams.edge_vocab)
        # graph inputs: non-trainable, non-persistent (rebuilt from the sidecars on load)
        for name in ("node_types", "node_mask", "adj", "edge_labels", "lap_pe", "seg",
                     "ext_label", "ext_dir", "ext_count", "n_diagrams"):
            self.register_buffer(name, getattr(diagrams, name), persistent=False)

        H = self.hidden_size
        if mode in ("pooled", "per_diagram"):
            self.node_type_embed = nn.Embedding(NODE_TYPE_VOCAB_SIZE, H)
            self.pe_proj = nn.Linear(int(laplacian_pe_dim), H)
            self.edge_label_bias = nn.Embedding(edge_vocab, num_heads)
            if edge_feature_mode == "bias_kv":
                rng = torch.get_rng_state()
                self.edge_key_embed = nn.Embedding(edge_vocab, H)
                self.edge_val_embed = nn.Embedding(edge_vocab, H)
                nn.init.zeros_(self.edge_key_embed.weight)
                nn.init.zeros_(self.edge_val_embed.weight)
                torch.set_rng_state(rng)
            self.cls_token = nn.Parameter(torch.zeros(H))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.layers = nn.ModuleList(GraphTransformerLayer(H, num_heads, dropout)
                                        for _ in range(num_layers))
            self.norm_out = nn.LayerNorm(H)
            self.out_proj = nn.Linear(H, self.out_dim)
            if mode == "pooled":
                self.pool_query = nn.Parameter(torch.zeros(self.out_dim))
                nn.init.trunc_normal_(self.pool_query, std=0.02)
        else:  # pooled_external
            self.external_pdg_embed = nn.Embedding(edge_vocab, H)
            self.external_dir_embed = nn.Embedding(3, H)
            with torch.no_grad():
                self.external_dir_embed.weight[0].zero_()
            self.external_set_norm = nn.LayerNorm(H)
            self.external_pool_query = nn.Parameter(torch.zeros(H))
            nn.init.trunc_normal_(self.external_pool_query, std=0.02)
            self.external_out_proj = nn.Linear(H, self.out_dim)

    # -- per-diagram readouts ------------------------------------------------------
    def encode_diagrams(self):
        """CLS readout of every diagram, (T, out_dim)."""
        dt = self.node_type_embed.weight.dtype
        T, N = self.node_types.shape
        H, h = self.hidden_size, self.num_heads
        x = self.node_type_embed(self.node_types) + self.pe_proj(self.lap_pe.to(dt))
        x = x * self.node_mask.unsqueeze(-1).to(dt)
        x = torch.cat([self.cls_token.to(dt).expand(T, 1, H), x], dim=1)        # (T, M, H)
        attn_mask = torch.cat([torch.ones(T, 1, dtype=torch.bool, device=x.device),
                               self.node_mask], dim=1)
        eb = self.edge_label_bias(self.edge_labels).to(dt) * self.adj.unsqueeze(-1).to(dt)
        edge_bias = x.new_zeros(T, h, N + 1, N + 1)
        edge_bias[:, :, 1:, 1:] = eb.permute(0, 3, 1, 2)
        key_table = val_table = None
        if self.edge_feature_mode == "bias_kv":
            key_table = self.edge_key_embed.weight.to(dt).reshape(-1, h, H // h)
            val_table = self.edge_val_embed.weight.to(dt).reshape(-1, h, H // h)
        for layer in self.layers:
            x = layer(x, edge_bias, attn_mask, labels=self.edge_labels, adj=self.adj,
                      key_table=key_table, val_table=val_table)
        return self.out_proj(self.norm_out(x[:, 0]))

    def _segment_softmax_pool(self, vals, scores, seg, n_groups):
        """sum_d softmax_d(scores) vals_d within each group; empty groups -> 0. Equal to
        their masked softmax over the padded diagram axis."""
        dt, dev = vals.dtype, vals.device
        smax = torch.full((n_groups,), float("-inf"), device=dev, dtype=dt)
        smax = smax.scatter_reduce(0, seg, scores, reduce="amax", include_self=True)
        ex = torch.exp(scores - smax[seg])
        den = torch.zeros(n_groups, device=dev, dtype=dt).index_add(0, seg, ex)
        w = ex / den[seg]
        return torch.zeros(n_groups, vals.shape[-1], device=dev, dtype=dt).index_add(
            0, seg, w.unsqueeze(-1) * vals)

    def _encode_pooled_external(self):
        dt = self.external_pdg_embed.weight.dtype
        E = self.ext_label.shape[1]
        mask = torch.arange(E, device=self.ext_label.device)[None] < self.ext_count[:, None]
        feat = self.external_pdg_embed(self.ext_label) + self.external_dir_embed(self.ext_dir)
        feat = feat * mask.unsqueeze(-1).to(dt)                                # (P, E, H)
        logits = (feat * self.external_pool_query).sum(-1) / math.sqrt(self.hidden_size)
        logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.nan_to_num(logits.softmax(dim=-1), nan=0.0).unsqueeze(-1)   # empty row -> 0
        out = self.external_out_proj(self.external_set_norm((feat * w).sum(1)))
        return out * (self.ext_count > 0).unsqueeze(-1).to(dt)

    def forward(self):
        if self.mode == "pooled_external":
            return self._encode_pooled_external()
        per_diag = self.encode_diagrams()                                        # (T, out)
        if self.mode == "per_diagram":
            return per_diag
        scores = (per_diag * self.pool_query.to(per_diag.dtype)).sum(-1) / math.sqrt(self.out_dim)
        return self._segment_softmax_pool(per_diag, scores, self.seg, self.n_proc)


def inject_at_positions(embeds, vectors, positions):
    """Their injection: ``embeds.index_put(positions, vectors, accumulate=True)``."""
    return embeds.index_put(positions, vectors.to(embeds.dtype), accumulate=True)


def process_token_add(proc_vectors, process_ids, ptr, n_tokens, width):
    """The per-token additive term for our flat particle batch: every particle of event
    ``b`` receives its process's vector ``proc_vectors[process_ids[b]]``, placed with
    :func:`inject_at_positions` on a zero (n_tokens, width) tensor, so the backbone adds
    it to its token embeddings. ``ptr`` (B+1,) event boundaries."""
    counts = ptr[1:] - ptr[:-1]
    vec = proc_vectors[process_ids].repeat_interleave(counts, dim=0)            # (n_tokens, W)
    base = torch.zeros(n_tokens, width, device=vec.device, dtype=vec.dtype)
    pos = torch.arange(n_tokens, device=vec.device)
    return inject_at_positions(base, vec, (pos,))
