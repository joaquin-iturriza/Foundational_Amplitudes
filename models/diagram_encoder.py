"""Graph-transformer encoder for a process's Feynman-diagram set.

Consumes the dense padded tensors from :mod:`diagram_graphs` and produces a single
conditioning vector ``E(process)`` that the LLoCa wrapper injects as a dedicated
"diagram token" (see ``wrappers.py``). It is the non-LLM, graph route of
arXiv:2606.23791: a small transformer trained from scratch.

Architecture
------------
Per diagram (a tiny graph), a Graphormer-style transformer:
  * node input  = [physical-property + type/state/order features | Laplacian PE]
  * **full** self-attention over nodes (not restricted to the adjacency); graph
    structure enters as a per-head **scalar edge bias** on the attention logits
    (learned from the edge's particle-property features) plus the LapPE. This is
    the well-tested Graphormer mechanism — we deliberately use it instead of the
    paper's additive key/value edge term (their Eq. 19) for simplicity/robustness.
  * a learnable **CLS readout token** per diagram collects it into one vector.
Then **attention pooling** over the diagrams of the process yields ``E``.

Parametrisation
---------------
This module is standard-parametrisation (like the particle encoder), *outside* the
muP width axis (``num_heads`` of the main transformer). Its width is fixed by
``d_model`` and does not scale with the backbone — mark it SP in muP finalisation.

Tier B
------
``forward`` accepts an optional ``edge_extra`` tensor of per-event, Lorentz-
invariant edge features (e.g. propagator virtualities ``s=(Σp)²``) concatenated to
the static edge features before the bias projection. With ``edge_extra=None`` the
encoder is the static Tier-A topology embedding.
"""

import torch
from torch import nn


class _DiagramLayer(nn.Module):
    """Pre-LN transformer layer with additive per-head edge bias and padding mask."""

    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_bias, key_mask):
        """
        x         : (D, M, d)        node embeddings (M = nodes + CLS)
        edge_bias : (D, H, M, M)     additive attention-logit bias per head
        key_mask  : (D, M) bool      True for real tokens (False = padding key)
        """
        D, M, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(D, M, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                       # each (D, M, H, dh)
        q = q.permute(0, 2, 1, 3)                         # (D, H, M, dh)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        logits = logits + edge_bias
        # mask padding *keys* (columns); rows for padding queries are harmless
        logits = logits.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)                       # (D, H, M, dh)
        out = out.permute(0, 2, 1, 3).reshape(D, M, d)
        x = x + self.drop(self.proj(out))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class DiagramEncoder(nn.Module):
    """Encode a process's diagram set into one conditioning vector ``E``.

    Parameters
    ----------
    f_node, f_edge : int       static node / edge feature widths (see diagram_graphs)
    k_pe : int                 Laplacian PE dimension
    d_model, n_heads, n_layers : transformer size (small; from-scratch)
    d_out : int                output ``E`` dimension (== d_diag injected as a token)
    f_edge_extra : int         width of optional per-event edge features (Tier B); 0 = Tier A
    """

    def __init__(self, f_node, f_edge, k_pe, d_model=64, n_heads=4, n_layers=3,
                 d_out=32, f_edge_extra=0, dropout=0.0, mlp_ratio=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.f_edge = f_edge
        self.f_edge_extra = f_edge_extra

        self.node_in = nn.Linear(f_node + k_pe, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        # per-head scalar edge bias; bias=False so non-edges (zero features) and the
        # CLS row/col contribute exactly zero (we also mask with edge_mask).
        self.edge_bias = nn.Linear(f_edge + f_edge_extra, n_heads, bias=False)

        self.layers = nn.ModuleList(
            _DiagramLayer(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        # attention pooling over diagrams: a learned query scores each diagram's CLS.
        self.pool_query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.pool_query, std=0.02)
        self.out = nn.Linear(d_model, d_out)

    def forward(self, pd, edge_extra=None):
        """Encode one :class:`diagram_graphs.ProcessDiagrams` → ``E`` of shape (d_out,).

        ``edge_extra`` (optional): (D, N, N, f_edge_extra) per-event edge features
        already aligned to the static ``pd.edge_feat`` node order. Must be supplied
        iff ``f_edge_extra > 0``.
        """
        node_feat, lap_pe = pd.node_feat, pd.lap_pe
        edge_feat, edge_mask, node_mask = pd.edge_feat, pd.edge_mask, pd.node_mask
        D, N = node_feat.shape[0], node_feat.shape[1]
        dev = node_feat.device

        # ---- node embeddings + CLS ----
        x_nodes = self.node_in(torch.cat([node_feat, lap_pe], dim=-1))   # (D, N, d)
        cls = self.cls.expand(D, 1, self.d_model).to(x_nodes.dtype)
        x = torch.cat([cls, x_nodes], dim=1)                              # (D, M, d), M=N+1
        key_mask = torch.cat(
            [torch.ones(D, 1, dtype=torch.bool, device=dev), node_mask], dim=1
        )

        # ---- edge bias (D, H, M, M) ----
        ef = edge_feat
        if self.f_edge_extra > 0:
            if edge_extra is None:
                raise ValueError("f_edge_extra>0 requires edge_extra")
            ef = torch.cat([ef, edge_extra.to(ef.dtype)], dim=-1)
        elif edge_extra is not None:
            raise ValueError("edge_extra given but f_edge_extra==0")
        eb = self.edge_bias(ef) * edge_mask[..., None].to(ef.dtype)       # (D, N, N, H)
        eb = eb.permute(0, 3, 1, 2)                                       # (D, H, N, N)
        # pad a zero CLS row/col at index 0 -> (D, H, M, M)
        M = N + 1
        eb_full = eb.new_zeros(D, self.n_heads, M, M)
        eb_full[:, :, 1:, 1:] = eb

        for layer in self.layers:
            x = layer(x, eb_full, key_mask)
        x = self.ln_f(x)

        diag_emb = x[:, 0]                                               # (D, d) CLS per diagram
        # ---- attention pool over diagrams ----
        scores = (diag_emb @ self.pool_query) / (self.d_model ** 0.5)    # (D,)
        weights = torch.softmax(scores, dim=0)                          # (D,)
        pooled = (weights[:, None] * diag_emb).sum(dim=0)               # (d,)
        return self.out(pooled)                                         # (d_out,)

    def encode_registry(self, registry, names=None, edge_extra_fn=None):
        """Encode several processes → ``{name: E}``.

        registry : {name: ProcessDiagrams} (already on the right device)
        names : iterable[str] | None   subset to encode (default: all in registry)
        edge_extra_fn : callable(name, pd) -> edge_extra tensor | None (Tier B hook)
        """
        names = list(registry.keys()) if names is None else names
        out = {}
        for name in names:
            pd = registry[name]
            extra = edge_extra_fn(name, pd) if edge_extra_fn is not None else None
            out[name] = self.forward(pd, edge_extra=extra)
        return out
