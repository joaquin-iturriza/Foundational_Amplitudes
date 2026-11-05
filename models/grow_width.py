"""Function-preserving width growth of a trained μP LLoCa model (plan warm_start step 4, D285).

A parent with num_heads H and a child with 2H (same attn_reps, blocks, mlp_factor) differ only in tensors whose
width dimensions double: the residual stream (H x attn_reps.dim channels), the attention heads (per-head dim
attn_reps.dim fixed) and the MLP hidden units (mlp_factor x the residual width). The child's state is built from the
parent's so that it computes the same function:

  - every residual channel, head and MLP hidden unit is duplicated (the producer's output rows copied);
  - a consumer of a duplicated dimension gets its input columns copied and halved, so its output is unchanged
    (W [x, x] / 2 = W x);
  - the μP readout (MuReadout) divides its input by width_mult, which doubles with the width: its input columns are
    copied without halving ([W, W] [x, x] / (2 m) = W x / m);
  - the layer norms (BaselineLayerNorm, no parameters) see [x, x], whose mean and variance are x's.

Layouts: a residual or MLP-hidden dimension is duplicated as [a, b] -> [a, b, a, b] (block tiling), the input of
an attention's out_linear is head-major (H, d) and tiles the same way; the output of MultiHeadQKVLinear is laid out
(3, d, H) with the head innermost (transformer_lloca_mup.MultiHeadQKVLinear.forward), so its heads are tiled along
that innermost axis.

Duplicates receive identical gradients and would stay identical under Adam and Muon; the second copy's incoming
weights (the rows a producer writes into a duplicated dimension) get Gaussian noise of `noise` times the child's own
initial std of that tensor, which breaks the symmetry at an O(noise) change of the function (HyperCloning, Samragh et
al. 2024, arXiv:2409.12903)."""
import torch
from mup import MuReadout


def _tile(t, dim):
    return torch.cat([t, t], dim=dim)


def _tile_qkv_rows(t, heads):
    """Rows of a MultiHeadQKVLinear weight or bias, laid out (3, d, H), with the H heads tiled to 2H."""
    rest = t.shape[1:]
    v = t.reshape(3, -1, heads, *rest)
    return torch.cat([v, v], dim=2).reshape(-1, *rest)


def _second_copy_rows(n_parent, heads=None):
    """Index of the second copy's rows in a doubled output dimension (block tiling, or qkv's head-innermost layout)."""
    if heads is None:
        return torch.arange(n_parent, 2 * n_parent)
    idx = torch.arange(2 * n_parent).reshape(3, -1, 2 * heads)
    return idx[..., heads:].reshape(-1)


def grow_state(parent_sd, child_model, parent_heads, noise=1e-3, seed=0):
    """The child's state dict built from the parent's (doubled width). `child_model` supplies the target shapes and
    the initial std of each tensor (it is not modified). Raises on any shape that is not equal or doubled."""
    own = child_model.state_dict()
    mods = dict(child_model.named_modules())
    g = torch.Generator().manual_seed(seed)
    out, grown = {}, []
    for k, c in own.items():
        if k not in parent_sd:
            raise KeyError(f"grow_width: {k} is not in the parent's state")
        p = parent_sd[k].to(device=c.device, dtype=c.dtype)
        if tuple(p.shape) == tuple(c.shape):
            out[k] = p.clone()
            continue
        dims = [d for d in range(p.dim()) if c.shape[d] != p.shape[d]]
        if p.dim() != c.dim() or any(c.shape[d] != 2 * p.shape[d] for d in dims):
            raise ValueError(f"grow_width: {k} {tuple(p.shape)} -> {tuple(c.shape)} is not a doubling")
        if ".q_linear." in k or ".k_linear." in k or ".v_linear." in k:
            raise NotImplementedError(f"grow_width: multi-query attention ({k}) is not supported")
        qkv = ".qkv_linear.linear." in k
        readout = isinstance(mods.get(k.rsplit(".", 1)[0]), MuReadout)
        t = p
        if 0 in dims:          # output rows duplicated
            t = _tile_qkv_rows(t, parent_heads) if qkv else _tile(t, 0)
        if 1 in dims:          # input columns duplicated; halved except at the μP readout
            t = _tile(t, 1)
            if not readout:
                t = t / 2
        if 0 in dims and noise and t.dim() == 2:
            rows = _second_copy_rows(p.shape[0], parent_heads if qkv else None)
            std = float(c.float().std()) if c.numel() > 1 else 0.0
            t = t.clone()
            # drawn on the CPU generator (reproducible whatever the device), then moved to the tensor's device: the
            # model lives on the GPU in a job (FA-ccin2p3-93b037bb and five others failed here, cuda:0 against cpu)
            t[rows] += (noise * std * torch.randn(len(rows), t.shape[1], generator=g)).to(device=t.device, dtype=t.dtype)
        out[k] = t
        grown.append(k)
    return out, grown
