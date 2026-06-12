"""μP-aware L-GATr-slim.

Reuses the unchanged sub-layers from ``lgatr.nets.lgatr_slim`` (the vector+scalar
``Linear``, ``GatedLinearUnit``, ``MLP``, ``RMSNorm``, ``Dropout``) and overrides
only the two things μP needs to control:

* attention: μP scales the logits by ``1/d`` instead of ``1/sqrt(d)``. Here the
  per-head key dimension grows with the width (``hidden_v/s_channels`` with a fixed
  ``num_heads``), so this is *load-bearing* (unlike the LLoCa transformer where the
  per-head dim is fixed). Implemented by pre-scaling the queries by ``d**-0.5`` so
  the backend's own ``d**-0.5`` composes to ``1/d`` -- backend-agnostic.
* readout: the final ``Linear`` divides its output by the input width multiplier
  (the μP ``1/width`` output scaling), like ``mup.MuReadout`` does for ``nn.Linear``.

The width axis is ``(hidden_v_channels, hidden_s_channels)``, scaled together. Base
shapes are computed automatically in ``__init__`` by the ``lloca.mup`` decorator
(base/delta = half/full of a fixed reference width), so no .bsh files or manual
base/delta models. Call ``lloca.mup.finalize(model)`` on the full wrapped model and
optimize with ``lloca.mup.MuAdamW``.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# Unchanged sub-layers from the stock slim net.
from lgatr.nets.lgatr_slim import (
    Dropout,
    Linear,
    MLP,
    RMSNorm,
    _post_attention_reshape,
)
from lgatr.primitives.attention import scaled_dot_product_attention
from lgatr.utils.misc import minimum_autocast_precision
from mup import MuReadout

from lloca.mup import mup_parametrized


@minimum_autocast_precision(torch.float32)
def _call_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)


class MuPSelfAttention(nn.Module):
    """Slim self-attention with μP (1/d) logit scaling via query pre-scaling."""

    def __init__(self, v_channels, s_channels, num_heads, attn_ratio=1, dropout_prob=None,
                 mup_attention=True):
        super().__init__()
        self.hidden_v_channels = max(attn_ratio * v_channels // num_heads, 1)
        self.hidden_s_channels = max(attn_ratio * s_channels // num_heads, 4)
        self.num_heads = num_heads
        self.mup_attention = mup_attention

        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]))

        self.linear_in = Linear(
            in_v_channels=v_channels,
            out_v_channels=3 * self.hidden_v_channels * self.num_heads,
            in_s_channels=s_channels,
            out_s_channels=3 * self.hidden_s_channels * self.num_heads,
            initialization="small",
        )
        self.linear_out = Linear(
            in_v_channels=self.hidden_v_channels * self.num_heads,
            out_v_channels=v_channels,
            in_s_channels=self.hidden_s_channels * self.num_heads,
            out_s_channels=s_channels,
            initialization="small",
        )
        self.norm = RMSNorm()
        self.dropout = Dropout(dropout_prob) if dropout_prob is not None else None

    def _pre_attention_reshape(self, qkv_v, qkv_s):
        qkv_v = (
            qkv_v.unflatten(-2, (3, self.hidden_v_channels, self.num_heads))
            .movedim(-4, 0)
            .movedim(-2, -4)
        )
        qkv_s = (
            qkv_s.unflatten(-1, (3, self.hidden_s_channels, self.num_heads))
            .movedim(-3, 0)
            .movedim(-1, -3)
        )
        qkv_v, qkv_s = self.norm(qkv_v, qkv_s)
        q_v, k_v, v_v = qkv_v.unbind(0)
        q_s, k_s, v_s = qkv_s.unbind(0)

        q_v_mod = q_v * self.metric.to(q_v.dtype)
        q = torch.cat([q_v_mod.flatten(start_dim=-2), q_s], dim=-1)
        k = torch.cat([k_v.flatten(start_dim=-2), k_s], dim=-1)
        v = torch.cat([v_v.flatten(start_dim=-2), v_s], dim=-1)
        return q, k, v

    def forward(self, vectors, scalars, **attn_kwargs):
        qkv_v, qkv_s = self.linear_in(vectors, scalars)
        q, k, v = self._pre_attention_reshape(qkv_v, qkv_s)
        # μP: q -> q / sqrt(d) so the backend's 1/sqrt(d) composes to 1/d (d = key dim).
        if self.mup_attention:
            q = q * q.shape[-1] ** -0.5
        out = _call_attention(q, k, v, **attn_kwargs)
        h_v, h_s = _post_attention_reshape(out, self.hidden_v_channels)
        out_v, out_s = self.linear_out(h_v, h_s)
        if self.dropout is not None:
            out_v, out_s = self.dropout(out_v, out_s)
        return out_v, out_s


class MuPBlock(nn.Module):
    """Slim block (pre-norm attention + MLP, residual) using μP attention."""

    def __init__(self, v_channels, s_channels, num_heads, nonlinearity="gelu",
                 mlp_ratio=2, attn_ratio=1, num_layers_mlp=2, dropout_prob=None,
                 mup_attention=True):
        super().__init__()
        self.norm = RMSNorm()
        self.attention = MuPSelfAttention(
            v_channels=v_channels, s_channels=s_channels, num_heads=num_heads,
            attn_ratio=attn_ratio, dropout_prob=dropout_prob, mup_attention=mup_attention,
        )
        self.mlp = MLP(
            v_channels=v_channels, s_channels=s_channels, nonlinearity=nonlinearity,
            mlp_ratio=mlp_ratio, num_layers=num_layers_mlp, dropout_prob=dropout_prob,
        )

    def forward(self, vectors, scalars, **attn_kwargs):
        h_v, h_s = self.norm(vectors, scalars)
        h_v, h_s = self.attention(h_v, h_s, **attn_kwargs)
        outputs_v = vectors + h_v
        outputs_s = scalars + h_s
        h_v, h_s = self.norm(outputs_v, outputs_s)
        h_v, h_s = self.mlp(h_v, h_s)
        return outputs_v + h_v, outputs_s + h_s


class MuPReadout(nn.Module):
    """μP output layer for vector+scalar features.

    The scalar output uses ``mup.MuReadout`` (handles the ``1/width`` scaling and the
    mup safety checks). The vector output is a zero-initialized ``weight_v`` matmul,
    with the same ``1/width`` scaling applied manually (there is no MuReadout for the
    raw vector parameter). Weights are zero-initialized, the μP readout convention.
    """

    def __init__(self, in_v_channels, out_v_channels, in_s_channels, out_s_channels):
        super().__init__()
        self.weight_v = nn.Parameter(torch.zeros(out_v_channels, in_v_channels))
        # scalar readout is optional (e.g. event generation has a vector-only output)
        self.readout_s = (
            MuReadout(in_s_channels, out_s_channels, readout_zero_init=True)
            if out_s_channels else None
        )

    def _fanin_mult(self):
        # width multiplier of the input (hidden) channels; 1.0 until shapes set. Use the
        # vector weight if it has rows, else the scalar readout weight (vector-only out
        # has an empty weight_v whose infshape is unreliable).
        w = self.weight_v if self.weight_v.shape[0] > 0 else (
            self.readout_s.weight if self.readout_s is not None else None)
        if w is not None and hasattr(w, "infshape"):
            return w.infshape[1].width_mult()
        return 1.0

    def forward(self, vectors, scalars):
        out_v = self.weight_v @ vectors
        mult = self._fanin_mult()
        if mult != 1.0:
            out_v = out_v / mult
        out_s = self.readout_s(scalars) if self.readout_s is not None else None
        return out_v, out_s


@mup_parametrized
class MuPLGATrSlim(nn.Module):
    """L-GATr-slim with self-contained μP (width = hidden_v_channels/hidden_s_channels)."""

    # Fixed base/delta reference widths (vector, scalar). Width multiplier grows with
    # the target width so μP transfers; structure is c*(base) for c = 1, 2, 4, ...
    DEFAULT_MUP_SHAPES = (
        {"hidden_v_channels": 4, "hidden_s_channels": 8},
        {"hidden_v_channels": 8, "hidden_s_channels": 16},
    )

    def __init__(
        self,
        in_v_channels: int,
        out_v_channels: int,
        hidden_v_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        hidden_s_channels: int,
        num_blocks: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        checkpoint_blocks: bool = False,
        *,
        parametrization: str = "mup",
        mup_base_shapes: dict | None = None,
        mup_delta_shapes: dict | None = None,
    ):
        super().__init__()
        self.parametrization = parametrization
        self._checkpoint_blocks = checkpoint_blocks

        self.linear_in = Linear(
            in_v_channels=in_v_channels, in_s_channels=in_s_channels,
            out_v_channels=hidden_v_channels, out_s_channels=hidden_s_channels,
        )
        self.blocks = nn.ModuleList(
            [
                MuPBlock(
                    v_channels=hidden_v_channels, s_channels=hidden_s_channels,
                    num_heads=num_heads, nonlinearity=nonlinearity, mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio, num_layers_mlp=num_layers_mlp,
                    dropout_prob=dropout_prob, mup_attention=(parametrization == "mup"),
                )
                for _ in range(num_blocks)
            ]
        )
        readout_cls = MuPReadout if parametrization == "mup" else Linear
        self.linear_out = readout_cls(
            hidden_v_channels, out_v_channels, hidden_s_channels, out_s_channels
        )

    def forward(self, vectors, scalars, **attn_kwargs):
        h_v, h_s = self.linear_in(vectors, scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_v, h_s = checkpoint(block, h_v, h_s, use_reentrant=False, **attn_kwargs)
            else:
                h_v, h_s = block(h_v, h_s, **attn_kwargs)
        return self.linear_out(h_v, h_s)
