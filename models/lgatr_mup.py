"""μP-aware full L-GATr (geometric-algebra transformer).

Wraps the stock ``lgatr.LGATr``. mup's ``set_base_shapes`` already identifies the
width dimensions of the geometric ``EquiLinear`` 3-D weights correctly (the channel
dims become "infinite", the basis dim stays finite), so the base-shape machinery is
automatic via the ``lloca.mup`` decorator. On top of that, μP needs:

* attention scaled by ``1/d`` instead of ``1/sqrt(d)`` -- here the per-head geometric
  key dimension grows with the hidden width, so this is load-bearing. We patch
  ``lgatr.primitives.attention.scaled_dot_product_attention`` to pre-scale the query
  by ``d**-0.5`` (the geometric ``sdp_attention`` calls it by name, so the patch
  reaches it). The patch is a no-op for standard parametrization.
* readout (``1/width``): the final ``EquiLinear`` is not an ``nn.Linear`` so it can't
  be a ``MuReadout``; we divide its output by the hidden width multiplier ourselves
  and set ``MUP_DO_ASSERT = False`` so mup skips its MuReadout check. The readout
  weights are zero-initialized (μP convention).

Width axis is ``(hidden_mv_channels, hidden_s_channels)``, scaled together.

.. note::
   μP-correctness for this geometric backbone has *not* been verified by a coordinate
   check from here -- validate on the cluster (see ``test_mup_coord_check_lgatr.py``)
   before trusting width transfer.
"""

import lgatr.primitives.attention as _attn_mod
import torch
from lgatr import LGATr, MLPConfig, SelfAttentionConfig
from torch import nn

from lloca.mup import mup_parametrized

_ORIG_SDP = _attn_mod.scaled_dot_product_attention


def _mup_sdp(query, key, value, **attn_kwargs):
    """μP scaled-dot-product attention: 1/d via query pre-scaling (d = key dim)."""
    return _ORIG_SDP(query * query.shape[-1] ** -0.5, key, value, **attn_kwargs)


def enable_mup_attention():
    """Globally switch lgatr's geometric attention to the μP (1/d) scaling.

    Affects every lgatr attention in the process; a training run uses a single model,
    so this only activates when a μP L-GATr is constructed.
    """
    _attn_mod.scaled_dot_product_attention = _mup_sdp


@mup_parametrized
class MuPLGATr(nn.Module):
    """Full L-GATr with self-contained μP (width = hidden_mv/hidden_s channels)."""

    # The readout is a geometric EquiLinear (not a MuReadout); we scale its output by
    # 1/width ourselves, so skip mup's MuReadout assertion.
    MUP_DO_ASSERT = False
    # Fixed base/delta reference widths; multiplier grows with the target width.
    DEFAULT_MUP_SHAPES = (
        {"hidden_mv_channels": 4, "hidden_s_channels": 4},
        {"hidden_mv_channels": 8, "hidden_s_channels": 8},
    )

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: int | None,
        out_s_channels: int | None,
        hidden_s_channels: int | None,
        num_blocks: int,
        num_heads: int,
        multi_query: bool = False,
        mlp_activation: str = "gelu",
        dropout_prob: float | None = None,
        checkpoint_blocks: bool = False,
        *,
        parametrization: str = "mup",
        mup_base_shapes: dict | None = None,
        mup_delta_shapes: dict | None = None,
    ):
        super().__init__()
        self.parametrization = parametrization
        if parametrization == "mup":
            enable_mup_attention()

        self.net = LGATr(
            num_blocks=num_blocks,
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            attention=SelfAttentionConfig(num_heads=num_heads, multi_query=multi_query),
            mlp=MLPConfig(activation=mlp_activation),
            dropout_prob=dropout_prob,
            checkpoint_blocks=checkpoint_blocks,
        )

        if parametrization == "mup":
            self._zero_init_readout()

    def _zero_init_readout(self):
        # μP readout convention: zero-initialize every weight component of the output
        # EquiLinear (mv, scalar->mv, mv->scalar, scalar->scalar, bias).
        for name, p in self.net.linear_out.named_parameters():
            nn.init.zeros_(p)

    def _readout_mult(self):
        # width multiplier of the hidden (input) channels of the readout EquiLinear;
        # 1.0 until base shapes are set. weight shape is (out, in=hidden, basis).
        w = self.net.linear_out.weight
        if hasattr(w, "infshape"):
            return w.infshape[1].width_mult()
        return 1.0

    def forward(self, multivectors, scalars=None, **attn_kwargs):
        out_mv, out_s = self.net(multivectors, scalars=scalars, **attn_kwargs)
        if self.parametrization == "mup":
            mult = self._readout_mult()
            if mult != 1.0:
                out_mv = out_mv / mult
                if out_s is not None:
                    out_s = out_s / mult
        return out_mv, out_s
