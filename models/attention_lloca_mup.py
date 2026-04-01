import math
from math import prod
from typing import Optional
import torch
from torch import Tensor
from torch.nn import functional as F

from lloca.framesnet.frames import Frames, InverseFrames, LowerIndicesFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform

try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False

# Cache BlockDiagonalMask objects by seq_lens pattern.
# BlockDiagonalMask is a lightweight descriptor that does not hold CUDA tensors
# itself — the actual CUDA work happens inside memory_efficient_attention — so
# caching it is safe and avoids repeated Python-level allocation on every step.
_mask_cache: dict = {}


class LLoCaAttention(torch.nn.Module):
    """"""

    def __init__(self, attn_reps, num_heads):
        super().__init__()
        self.transform = TensorRepsTransform(TensorReps(attn_reps))
        self.num_heads = num_heads

        self.frames = None
        self.inv_frames = None
        self.lower_inv_frames = None

    def prepare_frames(self, frames):
        """Prepare local frames for processing with LLoCa attention.
        For a single forward pass through the network, this method is
        called only once for efficiency.

        Parameters
        ----------
        frames: torch.tensor
            Local frames of reference for each particle of shape (..., N, 4, 4)
            where N is the number of particles.
        """
        self.frames = frames
        if not self.frames.is_global:
            # insert frames head dimension
            self.frames = self.frames.reshape(
                *frames.shape[:-3], 1, frames.shape[-3], 4, 4
            )
            self.frames = self.frames.repeat(
                *((1,) * len(frames.shape[:-3])), self.num_heads, 1, 1, 1
            )

            # create inv_frames and lower_inv_frames
            inv_frames = InverseFrames(self.frames)
            lower_inv_frames = LowerIndicesFrames(inv_frames)

            # qkv = (inv_frames, lower_inv_frames, inv_frames)
            # note that (lower_inv_frames, inv_frames, inv_frames) is equivalent
            self.frames_qkv = Frames(
                matrices=torch.stack(
                    [
                        inv_frames.matrices,
                        lower_inv_frames.matrices,
                        inv_frames.matrices,
                    ],
                    dim=0,
                ),
                is_identity=inv_frames.is_identity,
                is_global=inv_frames.is_global,
                det=torch.stack(
                    [inv_frames.det, lower_inv_frames.det, inv_frames.det], dim=0
                ),
                inv=torch.stack(
                    [inv_frames.inv, lower_inv_frames.inv, inv_frames.inv], dim=0
                ),
            )

            # flatten frames (preparation for tensorreps_transform)
            self.frames = self.frames.reshape(-1, 4, 4)
            self.frames_qkv = self.frames_qkv.reshape(-1, 4, 4)

    def forward(self, q_local, k_local, v_local, **attn_kwargs):
        """Execute LLoCa attention.

        Strategy
        1) Transform q, k, v into global frame
        2) Apply attention in global frame
        3) Transform output back into local frame

        Comments
        - dimensions: *dims (optional), H (head), N (particles), C (channels)
        - extension to cross-attention is trivial but we don't have this right now for convenience
          strategy: frames_q for queries (in contrast to frames=frames_kv)

        Parameters
        ----------
        q_local: torch.tensor
            Local queries of shape (*dims, H, N, C)
        k_local: torch.tensor
            Local keys of shape (*dims, H, N, C)
        v_local: torch.tensor
            Local values of shape (*dims, H, N, C)
        attn_kwargs: dict
            Optional arguments that are passed on to attention

        Returns
        -------
        out_local: torch.tensor
            Attention output in local frame of shape (*dims, H, N, C)
        """
        if self.frames.is_global:
            # fallback to standard attention for global frames
            return scaled_dot_product_attention(
                q_local,
                k_local,
                v_local,
                **attn_kwargs,
            )

        # check input shapes
        assert k_local.shape == v_local.shape == q_local.shape  # has to match perfectly
        assert 3 * prod(k_local.shape[:-1]) == self.frames_qkv.shape[-3]

        # transform q, k, v into global frame
        qkv_local = torch.stack([q_local, k_local, v_local], dim=0)
        qkv_global = self.transform(qkv_local, self.frames_qkv)
        q_global, k_global, v_global = torch.unbind(qkv_global, dim=0)

        # (B, H, N, C) format required for scaled_dot_product_attention
        shape_q, shape_k = q_global.shape, k_global.shape
        q_global = q_global.reshape(-1, *shape_q[-3:])
        k_global = k_global.reshape(-1, *shape_k[-3:])
        v_global = v_global.reshape(-1, *shape_k[-3:])

        # attention (in global frame)
        out_global = scaled_dot_product_attention(
            q_global,
            k_global,
            v_global,
            **attn_kwargs,
        )

        out_global = out_global.view(*shape_q)  # (*dims, H, N, C)

        # transform result back into local frame
        out_local = self.transform(out_global, self.frames)
        return out_local


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mup_scaling: bool = True,
    ptr: Optional[Tensor] = None,
) -> Tensor:
    """Execute μP-scaled dot-product attention (1/d instead of 1/sqrt(d)).

    When `ptr` is provided, uses xformers block-diagonal attention so that
    particles in each event only attend to other particles in the same event.
    This is both physically correct and O(B * N_particles^2) instead of
    O(N_total^2), giving a large speedup for batched variable-length sequences.

    Parameters
    ----------
    query : torch.Tensor
        (..., H, N_total, C)
    key : torch.Tensor
        (..., H, N_total, C)
    value : torch.Tensor
        (..., H, N_total, C)
    mup_scaling : bool
        If True, use μP scaling (divide by d instead of sqrt(d)).
    ptr : torch.Tensor, optional
        Event boundary pointer of shape (B+1,). If provided, block-diagonal
        attention is used via xformers.

    Returns
    -------
    torch.Tensor
        (..., H, N_total, C)
    """
    d = query.shape[-1]
    scale = 1.0 / d if mup_scaling else 1.0 / math.sqrt(d)

    if ptr is not None and _XFORMERS_AVAILABLE:
        seq_lens = tuple((ptr[1:] - ptr[:-1]).tolist())
        if seq_lens not in _mask_cache:
            _mask_cache[seq_lens] = BlockDiagonalMask.from_seqlens(list(seq_lens))
        attn_bias = _mask_cache[seq_lens]

        # xformers expects (1, N_total, H, C) — reshape from (..., H, N_total, C)
        *leading, H, N, C = query.shape
        q = query.reshape(1, H, N, C).permute(0, 2, 1, 3)  # (1, N, H, C)
        k = key.reshape(1, H, N, C).permute(0, 2, 1, 3)
        v = value.reshape(1, H, N, C).permute(0, 2, 1, 3)

        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=scale)
        # (1, N, H, C) → (..., H, N, C)
        return out.permute(0, 2, 1, 3).reshape(*leading, H, N, C)

    # Fallback: dense attention (cross-event attention will occur if ptr not given)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_weights, value)