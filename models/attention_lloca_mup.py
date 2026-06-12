import os
import math
import functools
from math import prod
from typing import Optional
import torch
from torch import Tensor
from torch.nn import functional as F

from lloca.framesnet.frames import Frames, InverseFrames, LowerIndicesFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.misc import minimum_autocast_precision

try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False

# Cache BlockDiagonalMask objects by seq_lens pattern, BOUNDED.
# Each mask holds small seqlen tensors; an UNBOUNDED cache keyed by seq_lens leaks
# because the multi-dataset balanced sampler yields a near-unique seq_lens tuple every
# step -> one new entry per step forever (host+GPU memory creep and a progressive
# slowdown as the dict grows). The LRU bound keeps it flat: single-dataset runs have a
# fixed seq_lens pattern (cache hits), while the multi-dataset case just rebuilds the
# (cheap) mask and evicts the oldest entry.
@functools.lru_cache(maxsize=128)
def _cached_block_diagonal_mask(seq_lens):
    return BlockDiagonalMask.from_seqlens(list(seq_lens))


def build_block_diagonal_bias(ptr, seq_lens=None):
    """Build (or fetch from cache) the xformers BlockDiagonalMask for `ptr`.

    Returns None if xformers is unavailable (caller falls back to dense attention).

    `seq_lens` (a CPU tuple of per-event particle counts) may be supplied to skip
    the GPU→CPU sync entirely. `ptr` is born on the CPU in the collate fn, so the
    caller (`_batch_loss_lloca`) can compute `seq_lens` from it *before* moving it
    to the GPU — turning the per-forward `.tolist()` sync (which drains the CUDA
    queue and caps CPU run-ahead) into a pure-CPU op. When `seq_lens` is None we
    fall back to deriving it from `ptr`, which forces the sync.

    NOTE on the sync: the mask depends only on `ptr`, fixed for an entire forward
    pass, so even the fallback MUST be called once per forward — not once per
    block. Computing it inside every block (as the old code did) issued
    `num_blocks` serializing syncs per step for the identical mask.
    """
    if not _XFORMERS_AVAILABLE:
        return None
    if seq_lens is None:
        seq_lens = tuple((ptr[1:] - ptr[:-1]).tolist())   # GPU→CPU sync
    else:
        seq_lens = tuple(seq_lens)
    return _cached_block_diagonal_mask(seq_lens)


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
        self._frames_broadcast = False
        if self.frames.is_global:
            return
        # Head-broadcast path (DEFAULT): keep one frame per particle and broadcast it
        # over the H heads inside the rep transform, instead of materializing num_heads
        # identical copies of every (4,4) matrix. Frame-bookkeeping memory drops by a
        # factor of num_heads (the μP width axis). Numerically identical to the repeat
        # path (guarded by test_amp.py check_frame_broadcast_equivalence).
        # Only valid for scalar+vector reps (max order == 1, e.g. "8x0n+2x1n");
        # higher-order reps fall back to repeat. Set LLOCA_FRAMES=repeat to force the
        # original path (A/B timing / equivalence checks).
        use_bcast = (
            os.environ.get("LLOCA_FRAMES", "broadcast") != "repeat"
            and self.transform.reps.max_rep.rep.order == 1
        )
        if use_bcast:
            self._prepare_frames_broadcast(frames)
        else:
            self._prepare_frames_repeat(frames)

    def _prepare_frames_repeat(self, frames):
        """Original path: replicate the per-particle frames across all heads."""
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

    def _prepare_frames_broadcast(self, frames):
        """Head-broadcast path: store one frame per particle (no head dimension).

        The qkv matrices use a leading axis of size 3 = (q, k, v) →
        (inv, lower_inv, inv) frames, exactly as the repeat path stacks them;
        the head dimension is introduced only as a broadcast inside
        _apply_frames_broadcast, so nothing is materialized num_heads times.
        """
        inv_frames = InverseFrames(frames)
        lower_inv_frames = LowerIndicesFrames(inv_frames)
        self._bcast_qkv_mats = torch.stack(
            [inv_frames.matrices, lower_inv_frames.matrices, inv_frames.matrices], dim=0
        )  # (3, ..., N, 4, 4)
        self._bcast_qkv_det = torch.stack(
            [inv_frames.det, lower_inv_frames.det, inv_frames.det], dim=0
        )  # (3, ..., N)
        self._bcast_out_mats = frames.matrices  # (..., N, 4, 4) — forward frames for output transform
        self._bcast_out_det = frames.det        # (..., N)
        self._frames_broadcast = True

    @minimum_autocast_precision(torch.float32)
    def _apply_frames_broadcast(self, tensor, mats, det):
        """Apply per-particle frames to `tensor`, broadcasting over the head dim.

        Equivalent to TensorRepsTransform._transform_only_scalars_and_vectors
        followed by transform_parity (valid for max order == 1), but with `mats`
        carrying no head axis — the same frame is shared across all heads via
        broadcasting, avoiding num_heads materialized copies.

        tensor : (*lead, H, N, C)   — *lead is () (output) or (3,) (qkv)
        mats   : (*lead, N, 4, 4)   — one frame per particle
        det    : (*lead, N)
        """
        tr = self.transform
        vec_start, vec_end = tr.start_end_idx[-1]   # vectors are the highest-order (last) rep block
        L = (vec_end - vec_start) // 4

        mats = mats.unsqueeze(mats.dim() - 3)        # insert size-1 head axis just before N
        if mats.dtype != tensor.dtype:
            mats = mats.to(tensor.dtype)

        *lead, H, N, C = tensor.shape
        vecs = tensor[..., vec_start:vec_end].reshape(*lead, H, N, L, 4)
        # out[...,n,l,i] = sum_j mats[...,n,i,j] vecs[...,n,l,j]   (mats broadcast over H)
        out_vecs = torch.einsum("...nij,...nlj->...nli", mats, vecs)
        out = tensor.clone()
        out[..., vec_start:vec_end] = out_vecs.reshape(*lead, H, N, vec_end - vec_start)

        if not tr.no_parity_odd:
            sign = det.sign().to(out.dtype).unsqueeze(det.dim() - 1)  # (*lead,1,N): broadcast over H
            out = torch.where(tr.parity_odd, sign.unsqueeze(-1) * out, out)
        return out

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

        # transform q, k, v into global frame
        qkv_local = torch.stack([q_local, k_local, v_local], dim=0)
        if self._frames_broadcast:
            qkv_global = self._apply_frames_broadcast(
                qkv_local, self._bcast_qkv_mats, self._bcast_qkv_det
            )
        else:
            assert 3 * prod(k_local.shape[:-1]) == self.frames_qkv.shape[-3]
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
        if self._frames_broadcast:
            out_local = self._apply_frames_broadcast(
                out_global, self._bcast_out_mats, self._bcast_out_det
            )
        else:
            out_local = self.transform(out_global, self.frames)
        return out_local


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mup_scaling: bool = True,
    ptr: Optional[Tensor] = None,
    attn_bias=None,
) -> Tensor:
    """Execute μP-scaled dot-product attention (1/d instead of 1/sqrt(d)).

    When a block-diagonal mask is supplied (either via a precomputed `attn_bias`
    or by deriving one from `ptr`), uses xformers block-diagonal attention so that
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
        Event boundary pointer of shape (B+1,). Legacy path: if given (and no
        `attn_bias`), the mask is built here — note this syncs once per call.
    attn_bias : xformers BlockDiagonalMask, optional
        Precomputed block-diagonal mask. Preferred: built once per forward by
        MuPTransformer.forward (see build_block_diagonal_bias), avoiding the
        per-block GPU→CPU sync that deriving it from `ptr` incurs.

    Returns
    -------
    torch.Tensor
        (..., H, N_total, C)
    """
    d = query.shape[-1]
    scale = 1.0 / d if mup_scaling else 1.0 / math.sqrt(d)

    # Legacy path: derive the mask from ptr here (one sync per call, i.e. per block).
    if attn_bias is None and ptr is not None:
        attn_bias = build_block_diagonal_bias(ptr)

    if attn_bias is not None:
        # xformers expects (1, N_total, H, C) — reshape from (..., H, N_total, C)
        *leading, H, N, C = query.shape
        q = query.reshape(1, H, N, C).permute(0, 2, 1, 3)  # (1, N, H, C)
        k = key.reshape(1, H, N, C).permute(0, 2, 1, 3)
        v = value.reshape(1, H, N, C).permute(0, 2, 1, 3)

        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=scale)
        # (1, N, H, C) → (..., H, N, C)
        return out.permute(0, 2, 1, 3).reshape(*leading, H, N, C)

    # Fallback: dense attention (cross-event attention will occur if no mask given)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_weights, value)