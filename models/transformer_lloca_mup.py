from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import mup  # Microsoft μP library
from mup import MuReadout, set_base_shapes, make_base_shapes

from .attention_lloca_mup import LLoCaAttention
from lloca.reps.tensorreps import TensorReps

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_


class BaselineLayerNorm(nn.Module):
    """Baseline layer norm over all dimensions except the first."""

    @staticmethod
    def forward(inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        outputs : Tensor
            Normalized inputs.
        """
        return torch.nn.functional.layer_norm(
            inputs, normalized_shape=inputs.shape[-1:]
        )


class MultiHeadQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-head attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads, encoder_var=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.linear = nn.Linear(in_channels, 3 * hidden_channels * num_heads)

        # fan-in of the K/V projections is in_channels (the full hidden dim, scales with width),
        # not hidden_channels (the per-head dim, constant). Using in_channels keeps K/V output
        # variance O(1) across widths.
        self.init_method = init_method_normal((encoder_var / in_channels)**0.5)

        self._reset_parameters()

    def _reset_parameters(self):
        #zero initialize querry head weights???
        self.init_method(self.linear.weight)
        q_weights = self.linear.weight[:self.hidden_channels * self.num_heads]
        nn.init.zeros_(q_weights)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        """Forward pass.

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        qkv = self.linear(inputs)  # (..., num_items, 3 * hidden_channels * num_heads)

        *leading, items, last = qkv.shape
        hidden_channels = last // (3 * self.num_heads)
        qkv = qkv.view(*leading, items, 3, hidden_channels, self.num_heads)
        qkv = qkv.movedim(-3, 0).movedim(-1, len(leading) + 1)
        q, k, v = qkv.unbind(dim=0)  # 3x (..., num_heads, num_items, hidden_channels)
        return q, k, v


class MultiQueryQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-query attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads, encoder_var=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.q_linear = nn.Linear(in_channels, hidden_channels * num_heads)
        self.k_linear = nn.Linear(in_channels, hidden_channels)
        self.v_linear = nn.Linear(in_channels, hidden_channels)

        # Same reasoning as MultiHeadQKVLinear: fan-in is in_channels, not hidden_channels.
        self.init_method = init_method_normal((encoder_var / in_channels)**0.5)

        self._reset_parameters()

    def _reset_parameters(self):
        #zero initialize querry head weights???
        self.init_method(self.k_linear.weight)
        self.init_method(self.v_linear.weight)
        nn.init.zeros_(self.q_linear.weight)
        if self.q_linear.bias is not None:
            nn.init.zeros_(self.q_linear.bias)
        if self.k_linear.bias is not None:
            nn.init.zeros_(self.k_linear.bias)
        if self.v_linear.bias is not None:
            nn.init.zeros_(self.v_linear.bias)

    def forward(self, inputs):
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        q = self.q_linear(inputs)

        *leading, items, last = q.shape
        hidden_channels = last // self.num_heads
        q = q.reshape(*leading, items, self.num_heads, hidden_channels)
        q = q.movedim(-2, -3)

        k = self.k_linear(inputs)[
            ..., None, :, :
        ]  # (..., head=1, item, hidden_channels)
        v = self.v_linear(inputs)[..., None, :, :]
        return q, k, v


class BaselineSelfAttention(nn.Module):
    """Baseline self-attention layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    attention
    num_heads : int
        Number of attention heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    dropout_prob : float
        Dropout probability for output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        attention,
        num_heads: int = 8,
        multi_query: bool = True,
        dropout_prob=None,
        encoder_var=1.0,
    ) -> None:
        super().__init__()

        # Store settings
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        self.attention = attention

        # Linear maps
        qkv_class = MultiQueryQKVLinear if multi_query else MultiHeadQKVLinear
        self.qkv_linear = qkv_class(in_channels, hidden_channels, num_heads)
        self.out_linear = nn.Linear(hidden_channels * num_heads, out_channels)

        if dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

        # out_linear fan-in is hidden_channels * num_heads (total concatenated heads),
        # not hidden_channels (per-head). Use the total to keep output variance O(1).
        self.init_method = init_method_normal((encoder_var / (hidden_channels * num_heads))**0.5)

        self._reset_parameters()

    def _reset_parameters(self):
        self.init_method(self.out_linear.weight)
        if self.out_linear.bias is not None:
            nn.init.zeros_(self.out_linear.bias)

    def forward(self, inputs: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        **attn_kwargs

        Returns
        -------
        outputs : Tensor
            Outputs
        """
        q, k, v = self.qkv_linear(
            inputs
        )  # each: (..., num_heads, num_items, num_channels)

        # Attention layer
        h = self.attention(
            q.contiguous(),
            k.expand_as(q).contiguous(),
            v.expand_as(q),
            **attn_kwargs,
        )

        # Concatenate heads and transform linearly
        *leading, num_heads, num_items, hidden_channels = h.shape
        h = h.permute(*range(len(leading)), -2, -3, -1)
        h = h.reshape(*leading, num_items, num_heads * hidden_channels)

        outputs = self.out_linear(h)  # (..., num_items, out_channels)

        if self.dropout is not None:
            outputs = self.dropout(outputs)

        return outputs


class BaselineTransformerBlock(nn.Module):
    """Baseline transformer block.

    Inputs are first processed by a block consisting of LayerNorm, multi-head self-attention, and
    residual connection. Then the data is processed by a block consisting of another LayerNorm, an
    item-wise two-layer MLP with GeLU activations, and another residual connection.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    attention
    num_heads : int
        Number of attention heads.
    attention_factor : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    mlp_factor : int
        Factor by which the activation size is increased over the default value of hidden_channels.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    dropout_prob : float
        Dropout probability for output.
    """

    def __init__(
        self,
        channels,
        attention,
        num_heads: int = 8,
        attention_factor: int = 1,
        multi_query: bool = True,
        mlp_factor: int = 4,
        dropout_prob=None,
        encoder_var=1.0,
    ) -> None:
        super().__init__()

        self.norm1 = BaselineLayerNorm()
        self.norm2 = BaselineLayerNorm()

        hidden_channels = channels // num_heads * attention_factor

        self.attention = BaselineSelfAttention(
            channels,
            channels,
            hidden_channels,
            attention,
            num_heads=num_heads,
            multi_query=multi_query,
            dropout_prob=dropout_prob,
        )

        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_factor * channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
            nn.GELU(),
            nn.Linear(mlp_factor * channels, channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
        )

        self.init_method = init_method_normal((encoder_var / channels)**0.5)

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                self.init_method(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        **attn_kwargs

        Returns
        -------
        outputs : Tensor
            Outputs
        """

        # Residual attention
        h = self.norm1(inputs)
        h = self.attention(h, **attn_kwargs)
        outputs = inputs + h

        # Residual MLP
        h = self.norm2(outputs)
        h = self.mlp(h)
        outputs = outputs + h

        return outputs


class MuPTransformer(nn.Module):
    """Baseline transformer.

    Combines num_blocks transformer blocks, each consisting of multi-head self-attention layers, an
    MLP, residual connections, and normalization layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    attn_reps : str
        Representation of each attention head.
    out_channels : int
        Number of output channels.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    checkpoint_blocks : bool
        Use gradient checkpointing for transformer blocks.
    attention_factor : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    mlp_factor : int
        Factor by which the activation size is increased over the default value of hidden_channels.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    dropout_prob : float
        Dropout probability for output.
    """

    def __init__(
        self,
        in_channels: int,
        attn_reps: str,
        out_channels: int,
        num_blocks: int,
        num_heads: int,
        checkpoint_blocks: bool = False,
        attention_factor: int = 1,
        mlp_factor: int = 4,
        multi_query: bool = False,
        dropout_prob=None,
        encoder_var=1.0,
    ) -> None:
        super().__init__()
        attn_reps = TensorReps(attn_reps)
        self.hidden_channels = attn_reps.dim * num_heads // attention_factor
        self.checkpoint_blocks = checkpoint_blocks
        self.attention = LLoCaAttention(attn_reps, num_heads)

        self.linear_in = nn.Linear(in_channels, self.hidden_channels)
        self.blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    self.hidden_channels,
                    attention=self.attention,
                    num_heads=num_heads,
                    attention_factor=attention_factor,
                    mlp_factor=mlp_factor,
                    multi_query=multi_query,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = MuReadout(self.hidden_channels, out_channels, bias=True)
        self.init_method = init_method_normal((encoder_var / in_channels)**0.5)

        self._reset_parameters()

    def _reset_parameters(self):
        self.init_method(self.linear_in.weight)
        if self.linear_in.bias is not None:
            nn.init.zeros_(self.linear_in.bias)
    def forward(self, inputs: torch.Tensor, frames, **attn_kwargs) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data with shape (..., num_items, in_channels)
        frames : Frames
            Local frames used for invariant particle attention
        **attn_kwargs

        Returns
        -------
        outputs : Tensor
            Outputs with shape (..., num_items, out_channels)
        """
        self.attention.prepare_frames(frames)

        h = self.linear_in(inputs)
        for block in self.blocks:
            if self.checkpoint_blocks:
                fn = partial(block, **attn_kwargs)
                h = checkpoint(fn, h)
            else:
                h = block(h, **attn_kwargs)
        outputs = self.linear_out(h)
        return outputs