"""Simple MLP module."""
from typing import List

import math
import torch
from torch import nn
import mup  # Microsoft μP library
from mup import MuReadout, set_base_shapes, make_base_shapes



class MLP(nn.Module):
    """A simple MLP.

    Flattens all dimensions except batch and uses GELU nonlinearities.
    """

    def __init__(
        self, in_shape, out_shape, hidden_channels, hidden_layers, dropout_prob=None
    ):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.in_shape = in_shape
        self.out_shape = out_shape

        layers: List[nn.Module] = [nn.Linear(prod(in_shape), hidden_channels)]
        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))
        for _ in range(hidden_layers - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_channels, prod(self.out_shape)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor):
        """Forward pass of MLP."""
        return self.mlp(inputs)

class MuMLP_LLoCa(nn.Module):
    """A simple MLP.

    Flattens all dimensions except batch and uses GELU nonlinearities.
    """

    def __init__(
        self, in_shape, out_shape, hidden_channels, hidden_layers, dropout_prob=None
    ):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.in_shape = in_shape
        self.out_shape = out_shape

        layers: List[nn.Module] = [nn.Linear(prod(in_shape), hidden_channels)]
        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))
        for _ in range(hidden_layers - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.GELU())
        layers.append(MuReadout(hidden_channels, prod(self.out_shape)))
        #layers.append(nn.Linear(hidden_channels, prod(self.out_shape)))
        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(lambda m: self.init_weights(m, activation='gelu', init_bias=True))

    @staticmethod
    def init_weights(m, activation='gelu', init_bias=True):
        """
        μP-compatible initialization for MLP layers.

        Args:
            m: layer (nn.Linear or MuReadout)
            activation: activation function name ('gelu', 'relu', etc.)
            init_bias: whether to initialize biases
        """
        if isinstance(m, MuReadout):
            # Readout layer: weights and biases initialized to zero
            if m.weight is not None:
                nn.init.zeros_(m.weight)
            if (m.bias is not None) and init_bias:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            # Hidden layers: Kaiming normal with activation gain
            #gain = nn.init.calculate_gain(activation)
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            #m.weight.data.mul_(gain)

            # Bias initialization
            if (m.bias is not None): #and init_bias:
                fan_in = m.weight.size(1)
                bound = 1 / (fan_in ** 0.5)
                nn.init.uniform_(m.bias, -bound, bound)


    def forward(self, inputs: torch.Tensor):
        """Forward pass of MLP."""
        return self.mlp(inputs)


def prod(shape):
    if isinstance(shape, int):
        return shape
    else:
        return math.prod(shape)