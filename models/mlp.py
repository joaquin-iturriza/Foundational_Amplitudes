from typing import List

import numpy as np
import torch
from torch import nn

from .activation import switchable_activation


class MLP(nn.Module):
    """A simple baseline MLP.

    Flattens all dimensions except batch and uses GELU nonlinearities.
    """

    def __init__(
        self,
        n_features,
        type_token_list,
        hidden_channels,
        hidden_layers,
        out_shape=1,
        activation="gelu",
        transforms=None,
        fv_input=True,
        num_groups=1,
        dropout_prob=None,
        gain=1.0
    ):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.in_shape = n_features
        self.out_shape = out_shape

        layers: List[nn.Module] = [nn.Linear(np.prod(self.in_shape), hidden_channels)]
        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))
        for _ in range(hidden_layers - 1):
            layers.append(switchable_activation(activation=activation, num_groups=1))
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(switchable_activation(activation=activation, num_groups=1))
        layers.append(nn.Linear(hidden_channels, np.prod(self.out_shape)))
        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(lambda m: self.init_weights(m, gain=gain))

    @staticmethod
    def init_weights(m,gain=1.0):
        """Initialize weights of the MLP."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                fan_in = m.weight.size(1)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(gain)

    def forward(self, inputs: torch.Tensor):
        """Forward pass of baseline MLP."""
        return self.mlp(inputs)
