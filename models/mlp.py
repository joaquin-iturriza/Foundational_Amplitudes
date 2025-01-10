from typing import List

import numpy as np
import torch
from torch import nn

from .activation import switchable_activation
from transformations import *


class MLP(nn.Module):
    """A simple baseline MLP.

    Flattens all dimensions except batch and uses GELU nonlinearities.
    """

    def __init__(
        self,
        type_token_list,
        hidden_channels,
        hidden_layers,
        activation="gelu",
        transforms=None,
        fv_input=True,
        num_groups=1,
        dropout_prob=None,
    ):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        n_particles = len(type_token_list)
        if fv_input:
            self.in_shape = 4*n_particles
        else:
            self.in_shape = 0
        if transforms:
            self.trafo_fns = [t['fn'] for t in transforms]
            for t in transforms:
                self.in_shape += t['out_dim']
            self.transforms = transforms
        else:
            self.in_shape = 4*len(type_token_list)
        self.out_shape = 1

        layers: List[nn.Module] = [nn.Linear(np.prod(in_shape), hidden_channels)]
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

    def forward(self, inputs: torch.Tensor):
        """Forward pass of baseline MLP."""
        return self.mlp(inputs)
