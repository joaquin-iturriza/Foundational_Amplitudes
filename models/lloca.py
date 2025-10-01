from typing import List

import numpy as np
import torch
from torch import nn

from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.utils import build_edge_index_fully_connected, get_edge_attr
from lloca.framesnet.nonequi_frames import IdentityFrames
from lloca.framesnet.equi_frames import LearnedPDFrames
from lloca.equivectors.equimlp import EquiMLP
from lloca.backbone.transformer import Transformer
from lloca.utils.rand_transforms import rand_lorentz
from lloca.framesnet.frames import Frames

class LLOCATransformer(nn.Module):
    """LLoCa Transformer baseline"""

    def __init__(
        self,
        num_scalars,
        hidden_channels_mlp,
        num_layers_mlp,
        in_channels,
        attn_reps,
        out_channels,
        num_blocks,
        num_heads,
        loss='MSE',
    ):
        super().__init__()

        def equivectors_constructor(n_vectors):
            return EquiMLP(
                n_vectors=n_vectors,
                num_blocks=1,
                num_scalars=num_scalars,
                hidden_channels=hidden_channels_mlp,
                num_layers_mlp=num_layers_mlp,
                fm_norm=True, 
                layer_norm=True, 
                nonlinearity="softmax",
            )

        self.framesnet = LearnedPDFrames(equivectors=equivectors_constructor)
        #self.framesnet = IdentityFrames()
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))
        self.net = Transformer(
            in_channels,
            attn_reps,
            out_channels,
            num_blocks,
            num_heads,
        )

    def forward(
        self, 
        fourmomenta: torch.Tensor, 
        particle_type: torch.Tensor, 
        mean: float,
        std: float,
    ):          
        """Forward pass of the LLoCa network."""
        frames = self.framesnet(
            fourmomenta, scalars=particle_type, ptr=None, return_tracker=False
        )
        fourmomenta_local = self.trafo_fourmomenta(fourmomenta, frames)  
        features_local = (fourmomenta_local - mean) / std


        # move everything to less safe dtype
        features_local = features_local.to(torch.float32)
        frames.to(torch.float32)
        features = torch.cat([features_local, particle_type], dim=-1)
        
        return self.net(features, frames)
