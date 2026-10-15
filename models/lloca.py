from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.utils import get_edge_attr
from lloca.framesnet.nonequi_frames import IdentityFrames
from lloca.framesnet.equi_frames import LearnedPDFrames
from .equimlp import EquiMLP
from lloca.backbone.transformer import Transformer
from .transformer_lloca_mup import MuPTransformer
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
        freeze_framesnet: bool = False,
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
        if freeze_framesnet:
            for param in self.framesnet.parameters():
                param.requires_grad = False
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

class LLOCAMuPTransformer(nn.Module):
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
        dropout_prob: float = 0.0,
        freeze_framesnet: bool = False,
        checkpoint_blocks: bool = False,
        parametrization: str = "mup",
        detach_sigma_backbone: bool = False,
        sigma_after_pool: bool = True,
        causal_mask: bool = False,
        causal_frames: str = "prefix",
        causal_frames_min: int = 4,
        bayesian_last_layer: bool = False,
        bnn_prior_prec: float = 1.0,
        bnn_logsig2_init: float = -9.0,
        # token_size: int = 0,
    ):
        super().__init__()
        self.parametrization = parametrization
        # Autoregressive attention (paper draft, Sec. "Extrapolating to more legs"): within an event, particle i
        # attends to particles 1..i only. Under LLoCa the local frames are a second path between particles: with
        # causal_frames="prefix" the frames net of particle i reads particles 1..max(i, causal_frames_min) only, so the
        # outputs of a prefix of length >= causal_frames_min do not depend on the particles after it (tests/
        # test_causal_mask.py); "full" leaves the frames net on the whole event.
        self.causal_mask = bool(causal_mask)
        assert causal_frames in ("prefix", "full"), causal_frames
        self.causal_frames = causal_frames
        self.causal_frames_min = int(causal_frames_min)
        assert self.causal_frames_min >= 2, "a prefix frame needs other particles to point at"
        # See forward(): softplus must be applied ONCE to the POOLED per-event sigma logit, not
        # per-particle before pooling. False restores the old (range-compressing) behaviour for A/B.
        self.sigma_after_pool = sigma_after_pool
        # HETEROSC only: feed the sigma readout DETACHED backbone features, so the
        # backbone receives only the mean's (pure-MSE, beta=1) gradient and sigma is a
        # read-only calibration head. Decouples the (mu, sigma) multi-task interference
        # that otherwise floors mu where the mean is easy to fit (single-process
        # finetune). Same weights/shape -> warm-starts from a non-detached checkpoint.
        self.detach_sigma_backbone = detach_sigma_backbone
        # Heteroscedastic head: the net emits 2*out_shape channels (mean, sigma);
        # sigma is softplus-ed for positivity in forward. out_shape stays the real
        # target dim so experiment.py's split (last out_shape = sigma) is unchanged.
        self.loss = loss
        self.out_shape = out_channels
        net_out_channels = 2 * out_channels if loss == "HETEROSC" else out_channels

        def equivectors_constructor(n_vectors):
            return EquiMLP(
                n_vectors=n_vectors,
                #num_blocks=1,
                num_scalars=num_scalars,
                hidden_channels=128, # FIXED TO 128 FOR MUP
                num_layers_mlp=num_layers_mlp,
                fm_norm=True,
                layer_norm=True,
                nonlinearity="softmax",
                dropout_prob=dropout_prob,
            )

        self.framesnet = LearnedPDFrames(equivectors=equivectors_constructor)
        if freeze_framesnet:
            for param in self.framesnet.parameters():
                param.requires_grad = False
        #self.framesnet = IdentityFrames()
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))
        self.net = MuPTransformer(
            in_channels,
            attn_reps,
            net_out_channels,
            num_blocks,
            num_heads,
            dropout_prob=dropout_prob,
            checkpoint_blocks=checkpoint_blocks,
            parametrization=parametrization,
            # BNN last layer (draft Sec. 5, fig:bnn): net.linear_out becomes a variational μP readout; the KL term
            # enters the training loss in experiment._reg_term, evaluation uses the posterior-mean weights
            **({"bayesian_readout": True, "bnn_prior_prec": bnn_prior_prec, "bnn_logsig2_init": bnn_logsig2_init}
               if bayesian_last_layer else {}),
        )

    def forward(
        self,
        fourmomenta: torch.Tensor,
        particle_type: torch.Tensor,
        mean: float,
        std: float,
        ptr: torch.Tensor,
        seq_lens=None,
        pair_ctx=None,
        token_add=None,
    ):
        """Forward pass of the LLoCa network."""
        frames_kwargs = {}
        if self.causal_mask:
            assert pair_ctx is None, "causal_mask and the pairwise attention bias are not combined"
            if self.causal_frames == "prefix":
                frames_kwargs["causal_min_prefix"] = self.causal_frames_min
        frames = self.framesnet(
            fourmomenta, scalars=particle_type, ptr=ptr, return_tracker=False, **frames_kwargs
        )
        fourmomenta_local = self.trafo_fourmomenta(fourmomenta, frames)
        features_local = (fourmomenta_local - mean) / std


        # move everything to less safe dtype
        features_local = features_local.to(torch.float32)
        frames.to(torch.float32)
        features = torch.cat([features_local, particle_type], dim=-1)

        # Pass ptr so the transformer can build a block-diagonal attention mask,
        # restricting each event's particles to attend only within that event.
        # seq_lens (CPU per-event lengths, optional) lets the mask builder skip a
        # GPU→CPU sync; see build_block_diagonal_bias.
        # sigma_after_pool (default, CORRECT): emit sigma as a RAW per-particle logit and let the
        # wrapper apply softplus ONCE to the pooled, per-event value -> sigma = softplus(mean_i z_i).
        #
        # This matches the reference implementation (heidelberg-hepml/amplitude_DSI), where HETEROSC
        # is only ever wired into the MLP: there the readout is ALREADY per-event and softplus is
        # applied to that per-event scalar. There is no pooling in its sigma path.
        #
        # The old path (sigma_after_pool=False) applied softplus PER PARTICLE and then mean-pooled,
        # giving sigma = mean_i softplus(z_i). That compresses sigma's dynamic range: making sigma
        # small needs EVERY particle's logit very negative, while one large logit makes it big. Net
        # effect measured: sigma spans 23x while the true error spans 91x -> reliability slope ~1.5
        # (under-dispersed), sigma over-predicting small errors and under-predicting the hard tail by
        # ~2.75x, and a hard accuracy-vs-calibration trade-off. mu was never affected, because
        # pooling is linear and commutes with its readout.
        if self.loss == "HETEROSC" and self.sigma_after_pool:
            if self.detach_sigma_backbone:
                h = self.net(features, frames, ptr=ptr, seq_lens=seq_lens, causal=self.causal_mask,
                             pair_ctx=pair_ctx, token_add=token_add, return_features=True)
                mu = self.net.linear_out(h)[..., : self.out_shape]
                sig_logit = self.net.linear_out(h.detach())[..., -self.out_shape:]
                return torch.cat([mu, sig_logit], dim=-1)
            # raw logits; the wrapper pools then softpluses
            return self.net(features, frames, ptr=ptr, seq_lens=seq_lens, causal=self.causal_mask, pair_ctx=pair_ctx, token_add=token_add)

        if self.loss == "HETEROSC" and self.detach_sigma_backbone:
            # Apply linear_out twice: mu from live features (grad -> backbone), sigma
            # from detached features (linear_out sigma-rows still train, backbone does
            # not see sigma's gradient). See __init__ note.
            h = self.net(features, frames, ptr=ptr, seq_lens=seq_lens, causal=self.causal_mask,
                         pair_ctx=pair_ctx, token_add=token_add, return_features=True)
            mu = self.net.linear_out(h)[..., : self.out_shape]
            sigma_raw = self.net.linear_out(h.detach())[..., -self.out_shape:]
            sigma = torch.clamp(F.softplus(sigma_raw), min=1e-6)
            return torch.cat([mu, sigma], dim=-1)
        out = self.net(features, frames, ptr=ptr, seq_lens=seq_lens, causal=self.causal_mask, pair_ctx=pair_ctx, token_add=token_add)
        if self.loss == "HETEROSC":
            # softplus the sigma channels for positivity (floored), matching MuMLP.
            mu = out[..., : self.out_shape]
            sigma = torch.clamp(F.softplus(out[..., -self.out_shape:]), min=1e-6)
            out = torch.cat([mu, sigma], dim=-1)
        return out
