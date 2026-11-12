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


        # the network's dtype (float32, or float64 for a training.dtype=float64 run): the local momenta keep the
        # precision the frames were computed in only when the network runs in it
        features_local = features_local.to(particle_type.dtype)
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
        layer_norm: bool = True,
        parametrization: str = "mup",
        detach_sigma_backbone: bool = False,
        sigma_after_pool: bool = True,
        causal_mask: bool = False,
        causal_frames: str = "prefix",
        causal_frames_min: int = 4,
        bayesian_last_layer: bool = False,
        bnn_prior_prec: float = 1.0,
        bnn_logsig2_init: float = -9.0,
        sign_channel: bool = False,
        sign_branch: bool = False,
        sign_branch_blocks: int = 1,
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
        # sign_channel (HETEROSC with training.sign_head; set by experiment.init_data): one more readout row, the sign
        # logit, between mu and sigma: rows [mu (out_shape), sign (1), sigma (out_shape)], so the experiment's splits
        # (sigma = the last out_shape channels, then the sign = the last remaining one) and the sigma stage (only the
        # last out_shape rows train) read it by role. The net is built so that every parameter but the sign row, and
        # the global RNG after construction, equal those of the same model without it (see _build_net).
        self.sign_channel = bool(sign_channel)
        assert not self.sign_channel or loss == "HETEROSC", "sign_channel is the HETEROSC readout's sign row"
        # rows read from the live (non-detached) features: mu, and the sign logit when present
        self.n_main = out_channels + (1 if self.sign_channel else 0)
        # sign_branch: the sign logit (readout row out_channels) is computed by a copy of the last transformer block on
        # detached features and frames (models/transformer_lloca_mup.py MuPTransformer.sign_block), so the sign loss
        # never reaches the shared trunk nor its clipped update (plan catalog_coverage step 73); sign_branch_blocks k:
        # a copy of the last k blocks fed block num_blocks - k's input detached (step 80)
        self.sign_branch = bool(sign_branch)
        assert not self.sign_branch or (self.sign_channel and loss == "HETEROSC" and sigma_after_pool
                                        and not detach_sigma_backbone and not bayesian_last_layer), \
            "sign_branch needs HETEROSC + training.sign_head (sign_channel) + sigma_after_pool, no detach_sigma_backbone " \
            "nor BNN readout"

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
        def _build_net(n_out):
            return MuPTransformer(
                in_channels,
                attn_reps,
                n_out,
                num_blocks,
                num_heads,
                dropout_prob=dropout_prob,
                checkpoint_blocks=checkpoint_blocks,
                parametrization=parametrization,
                **({"sign_branch": True, "sign_index": out_channels, "sign_branch_blocks": int(sign_branch_blocks)}
                   if self.sign_branch else {}),
                # BNN last layer (draft Sec. 5, fig:bnn): net.linear_out becomes a variational μP readout; the KL term
                # enters the training loss in experiment._reg_term, evaluation uses the posterior-mean weights
                **({"bayesian_readout": True, "bnn_prior_prec": bnn_prior_prec, "bnn_logsig2_init": bnn_logsig2_init}
                   if bayesian_last_layer else {}),
            )

        if not self.sign_channel:
            self.net = _build_net(net_out_channels)
        else:
            # The reference build (no sign row) draws the global RNG exactly as the het model without the head does;
            # the (2k+1)-row build, made from the same RNG state, supplies the module (its μP infshapes) and the sign
            # row; every other parameter and buffer is copied from the reference, and the RNG is left where the
            # reference build left it, so the backbone, the mu and sigma rows and everything initialised after this
            # (the wrapper's encoders) are bit-equal to the het model's at the same seed (tests/test_het_sign_head.py).
            k = out_channels
            rng = torch.random.get_rng_state()
            ref = _build_net(net_out_channels)
            rng_after = torch.random.get_rng_state()
            torch.random.set_rng_state(rng)
            self.net = _build_net(net_out_channels + 1)
            torch.random.set_rng_state(rng_after)
            ref_sd = ref.state_dict()
            with torch.no_grad():
                for name, t in self.net.state_dict(keep_vars=True).items():
                    r = ref_sd[name]
                    if t.shape == r.shape:
                        t.copy_(r)
                    else:                       # a readout tensor: rows [mu (k), sign (1), sigma (k)]
                        assert name.startswith("linear_out.") and t.shape[0] == r.shape[0] + 1, (name, t.shape, r.shape)
                        t[:k].copy_(r[:k])
                        t[k + 1:].copy_(r[k:])
            del ref
        if layer_norm is not True:
            # model.net.layer_norm=false: the blocks' pre-norms become identities (kit_benchmark: does normalisation
            # stop the transformer extrapolating its out-of-range log inputs, as a ReLU MLP does?); =rezero also gates
            # each residual branch by a learned scalar starting at 0 (ReZero, arXiv:2003.04887), which trains a norm-free
            # stack stably (false diverged at the LN run's lr, FA-ccin2p3-bba9a61e)
            assert layer_norm in (False, "rezero"), f"model.net.layer_norm={layer_norm!r}: true, false or rezero"
            from models.transformer_lloca_mup import BaselineTransformerBlock
            for m in self.net.modules():
                if isinstance(m, BaselineTransformerBlock):
                    m.norm1, m.norm2 = nn.Identity(), nn.Identity()
                    if layer_norm == "rezero":
                        m.gate1, m.gate2 = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))

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
        if getattr(self, "scalars_only", False):   # model.input_scalars_only: no momenta, no frames (wrappers.py)
            frames = IdentityFrames()(fourmomenta)
            features_local = torch.zeros_like(fourmomenta)
        else:
            frames = self.framesnet(
                fourmomenta, scalars=particle_type, ptr=ptr, return_tracker=False, **frames_kwargs
            )
            fourmomenta_local = self.trafo_fourmomenta(fourmomenta, frames)
            features_local = (fourmomenta_local - mean) / std
        if getattr(self, "local_log_eps", 0.0):   # model.input_local_log_eps: log-compressed local components (invariant)
            features_local = torch.asinh(features_local / self.local_log_eps) / float(np.arcsinh(1 / self.local_log_eps))

        # the network's dtype (float32, or float64 for a training.dtype=float64 run): the local momenta keep the
        # precision the frames were computed in only when the network runs in it
        features_local = features_local.to(particle_type.dtype)
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
                mu = self.net.linear_out(h)[..., : self.n_main]          # mu (and the sign logit)
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
            mu = self.net.linear_out(h)[..., : self.n_main]
            sigma_raw = self.net.linear_out(h.detach())[..., -self.out_shape:]
            sigma = torch.clamp(F.softplus(sigma_raw), min=1e-6)
            return torch.cat([mu, sigma], dim=-1)
        out = self.net(features, frames, ptr=ptr, seq_lens=seq_lens, causal=self.causal_mask, pair_ctx=pair_ctx, token_add=token_add)
        if self.loss == "HETEROSC":
            # softplus the sigma channels for positivity (floored), matching MuMLP.
            mu = out[..., : self.n_main]
            sigma = torch.clamp(F.softplus(out[..., -self.out_shape:]), min=1e-6)
            out = torch.cat([mu, sigma], dim=-1)
        return out
