import torch
from torch import nn
import numpy as np

from models import MLP


class DSI(nn.Module):
    """
    Deep set + invariant approach
    This network combines implicit bias from permutation invariance (deep sets)
    and Lorentz invariance (Lorentz inner products),
    but in a way that breaks both invariances when combining the two approaches

    There are two types of MLP networks:
    - prenet: MLP (one for each particle type) that processes fourmomenta,
      the results are combined in a deep set to form a permutation-invariant result
      This can be viewed as an optional preprocessing step
    - net: MLP that combines the deep set result with Lorentz invariants
      to extract the final result
    """

    def __init__(
        self,
        type_token_list,
        hidden_channels_prenet,
        hidden_layers_prenet,
        out_dim_prenet_sep,
        hidden_channels_net,
        hidden_layers_net,
        use_deepset=True,
        sum_deepset=True,
        use_invariants=True,
        incl_diag_invariants=True,
        use_only_invariants=False,
        activation="gelu",
        num_groups=1,
        dropout_prob=None,
    ):
        """
        Parameters
        ----------
        type_token_list : List[int]
            List of particles in the process, with an integer representing the particle type
            Example: [0,0,1,2,2] for q q > Z g g
        hidden_channels_prenet : int
        hidden_layers_prenet: int
        out_dim_prenet_sep : int
            Size of the latent space extract from the deep set
        hidden_channels_net : int
        hidden_layers_net : int
        use_deepset : bool
            whether to use the deep set part (affects prenet)
            We find the same results with use_deepset=False
        sum_deepset : bool
            whether to sum the deep set embeddings or concatenate them
            Permutation invariance is broken anyways if use_invariants=True,
            so one can also decide to break it at an earlier stage
            We find slightly worse results with sum_deepset=False
        use_invariants : bool
            whether to use the invariants part (affects net)
            We find significantly worse results with use_invariants=False
        use_only_invariants : bool
            whether to use only the invariants as input
        dropout_prob : float
        """
        super().__init__()
        assert use_deepset or use_invariants
        self.use_deepset = use_deepset
        self.use_invariants = use_invariants
        self.incl_diag_invariants = incl_diag_invariants
        self.use_only_invariants = use_only_invariants
        self.sum_deepset = sum_deepset

        n = len(type_token_list)
        if self.use_deepset:
            assert (
                len(np.unique(type_token_list)) == max(type_token_list) + 1
            ), f"Invalid type_token_list={type_token_list}"
            self.type_token_list = type_token_list

            self.prenets = nn.ModuleList(
                [
                    MLP(
                        in_shape=4,
                        out_shape=out_dim_prenet_sep,
                        hidden_channels=hidden_channels_prenet,
                        hidden_layers=hidden_layers_prenet,
                        dropout_prob=dropout_prob,
                    )
                    for _ in range(max(type_token_list) + 1)
                ]
            )
            mlp_inputs = out_dim_prenet_sep * (
                max(type_token_list) + 1 if sum_deepset else len(type_token_list)
            )
        else:
            mlp_inputs = n * 4

        if self.incl_diag_invariants:
            n_invariants = n * (n + 1) // 2
        else:
            n_invariants = n * (n - 1) // 2
        if self.use_invariants:
            if self.use_only_invariants:
                mlp_inputs = n_invariants
            else:
                mlp_inputs += n_invariants

        self.net = MLP(
            in_shape=mlp_inputs,
            out_shape=1,
            hidden_channels=hidden_channels_net,
            hidden_layers=hidden_layers_net,
            dropout_prob=dropout_prob,
            activation=activation,
            num_groups=num_groups,
        )

        # standardization parameters
        # (could evaluate them pre training,
        # but we have large batchsizes
        # so no big difference expected)
        self.register_buffer("inv_mean", torch.ones(1, 1, n_invariants))
        self.register_buffer("inv_std", torch.ones(1, 1, n_invariants))

    def _compute_invariants(self, particles, incl_diag_invariants=True):
        invariants = compute_invariants(
            particles, incl_diag_invariants=incl_diag_invariants)

        # standardize
        if torch.all(self.inv_mean == 1) or torch.all(self.inv_std == 1):
            self.inv_mean = invariants.mean(dim=-2, keepdim=True)
            self.inv_std = invariants.std(dim=-2, keepdim=True)
            # avoid division by zero
            self.inv_std[self.inv_std < 1e-5] = 1
        
        invariants = (invariants - self.inv_mean) / self.inv_std

        return invariants

    def forward(self, particles, type_token):

        # deep set preprocessing
        if self.use_deepset:
            assert len(type_token) == 1
            type_token = type_token[0]
            assert type_token.cpu().numpy().tolist() == self.type_token_list
            preprocessing = []
            for i in range(max(type_token) + 1):
                identical_particles = particles[..., type_token == i, :]
                embedding = self.prenets[i](identical_particles)
                embedding = (
                    embedding.sum(dim=-2, keepdim=True)
                    if self.sum_deepset
                    else embedding
                )
                preprocessing.append(embedding)
            preprocessing = torch.cat(preprocessing, dim=-2)
            preprocessing = preprocessing.view(*particles.shape[:-2], -1)
        else:
            preprocessing = particles.view(*particles.shape[:-2], -1)

        # invariants
        if self.use_invariants:
            invariants = self._compute_invariants(
                particles, incl_diag_invariants=self.incl_diag_invariants)
        else:
            invariants = torch.empty(
                *particles.shape[:-2],
                0,
                device=particles.device,
                dtype=particles.dtype,
            )

        # combine everything
        if self.use_only_invariants:
            latent_full = invariants
        else:
            latent_full = torch.cat((preprocessing, invariants), dim=-1)
        result = self.net(latent_full)
        return result
