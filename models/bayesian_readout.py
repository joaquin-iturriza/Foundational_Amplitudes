"""Bayesian (variational) μP readout: the "BNN last layer" of the multiplicity draft (Sec. 5, fig:bnn).

The layer is the Heidelberg ``VBLinear`` (heidelberg-hepml/jetgpt-splittings ``src/networks/vblinear.py``,
the autoregressive-transformer code of Butter:2024zbd whose ``Transformer.linear_out`` it replaces when
``bayesian=True``; the same class in heidelberg-hepml/arlo for the amplitude BNNs of Bahl:2024gyt), carried
over line for line except for the μP output scaling:

* posterior ``q(w) = N(mu_w, exp(logsig2_w))`` per weight, deterministic bias;
* prior ``N(0, 1/prior_prec)`` per weight, ``prior_prec = 1``;
* init ``mu_w ~ N(0, 1/fan_in)``, ``logsig2_w ~ N(-9, 0.001)``, bias 0;
* ``log sigma^2`` clamped to ``[-11, 11]`` in the forward and in the KL (jetgpt-splittings; arlo uses -30);
* training forward: local reparametrization (Kingma et al. arXiv:1506.02557), mean ``x W_mu^T + b`` plus
  ``sqrt(x^2 exp(logsig2)^T + 1e-8)`` times a standard normal per output;
* ``kl()``: the closed-form Gaussian KL summed over the weights; the caller divides by the number of training
  events (jetgpt-splittings ``JetGPT.batch_loss``: ``loss += KL() / traindata_size``; arlo ``kl / n_traindata``);
* evaluation: the posterior-mean weights (``map``; arlo's ``pred_batch`` and ``enable_map``). For a last layer
  this is the exact mean prediction: the output is linear in the weights and the features do not depend on
  them, so ``E_q[x W^T + b] = x mu_w^T + b`` (and the per-event mean over particles is linear too). Sampled
  weights (``sample_eval``) reproduce the reference's ``nsamples_BNN`` evaluations for a spread.

μP: the class is a ``mup.MuReadout`` whose ``weight`` *is* ``mu_w``, so every μP hook treats it as the readout it
replaces. ``set_base_shapes`` gives ``weight`` and ``logsig2_w`` (same ``(out, width)`` shape) the infshape of a
readout (finite fan-out, infinite fan-in: vector-like, so MuAdam leaves their learning rate unscaled) and calls
``_rescale_parameters`` (``weight``, bias times ``sqrt(width_mult)``: the reference ``N(0, 1/fan_in)`` init at the
base width, constant variance across widths). The forward divides the input by ``width_mult`` (times
``output_mult``) exactly as ``MuReadout.forward`` does, before both the mean and the variance paths, so a sampled
weight ``w ~ q`` acts as ``MuReadout`` with weight ``w``. ``logsig2_w`` and the prior live in the same stored
(constant-variance) units as ``weight`` and are not rescaled; at the base width (``width_mult = 1``) the layer is
the reference ``VBLinear``.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from mup import MuReadout


class MuBayesianReadout(MuReadout):
    LOGSIG2_CLAMP = (-11.0, 11.0)

    def __init__(self, in_features, out_features, bias=True, prior_prec=1.0, logsig2_init=-9.0, **kwargs):
        self.prior_prec = float(prior_prec)
        self.logsig2_init = float(logsig2_init)
        self.sample_eval = False   # True: evaluate with one weight draw held in self.random (reference nsamples_BNN)
        self.random = None
        super().__init__(in_features, out_features, bias=bias, **kwargs)   # reset_parameters: weight, bias
        self.logsig2_w = nn.Parameter(torch.empty(out_features, in_features))
        self.logsig2_w.data.normal_(self.logsig2_init, 0.001)

    def reset_parameters(self) -> None:
        # VBLinear.reset_parameters
        self.weight.data.normal_(0, 1.0 / math.sqrt(self.weight.size(1)))
        if self.bias is not None:
            self.bias.data.zero_()
        if "logsig2_w" in self._parameters:   # absent during Linear.__init__, created right after it
            self.logsig2_w.data.normal_(self.logsig2_init, 0.001)

    def _logsig2(self):
        return self.logsig2_w.clamp(*self.LOGSIG2_CLAMP)

    def kl(self):
        """KL(q || prior) summed over the weights (VBLinear.KL)."""
        logsig2 = self._logsig2()
        return 0.5 * (self.prior_prec * (self.weight.pow(2) + logsig2.exp())
                      - logsig2 - 1 - math.log(self.prior_prec)).sum()

    def reset_random(self):
        self.random = None

    def forward(self, x):
        x = self.output_mult * x / self.width_mult()   # MuReadout.forward's μP input scaling
        if self.training:
            mu_out = F.linear(x, self.weight, self.bias)
            var_out = F.linear(x.pow(2), self._logsig2().exp()) + 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)
        if not self.sample_eval:
            return F.linear(x, self.weight, self.bias)
        if self.random is None:
            self.random = torch.randn_like(self.logsig2_w)
        return F.linear(x, self.weight + self._logsig2().exp().sqrt() * self.random, self.bias)


def bayesian_readouts(model):
    return [m for m in model.modules() if isinstance(m, MuBayesianReadout)]
