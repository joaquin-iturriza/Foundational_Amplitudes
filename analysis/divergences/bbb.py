"""Full-network Bayes-by-backprop (BBB) for the muP LLoCa amplitude net.

A GENUINELY Bayesian network: a mean-field Gaussian posterior q(w)=N(mu_w, sigma_w^2) over EVERY
weight/bias (not MC-dropout, not a Laplace/last-layer approximation, not a deep ensemble). Trained
by maximising the ELBO = E_q[log p(y|x,w)] - KL(q||p) via the local reparameterisation trick
(Blundell et al. 2015, "Weight Uncertainty in Neural Networks"). The uncertainty that then steers
L2 online generation is EPISTEMIC -- the predictive spread of mu over posterior weight samples,
high exactly where the training set is sparse -- rather than the point heteroscedastic head's
learned aleatoric spread.

muP integration (the subtle part). We do NOT rebuild the architecture: we `variationalize()` an
ALREADY-BUILT, already-muP-finalised, optionally warm-started net. For each leaf Linear/MuReadout:

  * the existing `weight`/`bias` tensors are kept verbatim as the posterior MEAN. So muP's per-layer
    init, the MuAdam per-layer lr scaling on the means, AND a base22 warm-start (means = pretrained
    weights) are all preserved with zero disturbance -- the BBB net STARTS as the deterministic
    pretrained net and only has to learn the spread.
  * a sibling `weight_rho`/`bias_rho` (same shape) parameterises the posterior std via
    sigma = softplus(rho). Initialised so sigma_0 = sigma_rel * s_layer, where s_layer is the RMS of
    the muP-initialised mean weights -- i.e. the posterior std starts as a small (sigma_rel<<1)
    multiple of the layer's own muP weight scale (~1/sqrt(fan_in)), so the injected noise
    sigma*eps @ x keeps the muP activation-variance scaling across width.
  * the Gaussian PRIOR is N(0, s_layer^2) per layer (same muP scale), so the KL neither blows up nor
    vanishes with width. KL has the closed form for two diagonal Gaussians.

The forward samples w = mu + softplus(rho)*eps ONCE per call (standard BBB); set `.deterministic`
(mu only, no noise) for the mean prediction, or draw K stochastic forwards to get the epistemic
predictive std. rho params are registered AFTER muP finalisation and treated as standard-param
(width-independent) by MuAdam -- sigma is a small correction to the muP-scaled mean, so this is a
deliberate first-order choice, flagged for revisiting if width-scaling of the posterior matters.

CPU-testable: with rho -> -inf (sigma -> 0) a variationalized net reproduces the base net bitwise,
and total_kl() -> the mu^2/(2 s^2) prior term. See tests in test_bbb.py / harness --validate.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SOFTPLUS_INV_EPS = 1e-12


def _softplus_inv(y):
    """rho such that softplus(rho) = y, for y > 0. Stable: log(expm1(y))."""
    y = torch.clamp(y, min=SOFTPLUS_INV_EPS)
    return torch.log(torch.expm1(y))


def _copy_infshape(new_param, ref_param):
    """Copy muP's `infshape` from ref_param onto new_param (same shape). MuAdam asserts every
    parameter carries an infshape; the new rho params otherwise crash the optimizer build. Copying
    the sibling weight's infshape makes MuAdam scale rho with muP identically to that weight."""
    ish = getattr(ref_param, "infshape", None)
    if ish is not None:
        new_param.infshape = ish


class VariationalLinear(nn.Module):
    """Wraps a built nn.Linear/MuReadout: mean = the original weight/bias, plus a log-std rho each.

    Keeps a reference to the ORIGINAL module so any non-standard forward behaviour (e.g. MuReadout's
    output multiplier / width scaling) is reused -- we only swap in a SAMPLED weight via functional
    linear, then, for MuReadout, apply its output_mult/width_mult exactly as it would."""

    def __init__(self, lin: nn.Linear, sigma_rel: float = 0.05, prior_rel: float = 1.0,
                 rho_floor: float = 1e-6):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.has_bias = lin.bias is not None
        # posterior MEAN = the original parameters (verbatim -> muP init + warm-start preserved)
        self.weight = lin.weight
        self.bias = lin.bias if self.has_bias else None

        # per-layer muP scale: RMS of the initialised mean weight (~1/sqrt(fan_in) under muP).
        with torch.no_grad():
            s_w = float(lin.weight.detach().pow(2).mean().sqrt().clamp_min(rho_floor))
        self.register_buffer("prior_sigma_w", torch.tensor(prior_rel * s_w), persistent=True)
        self.weight_rho = nn.Parameter(
            _softplus_inv(torch.full_like(lin.weight, sigma_rel * s_w)))
        _copy_infshape(self.weight_rho, lin.weight)     # muP: rho scales like its weight (MuAdam needs it)
        if self.has_bias:
            with torch.no_grad():
                s_b = max(sigma_rel * s_w, rho_floor)   # bias has no fan-in; reuse the layer scale
            self.register_buffer("prior_sigma_b", torch.tensor(prior_rel * s_w), persistent=True)
            self.bias_rho = nn.Parameter(_softplus_inv(torch.full_like(lin.bias, s_b)))
            _copy_infshape(self.bias_rho, lin.bias)

        # MuReadout applies an output multiplier (output_mult / width_mult) after the linear. Detect
        # and reproduce it so the sampled-weight forward matches the deterministic path exactly.
        self._out_mult = None
        om = getattr(lin, "output_mult", None); wm = getattr(lin, "width_mult", None)
        if om is not None:
            try:
                self._out_mult = float(om) / float(wm() if callable(wm) else wm)
            except Exception:
                self._out_mult = None
        # Sampling policy: draw a posterior weight sample iff we are TRAINING (standard BBB) OR
        # `sample_in_eval` is forced on (for epistemic-std scoring in eval mode). Validation / held-out
        # eval run in eval mode with sample_in_eval=False -> the deterministic posterior MEAN, so
        # checkpoint selection and the reported metric are stable. `deterministic` hard-forces the mean
        # regardless of mode (bitwise-equivalence check against the pre-variationalized net).
        self.deterministic = False
        self.sample_in_eval = False

    def _sample(self, mu, rho):
        if self.deterministic or not (self.training or self.sample_in_eval):
            return mu
        sigma = F.softplus(rho)
        return mu + sigma * torch.randn_like(sigma)

    def forward(self, x):
        w = self._sample(self.weight, self.weight_rho)
        b = self._sample(self.bias, self.bias_rho) if self.has_bias else None
        out = F.linear(x, w, b)
        if self._out_mult is not None:
            out = out * self._out_mult
        return out

    def kl(self):
        """KL(q||prior) summed over this layer's weights (and bias), diagonal Gaussians."""
        kl = _kl_gaussian(self.weight, F.softplus(self.weight_rho), self.prior_sigma_w)
        if self.has_bias:
            kl = kl + _kl_gaussian(self.bias, F.softplus(self.bias_rho), self.prior_sigma_b)
        return kl


def _kl_gaussian(mu_q, sigma_q, sigma_p):
    """sum_i KL( N(mu_i, sig_qi^2) || N(0, sig_p^2) ) for a scalar prior std sigma_p."""
    sp2 = sigma_p ** 2
    return (0.5 * ((sigma_q ** 2 + mu_q ** 2) / sp2 - 1.0) - torch.log(sigma_q) + torch.log(sigma_p)).sum()


def variationalize(model: nn.Module, sigma_rel: float = 0.05, prior_rel: float = 1.0,
                   skip_substrings=("framesnet",)):
    """In-place: replace every leaf nn.Linear under `model` with a VariationalLinear (mean = the
    existing weights). Returns the list of VariationalLinear modules (for kl()/deterministic toggles).

    skip_substrings: module-name fragments to leave deterministic (default: the framesnet equivariant
    frames -- geometry, not the amplitude regressor; keep it a fixed feature map so the epistemic
    uncertainty is about the AMPLITUDE map, and to avoid perturbing the equivariance construction)."""
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if any(s in full for s in skip_substrings):
                continue
            vl = VariationalLinear(child, sigma_rel=sigma_rel, prior_rel=prior_rel)
            setattr(module, child_name, vl)
            replaced.append(vl)
    return replaced


def collect_variational(model: nn.Module):
    """All VariationalLinear submodules of `model`."""
    return [m for m in model.modules() if isinstance(m, VariationalLinear)]


def total_kl(model: nn.Module):
    """Sum of per-layer KL(q||prior) over all VariationalLinear layers (a scalar tensor)."""
    vls = collect_variational(model)
    if not vls:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack([vl.kl() for vl in vls]).sum()


def set_deterministic(model: nn.Module, flag: bool = True):
    """Hard-toggle mu-only (no weight noise) forward on every VariationalLinear, regardless of
    train/eval mode -- for the mean prediction or a bitwise-equivalence check against the base net."""
    for vl in collect_variational(model):
        vl.deterministic = flag


def set_sample_in_eval(model: nn.Module, flag: bool = True):
    """Force posterior sampling even in eval() mode (for epistemic-std scoring). Training-mode
    sampling is unaffected; remember to turn this back off after scoring."""
    for vl in collect_variational(model):
        vl.sample_in_eval = flag
