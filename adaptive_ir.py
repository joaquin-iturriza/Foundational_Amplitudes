"""Online adaptive IR-bin sampling for the ee->uug singularity study.

Estimates the optimal training-sample density OVER phase space *during* training,
instead of scanning a fixed family of static densities offline.

Setup: one dense pool (antenna, covers every y_min decade). Bin events by y_min into
log-decade bins. The sampler draws bins with an adaptable weight; each validation we
read the per-bin loss and push the weights toward the variance-optimal importance
proposal for a chosen test weighting Q_b.

Unbiased mode (default): the training objective is fixed to the Q_b-weighted mean loss
    J = Σ_b Q_b E_b ,     E_b = mean per-event loss in bin b ,   Σ_b Q_b = 1 .
Sampling bin b with probability π_b and reweighting each event's loss by Q_b/π_b makes
the batch loss an UNBIASED estimator of J for any π (proof: E_π[(Q_b/π_b)ℓ] = Σ_b π_b·
(Q_b/π_b)E_b = Σ_b Q_b E_b), and the mean correction weight is Σ_b Q_b = 1. So changing
π only changes the gradient VARIANCE, not the optimum — and the variance-minimizing
proposal is π*_b ∝ Q_b·√(E[‖∇ℓ‖²|b]).  We proxy the per-bin gradient scale by the bin
loss (for MSE, ‖∇ℓ‖ ∝ |residual|·‖∂f/∂θ‖, so the per-bin RMS gradient ∝ √E_b up to a
slowly-varying feature scale), giving

    π_b  ∝  Q_b · E_b^{α/2}         (α=1 → the √E_b variance-optimal proxy).

The converged π_b is then the online ESTIMATE of the optimal sampling density, directly
comparable to the offline mixture-fraction sweep. Q_b default = log-flat (1 per decade).
"""
import numpy as np


def assign_ir_bins(y_min, edges):
    """Map y_min -> bin index in [0, len(edges)-1). edges: increasing (n_bins+1,)."""
    b = np.digitize(np.asarray(y_min), np.asarray(edges)) - 1
    return np.clip(b, 0, len(edges) - 2).astype(np.int64)


class IRWeightController:
    """EMA per-bin loss -> (sampling weights, per-event loss-correction factors).

    target : (n_bins,) test weighting Q_b (renormalized to sum 1). None -> log-flat.
    alpha  : exponent; π_b ∝ Q_b·E_b^{α/2}. α=1 is the √E variance-optimal proxy;
             α=0 recovers static π_b=Q_b (no adaptation).
    ema    : EMA decay for the per-bin loss estimate E_b.
    """

    def __init__(self, n_bins, target=None, alpha=1.0, ema=0.9, floor=1e-12):
        self.n_bins = int(n_bins)
        t = np.ones(n_bins) if target is None else np.asarray(target, float)
        self.target = t / t.sum()
        self.alpha = float(alpha)
        self.ema = float(ema)
        self.floor = float(floor)
        self.L = None                       # EMA per-bin loss E_b
        self.pi = self.target.copy()        # current sampling distribution

    def update_loss(self, per_bin_loss, present):
        """Fold a fresh per-bin mean loss into the EMA (only for bins with events)."""
        pbl = np.asarray(per_bin_loss, float)
        present = np.asarray(present, bool)
        if self.L is None:
            self.L = np.where(present, np.clip(pbl, self.floor, None), self.floor)
        else:
            upd = self.ema * self.L + (1.0 - self.ema) * np.clip(pbl, self.floor, None)
            self.L = np.where(present, upd, self.L)
        return self.L

    def weights(self, present=None):
        """Variance-optimal proposal proxy π_b ∝ Q_b·E_b^{α/2}, restricted to present bins."""
        present = np.ones(self.n_bins, bool) if present is None else np.asarray(present, bool)
        L = self.L if self.L is not None else np.ones(self.n_bins)
        scale = np.clip(L, self.floor, None) ** (self.alpha / 2.0)
        w = self.target * scale * present
        if w.sum() <= 0:
            w = self.target * present
        self.pi = w / w.sum()
        return self.pi

    def loss_correction(self, bin_ids):
        """Per-event unbiased-IS weight Q_b/π_b (mean over the sampled dist = Σ Q_b = 1)."""
        c = self.target / np.clip(self.pi, self.floor, None)
        return c[np.asarray(bin_ids)]
