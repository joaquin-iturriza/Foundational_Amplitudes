#!/usr/bin/env python
"""Best-vs-last mu-MSE for the DETACHED-sigma uug finetune (ft_uug_het_detach).

Closes the loop on the bisection. bisect_head.sh showed a 2-ch HETEROSC net trained with
plain MSE on mu reaches 1.0e-4 (== the 1-ch MSE reference), so the MODEL is fine and the
only thing that wrecks mu is sigma's gradient into the shared trunk. But the detach run
(which removes exactly that gradient) reported 0.031, not ~1e-4 — seemingly contradicting it.

Suspected confound: HETEROSC checkpoint selection uses val_loss_no_reg = the beta-NLL
(base_experiment.py:866), NOT mu-MSE. So the detach run's mu may have trained fine and then
been REPORTED from a beta-NLL-selected checkpoint with poor mu. If last << best in mu-MSE,
that is confirmed and there are two independent bugs: (1) sigma's trunk gradient wrecks mu,
(2) beta-NLL checkpoint selection then hides/misreports it. GPU.
"""
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WT)
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))

from diag_ckpt_mu import eval_ckpt  # noqa


def main():
    d = os.path.join(WT, "runs/heterosc_foundation/ft_uug_het_detach")
    for ck in ["model_run0_best.pt", "model_run0.pt"]:
        r = eval_ckpt(d, ck)
        sm = r.get("sigma_med", float("nan"))
        print(f"  DETACH {ck:22s} mu-MSE={r['mse']:.4g}  mu-MAE={r['mae']:.4g}  sigma_med={sm:.3g}",
              flush=True)
    print("\nreference: het_mu_only (sigma no grad) = 9.99e-5 | mse_ref (1-ch MSE) = 1.11e-4")


if __name__ == "__main__":
    main()
