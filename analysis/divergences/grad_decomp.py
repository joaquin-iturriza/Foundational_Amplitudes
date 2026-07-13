#!/usr/bin/env python
"""Decompose the beta=1 NLL trunk gradient into its mu-term and sigma-term parts.

Why: het_mu_only (sigma gets no gradient) fits mu to 1.0e-4; the SAME net under the full
beta=1 NLL only reaches 8.6e-2. Loss math and model wiring are both verified correct, so
the suspect is sigma's gradient competing with mu's in the SHARED trunk. An ~860x loss to
MSE would be bizarre in a normal regression setting -- but amplitudes are DETERMINISTIC
(no aleatoric noise), so the Gaussian likelihood is degenerate: the true sigma is 0, the NLL
is unbounded below (sigma->0 => log sigma -> -inf), and the sigma-term's gradient never
decays to zero the way it does when it converges onto a real noise floor.

At beta=1 the per-event loss splits EXACTLY into two additive pieces:
    elem = (y-mu)^2 / 2                      <- the mu term (identical to MSE up to 1/2)
         + detach(sigma^2) * log(sigma)      <- the sigma term
This script evaluates BOTH at a well-fit-mu checkpoint and reports the norm of each one's
gradient w.r.t. the TRUNK parameters (everything except the readout). If ||g_sigma|| >>
||g_mu||, the NLL provably drags the trunk off the good mu solution -> mechanism confirmed.
Also reports the cosine between the full-NLL trunk gradient and the pure-MSE one. GPU.
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WT)
from experiment import AmplitudeExperiment  # noqa
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
from extract_preds import load_finetuned_state  # noqa


def build(run_dir, ckpt):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.ema = False; cfg.count_flops = False
        cfg.run_dir = os.path.join(REPO, "runs", "_grad_tmp")
        if cfg.get("fine_tune") is not None:
            cfg.fine_tune.pretrained_path = None
        cfg.training.heterosc_mu_only = False        # we want the FULL NLL here
        cfg.model.net.loss = "HETEROSC"
        cfg.training.loss = "HETEROSC"
    sys.path.insert(0, WT)
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    exp.model.load_state_dict(load_finetuned_state(os.path.join(run_dir, "models", ckpt))["model"])
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    return exp


def trunk_params(model):
    # everything EXCEPT the readout rows (linear_out) = the shared trunk
    return [p for n, p in model.named_parameters()
            if p.requires_grad and "linear_out" not in n]


def gnorm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = torch.cat([x.reshape(-1) for x in g if x is not None])
    return flat


def main():
    run_dir = os.path.join(WT, "runs/heterosc_bisect/het_mu_only")   # mu-MSE ~1.0e-4
    exp = build(run_dir, "model_run0_best.pt")
    tp = trunk_params(exp.model)

    data = next(iter(exp.train_loader))
    particles, y, tokens, order_labels, ptr, process_ids = data
    dev = exp.device
    particles, y = particles.to(dev), y.to(dev)
    tokens, order_labels = tokens.to(dev), order_labels.to(dev)
    ptr, process_ids = ptr.to(dev), process_ids.to(dev)

    for p in exp.model.parameters():
        p.requires_grad_(True)
    out = exp.model(particles, tokens, mean=exp.mom_mean[0], std=exp.mom_std[0],
                    ptr=ptr, order_labels=order_labels, process_ids=process_ids)
    mu    = out[..., :1]
    sigma = torch.clamp(out[..., -1:], min=1e-15, max=1e5)

    resid = (y - mu)
    # the two ADDITIVE pieces of the beta=1 per-event loss
    L_mu    = ((resid ** 2) / 2).mean()
    L_sigma = ((sigma.detach() ** 2) * torch.log(sigma)).mean()
    L_mse   = (resid ** 2).mean()

    g_mu    = gnorm(L_mu,    tp)
    g_sigma = gnorm(L_sigma, tp)
    g_mse   = gnorm(L_mse,   tp)
    g_full  = g_mu + g_sigma          # beta=1 NLL trunk gradient (linearity of grad)

    cos = torch.nn.functional.cosine_similarity(g_full, g_mse, dim=0).item()
    print("=== at the WELL-FIT mu checkpoint (het_mu_only, mu-MSE ~1.0e-4) ===")
    print(f"  residual RMS       = {resid.pow(2).mean().sqrt().item():.4g}")
    print(f"  sigma  median      = {sigma.median().item():.4g}   mean = {sigma.mean().item():.4g}")
    print()
    print(f"  ||g_mu||    (trunk, mu term)    = {g_mu.norm().item():.4g}")
    print(f"  ||g_sigma|| (trunk, sigma term) = {g_sigma.norm().item():.4g}")
    print(f"  ratio ||g_sigma|| / ||g_mu||    = {(g_sigma.norm()/g_mu.norm()).item():.4g}")
    print()
    print(f"  cos( g_full(NLL) , g_MSE )      = {cos:.4f}")
    print("  -> if the ratio >> 1 and cos ~ 0, the NLL trunk update is dominated by sigma")
    print("     and is nearly orthogonal to the direction that fits mu.")


if __name__ == "__main__":
    main()
