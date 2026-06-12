"""Closed-loop check of the significance-based plateau detector.

Drives the REAL AmplitudeExperiment._compute_sampler_weights / _alpha_and_se /
_robust_slope against synthetic per-process loss curves, with the sampling weights
feeding back into how fast each dataset accrues compute (exactly the loop that bites
in training).  No GPU / data needed — only the project env for the import.

Run on Jean Zay:
    cd $WORK/Foundational_Amplitudes        # /lustre/.../Foundational_Amplitudes
    python test_sampler_probe.py

Three synthetic datasets exercise the cases that matter:
  * fast            : steep, low floor, never stops improving -> must STAY 'scaling'
                      with the largest share. (This is the case that collapsed to
                      'plateaued' before the robustness fixes.)
  * true_plateau    : scales early then floors at a high irreducible loss; extra data
                      does NOT lower it -> probe should confirm 'plateaued'.
  * stalled_recover : scales, then a transient stall (loss flat over a band of its own
                      compute), then resumes -> the probe/re-probe should push it
                      through the stall and return it to 'scaling', not abandon it.
"""
import types
import numpy as np
from omegaconf import OmegaConf

from experiment import AmplitudeExperiment

# ── bare instance (skip __init__) + just the attributes the methods read ──────
exp = AmplitudeExperiment.__new__(AmplitudeExperiment)
exp.cfg = OmegaConf.create({"training": {
    "sampler_alpha_min": 0.05,
    "sampler_sig_k": 2.0,
    "sampler_warmup_vals": 10,
    "sampler_probe_max": 15,
    "sampler_reprobe_every": 20,
    "sampler_probe_boost": 1.0,
    "sampler_min_alpha_frac": 0.25,
    "sampler_alpha_window": 30,
}})
exp.train_sampler = types.SimpleNamespace(compute_snapshots=[])  # list[{proc_idx: n_samples}]
exp._proc_ema_losses = {}

names = ["fast", "true_plateau", "stalled_recover"]
truth = {
    #               floor    coeff  alpha  onset-scale  [stall band in own samples]
    "fast":            dict(E=0.003, C=1.0, alpha=0.80, scale=300),
    # hard flatline: scales until 8000 of its own samples, then frozen (dead) —
    # an open-ended stall (never resumes).  Must end 'plateaued' and stay there.
    "true_plateau":    dict(E=0.05,  C=1.0, alpha=0.70, scale=300, stall=(8000, 10**12)),
    "stalled_recover": dict(E=0.01,  C=1.0, alpha=0.60, scale=300, stall=(2500, 6500)),
}

def true_loss(name, c):
    t = truth[name]
    c_eff = c
    stall = t.get("stall")
    if stall:                       # freeze progress across [lo, hi], then resume
        lo, hi = stall
        if c <= lo:      c_eff = c
        elif c <= hi:    c_eff = lo
        else:            c_eff = c - (hi - lo)
    # L(c) = E + C (1 + c_eff/scale)^(-alpha): power law that floors at E
    return t["E"] + t["C"] * (1.0 + c_eff / t["scale"]) ** (-t["alpha"])

batch = 512
ema_decay = 0.6
ema = {}
compute = {n: 0 for n in names}
rng = np.random.default_rng(0)

# track status transitions for a compact end-of-run summary
prev_status = {n: "scaling" for n in names}
transitions = {n: [] for n in names}

print(f"{'step':>4} | " + " | ".join(f"{n:>22}" for n in names))
for step in range(200):
    exp.train_sampler.compute_snapshots.append({i: compute[n] for i, n in enumerate(names)})
    for n in names:
        L = true_loss(n, compute[n]) * (1.0 + 0.02 * rng.standard_normal())
        ema[n] = L if n not in ema else ema_decay * ema[n] + (1.0 - ema_decay) * L
        exp._proc_ema_losses.setdefault(n, []).append(ema[n])

    weights, status, alphas = exp._compute_sampler_weights(names)
    tot = sum(weights)
    for i, n in enumerate(names):
        compute[n] += int(batch * weights[i] / tot)
        if status[n] != prev_status[n]:
            transitions[n].append((step, prev_status[n], status[n]))
            prev_status[n] = status[n]

    if step % 20 == 0 or step in (10, 13):
        cells = [f"a={alphas[n]:.2f} {status[n][:4]} w={weights[i] / tot:.2f}"
                 for i, n in enumerate(names)]
        print(f"{step:>4} | " + " | ".join(f"{c:>22}" for c in cells))

sh = sum(compute.values())
print("\nfinal status :", status)
print("compute share:", {n: round(compute[n] / sh, 3) for n in names})
print("\ntransitions (step: from->to):")
for n in names:
    seq = ", ".join(f"{s}:{a[:4]}->{b[:4]}" for s, a, b in transitions[n]) or "(none)"
    print(f"  {n:>16}: {seq}")
print("\nEXPECT: fast stays 'scaling' (largest share, no false plateau); "
      "true_plateau ends 'plateaued';\n        stalled_recover dips into probe/plateau "
      "during the stall but returns to 'scaling' after it resumes.")
