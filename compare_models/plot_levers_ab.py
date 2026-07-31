#!/usr/bin/env python
"""Fig. levers_offshell_ab -- the flagship off-shellness A/B (44 datasets, full data, 30k it).

Both arms come from compare_models/run_levers_ab.sh, which differs only in the four
internal-mass levers (mass_from_momenta, coupling_scalars, internal_mass_scalars,
offshell_per_event). Per-dataset losses are read from each arm's
plots_0/per_process_metrics.json.

Left : the ee->mumu Z-mass scan. Without a mass feature the model can only learn one effective
       M_Z, so the loss traces a U around the middle of the scanned range; with the
       off-shellness feature s_prop - M^2 it is flat and ~5 orders of magnitude lower.
Right: median loss per resonance family, showing the lever generalises beyond the s-channel Z.

Metric is `val_loss_no_reg` read at each run's best checkpoint -- the repo's comparison rule
(the regularized loss confounds the metric with the tuned lambda). See best_per_process for
why that is not the same as a per-process minimum. CPU only.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

#: (dataset-name prefix, legend/tick name). Order is the argument: the s-channel Z scan is the
#: probe the lever was designed for, the other three are the generalisation test.
FAMILIES = [
    ("ee_mumu__mz",       r"$Z$ ($s$-channel)"),
    ("ee_mumumumu__mz4l", r"$Z$ in $4\ell$"),
    ("ee_wwbb__mt",       r"top"),
    ("ee_mumutautau__mh", r"Higgs"),
]


def best_per_process(run_dir):
    """{dataset: val_loss_no_reg at the run's BEST checkpoint}.

    The repo's comparison rule is the minimum of the COMBINED `val_loss_no_reg` series --
    the checkpoint-selection metric, see compare_models/aggregate_scan_ab.py:best_val_no_reg.
    Taking a per-process min instead reports a per-process epoch oracle that no single saved
    model realises: in this A/B the off arm's global best is validation index 7 and the
    offshell arm's is 14, while the per-process argmins scatter over indices 0-11. It also
    biases the comparison, and towards the baseline: ee->mumu median for `off` reads 0.558
    under per-process min against 0.704 at its own best checkpoint, a 21% flattering.
    """
    with open(os.path.join(run_dir, "plots_0", "per_process_metrics.json")) as f:
        d = json.load(f)
    series = d.get("val_loss_no_reg") or d.get("val_loss") or []
    vals = [(i, v) for i, v in enumerate(series) if v is not None and np.isfinite(v)]
    if not vals:
        raise SystemExit(f"[plot_levers_ab] no val_loss_no_reg series in {run_dir}")
    best_i = min(vals, key=lambda t: t[1])[0]
    P = d["proc_val_losses_no_reg"]
    return {k: float(v[best_i]) for k, v in P.items() if len(v) > best_i}


def scanned_masses(recipe, prefix, pdg):
    """{dataset: scanned mass} for one scan family, read from the recipe that generated it.

    The mass is a property of the DATASET, not of the run, so it has to come from the recipe;
    reconstructing it from the dataset index would silently break if the scan grid changed.
    """
    txt = open(recipe).read()
    out = {}
    for name in re.findall(rf"name:\s*({re.escape(prefix)}\d+)", txt):
        # the physics block for this entry: masses: {<pdg>: <value>}
        seg = txt[txt.index(f"name: {name}"):]
        m = re.search(rf"masses:\s*\{{\s*{pdg}:\s*([0-9.eE+-]+)", seg[:800].replace("\n", " "))
        if m:
            out[name] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--ab_dir", default=os.path.join(here, "_levers_ab"))
    ap.add_argument("--recipe", default=os.path.join(REPO, "recipes", "scan_levers.yaml"))
    ap.add_argument("--out_base", default=os.path.join(here, "_levers_ab", "levers_offshell_ab"))
    args = ap.parse_args()

    arms = {a: best_per_process(os.path.join(args.ab_dir, a)) for a in ("off", "offshell")}
    style = {"off": (ps.C.vermillion, "o-", "no mass feature"),
             "offshell": (ps.C.blue, "s-", r"off-shellness $s_{\mathrm{prop}}-M^2$")}

    fig, (axL, axR) = ps.figure(ncols=2)

    # --- left: the Z-mass scan -------------------------------------------------
    mz = scanned_masses(args.recipe, "ee_mumu__mz", 23)
    names = sorted(mz, key=lambda k: mz[k])
    for arm, (col, mk, lab) in style.items():
        axL.plot([mz[n] for n in names], [arms[arm][n] for n in names], mk,
                 color=col, label=lab)
    axL.set_yscale("log")
    axL.set_xlabel(r"scanned $M_Z$ [GeV]")
    axL.set_ylabel(r"$\mathrm{val\ loss}_{\mathrm{no\ reg}}$")
    # Bound the range from the DATA, with a decade of headroom, rather than hardcoding it:
    # a fixed 1e-6 floor sat above a real 6.1e-7 point, which make_room happened to rescue
    # here but would silently clip on any rerun where make_room does not fire. (make_room
    # still runs: the process label counts as a mover even with the legend moved out.)
    _all = [v for arm in arms.values() for v in arm.values() if v > 0]
    axL.set_ylim(10 ** np.floor(np.log10(min(_all)) - 0.3),
                 10 ** np.ceil(np.log10(max(_all)) + 0.3))
    ps.process_label(axL, r"$e^+e^-\to\mu^+\mu^-$", loc="lower right")

    # --- right: median per resonance family ------------------------------------
    x = np.arange(len(FAMILIES))
    w = 0.38
    for i, (arm, (col, _, lab)) in enumerate(style.items()):
        med = [np.median([v for k, v in arms[arm].items() if k.startswith(pref)])
               for pref, _ in FAMILIES]
        axR.bar(x + (i - 0.5) * w, med, w, color=col, label=lab)
    axR.set_yscale("log")
    axR.set_xticks(x)
    axR.set_xticklabels([nm for _, nm in FAMILIES], rotation=20, ha="right")
    axR.set_ylabel(r"median $\mathrm{val\ loss}_{\mathrm{no\ reg}}$")
    # A bar on a log axis is drawn from the axis floor, so an unbounded floor makes the
    # s-channel bar a 10-decade slab whose length says nothing. Clip to just below the
    # smallest median so every bar's length is readable against the others.
    axR.set_ylim(10 ** np.floor(np.log10(min(_all)) - 0.3), 3e0)

    # One legend across the top: both panels plot the same two arms, and any in-panel legend
    # here sits on either the scan curves or the bars.
    ps.shared_legend(fig, axL, ncol=2)
    ps.save(fig, args.out_base)

    for pref, nm in FAMILIES:
        a = np.median([v for k, v in arms["off"].items() if k.startswith(pref)])
        b = np.median([v for k, v in arms["offshell"].items() if k.startswith(pref)])
        print(f"{nm:22s} off={a:.4g}  offshell={b:.4g}  ratio={a / b:.1f}x")


if __name__ == "__main__":
    main()
