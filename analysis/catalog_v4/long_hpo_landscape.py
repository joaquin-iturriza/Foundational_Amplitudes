#!/usr/bin/env python3
"""The HPO landscape of the two long-run DyHPO sweeps (plan catalog_coverage step 32; Joaquin 2026-11-04: 'show me
the HPO information of the long runs, to see how the landscape looks'): cov_v4_long_hpo8_cc_17056 (4x, 17056 steps)
and cov_v4_long_hpo8_cc_68224 (16x, 68224 steps), 8 trials each, single fidelity, the template's 7 HPs.

    python3 analysis/catalog_v4/long_hpo_landscape.py pull   one read-only `site run` on CC-IN2P3: every finished
                                                             trial's searched HPs from its config.yaml
                                                             -> data/long_hpo_landscape.json
    python3 analysis/catalog_v4/long_hpo_landscape.py figs   figs/long_hpo_landscape_hps.{png,pdf}: each trial's
                                                             objective against each HP, one panel per HP, both sweeps,
                                                             best marked; figs/long_hpo_landscape_curves.{png,pdf}:
                                                             each trial's validation curve over steps

The objective is the sweep's own (results/*.json val_loss: the main run's best val_loss_no_reg, het NLL, read in
data/long_sweeps_best.json by size_figs.py best); the curves are data/size_curves.json (size_figs.py pull, the
'Val loss (combined)' lines of each trial's out_0.log). A trial that was rerun (_r1, _r2) has one config: the HPs
are read from the directory size_curves.json read its curve from.
"""
import json, os, re, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
OUT = os.path.join(DATA, "long_hpo_landscape.json")
SWEEPS = {"17056": "cov_v4_long_hpo8_cc_17056", "68224": "cov_v4_long_hpo8_cc_68224"}
CONFIGS = {"17056": "sweep/sweep_config_ccin2p3_cov_v4_long_hpo_x4.yaml",
           "68224": "sweep/sweep_config_ccin2p3_cov_v4_long_hpo_x16.yaml"}
# search-space name -> (path in the trial's config.yaml, axis label, log axis)
HPS = {"training.lr": (("training", "lr"), "learning rate", True),
       "training.muon_weight_decay": (("training", "muon_weight_decay"), "Muon weight decay", True),
       "training.regularization_lambda": (("training", "regularization_lambda"), "regularization", True),
       "training.cosanneal_warmup_frac": (("training", "cosanneal_warmup_frac"), "warm-up fraction", False),
       "training.cosanneal_eta_min": (("training", "cosanneal_eta_min"), r"$\eta_{\min}$", True),
       "ema": (("ema",), "EMA", False),
       "training.ema_decay": (("training", "ema_decay"), "EMA decay", False)}

REMOTE = r'''
import json, sys, yaml
out = {}
for d in sys.argv[1:]:
    c = yaml.safe_load(open(d + "/config.yaml"))
    out[d] = {k: c["training"][k] for k in ("lr", "muon_weight_decay", "regularization_lambda", "cosanneal_warmup_frac",
                                             "cosanneal_eta_min", "ema_decay")}
    out[d]["ema"] = c.get("ema")
print("LHPO " + json.dumps(out))
'''


def _rows():
    """per sweep: [(trial key in size_curves.json, hp index, objective)] of every finished trial"""
    best = json.load(open(os.path.join(DATA, "long_sweeps_best.json")))
    tr = json.load(open(os.path.join(DATA, "size_curves.json")))["trials"]
    rows = {}
    for steps in SWEEPS:
        rows[steps] = []
        for t in best[steps]["trials"]:
            k = [x for x in tr if re.match(rf"{steps}_trial_{t['hp']}(_r\d+)?$", x)]
            if len(k) != 1:
                sys.exit(f"{steps} hp{t['hp']}: {len(k)} curves in size_curves.json")
            rows[steps].append((k[0], t["hp"], t["val_loss"]))
    return rows, tr


def pull():
    rows, tr = _rows()
    dirs = [f"runs/{tr[k]['sweep']}/{tr[k]['trial']}" for v in rows.values() for k, _, _ in v]
    cmd = (f"F=$(mktemp /tmp/zz_probe_lhpo_XXXXXX.py); cat > $F <<'PYEOF'\n{REMOTE}PYEOF\n"
           f"python3 $F {' '.join(dirs)}; rm -f $F")
    res = subprocess.run(["site", "--timeout", "600", "run", "ccin2p3", "FA", "--", cmd], capture_output=True,
                         text=True, timeout=700)
    line = [l for l in res.stdout.splitlines() if l.startswith("LHPO ")]
    if not line:
        sys.exit("pull: no LHPO line\n" + res.stdout[-3000:] + res.stderr[-2000:])
    hp = json.loads(line[-1][len("LHPO "):])
    out = {"source": "config.yaml of each finished trial on ccin2p3 (read-only site run); objective from "
                     "data/long_sweeps_best.json", "sweeps": {}}
    for steps, v in rows.items():
        out["sweeps"][steps] = {"sweep": SWEEPS[steps], "trials": [
            {"key": k, "hp": h, "val_loss": y, **hp[f"runs/{tr[k]['sweep']}/{tr[k]['trial']}"]} for k, h, y in v]}
    json.dump(out, open(OUT, "w"), indent=1)
    print({s: len(v["trials"]) for s, v in out["sweeps"].items()})


def figs():
    sys.path.insert(0, ROOT)
    import yaml
    import plot_style as ps
    d = json.load(open(OUT))
    tr = json.load(open(os.path.join(DATA, "size_curves.json")))["trials"]
    col = {"17056": (ps.C.blue, "o", "17056 steps (4x)"), "68224": (ps.C.vermillion, "s", "68224 steps (16x)")}
    # (1) objective against each HP: one figure per HP (ps.panels), the legend as a strip of its own
    figs = ps.panels(len(HPS))
    for (fig, ax), (name, (path, lab, logx)) in zip(figs, HPS.items()):
        for steps, (c, mk, sl) in col.items():
            T = d["sweeps"][steps]["trials"]
            sp = {x["name"]: x for x in yaml.safe_load(open(os.path.join(ROOT, CONFIGS[steps])))["search_space"]}
            x = [float(t[path[-1]]) if name != "ema" else float(bool(t["ema"])) for t in T]
            y = [t["val_loss"] for t in T]
            ax.scatter(x, y, color=c, marker=mk, s=18, label=sl)
            b = int(np.argmin(y))
            ax.scatter([x[b]], [y[b]], facecolor="none", edgecolor="k", marker=mk, s=70, lw=1.2,
                       label="best of its sweep" if steps == "68224" else None)
            if "low" in sp.get(name, {}):
                for e in (float(sp[name]["low"]), float(sp[name]["high"])):
                    ax.axvline(e, color=c, lw=0.6, ls=":")
        if logx:
            from matplotlib.ticker import NullFormatter
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
        if name == "ema":
            ax.set_xticks([0, 1]); ax.set_xticklabels(["off", "on"]); ax.set_xlim(-0.5, 1.5)
        ax.set_xlabel(lab); ax.set_ylabel("best val. loss (het. NLL)")
    ax0 = figs[0][1]
    ax0.plot([], [], color=ps.C.grey, lw=0.6, ls=":", label="search range")
    base = os.path.join(FIGS, "long_hpo_landscape_hps")
    ps.legend_strip(ax0, base + "_legend", ncol=4)
    ps.save_panels(figs, base)
    # (2) validation curves; every validation drawn, those at or above YCAP at the cap (listed in the printout)
    YCAP, spikes = 1.0, {}
    fig, axes = ps.figure(ncols=2, sharey=True)
    for ax, (steps, (c, mk, sl)) in zip(axes, col.items()):
        T = d["sweeps"][steps]["trials"]
        b = min(T, key=lambda t: t["val_loss"])
        for t in T:
            v = np.array(tr[t["key"]]["vals"], float)
            off = v[:, 1] >= YCAP                      # spikes (warm-up and mid-run, up to ~1e11): kept, drawn at the cap
            if off.any():
                spikes.setdefault(steps, {})[t["hp"]] = [[int(a), float(f"{y:.3g}")] for a, y in v[off]]
            best = t is b
            ax.plot(v[:, 0], np.minimum(v[:, 1], YCAP), color=c if best else ps.C.grey, lw=1.6 if best else 0.8,
                    alpha=1 if best else 0.6, label=(f"{sl}: best, lr {t['lr']:.2g}" if best else None))
            ax.scatter(v[off, 0], np.full(off.sum(), YCAP), marker="^", s=14, color=c if best else ps.C.grey, zorder=3)
        ax.plot([], [], color=ps.C.grey, lw=0.8, label=f"{sl}: other trials")
        ax.scatter([], [], marker="^", s=14, color=ps.C.grey, label="validation off the scale")
        ax.set_ylim(None, YCAP + 0.3)
        ax.set_xlabel("step")
        ps.legend(ax, "upper right")
    axes[0].set_ylabel(r"val. loss (het. NLL)")
    ps.save(fig, os.path.join(FIGS, "long_hpo_landscape_curves"))
    print("validations off the scale (step, value) per sweep and trial:", json.dumps(spikes))
    for steps in SWEEPS:
        T = sorted(d["sweeps"][steps]["trials"], key=lambda t: t["val_loss"])
        print(steps, [(t["hp"], round(t["val_loss"], 3), t["lr"], t["ema"]) for t in T])


if __name__ == "__main__":
    {"pull": pull, "figs": figs}[sys.argv[1]]()
