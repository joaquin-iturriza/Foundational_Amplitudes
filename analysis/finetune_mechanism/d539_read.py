"""Plan finetune_mechanism D539 step: why the momentum log-rescale costs Z+4g scratch (design 2026-11-24, hp1961 HPs,
batch 2048, 1953 steps, seeds 42-44). Arms per process (zng = q q~ -> Z+4g, uug = e+e- -> u u~ g) and D (1e4, 1e5):
off, eps1e-8 and, for Z+4g at 1e4, eps1e-4, eps1e-2, eps1 and massless (the rescale on massless momenta only).
Z+4g 1e4 eps1e-8 is the one-setting search's hp1961 (seed 42 its trial, 43/44 ftmech_zng4g_scrsoftsrch_b2048_s4x).

Measure: best val MSE(log|M|^2) = min val mu-MSE x prepd_std^2 (each run's own data_stats.json); r = arm/off of the same
process and D, three-seed geometric means, 2-sigma interval from the arms' own seed log-sd (zng4g_d1e3_read.ratio).

    python3 analysis/finetune_mechanism/d539_read.py [--cached]   -> data/d539.json, figs/d539
    python3 analysis/finetune_mechanism/d539_read.py --set s [--cached]   -> the step-4 set at hp1333 (S's optimum),
        runs ftmech_d539s_*, every cell off/eps1e-8 and all new: data/d539s.json, figs/d539s
    python3 analysis/finetune_mechanism/d539_read.py --set u [--cached]   -> step 5, u u~ g only, best against best: each
        arm's own 16-trial search ftmech_d539_uug_<D>_<arm>_b2048 (seed 42 = its chosen trial, the lowest val_loss)
        and seeds 43/44 of that trial (runs ftmech_d539_uug_<D>_<arm>_b2048_s4x): data/d539u.json, figs/d539u
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from analysis.common.readers import site_python  # noqa: E402
from zng4g_readout import driven, CC  # noqa: E402
from zng4g_d1e3_read import REMOTE, ratio  # noqa: E402

CELLS = {("zng", "1e4"): ["off", "eps1e-8", "eps1e-4", "eps1e-2", "eps1", "massless"],
         ("zng", "1e5"): ["off", "eps1e-8"], ("uug", "1e4"): ["off", "eps1e-8"], ("uug", "1e5"): ["off", "eps1e-8"]}
SEEDS = (42, 43, 44)
SET = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else ""   # "" = hp1961 (step 3), "s" = hp1333
if SET:
    CELLS = {c: ["off", "eps1e-8"] for c in CELLS if SET != "u" or c[0] == "uug"}
TAG = f"d539{SET}"
REUSED = "ftmech_zng4g_scrsoftsrch_b2048"   # Z+4g 1e4 eps1e-8: the one-setting search's chosen trial hp1961


def collect():
    args = []
    for (p, d), arms in CELLS.items():
        for a in arms:
            for s in SEEDS:
                lab = f"{p}|{d}|{a}|{s}"
                if SET == "u":
                    n = f"ftmech_d539_uug_{d}_{a}_b2048"
                    if s == 42:
                        t = min(driven(n), key=lambda t: t[2])
                        args.append(f"'{lab}#{t[4]}#{t[5]}'")
                        continue
                    n = f"{n}_s{s}"
                elif (p, d, a) == ("zng", "1e4", "eps1e-8") and not SET:
                    if s == 42:
                        t = next(t for t in driven(REUSED) if t[1] == 1961)
                        args.append(f"'{lab}#{t[4]}#{t[5]}'")
                        continue
                    n = f"{REUSED}_s{s}"
                else:
                    n = f"ftmech_{TAG}_{p}_{d}_{a}_s{s}"
                args.append(f"'{lab}#{CC}/runs/{n}#{CC}/sweeps/{n}/results/fixed.json'")
    return site_python(REMOTE.replace("ZNG1K", "D539"), args, "D539")


def main():
    cache = os.path.join(HERE, "data", f"{TAG}_raw.json")
    raw = json.load(open(cache)) if "--cached" in sys.argv else collect()
    json.dump(raw, open(cache, "w"))
    best, curves, missing = {}, {}, []
    for k, v in raw.items():
        if not v.get("val_mse_curve") or "std" not in v:
            missing.append(k)
            continue
        c = np.array(v["val_mse_curve"]) * v["std"] ** 2
        p, d, a, s = k.split("|")
        if SET == "u" and s != "42":   # seeds of the chosen trial: same lr as its seed-42 trial run
            assert abs(v["lr"] / raw[f"{p}|{d}|{a}|42"]["lr"] - 1) < 1e-9, k
        best.setdefault((p, d, a), {})[int(s)] = float(c.min())
        curves[k] = c
    res = {"missing": missing, "cells": {}}
    for (p, d), arms in CELLS.items():
        cell = {}
        for a in arms:
            b = best.get((p, d, a), {})
            cell[a] = {"best": [b.get(s) for s in SEEDS], "gmean": float(np.exp(np.mean(np.log(list(b.values()))))) if b else None}
            off = best.get((p, d, "off"), {})
            if a != "off" and len(b) == 3 and len(off) == 3:
                cell[a]["over_off"] = ratio(np.array([b[s] for s in SEEDS]), np.array([off[s] for s in SEEDS]))
        res["cells"][f"{p}_{d}"] = cell
    res["blowups"] = [k for k, c in curves.items() if np.any(c > 10 * np.minimum.accumulate(c))]
    if SET == "u":   # each search's chosen lr: decades to the nearer edge of its lr window
        import yaml
        dec = {}
        for (p, d), arms in CELLS.items():
            for a in arms:
                w = next(x for x in yaml.safe_load(open(os.path.join(ROOT, "sweep", f"sweep_config_ftmech_d539_uug_{d}_{a}_b2048.yaml")))["search_space"] if x["name"] == "training.lr")
                lr = raw[f"{p}|{d}|{a}|42"]["lr"]
                dec[f"{d}_{a}"] = float(min(np.log10(lr / w["low"]), np.log10(w["high"] / lr)))
        res["lr_dec_from_edge"] = dec
        res["lr_dec_from_edge_min"] = min(dec.values())
    json.dump(res, open(os.path.join(HERE, "data", f"{TAG}.json"), "w"), indent=1)
    print(json.dumps(res))   # one line: verify_claims reads the last {...} line
    plot(res, curves)


def plot(res, curves):
    import plot_style as ps
    if SET == "u":
        return plot_u(res, curves)
    lab = {"off": "no rescale", "eps1e-8": r"rescale, $\epsilon=10^{-8}$", "eps1e-4": r"rescale, $\epsilon=10^{-4}$",
           "eps1e-2": r"rescale, $\epsilon=10^{-2}$", "eps1": r"rescale, $\epsilon=1$",
           "massless": "rescale of massless\nmomenta only"}
    cols = dict(zip(lab, [ps.C.blue] + ps.sequence(4) + [ps.C.vermillion]))
    fig, axs = ps.figure(ncols=2, nrows=2)
    (a, g), (h, cell) = axs
    blown = set(res["blowups"])   # open markers: a run whose val rose > 10x its best-so-far (D398), read at its best
    arms = CELLS[("zng", "1e4")]
    for i, k in enumerate(arms):
        for j, (s, x) in enumerate(zip(SEEDS, res["cells"]["zng_1e4"][k]["best"])):
            if x is not None:
                a.plot(i + (j - 1) * 0.15, x, "o", color=cols[k], label=lab[k] if j == 0 else None,
                       mfc="none" if f"zng|1e4|{k}|{s}" in blown else cols[k])
    a.set_xticks([]); a.set_xlim(-0.5, len(arms) - 0.5); a.set_yscale("log")
    a.set_xlabel(r"input, $D_{\rm train}=10^4$"); a.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(a, r"$q\bar q\to Z+4g$")
    names = {"zng": r"$Z+4g$", "uug": r"$u\bar ug$"}
    ticks = []
    for i, (p, d) in enumerate([c for c in CELLS]):
        for j, k in enumerate(("off", "eps1e-8")):
            for m, (s, x) in enumerate(zip(SEEDS, res["cells"][f"{p}_{d}"][k]["best"])):
                if x is not None:
                    g.plot(i + (j - 0.5) * 0.3 + (m - 1) * 0.05, x, "o", color=cols[k],
                           mfc="none" if f"{p}|{d}|{k}|{s}" in blown else cols[k])
        ticks.append(f"{names[p]}, $10^{{{d[-1]}}}$")
    g.set_xlim(-0.5, len(ticks) - 0.5); g.set_xticks(range(len(ticks))); g.set_xticklabels(ticks, fontsize="x-small"); g.set_yscale("log")
    g.set_xlabel(r"process, $D_{\rm train}$"); g.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    for k in arms:   # validation curves of seed 42, every Z+4g 1e4 arm
        c = curves.get(f"zng|1e4|{k}|42")
        if c is not None:
            h.plot(np.linspace(0, 1, len(c)), c, color=cols[k])
    h.set_yscale("log"); h.set_xlabel("training progress (step / 1953)")
    h.set_ylabel(r"val MSE$(\log|\mathcal{M}|^2)$"); ps.process_label(h, r"$q\bar q\to Z+4g$")
    a.plot([], [], "o", color="0.5", mfc="none", label="blew up (read\nat its best)")
    ps.legend_cell(cell, a)   # panel (b) plots off and eps1e-8 with panel (a)'s markers
    ps.save(fig, os.path.join(HERE, "figs", TAG))


def plot_u(res, curves):
    import plot_style as ps
    lab = {"off": "no rescale", "eps1e-8": r"rescale, $\epsilon=10^{-8}$"}
    cols = {"off": ps.C.blue, "eps1e-8": ps.C.vermillion}
    fig, axs = ps.figure(ncols=2, nrows=3)
    (g, h1), (h2, k1), (k2, cell) = axs
    blown = set(res["blowups"])   # open markers: val rose > 10x its best-so-far (D398), read at its best
    kit = json.load(open(os.path.join(HERE, "data", f"{TAG}_kit.json")))["cells"]   # d539_kit_read.py --set u
    fam = {"soft_x": "soft", "coll_y": "collinear", "bulk": "bulk"}
    for ax, d in ((k1, "1e4"), (k2, "1e5")):
        for j, a in enumerate(("off", "eps1e-8")):
            for i, f in enumerate(fam):
                v = [x for x in kit[d][a][f] if x is not None]
                ax.plot(i + (j - 0.5) * 0.3 + (np.arange(len(v)) - 1) * 0.05, v, "o", color=cols[a])
        ax.set_xticks(range(len(fam))); ax.set_xticklabels(list(fam.values())); ax.set_xlim(-0.5, len(fam) - 0.5)
        ax.set_yscale("log"); ax.set_xlabel("deepest kit bin")
        ax.set_ylabel(r"MSE$(\log_{10}|\mathcal{M}|^2)$")
        ps.process_label(ax, rf"$e^+e^-\to u\bar ug$, $D_{{\rm train}}=10^{d[-1]}$")
    for i, (p, d) in enumerate(CELLS):
        for j, k in enumerate(("off", "eps1e-8")):
            for m, (sd, x) in enumerate(zip(SEEDS, res["cells"][f"{p}_{d}"][k]["best"])):
                if x is not None:
                    g.plot(i + (j - 0.5) * 0.3 + (m - 1) * 0.05, x, "o", color=cols[k],
                           label=lab[k] if (i, m) == (0, 0) else None,
                           mfc="none" if f"{p}|{d}|{k}|{sd}" in blown else cols[k])
    g.set_xlim(-0.5, len(CELLS) - 0.5); g.set_xticks(range(len(CELLS)))
    g.set_xticklabels([f"$10^{{{d[-1]}}}$" for _, d in CELLS]); g.set_yscale("log")
    g.set_xlabel(r"$D_{\rm train}$"); g.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(g, r"$e^+e^-\to u\bar ug$")
    for ax, d in ((h1, "1e4"), (h2, "1e5")):
        for k in ("off", "eps1e-8"):
            for sd in SEEDS:
                c = curves.get(f"uug|{d}|{k}|{sd}")
                if c is not None:
                    ax.plot(np.linspace(0, 1, len(c)), c, color=cols[k])
        ax.set_yscale("log"); ax.set_xlabel("training progress (step / 1953)")
        ax.set_ylabel(r"val MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, rf"$e^+e^-\to u\bar ug$, $D_{{\rm train}}=10^{d[-1]}$")
    if blown:
        g.plot([], [], "o", color="0.5", mfc="none", label="blew up (read\nat its best)")
    ps.legend_cell(cell, g)
    ps.save(fig, os.path.join(HERE, "figs", TAG))


if __name__ == "__main__":
    main()
