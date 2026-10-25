"""The small-D fine-tuning study, first round (D120): every method on the 60 cells of analysis/transfer/data/
lowd_ft_spec.json (e+e- -> Z gamma, e+e- -> d dbar, u d -> u d, e+e- -> b bbar at one loop, u ubar -> g g; rungs 1, 7,
9; D = 10 to 10^2.5), each at its cell's best fine-tune HPs. Inputs, copied home:
  data/lowd_ft_runs.jsonl     the LOWD lines of scripts/job_lowd_ft.sh (tools/lowd_ft.py), one per method and cell
  data/lowd_ft_readout.jsonl  the READOUT lines of its readout job (tools/linear_readout.py on the same 60 cells)
Per cell, the MSE of log|M|^2 at the best checkpoint (validation) of
  default, lpft, l2sp, freeze   the four trained methods, rerun on one machine
  readout                       the frozen trunk with a ridge head (no training)
  choice                        per cell the better of readout and default on validation (chosen on the split it is
                                scored on, as the grid's own best-of-five is)
against the grid's recorded fine-tune and scratch. Gains in decades, log10(reference / method), so positive is better.
Medians over the 15 probe-and-rung cells per D, and per probe; every cell without a value listed.
    python3 analysis/transfer/lowd_ft.py  -> data/lowd_ft.json, figs/lowd_ft.{png,pdf}
"""
import collections, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
TRAINED = ("default", "lpft", "l2sp", "freeze")
METHODS = TRAINED + ("readout", "choice")
LABEL = {"default": "default fine-tune", "lpft": "linear probe, then fine-tune", "l2sp": "pulled toward the start",
         "freeze": "lower half frozen", "readout": "frozen read-out", "choice": "read-out or fine-tune"}


def load():
    spec = json.load(open(os.path.join(HERE, "data", "lowd_ft_spec.json")))
    key = lambda p, r, k: f"{p}/r{r}/d{k}"
    cells = {key(c["probe"], c["rung"], c["k"]): dict(c, values={}) for c in spec}
    missing = []
    f = os.path.join(HERE, "data", "lowd_ft_runs.jsonl")
    for line in (open(f) if os.path.exists(f) else []):
        r = json.loads(line)
        c = cells[key(r["probe"], r["rung"], r["k"])]
        if "mse_logM2" in r:
            c["values"][r["method"]] = r["mse_logM2"]
        else:
            missing.append((r["method"], key(r["probe"], r["rung"], r["k"]), r.get("error")))
    f = os.path.join(HERE, "data", "lowd_ft_readout.jsonl")
    by_sweep = {os.path.basename(c["config"])[len("sweep_config_"):-len(".yaml")]: c for c in cells.values()}
    for line in (open(f) if os.path.exists(f) else []):
        r = json.loads(line)
        if "readout_val_mse" in r and r["sweep"] in by_sweep:
            by_sweep[r["sweep"]]["values"]["readout"] = r["readout_val_mse"]
    for c in cells.values():
        v = c["values"]
        if "readout" in v and "default" in v:
            v["choice"] = min(v["readout"], v["default"])
        for m in METHODS:
            if m not in v and not any(x[0] == m and x[1] == key(c["probe"], c["rung"], c["k"]) for x in missing):
                missing.append((m, key(c["probe"], c["rung"], c["k"]), "no result copied home"))
    return cells, missing


def main():
    cells, missing = load()
    out = {"per_D": {}, "per_probe": {}, "cells": {}, "missing": missing}
    for k, c in cells.items():
        v = c["values"]
        out["cells"][k] = {"D": c["D"], "scratch": c["scratch"], "recorded_ft": c["recorded_ft"], **v,
                           "gain_over_scratch": {m: float(np.log10(c["scratch"] / v[m])) for m in v},
                           "gain_over_default": {m: float(np.log10(v["default"] / v[m])) for m in v if "default" in v}}
    for grp, keyf in (("per_D", lambda c: str(c["D"])), ("per_probe", lambda c: c["probe"])):
        g = collections.defaultdict(list)
        for k, c in cells.items():
            g[keyf(c)].append(out["cells"][k])
        for name, rows in sorted(g.items()):
            s = {"n": len(rows)}
            for m in METHODS:
                gs = [r["gain_over_scratch"][m] for r in rows if m in r["gain_over_scratch"]]
                gd = [r["gain_over_default"][m] for r in rows if m in r.get("gain_over_default", {})]
                s[m] = {"n": len(gs), "gain_over_scratch": float(np.median(gs)) if gs else None,
                        "gain_over_default": float(np.median(gd)) if gd else None,
                        "beats_default": int(sum(x > 0 for x in gd)), "beats_scratch": int(sum(x > 0 for x in gs))}
            rec = [r["recorded_ft"] for r in rows]
            s["recorded_ft_gain_over_scratch"] = float(np.median([np.log10(r["scratch"] / r["recorded_ft"]) for r in rows]))
            out[grp][name] = s
            print(grp, name, json.dumps({m: (s[m]["gain_over_scratch"], s[m]["gain_over_default"]) for m in METHODS}))
    print(f"missing: {len(missing)}", missing[:10])
    json.dump(out, open(os.path.join(HERE, "data", "lowd_ft.json"), "w"), indent=1)
    plot(out)


def plot(out):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    Ds = sorted(out["per_D"], key=float)
    x = np.log10([float(d) for d in Ds])
    cols = {"default": ps.C.blue, "lpft": ps.C.vermillion, "l2sp": ps.C.green, "freeze": ps.C.orange,
            "readout": ps.C.grey, "choice": ps.C.purple}
    marks = {"default": "o", "lpft": "s", "l2sp": "^", "freeze": "v", "readout": "D", "choice": "*"}
    for ax, what, yl in ((axs[0], "gain_over_scratch", r"gain over scratch (decades)"),
                         (axs[1], "gain_over_default", r"gain over the default (decades)")):
        for m in METHODS:
            if what == "gain_over_default" and m == "default":
                continue
            y = [out["per_D"][d][m][what] for d in Ds]
            if all(v is None for v in y):
                continue
            ax.plot(x, [np.nan if v is None else v for v in y], marker=marks[m], color=cols[m], label=LABEL[m])
        ax.plot([x.min(), x.max()], [0, 0], color=ps.C.grey, ls="--", label="equal")
        ax.set_xlabel(r"$\log_{10} D$"); ax.set_ylabel(yl)
    ps.legend(axs[0], "upper left", handles=axs[0].get_legend_handles_labels()[0])
    for ax in axs:
        ps.make_room(ax)
    ps.save(fig, os.path.join(HERE, "figs", "lowd_ft"))


if __name__ == "__main__":
    main()
