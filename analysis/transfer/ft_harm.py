"""Why a fine-tune loses to scratch at small D (D151, plan entry ft_harm): a bad start, or fine-tuning undoing a good one.

A cell loses when its fine-tune's best (MSE of log|M|^2 at the best validation checkpoint) is above scratch's best in
the same cell. Since the reported value is the curve's minimum, a losing curve never dips below scratch; what the
pretrained start holds is read off the frozen network instead:
  readout     a ridge head fitted on the cell's D training events over the frozen trunk's mean-pooled features, scored
              on validation (tools/linear_readout.py). It is exactly the step-0 validation of linear probe then
              fine-tune, whose head is written as this read-out before the first step.
  head_raw    the pretrained network as is at step 0 (its own head) on the probe's standardized target
  first       the first recorded validation of the fine-tune (after validate_every steps, 1% of the run; one step at
              D = 10 in the DyHPO round)
Per losing cell, with S scratch's best, R the read-out and F the fine-tune's best (the user's three classes, D151):
  (i)   bad start              R >= S and F <= R: the start holds less than scratch reaches; fine-tuning improves on it,
                               not enough
  (ii)  fine-tuning undoes it  R < S < F: the start alone beats scratch and the fine-tune loses it
  (iii) other                  "iii-worse": R >= S and F > R, the start is worse than scratch and the fine-tune ends
                               worse still than a linear head on its own frozen trunk; "iii-noreadout": no read-out
and the curve's shape: where its minimum sits (fraction of the run) and the last validation over the minimum.
Sources (all home; the curves of the first round copied from CC zz_lowd before any model deletion, D149):
  the transfer grid's fine-tunes, 8k (scratch_sweeps.json via cells.py; gains data/nogain.json), each cell's best
      trial; read-outs where they exist (lowd cells, rungs 1, 7, 9 at D = 10 to 10^2.5; data/nogain_readout_za.json
      for ee->Z gamma at D = 10^1.5, 10^2, every rung)
  the small-D study's first round (data/lowd_ft_curves.json, one run per method and cell) and DyHPO round
      (data/lowd_dyhpo_sweeps.json, each method's best trial), read-outs data/lowd_ft_readout.jsonl
    python3 analysis/transfer/ft_harm.py --collect ~/.local/share/ccorch/artifacts/FA/zz_lowd_curves/first_round
                                          -> data/lowd_ft_curves.json (the first round's result files, compacted)
    .venv-driver/bin/python analysis/transfer/ft_harm.py   -> data/ft_harm.json, figs/ft_harm.{png,pdf}
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
DATA = os.path.join(HERE, "data")
SMALL = (2, 3, 4, 5)


def collect(src):
    out = {}
    for f in sorted(glob.glob(os.path.join(src, "tp3_lowd_*", "r*_d*_result.json"))):
        arm = os.path.basename(os.path.dirname(f))[len("tp3_lowd_"):]
        method, probe = arm.split("_", 1)
        tag = os.path.basename(f)[:-len("_result.json")]
        r = json.load(open(f))
        std = json.load(open(os.path.join(os.path.dirname(f), tag, "data_stats.json")))["prepd_std"][0]
        out[f"{method}/{probe}/{tag}"] = {k: r[k] for k in ("val_loss", "best_step", "validate_every", "val_curve")}
        out[f"{method}/{probe}/{tag}"]["prepd_std"] = std
    json.dump(out, open(os.path.join(DATA, "lowd_ft_curves.json"), "w"))
    print(f"{len(out)} curves -> data/lowd_ft_curves.json")


def shape(curve, every, std, T=None):
    y = np.asarray(curve, float) * std ** 2
    i = int(np.argmin(y))
    T = T or every * len(y)
    return {"first": float(y[0]), "first_step": every, "best": float(y[i]), "best_frac": (i + 1) * every / T,
            "last_over_best": float(y[-1] / y[i]), "steps": T}


def readouts():
    """(probe, rung, k) -> {readout, head_raw}; rung 0 is the ee->uu pretraining."""
    R = {}
    spec = json.load(open(os.path.join(DATA, "lowd_ft_spec.json")))
    by_sweep = {os.path.basename(c["config"])[len("sweep_config_"):-len(".yaml")]: c for c in spec}
    for line in open(os.path.join(DATA, "lowd_ft_readout.jsonl")):
        r = json.loads(line)
        c = by_sweep.get(r["sweep"])
        if c and "readout_val_mse" in r:
            R[(c["probe"], c["rung"], c["k"])] = {"readout": r["readout_val_mse"], "head_raw": r.get("head_raw_val_mse")}
    za = json.load(open(os.path.join(DATA, "nogain_readout_za.json")))
    for k, v in za.items():
        for arm, a in v["arms"].items():
            rung = 0 if arm == "uufte" else int(arm[1]) if arm[0] == "r" and arm[1].isdigit() else None
            if rung is not None:
                R.setdefault(("ee_Za", rung, int(k)), {"readout": a["readout"], "head_raw": None})
    return R


def classify(F, S, R):
    if F <= S:
        return None
    if R is None:
        return "iii-noreadout"
    if F > R:
        return "ii" if R < S else "iii-worse"
    return "i"


def rows():
    from cells import best, scratch
    from rung_grid import P, fam
    R = readouts()
    ng = json.load(open(os.path.join(DATA, "nogain.json")))
    out = []
    # the grid: each cell's best fine-tune trial against scratch's best
    for p in P:
        for k in range(2, 9):
            S = scratch(p, k, steered=False)[0]
            if not S:
                continue
            for r in range(10):
                F, t = best(f"{fam(r)}_{p}_d{k}")
                if F is None or not t.get("val_curve"):
                    continue
                ro = R.get((p, r, k), {})
                row = dict(source="grid", method="default", probe=p, rung=r, k=k, scratch=S, ft=F,
                           readout=ro.get("readout"), head_raw=ro.get("head_raw"),
                           **shape(t["val_curve"], t["validate_every"], t["prepd_std"], t.get("T")))
                g = ng[p]["cells"].get(str(k), {}).get("gain", {}).get(str(r))
                assert g is None or abs(g - np.log10(S / F)) < 1e-6, (p, k, r, g)
                out.append(row)
    spec = {(c["probe"], c["rung"], c["k"]): c for c in json.load(open(os.path.join(DATA, "lowd_ft_spec.json")))}
    # the first round: one run per method and cell
    C = json.load(open(os.path.join(DATA, "lowd_ft_curves.json")))
    for key, v in C.items():
        m, p, tag = key.split("/")
        r, k = int(tag.split("_")[0][1:]), int(tag.split("_d")[1])
        c = spec[(p, r, k)]
        ro = R.get((p, r, k), {})
        out.append(dict(source="first round", method=m, probe=p, rung=r, k=k, scratch=c["scratch"],
                        ft=v["val_loss"] * v["prepd_std"] ** 2, readout=ro.get("readout"), head_raw=ro.get("head_raw"),
                        **shape(v["val_curve"], v["validate_every"], v["prepd_std"], c["steps"])))
    # the DyHPO round: each method's best trial
    Sw = json.load(open(os.path.join(DATA, "lowd_dyhpo_sweeps.json")))
    for (p, r, k), c in spec.items():
        for m in ("lpft", "l2sp", "freeze"):
            tr = [t for t in Sw.get(f"tp3_lowd_{m}_r{r}_{p}_d{k}", [])
                  if t.get("val_loss") is not None and t.get("prepd_std") and t.get("val_curve")]
            if not tr:
                continue
            t = min(tr, key=lambda t: t["val_loss"] * t["prepd_std"] ** 2)
            ro = R.get((p, r, k), {})
            out.append(dict(source="DyHPO round", method=m, probe=p, rung=r, k=k, scratch=c["scratch"],
                            ft=t["val_loss"] * t["prepd_std"] ** 2, readout=ro.get("readout"),
                            head_raw=ro.get("head_raw"), **shape(t["val_curve"], t["validate_every"], t["prepd_std"],
                                                                 t.get("T"))))
    for x in out:
        x["cls"] = classify(x["ft"], x["scratch"], x["readout"])
    return out


def summarise(sel):
    los = [x for x in sel if x["cls"]]
    n = {c: sum(x["cls"] == c for x in los) for c in ("i", "ii", "iii-worse", "iii-noreadout")}
    ro = [x for x in sel if x["readout"] is not None]
    lr = [x for x in los if x["readout"] is not None]
    med = lambda v: float(np.median(v)) if len(v) else None
    return {"cells": len(sel), "losing": len(los), "classes": n,
            "with_readout": len(ro), "ft_above_readout": sum(x["ft"] > x["readout"] for x in ro),
            "losing_with_readout": len(lr),
            # in decades: the fine-tune's best over its own read-out, and the read-out over scratch, losing cells
            "median_ft_over_readout": med([np.log10(x["ft"] / x["readout"]) for x in lr]),
            "median_readout_over_scratch": med([np.log10(x["readout"] / x["scratch"]) for x in lr]),
            "median_ft_over_scratch": med([np.log10(x["ft"] / x["scratch"]) for x in los]),
            "median_best_frac_losing": med([x["best_frac"] for x in los]),
            "median_best_frac_winning": med([x["best_frac"] for x in sel if not x["cls"]]),
            "median_last_over_best_losing": med([x["last_over_best"] for x in los]),
            "median_first_over_readout_losing": med([x["first"] / x["readout"] for x in lr])}


def main():
    if "--collect" in sys.argv:
        return collect(os.path.expanduser(sys.argv[sys.argv.index("--collect") + 1]))
    X = rows()
    small = [x for x in X if x["k"] in SMALL]
    summ = {}
    for src in ("grid", "first round", "DyHPO round"):
        summ[src] = summarise([x for x in small if x["source"] == src])
        for m in sorted({x["method"] for x in small if x["source"] == src}):
            summ[f"{src}/{m}"] = summarise([x for x in small if x["source"] == src and x["method"] == m])
    summ["grid, all D"] = summarise([x for x in X if x["source"] == "grid"])
    # the grid on the cells with a read-out (rungs 1, 7, 9 on the five lowd probes, every rung on ee->Z gamma at two D)
    summ["grid with read-out"] = summarise([x for x in small if x["source"] == "grid" and x["readout"] is not None])
    # by probe, all sources pooled at small D, cells with a read-out
    for p in sorted({x["probe"] for x in small if x["readout"] is not None}):
        summ[f"probe/{p}"] = summarise([x for x in small if x["probe"] == p and x["readout"] is not None])
    # linear probe then fine-tune: its step 0 is the read-out exactly, so its first validation against the read-out
    # is what the first steps do to a head that already fits
    lp = [x for x in small if x["method"] == "lpft"]
    summ["lpft first steps"] = {"runs": len(lp), "first_above_readout": sum(x["first"] > x["readout"] for x in lp),
                                "median_first_over_readout": float(np.median([x["first"] / x["readout"] for x in lp])),
                                "best_above_readout": sum(x["ft"] > x["readout"] for x in lp)}
    # every losing cell at small D, all sources pooled (one row per source, method and cell)
    summ["pooled"] = summarise(small)
    for k, v in summ.items():
        print(f"{k:28s}", json.dumps(v))
    json.dump({"summary": summ, "rows": X}, open(os.path.join(DATA, "ft_harm.json"), "w"), indent=1)
    plot(small)
    r2 = lambda v: None if v is None else round(v, 2)
    P, Z = summ["pooled"], summ["probe/ee_Za"]
    claims = {"losing": P["losing"], "cls": P["classes"], "with_readout": P["losing_with_readout"],
              "ft_over_readout": r2(P["median_ft_over_readout"]), "readout_over_scratch": r2(P["median_readout_over_scratch"]),
              "ft_over_scratch": r2(P["median_ft_over_scratch"]),
              "noreadout_grid_cells": summ["grid"]["classes"]["iii-noreadout"],
              "za": {"losing": Z["losing"], "cls": Z["classes"], "ft_over_readout": r2(Z["median_ft_over_readout"]),
                     "readout_over_scratch": r2(Z["median_readout_over_scratch"])},
              "probe": {p.split("/")[1]: {"losing": summ[p]["losing"], "cells": summ[p]["cells"],
                                          "i": summ[p]["classes"]["i"],
                                          "ft_over_readout": r2(summ[p]["median_ft_over_readout"]),
                                          "readout_over_scratch": r2(summ[p]["median_readout_over_scratch"])}
                        for p in summ if p.startswith("probe/")},
              "lpft": {"runs": summ["lpft first steps"]["runs"],
                       "first_above": summ["lpft first steps"]["first_above_readout"],
                       "first_over_readout": r2(summ["lpft first steps"]["median_first_over_readout"]),
                       "best_above": summ["lpft first steps"]["best_above_readout"]},
              "best_frac_losing": r2(P["median_best_frac_losing"]), "best_frac_winning": r2(P["median_best_frac_winning"]),
              "losing_cells": len({(x["probe"], x["rung"], x["k"]) for x in small if x["cls"] and x["source"] != "grid"}),
              "curves_first_round": len(json.load(open(os.path.join(DATA, "lowd_ft_curves.json"))))}
    print(json.dumps(claims))


def plot(small):
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    col = {"grid": ps.C.blue, "first round": ps.C.orange, "DyHPO round": ps.C.vermillion}
    lab = {"grid": "grid fine-tune", "first round": "first round", "DyHPO round": "own search"}
    for src in col:
        L = [x for x in small if x["source"] == src and x["readout"] is not None]
        a.scatter([np.log10(x["readout"] / x["scratch"]) for x in L], [np.log10(x["ft"] / x["scratch"]) for x in L],
                  color=col[src], label=lab[src], s=8, alpha=0.7)
    lim = [-2.5, 2.0]
    a.plot(lim, lim, ls="--", color=ps.C.grey, label="fine-tune = read-out")
    a.axhline(0, ls=":", color=ps.C.grey, label="scratch")
    a.axvline(0, ls=":", color=ps.C.grey)
    a.set_xlabel(r"$\log_{10}(\mathrm{read\!-\!out}/\mathrm{scratch})$")
    a.set_ylabel(r"$\log_{10}(\mathrm{fine\!-\!tune}/\mathrm{scratch})$")
    ps.legend(a, "upper left")
    ps.make_room(a)
    # linear probe then fine-tune on the losing cells: the curve over its step-0 value (the read-out)
    C = json.load(open(os.path.join(DATA, "lowd_ft_curves.json")))
    for key, v in C.items():
        m, p, tag = key.split("/")
        if m != "lpft":
            continue
        r, k = int(tag.split("_")[0][1:]), int(tag.split("_d")[1])
        x = next(z for z in small if z["source"] == "first round" and z["method"] == "lpft" and z["probe"] == p
                 and z["rung"] == r and z["k"] == k)
        y = np.asarray(v["val_curve"]) * v["prepd_std"] ** 2 / x["readout"]
        f = np.arange(1, len(y) + 1) / len(y)
        lose = x["cls"] is not None
        b.plot(f, np.log10(y), color=ps.C.vermillion if lose else ps.C.blue, alpha=0.5 if lose else 0.25,
               label="loses to scratch" if lose else "beats scratch")
    h, l = b.get_legend_handles_labels()
    keep = {l_: h_ for h_, l_ in zip(h, l)}
    b.axhline(0, ls="--", color=ps.C.grey, label="step 0 (read-out)")
    b.set_xlabel("fraction of the run")
    b.set_ylabel(r"$\log_{10}(\mathrm{validation}/\mathrm{read\!-\!out})$")
    ps.legend(b, "upper right", handles=list(keep.values()) + [b.lines[-1]],
              labels=list(keep) + ["step 0 (read-out)"])
    ps.make_room(b)
    ps.save(fig, os.path.join(HERE, "figs", "ft_harm"))


if __name__ == "__main__":
    main()
