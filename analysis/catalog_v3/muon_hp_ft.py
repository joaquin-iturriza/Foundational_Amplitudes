"""The Muon fine-tune HP search (muon_hp; the user 2026-10-26: "We may have to explore a bit the HP space for muon,
that's fine, the gain is clearly worth it"): sweep muon_hp_ft (sweep/sweep_config_ccin2p3_muon_hp_ft.yaml), a
single-fidelity DyHPO search over fine_tune.lr_scale (log-uniform [0.1, 10]) and fine_tune.layer_decay (uniform
[0.75, 1]), 12 trials. A trial is the 16-cell small-D grid of par:v3_muon (4 held-out probes x D = 10, 32, 100, 316)
fine-tuned with MuonAdamW from the Muon short pretraining's best checkpoint, the parent's standardization kept; its
objective is the geometric mean of the 16 cells' best-checkpoint val_loss_no_reg (sweep/run_trial.py _run_cells).
  collect (on the site): every trial summary and every cell's result file (best-checkpoint val_loss_no_reg, its
           curve, step-0 validation) with the HPs read back from the cell's run config, as JSON:
    site run --item muon_hp ccin2p3 FA -- python analysis/catalog_v3/muon_hp_ft.py collect \
        > analysis/catalog_v3/data/muon_hp_ft_trials.json
  main (local, from data/muon_hp_ft_trials.json and data/muon_ft.json):
    the untuned reference is the same 16 cells at lr_scale 1, layer_decay 1 from par:v3_muon's fine-tunes (Muon parent,
    Muon fine-tune, parent's standardization; data/muon_ft.json), objective their geometric mean; per trial the
    objective and its ratio to the reference; the best trial's per-cell, per-probe and per-D (geometric mean over the 4
    probes) ratio reference / best; every cell's curve after its best checkpoint (last validation
    over the best, and the best's position along the run; a non-finite validation fails the read); the best point's position in the search window (fraction of the
    log10 range for lr_scale, of the range for layer_decay); the step-0 check (each cell's step-0 validation against
    the reference cell's, same parent weights and events, so it should agree).
    python3 analysis/catalog_v3/muon_hp_ft.py   -> data/muon_hp_ft.json, figs/muon_hp_ft.{png,pdf}
Figure muon_hp_ft (1x2): (a) every trial in (lr_scale, layer_decay), colour its objective, the untuned point starred;
(b) the best trial's per-cell gain over the untuned point against D, one line per probe.
"""
import glob, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
SWEEP = "muon_hp_ft"
PROBES = {"v3_dg_dgwpwm": r"$dg\to dgW^+W^-$", "v3_usx_cxuwpz": r"$u\bar s\to \bar c uW^+Z$",
          "v3_gg_attx": r"$gg\to\gamma t\bar t$", "v3_udx_agwp_nlo": r"$u\bar d\to\gamma gW^+$, one loop"}
TAG_D = {"d2": 10, "d3": 32, "d4": 100, "d5": 316}
STEP0_TOL = 1e-3     # float32 reductions on a GPU agree far better; a changed target moves step 0 by far more
WINDOW = {"fine_tune.lr_scale": (0.1, 10.0, "log"), "fine_tune.layer_decay": (0.75, 1.0, "lin")}


def collect():
    import yaml
    res = os.path.join(ROOT, "sweeps", SWEEP, "results")
    trials = []
    for f in sorted(glob.glob(os.path.join(res, "hp*_t*.json"))):
        m = re.match(r"hp(\d{4})_t(\d+)_(\d+)\.json$", os.path.basename(f))
        if not m:
            continue                                        # a cell's file
        hp = int(m.group(1))
        s = json.load(open(f))
        cells = {}
        for name in s["cells"]:
            cf = sorted(glob.glob(os.path.join(res, f"hp{hp:04d}_t{m.group(2)}_{name}_*.json")))
            r = json.load(open(cf[-1]))
            d = os.path.join(ROOT, "runs", SWEEP, f"trial_{hp:04d}", name)
            c = yaml.safe_load(open(os.path.join(d, "config.yaml")))
            cells[name] = {"val_loss_no_reg": r["val_loss"], "best_step": r["best_step"],
                           "val_loss_step0": r.get("val_loss_step0"), "val_curve": r["val_curve"],
                           "lr_scale": float(c["fine_tune"]["lr_scale"]),
                           "layer_decay": float(c["fine_tune"]["layer_decay"]),
                           "optimizer": c["training"]["optimizer"], "target_stats": c["fine_tune"]["target_stats"],
                           "iterations": int(c["training"]["iterations"]),
                           "prepd_std": float(json.load(open(os.path.join(d, "data_stats.json")))["prepd_std"][0])}
        trials.append({"hp_idx": hp, "objective": s["val_loss"], "hp": s["hp"], "cells": cells})
    failed = [os.path.basename(e) for e in glob.glob(os.path.join(ROOT, "sweeps", SWEEP, "error", "*.err"))
              if "Trial FAILED" in open(e, errors="replace").read()]
    json.dump({"trials": trials, "failed_logs": failed,
               "n_jobs": len(glob.glob(os.path.join(ROOT, "sweeps", SWEEP, "jobs", "trial_*.sh")))}, sys.stdout)


def gmean(v):
    v = list(v)
    return math.exp(sum(math.log(x) for x in v) / len(v))


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def main():
    raw = json.load(open(os.path.join(HERE, "data", "muon_hp_ft_trials.json")))
    ft = json.load(open(os.path.join(HERE, "data", "muon_ft.json")))
    ref = {f"{c['probe']}_{c['tag']}": c for c in ft if c.get("target_stats") == "parent_shared"
           and c["parent"] == "dimensionless_muon" and c["optimizer"] == "MuonAdamW"}
    assert len(ref) == 16, len(ref)
    ref_obj = gmean(c["val_loss_no_reg"] for c in ref.values())
    trials, rise, step0 = [], [], []
    for t in raw["trials"]:
        cells = t["cells"]
        assert set(cells) == set(ref), t["hp_idx"]
        for n, c in cells.items():
            assert c["optimizer"] == "MuonAdamW" and c["target_stats"] == "parent_shared", (t["hp_idx"], n)
            assert abs(c["lr_scale"] / t["hp"]["fine_tune.lr_scale"] - 1) < 1e-6, (t["hp_idx"], n)
            assert c["iterations"] == ref[n]["steps"], (t["hp_idx"], n)
            v = c["val_curve"]
            assert all(math.isfinite(x) for x in v), (t["hp_idx"], n)
            # the curve after the best checkpoint: how far the last validation ends above it, and where the best is
            rise.append({"hp_idx": t["hp_idx"], "cell": n, "last_over_best": v[-1] / min(v),
                         "best_position": v.index(min(v)) / (len(v) - 1)})
            # step 0 is the parent's weights on the same events and target: it must equal the reference cell's,
            # else the target (or its standardization) changed between the reference and the search
            r0 = c["val_loss_step0"] / ref[n]["val_loss_step0"]
            assert abs(r0 - 1) < STEP0_TOL, (t["hp_idx"], n, r0)
            step0.append(r0)
        obj = gmean(c["val_loss_no_reg"] for c in cells.values())
        assert abs(obj / t["objective"] - 1) < 1e-9
        trials.append({"hp_idx": t["hp_idx"], "lr_scale": t["hp"]["fine_tune.lr_scale"],
                       "layer_decay": t["hp"]["fine_tune.layer_decay"], "objective": obj,
                       "ref_over": ref_obj / obj,
                       "cells": {n: c["val_loss_no_reg"] for n, c in cells.items()}})
    trials.sort(key=lambda r: r["objective"])
    best = trials[0]
    pos = {}
    for k, (lo, hi, sc) in WINDOW.items():
        v = best["lr_scale" if "lr_scale" in k else "layer_decay"]
        pos[k.split(".")[1]] = (math.log10(v / lo) / math.log10(hi / lo)) if sc == "log" else (v - lo) / (hi - lo)
    gain = {n: ref[n]["val_loss_no_reg"] / best["cells"][n] for n in ref}
    per_probe = {p: gmean(gain[f"{p}_{t}"] for t in TAG_D) for p in PROBES}
    per_D = {f"D{d}": gmean(gain[f"{p}_{t}"] for p in PROBES) for t, d in TAG_D.items()}
    out = {"sweep": SWEEP, "n_jobs": raw["n_jobs"], "n_observed": len(trials), "failed": raw["failed_logs"],
           "after_best": rise,
           "reference": {"lr_scale": 1.0, "layer_decay": 1.0, "objective": ref_obj,
                         "cells": {n: c["val_loss_no_reg"] for n, c in ref.items()}},
           "best": {"hp_idx": best["hp_idx"], "lr_scale": best["lr_scale"], "layer_decay": best["layer_decay"],
                    "objective": best["objective"], "ref_over_best": best["ref_over"], "window_position": pos,
                    "cell_gain": gain, "cell_gain_min": min(gain.values()), "cell_gain_max": max(gain.values()),
                    "n_cells_better": sum(g > 1 for g in gain.values()),
                    "per_probe_gain": per_probe, "per_D_gain": per_D},
           "worst": {"hp_idx": trials[-1]["hp_idx"], "lr_scale": trials[-1]["lr_scale"],
                     "layer_decay": trials[-1]["layer_decay"], "objective": trials[-1]["objective"],
                     "ref_over": trials[-1]["ref_over"]},
           "n_trials_better_than_reference": sum(r["objective"] < ref_obj for r in trials),
           "step0_ratio_range": [min(step0), max(step0)],
           "trials": trials}
    json.dump(out, open(os.path.join(HERE, "data", "muon_hp_ft.json"), "w"), indent=1)
    figure(out)
    claims = {"muon_hp_ft.n_observed": len(trials), "muon_hp_ft.reference": ref_obj,
              "muon_hp_ft.best.objective": best["objective"], "muon_hp_ft.best.lr_scale": best["lr_scale"],
              "muon_hp_ft.best.layer_decay": best["layer_decay"], "muon_hp_ft.best.ref_over_best": best["ref_over"],
              "muon_hp_ft.best.cell_gain_min": out["best"]["cell_gain_min"],
              "muon_hp_ft.best.cell_gain_max": out["best"]["cell_gain_max"],
              "muon_hp_ft.best.n_cells_better": out["best"]["n_cells_better"],
              "muon_hp_ft.worst.objective": out["worst"]["objective"],
              "muon_hp_ft.worst.lr_scale": out["worst"]["lr_scale"],
              "muon_hp_ft.worst.layer_decay": out["worst"]["layer_decay"],
              "muon_hp_ft.n_better": out["n_trials_better_than_reference"],
              "muon_hp_ft.all.last_over_best_median": median(r["last_over_best"] for r in rise),
              "muon_hp_ft.all.best_position_median": median(r["best_position"] for r in rise),
              "muon_hp_ft.all.n_cells": len(rise),
              "muon_hp_ft.all.n_nonfinite": 0,
              "muon_hp_ft.best.last_over_best_median": median(r["last_over_best"] for r in rise
                                                              if r["hp_idx"] == best["hp_idx"]),
              "muon_hp_ft.best.best_position_median": median(r["best_position"] for r in rise
                                                             if r["hp_idx"] == best["hp_idx"]),
              "muon_hp_ft.step0_maxdev": max(abs(r - 1) for r in step0)}
    for p, g in per_probe.items():
        claims[f"muon_hp_ft.best.probe.{p}"] = g
    for d, g in per_D.items():
        claims[f"muon_hp_ft.best.{d}"] = g
    for k, v in pos.items():
        claims[f"muon_hp_ft.best.window.{k}"] = v
    for r in trials:
        claims[f"muon_hp_ft.trials.hp{r['hp_idx']}.objective"] = r["objective"]
    print(json.dumps({k: (float(f"{v:.3g}") if isinstance(v, float) else v) for k, v in claims.items()}))


def figure(out):
    import numpy as np
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    tr = out["trials"]
    sc = a.scatter([r["lr_scale"] for r in tr], [r["layer_decay"] for r in tr], c=[r["objective"] for r in tr],
                   cmap="viridis", zorder=3)
    a.scatter([1.0], [1.0], marker="*", c=[out["reference"]["objective"]], cmap="viridis", norm=sc.norm,
              edgecolors="k", s=160, zorder=4)
    a.scatter([out["best"]["lr_scale"]], [out["best"]["layer_decay"]], marker="o", facecolors="none",
              edgecolors=ps.C.vermillion, s=120, zorder=5)
    a.set_xscale("log")
    a.set_xlim(0.1, 10)
    a.set_xlabel(r"fine-tune lr scale")
    a.set_ylabel(r"layer decay")
    ps.colorbar(a, sc, r"$\langle$val loss$\rangle_{\rm geo}$")
    cols = [ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange] if hasattr(ps.C, "green") else None
    Ds = list(TAG_D.values())
    for i, (p, lab) in enumerate(PROBES.items()):
        g = [out["best"]["cell_gain"][f"{p}_{t}"] for t in TAG_D]
        kw = {"color": cols[i]} if cols else {}
        b.plot(Ds, g, "o-", label=lab, **kw)
    b.axhline(1.0, color="0.5", ls=":")
    from matplotlib.lines import Line2D
    h = b.get_legend_handles_labels()[0] + [
        Line2D([], [], color="0.5", ls=":", marker="*", markerfacecolor="w", markeredgecolor="k",
               label="untuned (scale 1, decay 1)"),
        Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor=ps.C.vermillion,
               label="best trial")]
    b.set_xscale("log")
    b.set_xlabel(r"$D$ (training events)")
    b.set_ylabel(r"untuned / best")
    ps.legend(b, "upper left", handles=h)
    ps.make_room(b, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "muon_hp_ft"))


if __name__ == "__main__":
    collect() if sys.argv[1:] == ["collect"] else main()
