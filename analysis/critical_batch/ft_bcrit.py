"""critical_batch D560 step: why batch 128 beats 2048 on q q~ -> Z+4g at 10^4 training events, against the pretraining
scan's critical batch.

Cells (all 400 epochs of 10^4 events: 31250 steps at batch 128, 1953 at 2048):
  muon / adam = the D507 scratch searches (analysis/zng_uq/data/d507_search_v2.json, every trial, train and val curves);
  F, Fn, Sfx  = the fine-tune study's searches at both batches (analysis/finetune_mechanism/data, search trials 't<i>').
Per cell, the best trial (lowest validation MSE(log|M|^2) = min val mu-MSE x std^2): its best validation, the epoch of
it, the train MSE there, and train over validation at the last validation of the stable phase. The two-batch critical batch at matched loss,
    B_crit(L) = (2048 - 128 r) / (r - 1),  r = E_2048(L) / E_128(L)   (McCandlish et al.'s E = E_min (1 + B/B_crit)),
with E_B(L) the fewest events to first reach L over the cell's trials, read in the stable phase (first 7/8 of a WSD run; the whole run for the ReduceLROnPlateau Adam cells),
at the levels both batches reach there. The pretraining scan's epochs at each target: events of each batch's best
trial (cbs_readout.json) over the 1.4e7-event pool.

    python3 analysis/critical_batch/ft_bcrit.py  -> data/ft_bcrit.json, figs/ft_regimes.{png,pdf}

D563, the same cells retuned at 50 epochs (sweeps_local/*_ep50*, design passed 2026-11-25): per cell the best trial's
best validation MSE, its epoch and lr (flagged in the outer 5% of the log lr window), the best-to-third-best spread;
per arm r50 = best_2048 / best_128 against r400 (the batch-128 gain is the horizon if r50 <= 1.22), per cell
E50/E400 = best at 50 over best at 400 epochs (better below 1/1.22, worse above 1.22); the figure of both horizons.

    python3 analysis/critical_batch/ft_bcrit.py ep50 [--collect]  -> data/ft_bcrit_ep50.json, figs/ft_ep50.{png,pdf}
    (--collect reads every observed trial's curves from its site into data/ft_bcrit_ep50_raw.json first)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

FTM = os.path.join(ROOT, "analysis", "finetune_mechanism", "data")
D507 = os.path.join(ROOT, "analysis", "zng_uq", "data", "d507_search_v2.json")
OUT = os.path.join(HERE, "data", "ft_bcrit.json")
FIG = os.path.join(HERE, "figs", "ft_regimes")
EPOCHS = 400
EVENTS = EPOCHS * 10_000
STABLE = 7 / 8                 # wsd_cooldown_frac 0.125 of the WSD arms
STABLE_ARM = {"adam": 1.0}     # ReduceLROnPlateau has no cooldown: the Adam cells are read over the whole run
STD2 = 1.1197787003759445 ** 2  # target std of the Z+4g pools (zng4g_r1_read, d507_v2), shared by every cell
POOL = 1.4e7                   # catalog_v4 training pool of the scan (critical_batch finding 2026-11-24T01:13)
NTRIAL = 16


def d507_cells():
    """(val, train, epochs) per trial; epochs from each trial's validation spacing (best_step / (argmin + 1))."""
    cells = {}
    for name, v in json.load(open(D507)).items():
        opt, b = ("adam" if "adam" in name else "muon"), int(name.split("_b")[1].split("_")[0])
        steps = 31250 if b == 128 else 1953
        rows = []
        for t in v["trials"]:
            val = np.array(t["val"], float) * STD2
            i = int(np.nanargmin(val))
            every = t["best_step"] / (i + 1)
            assert abs(every - round(every)) < 1e-6, (name, t["file"])
            ep = np.arange(1, len(val) + 1) * every / steps * EPOCHS
            rows.append((val, np.array(t["train"], float) * STD2, ep))
        cells[(opt, b)] = rows
    return cells


def local_cells():
    files = {"F": "zng4g_readout_raw.json", "Fn": "zng4g_ftnoset_raw_b{b}.json", "Sfx": "zng4g_r1_raw_b{b}.json"}
    cells = {}
    for arm, f in files.items():
        for b in (128, 2048):
            d = json.load(open(os.path.join(FTM, f.format(b=b))))
            tr = {int(k.split("|")[2][1:]): v for k, v in d.items()
                  if k.split("|")[:2] == [arm, str(b)] and k.split("|")[2].startswith("t") and v.get("val_mse_curve")}
            rows = []
            for i in sorted(tr)[:NTRIAL]:
                val = np.array(tr[i]["val_mse_curve"], float) * STD2
                train = np.array(tr[i].get("train_mse_curve") or [np.nan] * len(val), float) * STD2
                rows.append((val, train, np.arange(1, len(val) + 1) / len(val) * EPOCHS))
            cells[(arm, b)] = rows
    return cells


def best_row(rows, stable=STABLE, epochs=EPOCHS):
    val, train, ep = min(rows, key=lambda r: np.nanmin(r[0]))[:3]
    i = int(np.nanargmin(val))
    return {"val_best": float(val[i]), "epoch_best": float(ep[i]), "frac_best": float(ep[i] / epochs),
            "train_at_best": float(train[i]), "train_over_val": float(train[i] / val[i]),
            "val_last_over_best": float(val[-1] / val[i]), "n_trials": len(rows),
            **stable_end(val, train, ep, stable, epochs)}


def stable_end(val, train, ep, stable=STABLE, epochs=EPOCHS):
    """Train over validation at the last validation of the stable phase (before the cooldown memorises)."""
    j = int(np.nonzero(in_stable(ep, stable, epochs))[0][-1])
    return {"epoch_stable_end": float(ep[j]), "val_stable_end": float(val[j]), "train_over_val_stable_end": float(train[j] / val[j])}


def in_stable(ep, stable, epochs=EPOCHS):
    return ep <= stable * epochs * (1 + 1e-9)


def events_to(val, ep, L, stable=STABLE):
    """Events at the first stable-phase validation <= L, log-interpolated from the one before; None if not reached
    there or reached at the first validation."""
    m = in_stable(ep, stable)
    c, e = val[m], ep[m]
    ok = np.isfinite(c) & (c <= L)
    if not ok.any() or int(np.argmax(ok)) == 0:
        return None
    i = int(np.argmax(ok))
    a, b = np.log(c[i - 1]), np.log(c[i])
    w = (a - np.log(L)) / (a - b) if np.isfinite(a) and a > b else 1.0
    return float((e[i - 1] + w * (e[i] - e[i - 1])) / EPOCHS * EVENTS)


def two_batch(r128, r2048, nlev=8, stable=STABLE):
    """B_crit at levels both batches reach in the stable phase (r <= 1: above any value; r >= 16: below 0)."""
    floor = {b: min(np.nanmin(v[in_stable(ep, stable)]) for v, _, ep in rows) for b, rows in ((128, r128), (2048, r2048))}
    start = min(np.nanmin(v[:2]) for v, _, _ in r128 + r2048)
    out = []
    for L in np.geomspace(start, max(floor.values()) * 1.001, nlev):
        E = {b: [x for x in (events_to(v, ep, L, stable) for v, _, ep in rows) if x] for b, rows in ((128, r128), (2048, r2048))}
        if E[128] and E[2048]:
            r = min(E[2048]) / min(E[128])
            out.append({"L": float(L), "E128": min(E[128]), "E2048": min(E[2048]), "r": r,
                        "B_crit": (2048 - 128 * r) / (r - 1) if r > 1 else None})
    return {"stable_floor": {str(b): float(f) for b, f in floor.items()}, "levels": out}


def plot(cells):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2, nrows=2)
    panels = (("muon", axs[0][0]), ("adam", axs[0][1]), ("Fn", axs[1][0]))
    for key, ax in panels:
        for b, col in ((2048, ps.C.blue), (128, ps.C.vermillion)):
            val, train, ep = min(cells[(key, b)], key=lambda r: np.nanmin(r[0]))
            ax.plot(ep, val, color=col, label=f"batch {b}, validation")
            ax.plot(ep, train, color=col, ls="--", label=f"batch {b}, train")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("epochs over the $10^4$ training events")
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, r"$q\bar q\to Z+4g$")
    ps.legend_cell(axs[1][1], axs[0][0])
    ps.save(fig, FIG)


EP50, EDGE, BAR = 50, 0.05, 1.22   # design (D563): horizon, outer 5% of the log lr window, the 1.22 spread
SW50 = {"muon": "tp_zng_d1e4_muon_b{b}_ep50_zgggg", "adam": "tp_zng_d1e4_adam_b{b}_ep50_zgggg",
        "Fn": "ftmech_zng4g_ftnoset_b{b}_ep50"}
STEPS50 = {128: 3906, 2048: 244}
RAW50 = os.path.join(HERE, "data", "ft_bcrit_ep50_raw.json")
REMOTE50 = r'''
import json, sys
out = {}
for a in sys.argv[1:]:
    lab, p = a.split("#")
    try:
        d = json.load(open(p))
        out[lab] = {k: d.get(k) for k in ("val_loss", "validate_every", "best_step", "val_mse_curve", "train_eval_mse_curve")}
    except Exception as e:
        out[lab] = {"error": repr(e)[:100]}
print("FTB50 " + json.dumps(out))
'''


def collect50():
    """Every observed trial of the six searches: its curves from its site, its lr from the local driver state."""
    sys.path.insert(0, os.path.join(ROOT, "analysis", "finetune_mechanism"))
    from zng4g_readout import driven
    from analysis.common.readers import site_python
    items, lr = {}, {}
    for arm, pat in SW50.items():
        for b in (128, 2048):
            name = pat.format(b=b)
            hp = {v["hp_idx"]: v["hp"]["training.lr"] for v in
                  json.load(open(os.path.join(ROOT, "sweeps_local", name, "driver_state.json")))["trials"].values()}
            for ti, h, v, site, rd, rp in driven(name):
                lab = f"{arm}|{b}|t{ti}"
                items.setdefault(site, []).append(f"'{lab}#{rp}'")
                lr[lab] = hp[h]
    raw = {}
    for site, args in items.items():
        raw.update({k: dict(v, lr=lr[k], site=site) for k, v in site_python(REMOTE50, args, "FTB50", site=site).items()})
    json.dump(raw, open(RAW50, "w"))


def cells50():
    """(val, train, epochs, lr) per trial; validation i sits at step (i + 1) x validate_every."""
    cells = {}
    for lab, t in json.load(open(RAW50)).items():
        arm, b, _ = lab.split("|")
        assert "error" not in t, (lab, t)
        val = np.array(t["val_mse_curve"], float)
        assert abs(np.nanmin(val) / t["val_loss"] - 1) < 1e-6, lab   # the searches' selection is the val mu-MSE
        ep = np.arange(1, len(val) + 1) * t["validate_every"] / STEPS50[int(b)] * EP50
        cells.setdefault((arm, int(b)), []).append((val * STD2, np.array(t["train_eval_mse_curve"], float) * STD2, ep, t["lr"]))
    return cells


def lr_window(arm, b):
    import yaml
    sp = yaml.safe_load(open(os.path.join(ROOT, "sweep", f"sweep_config_{SW50[arm].format(b=b)}.yaml")))["search_space"]
    w = next(x for x in sp if x["name"] == "training.lr")
    return w["low"], w["high"]


def read50(c400, c50):
    res = {"cells": {}, "arms": {}}
    for (arm, b), rows in sorted(c50.items()):
        r = best_row(rows, STABLE_ARM.get(arm, STABLE), EP50)
        best = sorted(np.nanmin(x[0]) for x in rows)
        lr = min(rows, key=lambda x: np.nanmin(x[0]))[3]
        lo, hi = lr_window(arm, b)
        pos = float(np.log(lr / lo) / np.log(hi / lo))
        e400 = best_row(c400[(arm, b)], STABLE_ARM.get(arm, STABLE))["val_best"]
        e = r["val_best"] / e400
        res["cells"][f"{arm}|{b}"] = {**r, "lr": lr, "lr_window": [lo, hi], "lr_pos": pos, "lr_edge": not EDGE <= pos <= 1 - EDGE,
                                      "third_over_best": float(best[2] / best[0]) if len(best) > 2 else None,
                                      "val_best_400": e400, "E50_over_E400": e,
                                      "read": "better" if e < 1 / BAR else "worse" if e > BAR else "equal"}
    for arm in SW50:
        if all(f"{arm}|{b}" in res["cells"] for b in (128, 2048)):
            r50 = res["cells"][f"{arm}|2048"]["val_best"] / res["cells"][f"{arm}|128"]["val_best"]
            r400 = res["cells"][f"{arm}|2048"]["val_best_400"] / res["cells"][f"{arm}|128"]["val_best_400"]
            res["arms"][arm] = {"r50": r50, "r400": r400, "read": "horizon" if r50 <= BAR else "persists"}
    res["n_trials"] = {k: v["n_trials"] for k, v in res["cells"].items()}
    return res


def plot50(c400, c50):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2, nrows=2)
    cols = {2048: ps.C.blue, 128: ps.C.vermillion}
    for (arm, ax) in (("muon", axs[0][0]), ("adam", axs[0][1]), ("Fn", axs[1][0])):
        for b in (2048, 128):
            for cs, ls in ((c400, "-"), (c50, "--")):
                if (arm, b) in cs:
                    val, _, ep = min(cs[(arm, b)], key=lambda r: np.nanmin(r[0]))[:3]
                    ax.plot(ep, val, color=cols[b], ls=ls)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("epochs over the $10^4$ training events")
        ax.set_ylabel(r"validation MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, r"$q\bar q\to Z+4g$")
    ax = axs[1][1]
    names = (("muon", "Muon"), ("adam", "Adam"), ("Fn", "fine-tuned"))
    for b, dx in ((2048, -0.12), (128, 0.12)):
        for cs, ls, mfc, ep in ((c400, "-", None, 400), (c50, "--", "none", 50)):
            xs = [i + dx for i, (k, _) in enumerate(names) if (k, b) in cs]
            ys = [min(np.nanmin(r[0]) for r in cs[(k, b)]) for k, _ in names if (k, b) in cs]
            ax.plot(xs, ys, "o", color=cols[b], mfc=mfc or cols[b])
            ax.plot([], [], "o", ls=ls, color=cols[b], mfc=mfc or cols[b], label=f"batch {b}, {ep} epochs")   # curves and bests
    ax.set_xticks(range(len(names)), [n for _, n in names])
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_yscale("log")
    ax.set_xlabel("arm")
    ax.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "ft_ep50"))


def main50():
    if "--collect" in sys.argv:
        collect50()
    c400, c50 = {**d507_cells(), **local_cells()}, cells50()
    res = read50(c400, c50)
    json.dump(res, open(os.path.join(HERE, "data", "ft_bcrit_ep50.json"), "w"), indent=1)
    print(json.dumps(res))
    plot50(c400, c50)


def main():
    cells = {**d507_cells(), **local_cells()}
    res = {"cells": {f"{k}|{b}": best_row(rows, STABLE_ARM.get(k, STABLE)) for (k, b), rows in cells.items()},
           "two_batch": {k: two_batch(cells[(k, 128)], cells[(k, 2048)], stable=STABLE_ARM.get(k, STABLE)) for k in ("muon", "adam", "F", "Fn", "Sfx")}}
    for k in ("muon", "adam", "F", "Fn", "Sfx"):
        res["cells"][f"{k}|ratio_2048_over_128"] = res["cells"][f"{k}|2048"]["val_best"] / res["cells"][f"{k}|128"]["val_best"]
    ro = json.load(open(os.path.join(HERE, "data", "cbs_readout.json")))
    res["scan_epochs"] = {t: {str(x["B"]): x["E"] / POOL for x in ro[t]["batches"] if x.get("E")}
                          for t in ro if t.startswith("-")}
    bc = [x["B_crit"] for x in res["two_batch"]["adam"]["levels"] if x["B_crit"] is not None]
    res["summary"] = {"adam_bcrit_min": min(bc), "adam_bcrit_max": max(bc), "adam_levels": len(bc),
                      **{f"{k}_r": res["two_batch"][k]["levels"][0]["r"] for k in ("muon", "Sfx", "F", "Fn")},
                      **{f"{k}_bcrit": res["two_batch"][k]["levels"][0]["B_crit"] for k in ("F", "Fn")}}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res))
    plot(cells)


if __name__ == "__main__":
    main50() if "ep50" in sys.argv[1:] else main()
