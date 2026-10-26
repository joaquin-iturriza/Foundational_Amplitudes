"""Audit of the fine-tune EMA start (the bug fixed in ddfc0314): which reported fine-tune values were read at a
validation that still carried the random initialisation.

The bug. Before ddfc0314, base_experiment.init_model built the EMA (torch_ema.ExponentialMovingAverage, its shadow a
copy of the parameters at that moment) on the fresh initialisation, and experiment.init_model loaded the parent's
weights afterwards; the shadow was never reset. torch_ema's update n (counting from 1) uses the decay
d_n = min(beta, (1 + n) / (10 + n)), so after N updates the shadow is
    sum_n w_n theta_n + W(N) theta_random,   W(N) = prod_{n=1}^{N} min(beta, (1 + n) / (10 + n)),
which, while (1 + n)/(10 + n) < beta (n < 80 at beta = 0.9, n < 142 at 0.941), is
    W(N) = 10! (N + 1)! / (N + 10)!:  2/11 after 1, 1/22 after 2, 1.4e-2 after 3, 5.0e-3 after 4, 2.0e-3 after 5,
                                      8.7e-4 after 6, 4.1e-4 after 7, 6.0e-5 after 10, 1.4e-11 after 80.
In that range W does not depend on beta, so for every fine-tune's early validations the decay is irrelevant; a trial
without a recorded decay takes config/default.yaml's.
Every validation of a run with EMA on scores the EMA weights, so validation j (every `validate_every` updates) carries
W(j * validate_every); the checkpoint choice and every reported fine-tune value use them. A fine-tune validates every
max(1, int(0.01 T)) updates (training.validate_frac 0.01): 1 at T = 176 (D = 10), 7 at 704 (D = 10^1.5), 28 and more
above.

What the audit reads (no GPU, no job): the stored validation curves of every fine-tune with its EMA flag
  analysis/transfer/scratch_sweeps.json        the transfer study's sweeps (ladder rungs, equal-D, final, synthetic,
                                               singular-limit probes, ...), every trial
  analysis/transfer/data/lowd_dyhpo_sweeps.json  the small-D study's per-method searches
  analysis/transfer/data/lowd_ft_curves.json   the small-D study's first round (each method at the cell's best HPs;
                                               EMA from lowd_ft_spec.json)
  analysis/transfer/data/ft_harm.json          par:ft_harm's rows (linear probe then fine-tune's first validation)
  analysis/catalog_v3/data/catft_curves.json   the Muon test's catalog_v3 fine-tunes and the reruns at ddfc0314
                                               (analysis/catalog_v3/collect_catft_curves.py); EMA on, the parents'
                                               decay 0.9412
No cut-off decides anything here. Per search (a sweep, or a fixed-HP cell) the reported value is its best trial at
its best validation; the audit records the update count of that validation and W there, and tabulates searches by
it. A trial whose EMA flag is recorded nowhere (the flag searched and not stored with the trial) is counted apart,
never as off. Listed for reading, not classified: per search, the trials with EMA on whose curve, from update 4 on,
is lowest at its first validation there (it rose from the start, so its clean minimum could lie in its first three
validations), with their value there over the search's reported value. Update 4 is this audit's own choice, open and
unsourced; the listing feeds no count or claim in docs/results.tex. The reruns at the fix had EMA on, restarted on
the parent's weights: their own state in the table ("restarted"), W = 0. The reruns at the fix
against the same cells before it are the measured size of the effect (run to run noise included: training is not
bitwise reproducible, the curves differ by up to 20 % late in the run); the script prints how many reruns are in.
    python3 analysis/transfer/ema_start_audit.py   -> analysis/transfer/data/ema_start_audit.json, summary on stdout
"""
import json, math, os, re
from collections import Counter, defaultdict

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_BETA = float(yaml.safe_load(open(os.path.join(ROOT, "config", "default.yaml")))["training"]["ema_decay"])
CATFT_BETA = 0.9412073      # the catalog_v3 short parents' ema_decay (their config.yaml)
RISE_FROM = 4               # the listing's start update: the audit's own choice, open (unsourced); it feeds no count
N_RERUN = 128               # the reruns at the fix: 2 kinds x 4 parent-and-optimizer pairs x 4 probes x 4 D


def W_curve(n_val, ve, beta):
    """W at each of n_val validations, every ve updates (the product carried along; below 1e-12 it is 0)."""
    ws, w, n = [], 1.0, 0
    for i in range(1, n_val + 1):
        while w > 1e-12 and n < i * ve:
            n += 1
            w *= min(beta, (1 + n) / (10 + n))
        ws.append(w if w > 1e-12 else 0.0)
    return ws


def as_bool(x):
    if isinstance(x, str):
        return {"true": True, "false": False}.get(x.lower())
    return None if x is None else bool(x)


def trial_rec(curve, ve, ema, beta):
    """ema: True, False or None (unknown)."""
    j = min(range(len(curve)), key=lambda i: curve[i])
    ws = W_curve(len(curve), ve, beta) if ema else [0.0] * len(curve)
    late = [i for i in range(len(curve)) if (i + 1) * ve >= RISE_FROM]
    rose = bool(ema) and ve < RISE_FROM and late and min(late, key=lambda i: curve[i]) == late[0]
    return dict(best=curve[j], best_update=(j + 1) * ve, W_best=ws[j], ema=ema, rose=bool(rose),
                rise_value=curve[late[0]] if rose else None)


def search(trials):
    tr = [t for t in trials if t["best"] is not None and math.isfinite(t["best"])]
    if not tr:
        return None
    win = min(tr, key=lambda t: t["best"])
    v = win["best"]
    rose = sorted(t["rise_value"] / v for t in tr if t["rose"] and t is not win)
    return dict(value=v, best_update=win["best_update"], W_best=win["W_best"], win_ema=win["ema"],
                n_trials=len(tr), n_ema_on=sum(t["ema"] is True for t in tr),
                n_ema_unknown=sum(t["ema"] is None for t in tr),
                win_rose=win["rose"], rose_over_value=rose)


def transfer(path, label):
    S = json.load(open(path))
    out = {}
    for name, trials in S.items():
        if name.endswith("_002"):
            continue
        trials = trials + S.get(name + "_002", [])
        is_ft = (any(t.get("fine_tune") is not None for t in trials)
                 or (re.search(r"_[a-z0-9]*ft[a-z0-9]*_", name) is not None and "scr" not in name))
        if not is_ft:
            continue
        recs = []
        for t in trials:
            cur, ve = t.get("val_curve"), t.get("validate_every")
            if not cur or not ve:
                continue
            h = t.get("hps") or {}
            ema = as_bool(h.get("ema")) if "ema" in h else as_bool(t.get("ema"))
            beta = float(h.get("ema_decay") or DEFAULT_BETA)
            recs.append(trial_rec(cur, int(ve), ema, beta) | {"ve": int(ve)})
        f = search(recs)
        if f:
            out[name] = f | dict(study=label, ve=min(r["ve"] for r in recs))
    return out


def lowd_round():
    spec = json.load(open(os.path.join(HERE, "data", "lowd_ft_spec.json")))
    ema = {f"{c['probe']}/r{c['rung']}_d{c['k']}":
           (as_bool(c["hp"].get("ema")), float(c["hp"].get("training.ema_decay", DEFAULT_BETA))) for c in spec}
    C = json.load(open(os.path.join(HERE, "data", "lowd_ft_curves.json")))
    out = {}
    for key, r in C.items():
        method, cell = key.split("/", 1)
        if method == "readout" or not r.get("val_curve"):
            continue
        e, beta = ema[cell]
        out["lowd_round/" + key] = search([trial_rec(r["val_curve"], int(r["validate_every"]), e, beta)]) | dict(
            study="small-D first round", ve=int(r["validate_every"]))
    return out


def catft():
    C = json.load(open(os.path.join(ROOT, "analysis", "catalog_v3", "data", "catft_curves.json")))
    out, pairs = {}, []
    for key, r in C.items():
        if not r.get("val_curve"):
            continue
        kind = key.split("/")[1]
        rerun = "emafix" in key
        f = search([trial_rec(r["val_curve"], int(r["validate_every"]), not rerun, CATFT_BETA)])
        if rerun:
            f["win_ema"] = "restarted"          # EMA on, restarted on the parent's weights (ddfc0314): W = 0
        f.update(study="Muon test catalog fine-tunes" + (", rerun at the fix" if rerun else ""),
                 ve=int(r["validate_every"]), scratch=kind.startswith("scratch"))
        if rerun:
            old = re.sub(r"^zz_catft[a-z_]*emafix/", "zz_catft/", key)
            if old in C:
                pairs.append(dict(cell=key.split("/", 1)[1], before=C[old]["val_loss"],
                                  before_update=C[old]["best_step"] * C[old]["validate_every"],
                                  after=r["val_loss"], after_update=r["best_step"] * r["validate_every"],
                                  step0=r.get("val_loss_step0"), ratio=C[old]["val_loss"] / r["val_loss"]))
        out["catft/" + key] = f
    return out, pairs, sum("emafix" in k for k in C)


def lpft_first_steps():
    """par:ft_harm's "first steps" count: linear probe then fine-tune starts at the read-out, and its first validation
    is compared with it; split by W at that first validation."""
    rows = json.load(open(os.path.join(HERE, "data", "ft_harm.json")))["rows"]
    spec = json.load(open(os.path.join(HERE, "data", "lowd_ft_spec.json")))
    ema1 = {(c["probe"], c["rung"], c["k"]): as_bool(c["hp"].get("ema")) for c in spec}
    sw = json.load(open(os.path.join(HERE, "data", "lowd_dyhpo_sweeps.json")))

    def ema_dyhpo(x):          # the method's search: its best trial, as ft_harm.py takes it
        n = f"tp3_lowd_lpft_r{x['rung']}_{x['probe']}_d{x['k']}"
        tr = [t for t in sw.get(n, []) + sw.get(n + "_002", []) if t.get("val_loss") is not None]
        return as_bool((min(tr, key=lambda t: t["val_loss"]).get("hps") or {}).get("ema")) if tr else None

    lp = [x for x in rows if x["method"] == "lpft" and x.get("readout") is not None]
    for x in lp:
        x["ema"] = ema1.get((x["probe"], x["rung"], x["k"])) if x["source"] == "first round" else ema_dyhpo(x)
        x["W_first"] = W_curve(1, x["first_step"], DEFAULT_BETA)[0] if x["ema"] else 0.0
    med = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else 0.5 * sum(sorted(v)[len(v) // 2 - 1:len(v) // 2 + 1])
    summ = lambda L: dict(runs=len(L), first_above_readout=sum(x["first"] > x["readout"] for x in L),
                          median_first_over_readout=med([x["first"] / x["readout"] for x in L]) if L else None)
    by = defaultdict(list)
    for x in lp:
        by["EMA unknown" if x["ema"] is None else f"first validation W {x['W_first']:.1e}"].append(x)
    # exposed: EMA on and a first validation after one update (the D = 10 fine-tunes, validate_every 1), where the
    # random initialisation still weighs W(1) = 2/11; every other run's first validation is at update 7 or later
    exposed = [x for x in lp if x["ema"] and x["first_step"] == 1]
    clean = [x for x in lp if not (x["ema"] and x["first_step"] == 1)]
    return dict(all=summ(lp), exposed=summ(exposed), clean=summ(clean),
                **{k: summ(v) for k, v in sorted(by.items())})


def main():
    res = {}
    res.update(transfer(os.path.join(HERE, "scratch_sweeps.json"), "transfer study"))
    res.update(transfer(os.path.join(HERE, "data", "lowd_dyhpo_sweeps.json"), "small-D searches"))
    res.update(lowd_round())
    cat, pairs, n_rerun = catft()
    res.update(cat)
    print("W(N): " + ", ".join(f"{N}: {W_curve(1, N, DEFAULT_BETA)[0]:.2e}" for N in (1, 2, 3, 4, 5, 6, 7, 10, 28, 80)))
    upd = lambda u: str(u) if u <= 7 else ">7"
    table = defaultdict(Counter)
    for name, f in res.items():
        if f.get("scratch"):
            continue
        col = ("EMA off" if f["win_ema"] is False else "EMA unknown" if f["win_ema"] is None
               else "restarted" if f["win_ema"] == "restarted" else upd(f["best_update"]))
        table[(f["study"], f["ve"])][col] += 1
    cols = ["1", "2", "3", "4", "5", "6", "7", ">7", "EMA off", "EMA unknown", "restarted"]
    print("\nsearches by the update count of their reported validation (winner's EMA on), else the winner's EMA state")
    print(f"  {'study':45s} {'every':>5s} " + " ".join(f"{c:>6s}" for c in cols))
    for (study, ve), c in sorted(table.items()):
        print(f"  {study:45s} {ve:5d} " + " ".join(f"{c.get(k, 0):6d}" for k in cols))
    print("\nsearches with any trial of unknown EMA:")
    for name, f in sorted(res.items()):
        if f["n_ema_unknown"]:
            print(f"  {name:55s} {f['n_ema_unknown']} of {f['n_trials']} trials, every {f['ve']}")
    print(f"\nreported value at update <= 7 with EMA on (update, W there):")
    for name, f in sorted(res.items()):
        if f["win_ema"] is True and f["best_update"] <= 7 and not f.get("scratch") and "rerun" not in f["study"]:
            print(f"  {name:65s} {f['value']:.4g} at update {f['best_update']} (W {f['W_best']:.1e})")
    print(f"\ntrials with EMA on that rose from update {RISE_FROM} on (value there over the search's reported value;"
          f" update {RISE_FROM} is the audit's own choice, open, feeding no count):")
    for name, f in sorted(res.items()):
        r = f["rose_over_value"] + ([1.0] if f["win_rose"] else [])
        if r and not f.get("scratch"):
            print(f"  {name:55s} " + ", ".join(f"{x:.2f}" for x in sorted(r)[:4])
                  + (" (the winner itself)" if f["win_rose"] else ""))
    print(f"\nbefore and after the fix, same cells ({n_rerun} of {N_RERUN} reruns in"
          + (", partial" if n_rerun < N_RERUN else "") + "; ratio = before / after):")
    for p in pairs:
        print(f"  {p['cell']:72s} {p['before']:.4g} (update {p['before_update']}) -> {p['after']:.4g} "
              f"(update {p['after_update']})  ratio {p['ratio']:.2f}  step 0 {p['step0']}")
    lp = lpft_first_steps()
    print("\nlinear probe then fine-tune, first validation against the read-out (par:ft_harm):")
    for k, v in lp.items():
        print(f"  {k:30s} " + json.dumps(v))
    flagged = {n: f for n, f in res.items() if (f["win_ema"] is True and f["best_update"] <= 7) or f["rose_over_value"]
               or f["win_rose"] or f["n_ema_unknown"]}
    t1 = Counter()
    for (study, ve), c in table.items():
        if study == "transfer study" and ve == 1:
            t1 += c
    cells = json.load(open(os.path.join(HERE, "data", "lowd_ft.json")))["cells"]
    za = cells["ee_Za/r1/d2"]
    za_audit = res["lowd_round/lpft/ee_Za/r1_d2"]
    claims = dict(W1=W_curve(1, 1, DEFAULT_BETA)[0], W2=W_curve(1, 2, DEFAULT_BETA)[0],
                  W3=W_curve(1, 3, DEFAULT_BETA)[0],
                  transfer_d10=dict(searches=sum(t1.values()), at_update_1_to_3=sum(t1.get(k, 0) for k in "123")),
                  lpft=dict((k, lp[k]) for k in ("all", "exposed", "clean")),
                  lowd_za_r1_d10=dict(lpft=za["lpft"], readout=za["readout"], update=za_audit["best_update"],
                                      W=za_audit["W_best"], gain_over_default=za["gain_over_default"]["lpft"]))
    print("\nclaims:")
    print(json.dumps(claims))
    json.dump(dict(W={N: W_curve(1, N, DEFAULT_BETA)[0] for N in range(1, 11)}, claims=claims,
                   table={f"{s} / every {ve}": dict(c) for (s, ve), c in table.items()},
                   listed=flagged, pairs=pairs, reruns_in=n_rerun, lpft_first_steps=lp),
              open(os.path.join(HERE, "data", "ema_start_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
