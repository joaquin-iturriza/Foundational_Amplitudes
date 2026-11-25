"""critical_batch D560 step: the critical batch of the q q~ -> Z+4g runs at 10^4 events, read from the existing pairs of
batch 128 and 2048 searches (same 400 epochs), against the pretraining scan's.

For each arm searched at both batches: E_B(L) = events seen when the validation MSE first reaches L (log-interpolated
between validations), the lowest over the arm's search trials at that batch (as the scan's best trial per target), read
only in the stable phase (validations up to 7/8 of the run: WSD cooldown 1/8; the same cut for Adam's plateau schedule).
McCandlish et al.'s law E = E_min (1 + B / B_crit) through the two batches gives
    B_crit(L) = (2048 - 128 r) / (r - 1),  r = E_2048(L) / E_128(L)
(r <= 1: no gain from the small batch, B_crit above any value; r >= 16: B_crit below 0, flagged). The same two-batch
reading on the scan's batches 2048 and 32768 checks it against the scan's five-batch fit.

    python3 analysis/critical_batch/ft_bcrit.py [--cached]  -> data/ft_bcrit.json, figs/ft_bcrit.{png,pdf}
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

FTM = os.path.join(ROOT, "analysis", "finetune_mechanism", "data")
RAW = os.path.join(HERE, "data", "ft_bcrit_raw.json")
OUT = os.path.join(HERE, "data", "ft_bcrit.json")
EVENTS = 400 * 10_000          # 400 epochs of 10^4 events, every arm at both batches (sweep configs t_steps 1953 / 31250)
STABLE = 7 / 8                 # wsd_cooldown_frac 0.125
NTRIAL = 16                    # search trials per arm and batch (the D507 searches' 8 + 8)
D507 = {"muon": "tp_zng_d1e4_muon_b{b}_zgggg", "adam": "tp_zng_d1e4_adam_b{b}_zgggg"}

REMOTE = r'''
import glob, json, os, sys, yaml
sys.path.insert(0, os.getcwd()); import siteconf
out = {}
for a in sys.argv[1:]:
    lab, name = a.split("#")
    sw = os.path.join(siteconf.SWEEP_DIR, name)
    for f in sorted(glob.glob(os.path.join(sw, "results", "hp*_t*.json"))):
        hp = int(os.path.basename(f)[2:6]); j = json.load(open(f))
        try:
            lr = yaml.safe_load(open(os.path.join(siteconf.PROJECT_DIR, "runs", name, "trial_%04d" % hp, "config.yaml")))["training"]["lr"]
        except Exception as e:
            lr = None
        out[f"{lab}|hp{hp}"] = {"val_loss": j.get("val_loss"), "val_mse_curve": j.get("val_mse_curve"), "lr": lr}
print("FTBC " + json.dumps(out))
'''


def collect():
    from analysis.common.readers import site_python
    args = [f"'{opt}|{b}#{name.format(b=b)}'" for opt, name in D507.items() for b in (128, 2048)]
    return site_python(REMOTE, args, "FTBC")


def local_arms():
    """Fine-tune-study arms searched at both batches: the search trials ('t<i>' keys), first NTRIAL by trial index."""
    files = {"F": "zng4g_readout_raw.json", "Fn": "zng4g_ftnoset_raw_b{b}.json", "Sfx": "zng4g_r1_raw_b{b}.json"}
    arms = {}
    for arm, f in files.items():
        for b in (128, 2048):
            d = json.load(open(os.path.join(FTM, f.format(b=b))))
            tr = {int(k.split("|")[2][1:]): v["val_mse_curve"] for k, v in d.items()
                  if k.split("|")[:2] == [arm, str(b)] and k.split("|")[2].startswith("t") and v.get("val_mse_curve")}
            arms.setdefault(arm, {})[b] = [tr[i] for i in sorted(tr)[:NTRIAL]]
    return arms


def events_to(curve, L, n_stable):
    """Events at the first validation <= L, log-interpolated from the one before; None if never in the stable phase or
    already at the first validation (unresolved)."""
    c = np.asarray(curve, float)[:n_stable]
    n = len(curve)
    ok = np.isfinite(c) & (c <= L)
    if not ok.any():
        return None
    i = int(np.argmax(ok))
    if i == 0:
        return None
    f0, f1 = i / n, (i + 1) / n
    a, b = np.log(c[i - 1]), np.log(c[i])
    w = (a - np.log(L)) / (a - b) if np.isfinite(a) and a > b else 1.0
    return (f0 + w * (f1 - f0)) * EVENTS


def bcrit(r, b_small, b_big):
    return (b_big - b_small * r) / (r - 1) if r > 1 else float("inf")


def read_pair(curves, b_small=128, b_big=2048, nlev=12):
    n = {b: int(round(STABLE * len(curves[b][0]))) for b in (b_small, b_big)}
    reach = {b: min(min(np.asarray(c, float)[:n[b]][np.isfinite(np.asarray(c, float)[:n[b]])]) for c in curves[b])
             for b in (b_small, b_big)}
    first = {b: min(np.asarray(c, float)[1] for c in curves[b] if np.isfinite(c[1])) for b in (b_small, b_big)}
    hi, lo = min(first.values()), max(reach.values()) * 1.001
    rows = []
    for L in np.geomspace(hi, lo, nlev):
        E = {b: [e for e in (events_to(c, L, n[b]) for c in curves[b]) if e is not None] for b in (b_small, b_big)}
        if not E[b_small] or not E[b_big]:
            continue
        e_s, e_b = min(E[b_small]), min(E[b_big])
        r = e_b / e_s
        rows.append({"L": float(L), "E_small": e_s, "E_big": e_b, "r": r, "B_crit": bcrit(r, b_small, b_big),
                     "flag": "r>=B_big/B_small" if r >= b_big / b_small else ("r<=1" if r <= 1 else "")})
    return {"reach": {str(b): float(v) for b, v in reach.items()}, "rows": rows}


def scan_check():
    """The same two-batch formula on the scan's best S_B (steps) at 2048 and 32768, against its five-batch fit."""
    ro = json.load(open(os.path.join(HERE, "data", "cbs_readout.json")))
    return ro


def main():
    if "--cached" in sys.argv:
        raw = json.load(open(RAW))
    else:
        raw = collect()
        json.dump(raw, open(RAW, "w"))
    arms = {}
    for opt in D507:
        for b in (128, 2048):
            tr = {int(k.split("|hp")[1]): v["val_mse_curve"] for k, v in raw.items()
                  if k.startswith(f"{opt}|{b}|") and v.get("val_mse_curve")}
            arms.setdefault(f"S_{opt}", {})[b] = [tr[h] for h in sorted(tr)]
    arms.update(local_arms())
    res = {"n_trials": {a: {str(b): len(c) for b, c in v.items()} for a, v in arms.items()}}
    res["pairs"] = {a: read_pair(v) for a, v in arms.items()}
    json.dump(res, open(OUT, "w"), indent=1)
    for a, p in res["pairs"].items():
        print(a, res["n_trials"][a], "reach", {k: round(v, 5) for k, v in p["reach"].items()})
        for r in p["rows"]:
            print(f"   L {r['L']:.4g}  E128 {r['E_small']:.3g}  E2048 {r['E_big']:.3g}  r {r['r']:.2f}  "
                  f"B_crit {r['B_crit']:.4g} {r['flag']}")


if __name__ == "__main__":
    main()
