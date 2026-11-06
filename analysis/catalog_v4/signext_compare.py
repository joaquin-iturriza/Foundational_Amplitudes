"""The one-loop sign extension on catalog_v4 against its control, as the passed design of v3_prep_explore reads it.

Arms (scripts/job_catalog_v3_short.sh, both at trial_1860's HPs with Muon, seeds 42 to 45):
  control    v4_dimensionless_muon, v4_dimensionless_muon_seed2, _seed3, _seed4
  extension  v4_dimensionless_muon_more_signed, v4_dimensionless_muon_more_signed_seed2, _seed3, _seed4 (more sign-changing one-loop
             training pools, recipes/catalog_v4_train_signext.yaml)
Decision measure: the zero-shot wrong-sign share on the held-out one-loop pools the scorer reads as sign-changing
(H5_signed.txt, 2000 validation events each, best checkpoint); per pool the mean over the seeds of each arm; the
number of pools (ties excluded) where the extension is lower, with the one-sided sign test p; the seed-mean medians;
each seed alone. Secondary: zero-shot MSE of log|M|^2 per probe (mean (q - t)^2 of the saved .npz) on every held-out
axis, the per-probe ratio extension over control (seed means), its median, and the control's own seed-43-over-42
median ratio as the seed spread.
    python3 analysis/catalog_v4/signext_compare.py   -> data/signext_compare.json

  warm    plan warm_start step 7, change C6 (superset design passed 2026-11-04T21:51; D283, D300): the control's
          model (v4_dimensionless_muon[_seedN]) warm-started on the extension's recipe for k in 533/1066/2132 steps
          (W_k, arms v4_mse_warmsx_k<k>[_seed2|3|4]) against the extension from scratch (B = the extension arm above),
          the parent (A = the control arm) and the matched control C_k (the same warm start on the control's own
          recipe, v4_mse_cont_k<k>[...]); seeds 42-45, each warm start from the parent at its seed. On the pools all
          arms score: per pool the seed-mean wrong-sign share; the ratio of medians over pools W_k/B with a 95% pool
          bootstrap (10000 resamples, seed 0) widened on each side by |ln| of B's own seed spread (the median over
          pools of B's seeds 43-45 mean over the median of B's seed 42); MATCH when the widened upper end <= 1.35,
          SHORT when the widened lower end > 1, INCONCLUSIVE otherwise; g_k = ln(W_k/A) / ln(B/A) (ratios of
          medians); carried when W_k is lower than C_k on more pools than higher with the one-sided sign test
          p < 0.05; YES at the smallest k with MATCH and carried. Beside: the MSE of log|M|^2 on H1-H3, median per-probe
          ratio of seed means W_k/B and W_k/C_k. -> data/warm_start_C6.json
"""
import glob
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
READS = os.path.join(os.path.dirname(HERE), "catalog_v3", "data", "short_reads")
SEEDS = (42, 43, 44, 45)  # seeds 44 and 45 by D195 (the user 2026-10-30: "Okay, yes to both")
CTRL = ("v4_dimensionless_muon", "v4_dimensionless_muon_seed2", "v4_dimensionless_muon_seed3", "v4_dimensionless_muon_seed4")
EXT = ("v4_dimensionless_muon_more_signed", "v4_dimensionless_muon_more_signed_seed2",
       "v4_dimensionless_muon_more_signed_seed3", "v4_dimensionless_muon_more_signed_seed4")
AXES = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")


def signed(arm, fields=None):
    """{pool: wrong_sign} of an arm's H5_signed.txt; with `fields`, {pool: {field: value}} of those record fields"""
    out = {}
    for line in open(os.path.join(READS, arm, "H5_signed.txt")):
        if line.startswith("SIGNED_AB "):
            r = json.loads(line[len("SIGNED_AB "):])
            if "wrong_sign" in r:
                out[r["pool"]] = r["wrong_sign"] if fields is None else {k: r.get(k) for k in fields}
    return out


def mse(arm, axis):
    out = {}
    for f in glob.glob(os.path.join(READS, arm, axis, "*.npz")):
        z = np.load(f)
        out[os.path.basename(f)[:-4]] = float(np.mean((z["q"] - z["t"]) ** 2))
    return out


def sign_p(k, n):  # one-sided: P(at least k lower of n) under p = 1/2
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    s = {a: signed(a) for a in CTRL + EXT}
    pools = sorted(set.intersection(*(set(v) for v in s.values())))
    c = np.array([[s[a][p] for p in pools] for a in CTRL])
    e = np.array([[s[a][p] for p in pools] for a in EXT])
    cm, em = c.mean(0), e.mean(0)
    d = em - cm
    lower, higher = int((d < 0).sum()), int((d > 0).sum())
    res = {"pools": len(pools),
           "sign_changing_per_arm": {a: len(v) for a, v in s.items()},
           "seed_mean": {"ext_lower": lower, "ext_higher": higher, "ties": len(pools) - lower - higher,
                         "p_one_sided": round(sign_p(lower, lower + higher), 3),
                         "median_ctrl_pct": round(100 * float(np.median(cm)), 2),
                         "median_ext_pct": round(100 * float(np.median(em)), 2)},
           "per_seed": {str(sd): {"ext_lower": int((e[i] < c[i]).sum()), "ext_higher": int((e[i] > c[i]).sum()),
                                  "median_ctrl_pct": round(100 * float(np.median(c[i])), 2),
                                  "median_ext_pct": round(100 * float(np.median(e[i])), 2)}
                        for i, sd in enumerate(SEEDS)},
           "per_pool_pct": {p: {"ctrl": [round(100 * x, 2) for x in c[:, j]], "ext": [round(100 * x, 2) for x in e[:, j]]}
                            for j, p in enumerate(pools)},
           "mse_ratio": {}}
    for ax in AXES:
        m = {a: mse(a, ax) for a in CTRL + EXT}
        probes = sorted(set.intersection(*(set(v) for v in m.values())))
        if not probes:
            continue
        cmse = np.array([[m[a][p] for p in probes] for a in CTRL]).mean(0)
        emse = np.array([[m[a][p] for p in probes] for a in EXT]).mean(0)
        seed = np.array([m[CTRL[1]][p] / m[CTRL[0]][p] for p in probes])
        res["mse_ratio"][ax] = {"probes": len(probes),
                                "median_ext_over_ctrl": round(float(np.median(emse / cmse)), 3),
                                "per_seed": [round(float(np.median(np.array([m[EXT[i]][p] / m[CTRL[i]][p] for p in probes]))), 3)
                                             for i in range(len(SEEDS))],
                                "ctrl_seed43_over_42": round(float(np.median(seed)), 3),
                                "median_mse_ctrl": round(float(np.median(cmse)), 4),
                                "median_mse_ext": round(float(np.median(emse)), 4)}
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "data", "signext_compare.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_pool_pct"}))


WARM_K = (533, 1066, 2132)
SFX = ("", "_seed2", "_seed3", "_seed4")
MATCH_HI = 1.35


def _seed_mean(arms, pools):
    return np.array([[signed(a)[p] for p in pools] for a in arms]).mean(0)


def warm():
    out = {"design": "plan warm_start step 7, change C6 (superset design passed 2026-11-04T21:51; D283, D300)",
           "B": list(EXT), "A": list(CTRL), "k": {}}
    have = lambda a: os.path.exists(os.path.join(READS, a, "H5_signed.txt"))
    for k in WARM_K:
        W = tuple(f"v4_mse_warmsx_k{k}{x}" for x in SFX)
        C = tuple(f"v4_mse_cont_k{k}{x}" for x in SFX)
        miss = [a for a in W + C if not have(a)]
        if miss:
            out["k"][k] = {"missing": miss}
            continue
        s = {a: signed(a) for a in W + C + CTRL + EXT}
        pools = sorted(set.intersection(*(set(v) for v in s.values())))
        w, c, b, a = (_seed_mean(x, pools) for x in (W, C, EXT, CTRL))
        b42 = np.array([s[EXT[0]][p] for p in pools])
        b_rest = np.array([[s[x][p] for p in pools] for x in EXT[1:]]).mean(0)
        spread = abs(np.log(np.median(b_rest) / np.median(b42)))
        r = float(np.median(w) / np.median(b))
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(10000):
            i = rng.integers(0, len(pools), len(pools))
            mb = np.median(b[i])
            if mb > 0:
                boot.append(np.median(w[i]) / mb)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        wl, wh = float(lo * np.exp(-spread)), float(hi * np.exp(spread))
        outcome = "MATCH" if wh <= MATCH_HI else "SHORT" if wl > 1 else "INCONCLUSIVE"
        d = w - c
        lower, higher = int((d < 0).sum()), int((d > 0).sum())
        p_c = sign_p(lower, lower + higher)
        ra = lambda x: float(np.median(x) / np.median(a))
        row = {"pools": len(pools), "median_pct": {n: round(100 * float(np.median(x)), 2)
                                                   for n, x in (("W", w), ("C", c), ("B", b), ("A", a))},
               "W_over_B": {"R": round(r, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                            "widened": [round(wl, 4), round(wh, 4)], "B_seed_spread_abs_ln": round(float(spread), 4)},
               "W_over_A": round(ra(w), 4), "B_over_A": round(ra(b), 4), "C_over_A": round(ra(c), 4),
               "g": round(float(np.log(ra(w)) / np.log(ra(b))), 4),
               "vs_C": {"W_lower": lower, "W_higher": higher, "p_one_sided": round(p_c, 4)},
               "outcome": outcome, "carried": bool(lower > higher and p_c < 0.05), "mse_ratio": {}}
        for ax in ("H1", "H2", "H2p5", "H3"):
            m = {x: mse(x, ax) for x in W + C + EXT}
            probes = sorted(set.intersection(*(set(v) for v in m.values())))
            if probes:
                sm = lambda arms: np.array([[m[x][q] for q in probes] for x in arms]).mean(0)
                mw, mc, mb = sm(W), sm(C), sm(EXT)
                row["mse_ratio"][ax] = {"probes": len(probes), "median_W_over_B": round(float(np.median(mw / mb)), 3),
                                        "median_W_over_C": round(float(np.median(mw / mc)), 3)}
        out["k"][k] = row
    done = [k for k in WARM_K if "outcome" in out["k"][k]]
    yes = next((k for k in done if out["k"][k]["outcome"] == "MATCH" and out["k"][k]["carried"]), None)
    out["verdict"] = (f"YES at k={yes} ({yes / 4264:.3f} of the from-scratch steps)" if yes else
                      "NO (no k up to 2132 matches)" if len(done) == 3 and all(out["k"][k]["outcome"] == "SHORT" for k in done)
                      else "not decided" + ("" if len(done) == 3 else " (reads missing)"))
    json.dump(out, open(os.path.join(HERE, "data", "warm_start_C6.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    import sys
    warm() if sys.argv[1:] == ["warm"] else main()
