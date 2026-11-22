"""Plan finetune_mechanism step 2 readout on q q~ -> Z+4g (D515 design passed 2026-11-20T16:10, arms released by D516).

Arms per batch (2048, 128), 8-trial DyHPO searches, seed 42, same space, waves and DyHPO seed as arm (1):
  P  = the W parent's pre-cooldown snapshot (step 56000 of the fp32 rerun FA-ccin2p3-74805752), fresh optimizer
       (sweeps_local/ftmech_zng4g_snap_b<B>);
  Po = the same snapshot with its optimizer state, no warmup (ftmech_zng4g_snapopt_b<B>).
Against arm (1) F (template fine-tune from the ORIGINAL parent FA-ccin2p3-47e14eae's final model): P/F8 and Po/F8 =
the arm's search best over F's first-8-trials best (same budget, start-up HPs), the 1.41 bar of the design's
single-seed reads (Sg/F8); beside it the ratio to F's 16-trial best and to scratch S's three-seed mean. Caveat: P and Po
branch from the rerun parent, which matches the original on 23 of 28 processes (parent_rerun_check_snapfp32).
Best val MSE(log|M|^2) = min val mu-MSE x std^2 (std from each run's data_stats.json, checked equal). Head start: each
chosen trial's best-so-far curve, the step fraction where it first reaches 1.2 x S's three-seed mean best. Step 0: het.
NLL at the first train_steps.npz record and the max pre-clip grad norm over steps 0-40, median over the arm's trials.
Diverged trials (no val) and blow-ups (val > 10x its best-so-far) are listed apart.

    python3 analysis/finetune_mechanism/zng4g_snap_read.py [--cached]  -> data/zng4g_snap_read.json, figs/zng4g_snap_*
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import zng4g_readout as R  # noqa: E402
from analysis.common.readers import site_python  # noqa: E402

BAR = 1.41
ARMS = {"P": "snap", "Po": "snapopt"}
STD_REF = 1.11978


def collect():
    items, meta = {"ccin2p3": [], "lxplus": []}, {}
    for b in ("2048", "128"):
        for arm, tag in ARMS.items():
            for ti, hp, v, site, rd, rp in R.driven(f"ftmech_zng4g_{tag}_b{b}"):
                lab = f"{arm}|{b}|t{ti}"
                items[site].append(f"'{lab}|{rd}|{rp}'".replace("|", "#", 2))
                meta[lab] = {"hp": hp, "trial": ti, "val_loss": v}
    out = {}
    for site, args in items.items():
        if args:
            r = site_python(R.REMOTE, args, "ZNG4G", site=site)
            out.update({k.replace("#", "|"): dict(v, site=site, **meta[k.replace("#", "|")]) for k, v in r.items()})
    assert len(out) == len(meta), (len(out), len(meta))
    return out


def analyse(raw, ft):
    res = {}
    for b in ("2048", "128"):
        f = ft[b]
        std = f["std"]
        Smean = f["mean"]["S"]
        r = {"std": std, "S_seeds": f["best"]["S"], "S_mean": Smean, "arms": {}}
        Ftr = sorted((v for k, v in R.arm_runs(R_RAW, "F", b).items() if k.startswith("t")), key=lambda v: v["trial"])
        groups = {"F": Ftr}
        for arm in ARMS:
            groups[arm] = sorted(R.arm_runs(raw, arm, b).values(), key=lambda v: v["trial"])
        stds = [v["std"] for a in ARMS for v in groups[a] if "std" in v]
        r["std_max_rel_dev"] = max(abs(s / std - 1) for s in stds)
        for a, ts in groups.items():
            done = [t for t in ts if t.get("val_loss") is not None]
            ch = min(done, key=lambda t: t["val_loss"])
            curve = np.minimum.accumulate(np.array(ch["val_mse_curve"]) * std ** 2)
            i = np.nonzero(curve <= 1.2 * Smean)[0]
            first8 = [t["val_loss"] * std ** 2 for t in done if t["trial"] < 8]
            r["arms"][a] = {
                "trials": [t["val_loss"] * std ** 2 if t.get("val_loss") is not None else None for t in ts],
                "hp_chosen": ch["hp"], "best": ch["val_loss"] * std ** 2, "best8": min(first8),
                "diverged_hp": [t["hp"] for t in ts if t.get("val_loss") is None],
                # blow-up: the val curve rises > 10x above its best-so-far (D398: listed apart, never retried)
                "blowup_hp": [t["hp"] for t in done if max(c / min(t["val_mse_curve"][:i + 1])
                                                          for i, c in enumerate(t["val_mse_curve"])) > 10],
                "curve": curve.tolist(), "reach_1p2S": float((i[0] + 1) / len(curve)) if len(i) else None,
                "nll0": float(np.median([t["loss"][0] for t in ts if t.get("loss")])),
                "gmax": float(np.median([max(t["grad_norm"]) for t in ts if t.get("grad_norm")])),
                "n_steps_rec": sum(1 for t in ts if t.get("loss"))}
        F8, F16 = r["arms"]["F"]["best8"], r["arms"]["F"]["best"]
        for a in ARMS:
            x = r["arms"][a]["best"]
            r["arms"][a].update({"over_F8": x / F8, "over_F16": x / F16, "over_Smean": x / Smean,
                                 "verdict_F8": "within" if 1 / BAR <= x / F8 <= BAR else
                                 ("higher" if x / F8 > BAR else "lower")})
        res[b] = r
    return res


R_RAW = json.load(open(os.path.join(HERE, "data", "zng4g_readout_raw.json")))


def main():
    cache = os.path.join(HERE, "data", "zng4g_snap_raw.json")
    raw = json.load(open(cache)) if "--cached" in sys.argv else collect()
    if "--cached" not in sys.argv:
        json.dump(raw, open(cache, "w"))
    errs = {k: [e for e in ("std_error", "steps_error", "res_error") if e in v] for k, v in raw.items()}
    print("read errors:", {k: v for k, v in errs.items() if v})
    ft = R.analyse(R_RAW)
    res = analyse(raw, ft)
    for b, r in res.items():
        r["S_curve"] = list(np.mean([np.minimum.accumulate(c) for c in ft[b]["curves"]["S"]], axis=0))
        r["arms"]["F"]["reach_1p2S_3seed"] = ft[b]["reach_1p2S"]["F"]
        r["step0"] = {"S": ft[b]["step0"]["S"], "F": ft[b]["step0"]["F"],
                      **{a: {k: r["arms"][a][k] for k in ("nll0", "gmax")} for a in ARMS}}
        assert abs(r["std"] / STD_REF - 1) < 1e-4 and r["std_max_rel_dev"] < 1e-6, (r["std"], r["std_max_rel_dev"])
        print(f"b{b}: std {r['std']:.5f} (snapshot runs' max rel dev {r['std_max_rel_dev']:.1e}); "
              f"S mean {r['S_mean']:.4g} seeds {[round(x, 4) for x in r['S_seeds']]}")
        for a, v in r["arms"].items():
            print(f"  {a}: best {v['best']:.4g} (hp{v['hp_chosen']}), first-8 best {v['best8']:.4g}, "
                  f"diverged {v['diverged_hp']}, blow-ups {v['blowup_hp']}, reach 1.2S {v['reach_1p2S']}, nll0 {v['nll0']:.3g}, "
                  f"gmax {v['gmax']:.3g}" + (f"; /F8 {v['over_F8']:.3f} ({v['verdict_F8']}), /F16 {v['over_F16']:.3f},"
                                              f" /S {v['over_Smean']:.3f}" if a in ARMS else ""))
    json.dump({b: {k: v for k, v in r.items() if k != "S_curve"} for b, r in res.items()},
              open(os.path.join(HERE, "data", "zng4g_snap_read.json"), "w"), indent=1)
    R.plot({b: {"curves": {"S": ft[b]["curves"]["S"], **{a: [r["arms"][a]["curve"]] for a in ("F", *ARMS)}},
                "best": {"S": r["S_seeds"], **{a: [x for x in r["arms"][a]["trials"] if x is not None]
                                               for a in ("F", *ARMS)}},
                "step0": r["step0"]} for b, r in res.items()}, arms=("S", "F", "P", "Po"), tag="snap")


if __name__ == "__main__":
    main()
