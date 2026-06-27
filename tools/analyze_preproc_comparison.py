#!/usr/bin/env python3
"""Compare the global vs per-dataset amplitude-preprocessing sweeps.

prepd-space losses (val_loss/proc_val_losses) are NOT comparable across the two
modes (per-dataset normalizes each process to unit variance), so the comparison
uses the scale-invariant RAW relative error parsed from each trial's eval log.
"""
import glob, json, os, re, sys

# Optional tag suffix (e.g. "_t1000") selects a sweep variant.
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
SWEEPS = {
    "global": f"sweeps/preproc_comparison/preproc_global{TAG}",
    "perds":  f"sweeps/preproc_comparison/preproc_perds{TAG}",
}
RUNS = {"global": f"runs/preproc_global{TAG}", "perds": f"runs/preproc_perds{TAG}"}
PROCS = ["ee_wwz","ee_WW","ee_ttbar","ee_uug","ee_uugg","ee_aa","ee_aaa","ee_uu"]

# log lines:
#  "Mean |rel err| on 1%% largest amplitudes <proc> test_<proc>: <v>"
#  "Mean |rel err| on 1%% largest amplitudes combined test: <v>"
RE_PROC_1PCT = re.compile(
    r"Mean \|rel err\| on 1%% largest amplitudes (\S+) (test|val)_\1: ([\d.eE+-]+)")
RE_COMB_1PCT = re.compile(
    r"Mean \|rel err\| on 1%% largest amplitudes combined (test|val): ([\d.eE+-]+)")
RE_PROC_FULL = re.compile(
    r"Mean \|rel err\| (test|val)_(\S+) (\S+): ([\d.eE+-]+)")


def hp_idx_from_result(path):
    m = re.search(r"hp(\d+)_t", os.path.basename(path))
    return int(m.group(1))


def load_results(sweep_dir):
    out = []
    for f in glob.glob(os.path.join(sweep_dir, "results", "*.json")):
        d = json.load(open(f))
        out.append((hp_idx_from_result(f), d))
    return out


def parse_log_raw(log_path, split="test"):
    """Return {proc: rel_err_1pct} and combined_1pct for the given split."""
    proc1, comb1 = {}, None
    with open(log_path) as fh:
        for line in fh:
            m = RE_PROC_1PCT.search(line)
            if m and m.group(2) == split:
                proc1[m.group(1)] = float(m.group(3))
                continue
            m = RE_COMB_1PCT.search(line)
            if m and m.group(1) == split:
                comb1 = float(m.group(2))
    return proc1, comb1


def main():
    summary = {}
    for mode, sdir in SWEEPS.items():
        res = load_results(sdir)
        # best trial by the HPO objective (prepd val_loss, each sweep's own scale)
        best_idx, best = min(res, key=lambda kv: kv[1]["val_loss"])
        vals = sorted(r["val_loss"] for _, r in res)
        med = vals[len(vals)//2]
        log = os.path.join(RUNS[mode], f"trial_{best_idx:04d}", "out_0.log")
        proc1, comb1 = parse_log_raw(log, split="test")
        # also: best raw rel-err per process ACROSS all trials in the sweep
        best_proc_across = {p: float("inf") for p in PROCS}
        for idx, _ in res:
            lp = os.path.join(RUNS[mode], f"trial_{idx:04d}", "out_0.log")
            if not os.path.exists(lp):
                continue
            pp, _ = parse_log_raw(lp, split="test")
            for p, v in pp.items():
                if p in best_proc_across:
                    best_proc_across[p] = min(best_proc_across[p], v)
        summary[mode] = dict(best_idx=best_idx, best_val=best["val_loss"],
                             med_val=med, proc1=proc1, comb1=comb1,
                             best_across=best_proc_across, n=len(res))

    g, p = summary["global"], summary["perds"]
    print("="*78)
    print(f"GLOBAL vs PER-DATASET amplitude preprocessing  (8-proc D1e3p5, tag='{TAG or 't100'}', 20 trials each)")
    print("="*78)
    print(f"\nprepd val_loss (HPO objective, NOT cross-comparable scales):")
    print(f"  global : best={g['best_val']:.4f}  median={g['med_val']:.4f}  (best trial {g['best_idx']})")
    print(f"  perds  : best={p['best_val']:.4f}  median={p['med_val']:.4f}  (best trial {p['best_idx']})")

    print(f"\nRAW rel-err on 1% largest amplitudes (TEST), scale-invariant — best-by-val trial:")
    print(f"  {'process':9s} {'global':>10s} {'perds':>10s}  {'winner':>8s}")
    for proc in PROCS:
        gv = g["proc1"].get(proc, float('nan'))
        pv = p["proc1"].get(proc, float('nan'))
        win = "perds" if pv < gv else "global"
        print(f"  {proc:9s} {gv:10.4f} {pv:10.4f}  {win:>8s}")
    print(f"  {'COMBINED':9s} {g['comb1']:10.4f} {p['comb1']:10.4f}"
          f"  {'perds' if p['comb1']<g['comb1'] else 'global':>8s}")

    print(f"\nRAW rel-err on 1% largest (TEST) — BEST across all 20 trials per process:")
    print(f"  {'process':9s} {'global':>10s} {'perds':>10s}  {'winner':>8s}")
    gw = pw = 0
    for proc in PROCS:
        gv = g["best_across"].get(proc, float('nan'))
        pv = p["best_across"].get(proc, float('nan'))
        win = "perds" if pv < gv else "global"
        if pv < gv: pw += 1
        else: gw += 1
        print(f"  {proc:9s} {gv:10.4f} {pv:10.4f}  {win:>8s}")
    print(f"\n  per-process best-of-20 winners: global={gw}  perds={pw}")

    # Equal-weight geometric mean of the 8 per-process raw rel-errs (1% largest,
    # test) per trial — the scale-invariant overall metric (mirrors the sweep's
    # geometric_mean loss aggregation, but on raw physics error).
    import math
    print(f"\nGeo-mean of per-process raw rel-err (1% largest, TEST) across trials:")
    geo = {}
    for mode in SWEEPS:
        res = load_results(SWEEPS[mode])
        gms = []
        for idx, _ in res:
            lp = os.path.join(RUNS[mode], f"trial_{idx:04d}", "out_0.log")
            pp, _ = parse_log_raw(lp, split="test")
            vals = [pp[p] for p in PROCS if p in pp and pp[p] > 0]
            if len(vals) == len(PROCS):
                gms.append(math.exp(sum(math.log(v) for v in vals) / len(vals)))
        gms.sort()
        geo[mode] = gms
        print(f"  {mode:7s}: best={gms[0]:.4f}  median={gms[len(gms)//2]:.4f}  "
              f"worst={gms[-1]:.4f}  (n={len(gms)})")
    print(f"\n  median geo-mean ratio global/perds = "
          f"{geo['global'][len(geo['global'])//2] / geo['perds'][len(geo['perds'])//2]:.2f}x")


if __name__ == "__main__":
    main()
