"""critical_batch: the batch-size scan as it stands, while its sweeps run.

    python analysis/critical_batch/scan_state.py --collect   # reads the cbs_b<B> sweeps on CC (a CPU job: call twice)
    python analysis/critical_batch/scan_state.py             # draws from data/cbs_state.json

(a) steps to val_loss_no_reg -2.08 of each batch's best trial (sweep/run_trial.py steps_to_target, the sweeps'
objective) against batch, with S = S_min (1 + B_crit / B) (cbs_read.fit_bcrit) over the batches drawn; (b) seconds
per training step against batch, from the validation timestamps of each trial's log (validation included), per GPU
model, beside the design's cost model 0.12 s + 0.86 s x B / 16384."""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))
DATA = f"{HERE}/data/cbs_state.json"
BATCHES = (2048, 4096, 8192, 16384, 32768)
TARGET = -2.08          # the design's objective target
EVERY = 85              # validate_every_n_steps of every cbs sweep config

REMOTE = r'''
import glob, json, os, re, sys
from datetime import datetime
out = {}
for B in sys.argv[1:]:
    d = f"sweeps/cbs_b{B}"
    if not os.path.isdir(d):
        continue
    res = {}
    for f in glob.glob(f"{d}/results/*.json"):
        r = json.load(open(f))
        res[int(re.match(r"hp(\d+)_", os.path.basename(f)).group(1))] = {"val_curve": r["val_curve"],
                                                                        "validate_every": r["validate_every"]}
    trials = []
    for f in sorted(glob.glob(f"{d}/error/*.err")):
        o = f.replace("/error/", "/output/")[:-4] + ".out"
        txt = open(f, errors="replace").read()
        summ = next((json.loads(l.split(" ", 1)[1]) for l in open(o, errors="replace") if l.startswith("RUN_SUMMARY ")),
                    {}) if os.path.exists(o) else {}
        m = re.search(r"trial_(\d+)$", summ.get("run_dir", ""))
        vals = re.findall(r"^\[(\S+ \S+) I\] Val loss \(combined\): \S+ \| val_loss_no_reg: (\S+)", txt, re.M)
        ts = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() for t, _ in vals]
        trials.append({"log": os.path.basename(f), "hp": int(m.group(1)) if m else None, "gpu": summ.get("gpu"),
                       "lr": summ.get("lr"), "log_curve": [float(v) for _, v in vals],
                       "sec_per_step": (ts[-1] - ts[0]) / ((len(ts) - 1) * {EVERY}) if len(ts) > 2 else None,
                       "stopped": "STOP_AT_VAL_LOSS" in txt, "diverged": "Training diverged" in txt,
                       "timeout": "DUE TO TIME LIMIT" in txt})
    out[B] = {"results": res, "trials": trials}
print("CBSSTATE " + json.dumps(out))
'''.replace("{EVERY}", str(EVERY))


def collect():
    from analysis.common.readers import site_python
    d = site_python(REMOTE, [str(b) for b in BATCHES], "CBSSTATE")
    json.dump(d, open(DATA, "w"))
    print(f"wrote {DATA}: " + ", ".join(f"{b}: {len(v['trials'])} logs, {len(v['results'])} results"
                                        for b, v in d.items()))


def best_steps(v):
    """(S, hp, n_with_outcome) of a batch: its trials' result curves, a log's curve where a trial left no result."""
    from run_trial import steps_to_target
    curves = {hp: r for hp, r in v["results"].items()}
    for t in v["trials"]:
        if t["hp"] is not None and str(t["hp"]) not in curves and (t["timeout"] or t["diverged"]):
            curves[str(t["hp"])] = {"val_curve": t["log_curve"], "validate_every": EVERY}
    S = [(steps_to_target(c, TARGET), hp) for hp, c in curves.items()]
    S = [x for x in S if x[0] is not None]
    return (min(S) if S else (None, None)) + (len(curves),)


def draw():
    import plot_style as ps
    from cbs_read import fit_bcrit
    d = json.load(open(DATA))
    rows = {int(b): best_steps(v) for b, v in d.items()}
    for b, (s, hp, n) in sorted(rows.items()):
        print(f"B {b}: best S({TARGET}) {s} (hp {hp}), {n} trials with an outcome of 8")
    fig, (axS, axT) = ps.figure(ncols=2)
    full = [(b, s) for b, (s, _, n) in rows.items() if s is not None and n >= 6]
    part = [(b, s, n) for b, (s, _, n) in rows.items() if s is not None and n < 6]
    axS.plot(*zip(*full), "o", color=ps.C.blue, label="best of 6-8 trials")
    for b, s, n in part:
        axS.plot([b], [s], "o", mfc="none", color=ps.C.blue, label=f"best of {n} trials so far")
    pts = full + [(b, s) for b, s, _ in part]
    if len(pts) >= 3:
        smin, bc = fit_bcrit(*zip(*pts))
        x = np.geomspace(1500, 40000, 100)
        print(f"fit over {len(pts)} batches: S_min {smin:.0f}, B_crit {bc:.0f}")
        axS.plot(x, smin * (1 + bc / x), "-", color=ps.C.vermillion,
                 label=rf"$S_{{\min}}(1+B_{{\rm crit}}/B)$, $B_{{\rm crit}}={bc:.0f}$")
    axS.set_xscale("log"); axS.set_yscale("log")
    axS.set_xlabel(r"batch size $B$")
    axS.set_ylabel(r"steps $S$ to $\mathcal{L}_{\rm val}=-2.08$")
    ps.legend(axS, "upper right")
    gpus = {}
    for b, v in d.items():
        for t in v["trials"]:
            if t["sec_per_step"] and t["gpu"]:
                gpus.setdefault("H100" if "H100" in t["gpu"] else "V100", []).append((int(b), t["sec_per_step"]))
    for (g, xs), c in zip(sorted(gpus.items(), key=lambda x: x[0] != "V100"), (ps.C.blue, ps.C.green)):
        axT.plot(*zip(*xs), "o", color=c, alpha=0.6, label=f"measured, {g}")
        for b in sorted({x for x, _ in xs}):
            print(f"{g} B {b}: median {np.median([y for x, y in xs if x == b]):.3f} s/step "
                  f"over {sum(x == b for x, _ in xs)} trials")
    x = np.geomspace(1500, 40000, 100)
    axT.plot(x, 0.12 + 0.86 * x / 16384, "--", color=ps.C.vermillion, label="design's cost model")
    axT.set_xscale("log"); axT.set_yscale("log")
    axT.set_xlabel(r"batch size $B$")
    axT.set_ylabel("seconds per training step")
    ps.legend(axT, "upper left")
    ps.save(fig, f"{HERE}/figs/cbs_scan_state")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    collect() if a.collect else draw()
