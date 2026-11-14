"""pretrain_data_saving step 6: the WSD-trained b b~ > Z Z held-out parent against its cosine twin (same pools, steps,
lr, batch, inputs, seed 42; only the schedule differs). Per parent, from its job log (`site logs <run> -n 100000`):
the combined val_loss_no_reg (HETEROSC) at every validation, the best (lowest) and its step, the last, and spikes
(a validation more than 1.0 above the running best, a factor e in sigma-weighted error, is listed apart, not dropped). Per process at each parent's best
validation: the paired difference WSD - cosine of the per-process val loss (HETEROSC NLL, log units).
    python analysis/data_saving/parent_curves.py <cosine log> <wsd log>  -> data/parent_curves.json, figs/parent_curves
"""
import json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import plot_style as ps  # noqa: E402

VAL = re.compile(r"Val loss \(combined\): \S+ \| val_loss_no_reg: (\S+) \| (.*)$")
EVERY = re.compile(r"for (\d+) iterations.*validating every (\d+) steps")


def read(path):
    every, vals, procs = None, [], []
    for line in open(path, errors="replace"):
        m = EVERY.search(line)
        if m and every is None:
            T, every = int(m.group(1)), int(m.group(2))
        m = VAL.search(line)
        if m and every:
            vals.append(float(m.group(1)))
            procs.append({k: float(v) for k, v in (kv.split("=") for kv in m.group(2).split(", ") if "=" in kv)})
        if "Finished training" in line and vals:
            break                               # the sigma stage's validations follow; the parent's curve ends here
    steps = np.minimum(np.arange(1, len(vals) + 1) * every, T)     # the last check is at the end of training
    return steps, np.array(vals), procs


def summary(steps, v):
    run_best = np.minimum.accumulate(v)
    i = int(np.argmin(v))
    # a spike: a check whose loss sits a decade above the running best on a positive-loss scale is ill-defined for the
    # signed HETEROSC NLL, so a spike is a check more than 1.0 above the running best (a factor e in sigma-weighted error)
    sp = [int(s) for s, x, b in zip(steps, v, run_best) if x - b > 1.0]
    return {"best": float(v[i]), "best_step": int(steps[i]), "last": float(v[-1]), "n_checks": len(v), "spikes_at": sp}


if __name__ == "__main__":
    cos, wsd = read(sys.argv[1]), read(sys.argv[2])
    out = {"cosine": summary(*cos[:2]), "wsd": summary(*wsd[:2])}
    pc, pw = cos[2][int(np.argmin(cos[1]))], wsd[2][int(np.argmin(wsd[1]))]
    d = np.array([pw[k] - pc[k] for k in pc if k in pw])
    out["per_process_wsd_minus_cosine_at_best"] = {"n": int(len(d)), "median": float(np.median(d)),
                                                   "q10_q90": [float(x) for x in np.percentile(d, [10, 90])],
                                                   "share_wsd_lower": float(np.mean(d < 0))}
    json.dump(out, open(os.path.join(HERE, "data", "parent_curves.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    fig, ax = ps.figure()
    for (s, v, _), c, lab in ((cos, "k", "cosine"), (wsd, ps.C.blue, "WSD")):
        ax.plot(s, v, "-", color=c, label=lab)
    # every check is drawn; the y axis tops at the highest non-spike check of the two runs (spikes are the json's
    # spikes_at, the rule above), so spikes run off the axis, not out of the data
    top = max(float(max(x for st, x in zip(s_, v_) if int(st) not in out[k]["spikes_at"]))
              for k, (s_, v_, _) in (("cosine", cos), ("wsd", wsd)))
    ax.set_ylim(top=top)
    ax.set_xlabel("training step"); ax.set_ylabel(r"val loss (HETEROSC NLL)")
    ps.legend(ax, "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "parent_curves"))
