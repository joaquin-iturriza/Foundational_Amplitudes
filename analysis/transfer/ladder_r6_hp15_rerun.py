"""Transfer study, the ladder: rung 6's hp15 (tp3_ladder_r6 trial 15) rerun cold from its own config on CC
(FA-ccin2p3-2ed18ab8, V100, scripts/job_rerun_trial.sh), set against its lxplus run's record and against the
rescoring of that run's checkpoint (tools/eval_best_val.py, ladder_eval_best.json).
Data: ladder_pretrain.json (the record and its validation curve), ladder_eval_best.json (the rescoring),
ladder_r6_hp15_rerun.json (the rerun's result.json and its own checkpoint rebuilt by tools/eval_delta.py).
Every value is val_loss_no_reg at the best checkpoint (CLAUDE.md, reported values).
  figs/ladder_r6_hp15_rerun   (a) both validation curves, best checkpoints marked, the rescoring as a level;
                              (b) the same over the last quarter of the anneal
Prints one JSON line with the numbers the notes quote.
    python3 analysis/transfer/ladder_r6_hp15_rerun.py
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

L = json.load(open(os.path.join(HERE, "ladder_pretrain.json")))
rec = next(t for t in L["tp3_ladder_r6"] if t["hp"] == 15)
resc = json.load(open(os.path.join(HERE, "ladder_eval_best.json")))["tp3_ladder_r6"]["15"]["val_loss"]
rr = json.load(open(os.path.join(HERE, "ladder_r6_hp15_rerun.json")))

curves = []
for t, lab, col in ((rec, "lxplus run (record)", ps.C.blue), (rr, "CC rerun, V100", ps.C.vermillion)):
    y = np.array(t["val_curve"]); x = (np.arange(len(y)) + 1) * t["validate_every"]
    curves.append((x, y, t["best_step"], t["val_loss"], lab, col))

fig, axes = ps.figure(ncols=2)
for ax, x0 in zip(axes, (0, 48000)):
    for x, y, bs, bv, lab, col in curves:
        m = x >= x0
        ax.plot(x[m], y[m], "-", color=col, label=lab)
        ax.plot(bs, bv, "o", color=col)
    ax.axhline(resc, color=ps.C.grey, ls="--", label="record's checkpoint, rescored")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel(r"$\mathcal{L}_{\rm val}$")
    ps.make_room(ax)
ps.legend(axes[0], "upper right")
ps.make_room(axes[0])
ps.save(fig, os.path.join(HERE, "figs", "ladder_r6_hp15_rerun"))

last = lambda t: t["val_curve"][-5:]
print(json.dumps({
    "record": rec["val_loss"], "record_step": rec["best_step"], "rescored": resc,
    "rerun": rr["val_loss"], "rerun_step": rr["best_step"], "rerun_rebuilt": rr["rebuilt"]["val_loss_rebuilt"],
    "rerun_over_record_pct": 100 * (rr["val_loss"] / rec["val_loss"] - 1),
    "rerun_under_rescored_pct": 100 * (1 - rr["val_loss"] / resc),
    "rescored_over_record_pct": 100 * (resc / rec["val_loss"] - 1),
    "rebuilt_off_pct": 100 * abs(rr["rebuilt"]["val_loss_rebuilt"] / rr["val_loss"] - 1),
    "record_last5_max": max(last(rec)), "rerun_last5_max": max(last(rr)),
    "proc_ratio_min": min(rr["proc_val_losses_no_reg"][p] / v for p, v in rec["proc_val_losses_no_reg"].items()),
    "proc_ratio_max": max(rr["proc_val_losses_no_reg"][p] / v for p, v in rec["proc_val_losses_no_reg"].items()),
}))
