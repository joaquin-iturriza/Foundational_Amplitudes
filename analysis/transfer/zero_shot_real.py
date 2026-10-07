"""Transfer study, real zero-shot: the pretraining with one shared standardization (tp3_gstd_r9, rung 9 at hp73, 64k
steps, data.shared_standardization) predicts a probe's log|M|^2 directly, mu + sigma h, with no data of the probe. Per
probe, its validation MSE of log|M|^2 divided by the probe's variance sigma_p^2 (1 = as good as the probe's exact mean,
which the model is not given), next to the same parent up to the best affine map (val_loss_affine, the shape only) and
to scratch trained on D = 10 and 10^2 events (cells.scratch, 8k grid), both also over sigma_p^2.
Input: analysis/transfer/zero_shot_gstd.json (ZERO_SHOT lines of scripts/job_zero_shot_ft.sh --parent).
    python analysis/transfer/zero_shot_real.py    -> figs/zero_shot_real
"""
import io, contextlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, scratch  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from rung_grid import LAB, P  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

rows = {re.search(r"fte_(.+)_d\d", r["run_dir"]).group(1): r
        for r in json.load(open(os.path.join(ROOT, "analysis", "transfer", "zero_shot_gstd.json")))}
fig, ax = ps.figure()
for i, p in enumerate(P):
    r = rows[p]; s2 = r["prepd_std"][0] ** 2
    for x, lab, c, m in ((r["mse_logm2_real"] / s2, "zero-shot", ps.C.vermillion, "D"),
                         (r["val_loss_affine"], "zero-shot up to an affine map", ps.C.orange, "o"),
                         (scratch(p, 2, steered=False)[0] / s2, r"scratch, $D=10$", "k", "x"),
                         (scratch(p, 4, steered=False)[0] / s2, r"scratch, $D=10^2$", ps.C.grey, "+")):
        ax.scatter([x], [i], color=c, marker=m, label=lab if i == 0 else None)
ax.axvline(1, color="0.5", ls="--", label=r"the probe's mean, $\sigma_p^2$")
ax.set_xscale("log")
ax.set_yticks(range(len(P))); ax.set_yticklabels([LAB[p] for p in P]); ax.invert_yaxis()
ax.set_xlabel(r"MSE$(\log|\mathcal{M}|^2)\,/\,\sigma_p^2$")
ps.shared_legend(fig, ax, ncol=2)
ps.save(fig, os.path.join(ROOT, "analysis", "transfer", "figs", "zero_shot_real"))
