"""Transfer study, real zero-shot, the distributions behind the scores of zero_shot_real.py: per probe, the validation
split's log|M|^2 (truth), the shared-standardization pretraining's prediction mu + sigma h with no data of the probe
(zero-shot), and that prediction after the best affine map onto the truth (the shape only), as histograms on one
binning; and truth against zero-shot prediction as a 2-D histogram with the diagonal.
Input: analysis/transfer/zero_shot_gstd_hist.txt (ZERO_SHOT_HIST lines of scripts/job_zero_shot_ft.sh --parent).
    python analysis/transfer/zero_shot_dist.py    -> figs/zero_shot_dist{1,2}_[a-f], figs/zero_shot_2d{1,2}_[a-f], legends
(two sets of six probes each: a panel set holds at most nine)
"""
import io, contextlib, json, os, re, sys
import numpy as np
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from rung_grid import LAB, P  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

H = {}
for line in open(os.path.join(ROOT, "analysis", "transfer", "zero_shot_gstd_hist.txt")):
    if line.startswith("ZERO_SHOT_HIST "):
        r = json.loads(line.split(" ", 1)[1])
        H[re.search(r"fte_(.+)_d\d", r["run_dir"]).group(1)] = r
F = os.path.join(ROOT, "analysis", "transfer", "figs")

for half, PP in ((1, P[:6]), (2, P[6:])):
  figs = ps.panels(len(PP))
  for (fig, ax), p in zip(figs, PP):
      r = H[p]; e = np.asarray(r["edges"])
      for key, lab, col, ls in (("truth", "truth", "k", "-"), ("pred", "zero-shot", ps.C.vermillion, "-"),
                                ("pred_affine", "zero-shot, affine map", ps.C.orange, "--")):
          c = np.asarray(r[key], float)
          ax.stairs(c / c.sum(), e, color=col, ls=ls, label=lab)
      ax.set_xlabel(r"$\log|\mathcal{M}|^2$"); ax.set_ylabel("fraction of events")
      ps.process_label(ax, LAB[p])
  ps.legend_strip(figs[0][1], os.path.join(F, "zero_shot_dist_legend"), ncol=3)
  ps.save_panels(figs, os.path.join(F, f"zero_shot_dist{half}"))

for half, PP in ((1, P[:6]), (2, P[6:])):
  figs = ps.panels(len(PP))
  for (fig, ax), p in zip(figs, PP):
      r = H[p]; lo, hi = r["h2_range"]; h2 = np.asarray(r["h2"], float)
      h2[h2 == 0] = np.nan
      m = ax.imshow(h2.T, origin="lower", extent=[lo, hi, lo, hi], aspect="auto", norm=LogNorm(), cmap="viridis")
      ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", label="prediction = truth")
      ax.set_xlabel(r"true $\log|\mathcal{M}|^2$"); ax.set_ylabel(r"zero-shot $\log|\mathcal{M}|^2$")
      ps.colorbar(ax, m, "events")
      ps.process_label(ax, LAB[p])
  ps.legend_strip(figs[0][1], os.path.join(F, "zero_shot_2d_legend"), ncol=1)
  ps.save_panels(figs, os.path.join(F, f"zero_shot_2d{half}"))
