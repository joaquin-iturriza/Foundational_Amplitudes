"""Paired comparison of the PID one-hot vs quantum-number (physics) particle
encoding on the 25-process joint pretrain: sweeps pretrain25_zs_pid_002 /
pretrain25_zs_phys_002 share trial ids and HPs, so trials pair exactly."""
import re, glob, os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "runs")
PAT = re.compile(r"Val loss \(combined\): ([0-9.eE+-]+)")

def curves(sweep):
    out = {}
    for log in sorted(glob.glob(os.path.join(ROOT, sweep, "trial_*", "out_0.log"))):
        vals = [float(m.group(1)) for m in PAT.finditer(open(log).read())]
        if vals:
            out[log.split(os.sep)[-2]] = np.array(vals)
    return out

pid, phys = curves("pretrain25_zs_pid_002"), curves("pretrain25_zs_phys_002")
common = sorted(set(pid) & set(phys))
best_pid = np.array([pid[t].min() for t in common])
best_phys = np.array([phys[t].min() for t in common])
print(f"{len(common)} paired trials; phys better in {np.sum(best_phys < best_pid)}")
print(f"best-vs-best: pid {best_pid.min():.4g}  phys {best_phys.min():.4g}  "
      f"ratio {best_pid.min()/best_phys.min():.2f}x; median ratio {np.median(best_pid/best_phys):.2f}x")

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
tp, tf = common[int(np.argmin(best_pid))], common[int(np.argmin(best_phys))]
a.plot(np.linspace(0, 3000, len(pid[tp])), pid[tp], color="#d62728", label="particle-ID one-hot")
a.plot(np.linspace(0, 3000, len(phys[tf])), phys[tf], color="#1f77b4", label="quantum numbers")
a.set(yscale="log", xlabel="training step", ylabel="validation loss (25-process joint)")
a.legend(frameon=False); a.set_title("best trial of each encoding")
lo = min(best_pid.min(), best_phys.min()) * 0.7
hi = max(best_pid.max(), best_phys.max()) * 1.4
b.scatter(best_pid, best_phys, s=28, color="#1f77b4", zorder=3)
b.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
b.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
      xlabel="best val loss, particle-ID one-hot", ylabel="best val loss, quantum numbers")
b.set_title(f"{len(common)} paired trials (same hyperparameters)")
b.text(0.96, 0.08, "below the line:\nquantum numbers better", transform=b.transAxes,
       ha="right", fontsize=8, color="gray")
fig.tight_layout()
base = os.path.join(os.path.dirname(__file__), "pid_vs_phys")
fig.savefig(base + ".png", dpi=200); fig.savefig(base + ".pdf")
print("saved", base + ".{png,pdf}")
