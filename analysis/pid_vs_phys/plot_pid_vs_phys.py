"""Paired comparison of the PID one-hot vs quantum-number (physics) particle
encoding on the 25-process joint pretrain: sweeps pretrain25_zs_pid_002 /
pretrain25_zs_phys_002 share trial ids and HPs, so trials pair exactly.
Left: tracked (regularized) validation curve of the best trial of each
encoding, scraped from the run logs. Right: best NON-regularized validation
loss of every paired trial, read from the sweep results JSONs (the comparison
metric; the tracked loss includes lambda*L2 with lambda a swept HP)."""
import re, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

ROOT = os.path.join(REPO, "runs")
SWEEPS = os.path.join(REPO, "sweeps")
PAT = re.compile(r"Val loss \(combined\): ([0-9.eE+-]+)")
STEPS = 3000


def curves(sweep):
    out = {}
    for log in sorted(glob.glob(os.path.join(ROOT, sweep, "trial_*", "out_0.log"))):
        vals = [float(m.group(1)) for m in PAT.finditer(open(log).read())]
        if vals:
            out[log.split(os.sep)[-2]] = np.array(vals)
    return out


def best_no_reg(sweep):
    """trial id -> smallest val_loss_no_reg, from <sweep>/results/hpNNNN_t3000_*.json."""
    out = {}
    stem = sweep[: -len("_002")]
    for f in glob.glob(os.path.join(SWEEPS, stem, sweep, "results", "hp*_t3000_*.json")):
        hp = int(os.path.basename(f)[2:6])
        out[f"trial_{hp:04d}"] = json.load(open(f))["val_loss"]
    return out


pid, phys = curves("pretrain25_zs_pid_002"), curves("pretrain25_zs_phys_002")
nr_pid, nr_phys = best_no_reg("pretrain25_zs_pid_002"), best_no_reg("pretrain25_zs_phys_002")
common = sorted(set(pid) & set(phys) & set(nr_pid) & set(nr_phys))
best_pid = np.array([nr_pid[t] for t in common])
best_phys = np.array([nr_phys[t] for t in common])
print(f"{len(common)} paired trials; phys better in {np.sum(best_phys < best_pid)}")
print(f"best-vs-best: pid {best_pid.min():.4g}  phys {best_phys.min():.4g}  "
      f"ratio {best_pid.min()/best_phys.min():.2f}x; median ratio {np.median(best_pid/best_phys):.2f}x")

fig, (a, b) = ps.figure(ncols=2)
tp, tf = common[int(np.argmin(best_pid))], common[int(np.argmin(best_phys))]
a.plot(np.linspace(0, STEPS, len(pid[tp])), pid[tp], color=ps.C.blue, label="one-hot PID")
a.plot(np.linspace(0, STEPS, len(phys[tf])), phys[tf], color=ps.C.vermillion, label="quantum numbers")
a.set(yscale="log", xlabel="training step", ylabel="validation loss (tracked)")
ps.legend(a, "upper right")

lo = min(best_pid.min(), best_phys.min()) * 0.7
hi = max(best_pid.max(), best_phys.max()) * 1.4
b.plot([lo, hi], [lo, hi], "--", color=ps.C.grey, label=r"$y=x$")
b.scatter(best_pid, best_phys, color=ps.C.blue, zorder=3, label="paired trials")
b.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
      xlabel="best val loss, one-hot PID",      # non-regularized; see docstring
      ylabel="best val loss, quantum numbers")
ps.legend(b, "upper left")
ps.save(fig, os.path.join(HERE, "pid_vs_phys"))
