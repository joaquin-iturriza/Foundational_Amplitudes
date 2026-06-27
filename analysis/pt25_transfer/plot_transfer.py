"""NLO transfer comparison, ALL nh=8, layer-decay finetune method.

Three lines, matched in width (nh8), batch (8192) and method:
  - pt25 FT   : 25-process foundation (runs/pretrain25), finetuned
  - old FT    : 8-process foundation (runs/pretrain_full_nh8), finetuned
                (sweeps finetune_scaling_virt[_002])
  - solo nh8  : from-scratch (scaling_solo_nh8_anchor2), single high-compute anchor

x-axis plotted two ways (separate panels): training COMPUTE (FLOPs) and WALLTIME (h).
y = held-out test_loss of the best-val trial per cell (from stored results).
"""
import glob, json, re, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# FLOPs/step (fwd+bwd, MACs) for the LLoCa muP net — established formula.
def f_step(nh, n_avg, bs):
    d = 16 * nh; L = 8
    f_tr = L * n_avg * (24 * d**2 + 2 * n_avg * d)
    f_fr = n_avg * 131072
    return 3.0 * bs * (f_fr + f_tr)

N_AVG = 4        # ee_uu / ee_ttbar NLO are 2->2 (4 particles)
NH, BS = 8, 8192
FSTEP = f_step(NH, N_AVG, BS)

def _load(f):
    for _ in range(3):                       # Lustre can throw transient EIO
        try:
            return json.load(open(f))
        except OSError:
            continue
    return None

def cells(glob_pat):
    """{t_steps: (held_out_test_loss, walltime_h)} over matching cells.

    Proper protocol, straight from the stored results/*.json: per cell, select the
    trial with the best val_loss (model selection), then report ITS test_loss
    (held-out) and walltime. Avoids the selection bias of min(val_loss), which
    picks the luckiest val and reads ~37x lower than its own held-out test."""
    out = {}
    for d in glob.glob(os.path.join(ROOT, glob_pat)):
        m = re.search(r"_t0*(\d+)$", d)
        if not m:
            continue
        t = int(m.group(1))
        recs = [r for r in (_load(f) for f in glob.glob(d + "/results/*.json")) if r]
        recs = [r for r in recs if "test_loss" in r]
        if not recs:
            continue
        sel = min(recs, key=lambda r: r["val_loss"])     # select on val
        cand = (sel["test_loss"], sel.get("traintime_hours", float("nan")))  # report test
        prev = out.get(t)
        if prev is None or cand[0] < prev[0]:
            out[t] = cand
    return out

def merge(*ds):
    o = {}
    for d in ds:
        for t, v in d.items():
            if t not in o or v[0] < o[t][0]:
                o[t] = v
    return o

datasets = [("ee_uu NLO-virt", "eeuunlovirte4"), ("ee_ttbar NLO-virt", "eettbarnlovirte4")]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for row, (title, key) in enumerate(datasets):
    pt25 = cells(f"sweeps/finetune_pt25_*{key}_t*")
    old  = merge(cells(f"sweeps/finetune_scaling_virt_{key}_t*"),
                 cells(f"sweeps/finetune_scaling_virt_002_{key}_t*"))
    # full from-scratch nh8 curve = all nh8 solo families merged (best per t_steps)
    solo = merge(*[cells(f"sweeps/{fam}_{key}_t*") for fam in (
        "scaling_solo_nh8_curve_virt", "scaling_solo_nh8_lowt_virt",
        "scaling_solo_nh8_anchor_virt_002", "scaling_solo_nh8_anchor2_virt",
        "solo_nh8_10k_virt")])

    series = [
        (pt25, "pt25 FT (25-process foundation)", dict(marker="o", color="C0", lw=2.4)),
        (old,  "old FT (8-process foundation)",   dict(marker="s", color="C1", lw=2.0, ls="--")),
        (solo, "from-scratch solo (nh8)",         dict(marker="^", color="C2", lw=2.0, ls=":")),
    ]
    for col, (xlab, xof) in enumerate([
        ("training compute [FLOP]", lambda t, w: 2 * FSTEP * t),   # 2x MACs -> FLOPs
        ("walltime [h]",           lambda t, w: w),
    ]):
        ax = axes[row][col]
        for data, label, st in series:
            if not data:
                continue
            ts = sorted(data)
            xs = [xof(t, data[t][1]) for t in ts]
            ys = [data[t][0] for t in ts]
            ax.plot(xs, ys, label=label, **st)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlab); ax.set_ylabel("held-out test_loss (best-val trial)")
        ax.set_title(f"{title}  —  vs {xlab.split(' [')[0]}")
        ax.grid(True, which="both", alpha=0.25)
        if row == 0 and col == 0:
            ax.legend(fontsize=8.5, framealpha=0.9)

fig.suptitle("NLO transfer (all nh=8, layer-decay finetune): 25- vs 8-process foundation vs from-scratch",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(HERE, "transfer_nh8.pdf"))
fig.savefig(os.path.join(HERE, "transfer_nh8.png"), dpi=130)
print("saved", os.path.join(HERE, "transfer_nh8.{pdf,png}"))
print(f"FSTEP (MACs/step, nh8 n_avg4 bs8192) = {FSTEP:.3e}")
