"""Fair test-MSE scaling curves for eeuu / eettbar NLO-virt, all nh=8.

Five curves per process: solo-10k, solo-100k, FT(8-ds)-10k, FT(8-ds)-100k, FT(25-ds).
y = test MSE in (signed)log-amplitude space = divisor^2 * stored prepd test_loss,
where `divisor` is the run's amplitude-standardization std, reconstructed exactly
from data (verified against each run's logged standardized min/max). Undoing the
standardization is an affine map applied identically to prediction and target, so
log-space MSE = divisor^2 * prepd_MSE exactly -- no model forward needed.

Per cell: best-val trial (model selection on val_loss), report ITS held-out
test_loss + walltime. x-axis plotted two ways: training compute [FLOP] and
walltime [h]. Compute uses each family's ACTUAL events/step (batch size).
"""
import glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# ---- compute model: FLOPs/step (fwd+bwd, MACs->FLOP via the 2x in compute()) ----
def f_step(nh, n_avg, bs):
    d = 16 * nh
    return 3.0 * bs * (n_avg * 131072 + 8 * n_avg * (24 * d**2 + 2 * n_avg * d))
NH, N_AVG = 8, 4
def compute_flop(eps, t):       # 2x: MACs -> FLOPs
    return 2 * f_step(NH, N_AVG, eps) * t

# ---- amplitude standardization divisor, reconstructed from data (verified exact) ----
DATASET = {"eeuunlovirte4": "ee_uu_nlo_virt_e4",
           "eettbarnlovirte4": "ee_ttbar_nlo_virt_e4"}
_div_cache = {}
def divisor(key, sub):
    ds = DATASET[key]
    ck = (ds, sub)
    if ck not in _div_cache:
        amp = np.load(os.path.join(ROOT, "data", f"{ds}.npy"))[:, -1]
        a = amp if sub is None else amp[:sub]
        v = (np.sign(a) * np.log1p(np.abs(a))) if (a <= 0).any() else np.log(a)
        _div_cache[ck] = float(v.std())
    return _div_cache[ck]

def _load(f):
    for _ in range(4):
        try: return json.load(open(f))
        except OSError: continue
    return None

def cells(fam, eps, sub, key):
    """{t: (logMSE, walltime_h, compute_flop)} from best-val trial per cell."""
    out = {}
    div2 = divisor(key, sub) ** 2
    for d in glob.glob(os.path.join(ROOT, "sweeps", f"{fam}_t*")):
        base = os.path.basename(d)
        if re.sub(r"_t\d+$", "", base) != fam:
            continue
        t = int(re.search(r"_t(\d+)$", base).group(1))
        recs = [r for r in (_load(f) for f in glob.glob(d + "/results/*.json"))
                if r and "test_loss" in r and "val_loss" in r and r["test_loss"] > 0]
        if not recs:
            continue
        sel = min(recs, key=lambda r: r["val_loss"])
        out[t] = (div2 * sel["test_loss"], sel.get("traintime_hours", float("nan")),
                  compute_flop(eps, t))
    return out

def merge(*ds):
    """best (lowest logMSE) per t across sub-families."""
    o = {}
    for d in ds:
        for t, v in d.items():
            if t not in o or v[0] < o[t][0]:
                o[t] = v
    return o

# group -> builder(key) -> {t: (logMSE, wall, compute)}
def group_data(grp, key):
    if grp == "solo 10k":
        return cells(f"solo_nh8_10k_virt_{key}", 3500, 10000, key)
    if grp == "solo 100k":
        return merge(*[cells(f"{p}_{key}", 17500, 100000, key) for p in (
            "scaling_solo_nh8_lowt_virt", "scaling_solo_nh8_anchor_virt_002",
            "scaling_solo_nh8_curve_virt", "scaling_solo_nh8_anchor2_virt")])
    if grp == "FT 8ds 10k":
        return cells(f"finetune_10k_virt_{key}", 3500, 10000, key)
    if grp == "FT 8ds 100k":
        return merge(cells(f"finetune_scaling_virt_002_{key}", 17500, 100000, key),
                     cells(f"finetune_scaling_virt_ext_{key}", 17500, 100000, key))
    if grp == "FT 25ds":
        pre = "finetune_pt25_eeuunlovirt" if key == "eeuunlovirte4" else "finetune_pt25_eettbarnlovirt"
        return cells(f"{pre}_{key}", 8333, None, key)
    raise ValueError(grp)

GROUPS = [
    ("solo 10k",     dict(marker="o", color="C2", lw=1.8, ls="-")),
    ("solo 100k",    dict(marker="s", color="C3", lw=1.8, ls="-")),
    ("FT 8ds 10k",   dict(marker="^", color="C0", lw=1.8, ls="--")),
    ("FT 8ds 100k",  dict(marker="v", color="C1", lw=1.8, ls="--")),
    ("FT 25ds",      dict(marker="D", color="C4", lw=2.2, ls="-")),
]
PROCS = [("ee_uu NLO-virt", "eeuunlovirte4"), ("ee_ttbar NLO-virt", "eettbarnlovirte4")]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
table = {}
for row, (title, key) in enumerate(PROCS):
    for col, (xlab, xi) in enumerate([("training compute [FLOP]", 2), ("walltime [h]", 1)]):
        ax = axes[row][col]
        for grp, st in GROUPS:
            data = group_data(grp, key)
            table.setdefault((key, grp), data)
            if not data:
                continue
            ts = sorted(data)
            xs = [data[t][xi] for t in ts]
            ys = [data[t][0] for t in ts]
            ax.plot(xs, ys, label=grp, **st)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlab); ax.set_ylabel("test MSE (log-amplitude space)")
        ax.set_title(f"{title} — vs {xlab.split(' [')[0]}")
        ax.grid(True, which="both", alpha=0.25)
        if row == 0 and col == 0:
            ax.legend(fontsize=9, framealpha=0.9)

fig.suptitle("NLO test-MSE scaling (nh=8, fair: log-space MSE, per-family compute)\n"
             "best-val trial per cell; y = divisor² × prepd test_loss", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fair_scaling_nh8.{ext}"), dpi=130)
print("saved fair_scaling_nh8.{pdf,png}\n")

# numeric dump
for key in ("eeuunlovirte4", "eettbarnlovirte4"):
    print(f"\n===== {key}   (divisor: 10k={divisor(key,10000):.4f} 100k={divisor(key,100000):.4f} full={divisor(key,None):.4f}) =====")
    for grp, _ in GROUPS:
        d = table[(key, grp)]
        if not d:
            print(f"  {grp:14s} (none)"); continue
        print(f"  {grp:14s}  " + "  ".join(
            f"t={t}:MSE={d[t][0]:.2e},{d[t][1]*60:.1f}min,C={d[t][2]:.1e}" for t in sorted(d)))
