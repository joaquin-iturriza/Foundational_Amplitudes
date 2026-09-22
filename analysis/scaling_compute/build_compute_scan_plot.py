"""Compute-scan curves: solo / ft8 / ft25  x  {1k,10k,100k,1M} datapoints, all nh=8.

y = test MSE in (signed)log space = divisor(dataset,sub)^2 * best-val test_loss
    (fair across preprocessing; divisor reconstructed from data, verified exact).
x = training compute [FLOP] = 2*f_step(batchsize(D)) * t_steps, and walltime [h].

Pulls the new cscan_* sweeps and reuses the existing solo 10k/100k families.
Robust to partial results: a cell with no finished trial is just skipped.
"""
import glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
DATASET = {"eeuunlovirte4": "ee_uu_nlo_virt_e4", "eettbarnlovirte4": "ee_ttbar_nlo_virt_e4"}
SUB = {"1k": 1000, "10k": 10000, "100k": 100000, "1M": 1000000}

def f_step(bs): return 3.0 * bs * 13139968
def batchsize(D): return int(min(16384, 0.7 * D / 2))
def perstep(D):  return 2 * f_step(batchsize(D))

_div = {}
def divisor(key, sub):
    ck = (key, sub)
    if ck not in _div:
        amp = np.load(f"{ROOT}/data/{DATASET[key]}.npy")[:, -1]
        a = amp if sub is None else amp[:sub]
        v = (np.sign(a) * np.log1p(np.abs(a))) if (a <= 0).any() else np.log(a)
        _div[ck] = float(v.std())
    return _div[ck]

def _load(f):
    for _ in range(4):
        try: return json.load(open(f))
        except OSError: continue
    return None

# family -> D -> list of sweep-name prefixes (cell dirs = <prefix>*_<dstag>_t<t>)
CURVES = {
    "solo": {
        "1k":   ["cscan_solo_D1k"],
        "10k":  ["solo_nh8_10k_virt", "cscan_solo_D10kext"],
        "100k": ["scaling_solo_nh8_lowt_virt", "scaling_solo_nh8_anchor_virt_002",
                 "scaling_solo_nh8_curve_virt", "scaling_solo_nh8_anchor2_virt"],
        "1M":   ["cscan_solo_D1M"],
    },
    "ft8":  {D: [f"cscan_ft8_D{D}"]  for D in SUB},
    "ft25": {D: [f"cscan_ft25_D{D}"] for D in SUB},
}

def cells(prefix, key):
    """yield (t_steps, cell_dir) for cells matching this prefix+process exactly."""
    pat = re.compile(rf"^{re.escape(prefix)}(_\d+)?_{key}_t(\d+)$")
    for d in glob.glob(f"{ROOT}/sweeps/{prefix}*_{key}_t*"):
        m = pat.match(os.path.basename(d))
        if m:
            yield int(m.group(2)), d

def curve(prefixes, key, D):
    """{t: (logMSE, walltime_h, compute)} merged over prefixes, best-val per cell."""
    div2 = divisor(key, SUB[D]) ** 2
    out = {}
    for pre in prefixes:
        for t, d in cells(pre, key):
            recs = [r for r in (_load(f) for f in glob.glob(d + "/results/*.json"))
                    if r and "test_loss" in r and "val_loss" in r and r["test_loss"] > 0]
            if not recs:
                continue
            sel = min(recs, key=lambda r: r["val_loss"])
            pt = (div2 * sel["test_loss"], sel.get("traintime_hours", float("nan")), perstep(SUB[D]) * t)
            if t not in out or pt[0] < out[t][0]:
                out[t] = pt
    return out

FAM_STYLE = {"solo": dict(color="C2", marker="o"),
             "ft8":  dict(color="C1", marker="s"),
             "ft25": dict(color="C0", marker="D")}
FAM_LABEL = {"solo": "solo (scratch)", "ft8": "FT ← 8-proc", "ft25": "FT ← 25-proc"}
DORDER = ["1k", "10k", "100k", "1M"]

coverage = {}
for proc_title, key in [("ee_uu NLO-virt", "eeuunlovirte4"), ("ee_ttbar NLO-virt", "eettbarnlovirte4")]:
    fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharey=True)
    for row, D in enumerate(DORDER):
        for col, (xlab, xi) in enumerate([("training compute [FLOP]", 2), ("walltime [h]", 1)]):
            ax = axes[row][col]
            for fam in ("solo", "ft8", "ft25"):
                data = curve(CURVES[fam][D], key, D)
                coverage[(key, fam, D)] = len(data)
                if not data:
                    continue
                ts = sorted(data)
                ax.plot([data[t][xi] for t in ts], [data[t][0] for t in ts],
                        label=FAM_LABEL[fam], lw=1.9, ms=5, **FAM_STYLE[fam])
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.set_xlabel(xlab)
            if col == 0:
                ax.set_ylabel(f"D={D}\ntest MSE (log space)")
            ax.set_title(f"D={D} — vs {xlab.split(' [')[0]}")
            if row == 0 and col == 0:
                ax.legend(fontsize=9, framealpha=0.9)
    fig.suptitle(f"{proc_title}: compute scan (nh=8) — solo vs FT(8) vs FT(25) at 1k/10k/100k/1M\n"
                 f"y = divisor²·best-val test_loss (fair log-space MSE)  [PARTIAL — runs in progress]",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    tag = key.replace("nlovirte4", "")
    for ext in ("png", "pdf"):
        fig.savefig(f"{os.path.dirname(os.path.abspath(__file__))}/compute_scan_{tag}.{ext}", dpi=120)
    print(f"saved compute_scan_{tag}.{{png,pdf}}")

print("\n=== coverage (cells with >=1 finished trial) ===")
print(f"{'family/D':16s} " + "  ".join(f"{k:>9s}" for k in ("eeuu", "eett")))
for fam in ("solo", "ft8", "ft25"):
    for D in DORDER:
        u = coverage.get(("eeuunlovirte4", fam, D), 0)
        t = coverage.get(("eettbarnlovirte4", fam, D), 0)
        print(f"{fam+'/'+D:16s} {u:>9d}  {t:>9d}")
