"""Walltime-only talk variant of build_compute_scan_plot.py — identical data/logic,
but a single 2x2 grid per process (the four dataset sizes 1k/10k/100k/1M), each
panel = fair log-space test MSE vs WALLTIME [h], curves solo / FT8 / FT25. Large
fonts/markers. Saves to plots/compute_scan_{eeuu,eettbar}_wt.pdf.

With --clean: omit the 1h feature-ladder markers and the raw-encoding 416-proc
curve (talk variant), saving to plots/compute_scan_{eeuu,eettbar}_wt_clean.pdf.
"""
import glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15,
    "axes.linewidth": 1.1,
})

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
    "ft416raw":  {D: [f"cscan_ft416raw_D{D}", f"cscan_ft416raw_D{D}_hpo"]  for D in SUB},
    "ft416best": {D: [f"cscan_ft416best_D{D}", f"cscan_ft416best_D{D}_hpo"] for D in SUB},
    "ft352lo":   {D: [f"cscan_ft352lo_D{D}", f"cscan_ft352lo_D{D}_hpo"]   for D in SUB},
}

def cells(prefix, key):
    pat = re.compile(rf"^{re.escape(prefix)}(_\d+)?_{key}_t(\d+)$")
    for d in glob.glob(f"{ROOT}/sweeps/{prefix}*_{key}_t*"):
        m = pat.match(os.path.basename(d))
        if m:
            yield int(m.group(2)), d

def curve(prefixes, key, D):
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
             "ft25": dict(color="C0", marker="D"),
             "ft416raw":  dict(color="C5", marker="v"),
             "ft416best": dict(color="C3", marker="^"),
             "ft352lo":   dict(color="C6", marker="P")}
FAM_LABEL = {"solo": "solo (scratch)", "ft8": "FT ← 8-proc", "ft25": "FT ← 25-proc",
             "ft416raw":  "FT ← 416-proc (raw enc.)",
             "ft416best": "FT ← 416-proc (best)",
             "ft352lo":   "FT ← 352-proc (LO only)"}
CLEAN = "--clean" in sys.argv[1:]
FAMS = ("solo", "ft8", "ft25", "ft416best", "ft352lo") if CLEAN else \
       ("solo", "ft8", "ft25", "ft416raw", "ft416best", "ft352lo")
DORDER = ["1k", "10k", "100k", "1M"]
WT_IDX = 1   # (logMSE, walltime_h, compute)  ->  walltime

for proc_title, key in [("ee_uu NLO-virt", "eeuunlovirte4"), ("ee_ttbar NLO-virt", "eettbarnlovirte4")]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharey=True)
    for i, D in enumerate(DORDER):
        ax = axes[i // 2][i % 2]
        for fam in FAMS:
            data = curve(CURVES[fam][D], key, D)
            if not data:
                continue
            ts = sorted(data)
            ax.plot([data[t][WT_IDX] for t in ts], [data[t][0] for t in ts],
                    label=FAM_LABEL[fam], lw=2.8, ms=9, **FAM_STYLE[fam])
        # 1h-pretrain feature-ladder markers (D=100k panel only): raw-era encoding
        # -> +onehots -> +mass/coupling scalars -> full adopted config, each
        # pretrained ~1h on the 416 sets, fine-tuned at the deepest cell. Same
        # x-axis convention as the curves (fine-tune walltime only).
        if D == "100k" and not CLEAN:
            for j, (fam, lbl) in enumerate([("ftraw1h", "raw"), ("ftrung2", "+onehots"),
                                            ("ftrung3", "+scalars"), ("ftbest1h", "full")]):
                for t, pt in curve([f"cscan_{fam}_D100k", f"cscan_{fam}_hpo"], key, D).items():
                    ax.scatter([pt[WT_IDX]], [pt[0]], s=150, marker="*",
                               color=f"C{j+4}", zorder=5,
                               label=f"1h ladder: {lbl}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3, lw=0.6)
        ax.set_title(f"D = {D} points")
        if i // 2 == 1:
            ax.set_xlabel("walltime [h]")
        if i % 2 == 0:
            ax.set_ylabel("test MSE (log space)")
        if i == 0:
            ax.legend(framealpha=0.9)
    # suptitle removed for talk slides (slide caption explains the figure)
    fig.tight_layout()
    tag = key.replace("nlovirte4", "")
    out = f"{ROOT}/plots/compute_scan_{tag}_wt{'_clean' if CLEAN else ''}.pdf"
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=180)
    print(f"saved {out} (+.png)")
