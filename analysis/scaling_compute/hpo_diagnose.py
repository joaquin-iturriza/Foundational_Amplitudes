"""Diagnose whether each DyHPO cell was well-optimized.

Per cell (family x t_steps): read every trial's searched LR knob (fine_tune.lr_scale
for finetune cells, training.lr for solo cells), its val_loss, and the wall-clock
start time of its SLURM job (first log line). Report:
  - LR boundary: where the best-val LR sits in the searched log-range (0=low edge,
    1=high edge). <0.15 or >0.85 => optimum likely against the search boundary.
  - whether val_loss vs LR shows a clear interior minimum.
  - start-time spread vs train time => did trials overlap (DyHPO surrogate empty
    => effectively random search) or run staggered (BO could use observations)?
"""
import glob, json, os, re, yaml
from datetime import datetime
import numpy as np

ROOT = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"

FAMILIES = {
    "solo 10k":    ("solo_nh8_10k_virt_{k}", "lr"),
    "solo 100k":   (["scaling_solo_nh8_lowt_virt_{k}", "scaling_solo_nh8_anchor_virt_002_{k}",
                     "scaling_solo_nh8_curve_virt_{k}", "scaling_solo_nh8_anchor2_virt_{k}"], "lr"),
    "FT 8ds 10k":  ("finetune_10k_virt_{k}", "lr_scale"),
    "FT 8ds 100k": (["finetune_scaling_virt_002_{k}", "finetune_scaling_virt_ext_{k}"], "lr_scale"),
    "FT 25ds":     ("finetune_pt25_{p}_{k}", "lr_scale"),
}
KEYS = ["eeuunlovirte4", "eettbarnlovirte4"]
TS_RE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")

def fams_for(spec, key):
    pre = "eeuunlovirt" if key == "eeuunlovirte4" else "eettbarnlovirt"
    spec = spec if isinstance(spec, list) else [spec]
    return [s.format(k=key, p=pre) for s in spec]

def lrknob(cfg, which):
    if which == "lr_scale":
        return (cfg.get("fine_tune") or {}).get("lr_scale")
    return cfg["training"]["lr"]

def start_time(logpath):
    try:
        with open(logpath) as f:
            for line in f:
                m = TS_RE.search(line)
                if m:
                    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except OSError:
        pass
    return None

def cell_trials(fam, t, which):
    """list of dicts: {lr, val, test, start, traintime_h} for each trial in the cell."""
    sd = os.path.join(ROOT, "sweeps", f"{fam}_t{t:05d}")
    rows = []
    for rf in glob.glob(sd + "/results/*.json"):
        m = re.search(r"hp(\d+)_t", os.path.basename(rf))
        idx = int(m.group(1))
        try: r = json.load(open(rf))
        except OSError: continue
        if "val_loss" not in r: continue
        td = os.path.join(ROOT, "runs", f"{fam}_t{t:05d}", f"trial_{idx:04d}")
        cfgp = os.path.join(td, "config.yaml")
        if not os.path.exists(cfgp): continue
        cfg = yaml.safe_load(open(cfgp))
        rows.append(dict(idx=idx, lr=lrknob(cfg, which), val=r["val_loss"],
                         test=r.get("test_loss"), traintime_h=r.get("traintime_hours"),
                         start=start_time(os.path.join(td, "out_0.log"))))
    return rows

def grid_for(fam, t):
    sd = os.path.join(ROOT, "sweeps", f"{fam}_t{t:05d}")
    return os.path.isdir(sd)

print(f"{'cell':40s} {'n':>3} {'lr_lo':>9} {'lr_hi':>9} {'best_lr':>9} {'edge':>5} {'Δval/dec':>8} {'tspread':>8} {'train':>7} {'overlap'}")
for key in KEYS:
    for gname, (spec, which) in FAMILIES.items():
        for fam in fams_for(spec, key):
            # all t levels for this fam
            ts = sorted(int(re.search(r"_t(\d+)$", os.path.basename(d)).group(1))
                        for d in glob.glob(os.path.join(ROOT, "sweeps", f"{fam}_t*"))
                        if re.sub(r"_t\d+$", "", os.path.basename(d)) == fam)
            for t in ts:
                rows = [r for r in cell_trials(fam, t, which) if r["lr"]]
                if len(rows) < 2: continue
                lrs = np.array([r["lr"] for r in rows]); vals = np.array([r["val"] for r in rows])
                best = rows[int(np.argmin(vals))]
                lo, hi = lrs.min(), lrs.max()
                # position of best lr in the sampled log-range
                edge = (np.log(best["lr"]) - np.log(lo)) / (np.log(hi) - np.log(lo) + 1e-12)
                # sensitivity: spread of val across the lr range (how much does lr matter?)
                # slope of log(val) vs log(lr) near optimum is noisy; use ratio worst/best
                dval = np.log10(vals.max() / vals.min())
                # timing
                starts = [r["start"] for r in rows if r["start"]]
                if len(starts) >= 2:
                    spread = (max(starts) - min(starts)).total_seconds()
                else:
                    spread = float("nan")
                traintime_s = np.median([r["traintime_h"] for r in rows if r["traintime_h"]]) * 3600
                overlap = "YES(~rand)" if (spread == spread and traintime_s and spread < traintime_s) else "staggered"
                flag = " <<EDGE" if (edge < 0.15 or edge > 0.85) else ""
                cell = f"{key[:6]}/{gname}/t{t}"
                print(f"{cell:40s} {len(rows):>3} {lo:>9.2e} {hi:>9.2e} {best['lr']:>9.2e} "
                      f"{edge:>5.2f} {dval:>8.2f} {spread:>8.0f} {traintime_s:>7.0f} {overlap}{flag}")
