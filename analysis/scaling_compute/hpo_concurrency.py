"""Did DyHPO actually inform later trials, or did trials run concurrently against a
stale/empty surrogate (=> effectively random search)?

A trial only benefits from BO if >= n_startup (4) earlier trials have OBSERVED
(finished) before it SUGGESTS (starts). Using each trial's job start time (first
log timestamp) and an end estimate (start + dataload + traintime + eval), compute
per cell:
  - max concurrency (how many ran at once)
  - n_informed = trials that started after >=4 others had finished
  - the lr(start-order) trace for a couple of cells to see if BO concentrated.
"""
import glob, json, os, re, yaml
from datetime import datetime, timedelta
import numpy as np

ROOT = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
TS_RE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
N_STARTUP = 4

def start_time(lp):
    try:
        for line in open(lp):
            m = TS_RE.search(line)
            if m: return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except OSError: pass
    return None

def lrknob(cfg, which):
    return (cfg.get("fine_tune") or {}).get("lr_scale") if which == "lr_scale" else cfg["training"]["lr"]

def cell(fam, t, which):
    sd = os.path.join(ROOT, "sweeps", f"{fam}_t{t:05d}")
    rows = []
    for rf in glob.glob(sd + "/results/*.json"):
        idx = int(re.search(r"hp(\d+)_t", os.path.basename(rf)).group(1))
        try: r = json.load(open(rf))
        except OSError: continue
        if "val_loss" not in r: continue
        td = os.path.join(ROOT, "runs", f"{fam}_t{t:05d}", f"trial_{idx:04d}")
        if not os.path.exists(td + "/config.yaml"): continue
        st = start_time(td + "/out_0.log")
        if st is None: continue
        cfg = yaml.safe_load(open(td + "/config.yaml"))
        # end estimate: train time + a flat 90s for dataload+eval overhead
        tt = (r.get("traintime_hours") or 0) * 3600
        rows.append(dict(idx=idx, lr=lrknob(cfg, which), val=r["val_loss"],
                         start=st, end=st + timedelta(seconds=tt + 90)))
    rows.sort(key=lambda d: d["start"])
    return rows

def analyze(rows):
    n = len(rows)
    informed = 0
    for i, r in enumerate(rows):
        done_before = sum(1 for j, o in enumerate(rows) if j != i and o["end"] <= r["start"])
        if done_before >= N_STARTUP: informed += 1
    # max concurrency via sweep line
    evs = []
    for r in rows:
        evs.append((r["start"], 1)); evs.append((r["end"], -1))
    evs.sort()
    cur = mx = 0
    for _, d in evs:
        cur += d; mx = max(mx, cur)
    return n, mx, informed

FAMS = [
    ("solo10k", "solo_nh8_10k_virt_eeuunlovirte4", "lr",
     [4,13,40,126,400,1265,4000,8000]),
    ("solo100k-curve", "scaling_solo_nh8_curve_virt_eeuunlovirte4", "lr", [400,1265,4000]),
    ("FT10k", "finetune_10k_virt_eeuunlovirte4", "lr_scale",
     [4,13,40,126,400,1265,4000,8000]),
    ("FT100k_002", "finetune_scaling_virt_002_eeuunlovirte4", "lr_scale",
     [4,13,40,126,400,1265,4000,20000,40000]),
    ("FT100k_ext", "finetune_scaling_virt_ext_eeuunlovirte4", "lr_scale",
     [4,40,400,4000,8000]),
    ("FT25", "finetune_pt25_eeuunlovirt_eeuunlovirte4", "lr_scale", [10,32,100,316]),
]

print(f"{'cell':28s} {'n':>3} {'maxConc':>7} {'informed':>9}  {'verdict'}")
for tag, fam, which, ts in FAMS:
    for t in ts:
        rows = cell(fam, t, which)
        if len(rows) < 3: continue
        n, mx, inf = analyze(rows)
        verdict = "RANDOM (all concurrent)" if inf == 0 else (
                  "mostly random" if inf < n/3 else "partly BO-informed")
        print(f"{tag+'/t'+str(t):28s} {n:>3} {mx:>7} {inf:>4}/{n:<4} {verdict}")

# show lr trace (start order) for 3 telling cells
print("\n--- lr in start-order (first 4 = random startup); does it concentrate? ---")
for tag, fam, which, t in [("FT100k_002", "finetune_scaling_virt_002_eeuunlovirte4", "lr_scale", 4000),
                           ("solo10k", "solo_nh8_10k_virt_eeuunlovirte4", "lr", 4000),
                           ("FT25", "finetune_pt25_eeuunlovirt_eeuunlovirte4", "lr_scale", 100)]:
    rows = cell(fam, t, which)
    t0 = rows[0]["start"]
    best = min(rows, key=lambda r: r["val"])
    print(f"\n{tag}/t{t}  (best lr={best['lr']:.3g} val={best['val']:.2e})")
    for i, r in enumerate(rows):
        dt = (r["start"] - t0).total_seconds() / 60
        mark = " <-best" if r is best else ""
        tag2 = "[startup]" if i < N_STARTUP else "[BO?]    "
        print(f"  {i:2d} {tag2} +{dt:6.1f}min  lr={r['lr']:.3e}  val={r['val']:.2e}{mark}")
