#!/usr/bin/env python3
"""
quantify_fast_vs_original.py — put numbers on the original-vs-fast comparison.

Answers two questions the plot can't show cleanly:
  (1) Variability: how reproducible is val_loss? (fast vs original, same HP)
  (2) Speedup vs model size: is the wall-time speedup independent of num_heads
      (and how does it depend on batch size, the more likely driver)?

RUN ON JEAN ZAY:
    python sweep/quantify_fast_vs_original.py [--sweep-base DIR] [--suffix _fast]
"""
import argparse
import json
import os
import re

import numpy as np
import yaml


def scan_best(results_dir):
    if not os.path.isdir(results_dir):
        return None
    best = None
    for fn in os.listdir(results_dir):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(results_dir, fn)) as f:
                r = json.load(f)
        except Exception:
            continue
        vl = r.get("val_loss")
        if vl is None:
            continue
        tt = r.get("traintime_hours")
        if best is None or vl < best[0]:
            best = (float(vl), float(tt) if tt else float("nan"))
    return best


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _bs_from_cfg(cell_dir):
    try:
        with open(os.path.join(cell_dir, "sweep_config.yaml")) as f:
            cfg = yaml.safe_load(f)
        return int(cfg["fixed_params"]["training.batchsize"])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-base",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "sweeps", "pretraining_scaling"))
    ap.add_argument("--suffix", default="_fast")
    args = ap.parse_args()
    base = args.sweep_base
    names = set(os.listdir(base))

    rows = []   # (nh, bs, t, orig_val, fast_val, orig_t, fast_t)
    for cell in sorted(names):
        if cell.endswith(args.suffix) or (cell + args.suffix) not in names:
            continue
        m_nh, m_t = re.search(r"nh(\d+)", cell), re.search(r"_t(\d+)", cell)
        if not (m_nh and m_t):
            continue
        orig = scan_best(os.path.join(base, cell, "results"))
        fast = scan_best(os.path.join(base, cell + args.suffix, "results"))
        if orig is None or fast is None:
            continue
        bs = _bs_from_cfg(os.path.join(base, cell))
        rows.append((int(m_nh.group(1)), bs, int(m_t.group(1)),
                     orig[0], fast[0], orig[1], fast[1]))

    if not rows:
        print("No matched original/_fast pairs with results yet.")
        return

    nh   = np.array([r[0] for r in rows])
    bs   = np.array([r[1] if r[1] else 0 for r in rows])
    ov   = np.array([r[3] for r in rows]); fv = np.array([r[4] for r in rows])
    ot   = np.array([r[5] for r in rows]); ft = np.array([r[6] for r in rows])
    ok   = np.isfinite(ot) & np.isfinite(ft) & (ft > 0)

    speed = ot[ok] / ft[ok]          # wall-time speedup factor
    saved = ot[ok] - ft[ok]          # absolute hours saved
    nh_t, bs_t = nh[ok], bs[ok]
    print(f"Matched pairs with timing: {ok.sum()} / {len(rows)}\n")

    # ── (1) val_loss reproducibility ────────────────────────────────────────
    vr = fv / ov                     # fast / original, same HP -> ideally ~1
    lr = np.log(vr)
    fac = np.exp(np.abs(lr))         # symmetric factor difference (>=1)
    print("=== (1) val_loss reproducibility (fast vs original, SAME HP) ===")
    print(f"  ratio fast/orig:  median={np.median(vr):.3f}  "
          f"IQR=[{np.percentile(vr,25):.3f}, {np.percentile(vr,75):.3f}]  "
          f"min={vr.min():.3f}  max={vr.max():.3f}")
    print(f"  |log-ratio| as a factor:  median={np.median(fac):.2f}x  "
          f"90th pct={np.percentile(fac,90):.2f}x")
    for thr in (1.25, 1.5, 2.0):
        frac = np.mean(fac <= thr) * 100
        print(f"  within {thr:.2f}x: {frac:.0f}% of cells")
    print("  (1.0 = perfectly reproducible; larger = more run-to-run noise)\n")

    # ── (2) speedup vs model size / batch size ──────────────────────────────
    print("=== (2) wall-time speedup (original / fast) ===")
    print(f"  overall: median={np.median(speed):.2f}x  "
          f"IQR=[{np.percentile(speed,25):.2f}, {np.percentile(speed,75):.2f}]  "
          f"min={speed.min():.2f}  max={speed.max():.2f}")
    print(f"  CV(speedup)={np.std(speed)/np.mean(speed):.2f}  (variability of the speedup itself)\n")

    print("  by num_heads:")
    print(f"    {'nh':>4} {'n':>3} {'median x':>9} {'IQR':>16} {'med saved (h)':>13}")
    for v in sorted(set(nh_t.tolist())):
        s = speed[nh_t == v]; sv = saved[nh_t == v]
        print(f"    {v:>4} {len(s):>3} {np.median(s):>8.2f}x "
              f"[{np.percentile(s,25):>5.2f},{np.percentile(s,75):>5.2f}] {np.median(sv):>12.3f}")

    print("\n  by batch size:")
    print(f"    {'bs':>6} {'n':>3} {'median x':>9}")
    for v in sorted(set(bs_t.tolist())):
        s = speed[bs_t == v]
        print(f"    {v:>6} {len(s):>3} {np.median(s):>8.2f}x")

    print("\n  rank correlation of speedup with:")
    print(f"    num_heads : Spearman r = {spearman(nh_t, speed):+.2f}")
    print(f"    batch size: Spearman r = {spearman(bs_t, speed):+.2f}")
    print("    (~0 = independent; negative w.r.t. nh = bigger models gain less)")


if __name__ == "__main__":
    main()
