#!/usr/bin/env python3
"""Step 70 of plan entry catalog_coverage: the read of the width x length scaling grid at >= 1 epoch, under the design
passed 2026-11-06T22:46:58 (plan design of catalog_coverage, D292, D293, D314, D315).

Cells: arm v4_het_default25 of scripts/job_catalog_v3_short.sh with LONG_HP=1262 (trial_1262's HPs, lr = 7.82114e-3
(steps/17056)^-0.55), LONG_HEADS 8/16/32, LONG_FACTOR 2/4/8/16 (8528, 17056, 34112, 68224 steps), seed 42, run dirs
runs/catalog_v4_long/v4_het_default25_w<W>_x<F>_hp1262; widths 8 at 4x and 16x are the long sweeps' trial_1262
(reused, not rerun: earlier code, H100 with TF32).

  pull       per training pool the validation MSE of ln|M|^2 of each finished new cell from its sigma stage's saved
             predictions (sigma_stage/preds_val.npz: mu frozen at the main run's best checkpoint), mean over events
             of (ln|raw_pred| - ln|raw_truth|)^2, as size_figs.pull_preds; one read-only `site run` on CC-IN2P3
             -> data/scaling_grid_preds.json (the reused cells come from data/size_mse_preds.json)
  read       per cell: L = geometric mean over every training pool of that MSE (size_figs' mse_logm2_gmean_all), best
             val_loss_no_reg (het NLL, main block, data/reads/<run>_val.json), blow-up check (mixture_ab.blowups),
             GPU and training minutes (the run's log), and the zero-shot reads (PP4, EE4, HD4, HD5, H1-H7c; the
             final model's, data/reads/<run>_<axis>.json) as R over the width-8 2x cell: the median over classes of
             the class's geometric mean of cell / reference, class bootstrap 95% (10000, seed 0).
             Fits of L(C) = A C^-alpha + L_inf, C = 6 N S B, fixed by the design: (i) joint over the 12 cells (one
             A, alpha, L_inf; 9 dof), (ii) per width over length (4 points, 1 dof), (iii) per length over width (3
             points, 0 dof: a curve, not a test), (iv) shared exponent (one alpha and L_inf, an A per width; 7 dof).
             Errors: relative, per width, the 1x seed pair's |ln(MSE_42 / MSE_43)| (size_mse_preds.json: w8 0.032,
             w16 0.077, w32 0.062), the design's seed-spread error. A fit runs only with all its cells in; (i) and
             (iv) need all 12 (an INTERIM fit over the cells in hand is printed beside, labelled, never a verdict).
             -> data/scaling_grid.json
             Beside the axes, held_default: the default-sampled events of the held-out classes pooled (the PP4
             and EE4 probes whose class is held out, plus HD4 and HD5; keys (axis, probe)), pooled and per held-out
             axis, as mixture_ab._held_default builds it for two seeds, here on one seed per side.
  fig        four figures (png + pdf; the passed design names the measures, not a figure layout):
             figs/scaling_grid_compute  L against C per width, per-width fits drawn where determined;
             figs/scaling_grid_loss     best val_loss_no_reg (het NLL, main block) against C per width;
             figs/scaling_grid_hours    L against the main block's training GPU-hours (extra: cost per width);
             figs/scaling_grid_reads    R over the width-8 2x cell against C, on PP4 and on held_default pooled
"""
import json, math, os, re, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
sys.path.insert(0, HERE)
BATCH = 16384
N_PARAMS = {8: 1775829, 16: 6516181, 32: 25434069}     # 'learnable parameters' of each log
STEPS = {2: 8528, 4: 17056, 8: 34112, 16: 68224}
# (width, factor): run id; plan catalog_coverage step 70 (cells read from each log's 'LONG_HP 1262: steps' line and
# its parameter count)
NEW = {(8, 2): "FA-ccin2p3-34cfff0c", (16, 2): "FA-ccin2p3-c53b8e51", (32, 2): "FA-ccin2p3-ff309c2d",
       (16, 4): "FA-ccin2p3-1b1a8f6e", (32, 4): "FA-ccin2p3-b07d6237",
       (8, 8): "FA-ccin2p3-4ba33195", (16, 8): "FA-ccin2p3-8a523b67", (32, 8): "FA-ccin2p3-8b543518",
       (16, 16): "FA-ccin2p3-4d6e9b06", (32, 16): "FA-ccin2p3-a77e20b8"}
# reused: the long sweeps' trial_1262, read by their eval re-reads (mixture_ab.LONG_BEST)
REUSED = {(8, 4): ("FA-ccin2p3-6f8a1eed", "runs/cov_v4_long_hpo8_cc_17056/trial_1262", "trial_17056_trial_1262"),
          (8, 16): ("FA-ccin2p3-1a46f3e0", "runs/cov_v4_long_hpo8_cc_68224/trial_1262", "trial_68224_trial_1262")}
REF = (8, 2)
AXES = ("PP4", "EE4", "HD4", "HD5", "H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")
X1 = {8: ("runs/catalog_v4_long/v4_het_default25_w8_x1", "runs/catalog_v4_long/v4_het_default25_seed2_w8_x1"),
      16: ("runs/catalog_v4_long/v4_het_default25_w16_x1", "runs/catalog_v4_long/v4_het_default25_seed2_w16_x1"),
      32: ("runs/catalog_v4_long/v4_het_default25_w32_x1", "runs/catalog_v4_long/v4_het_default25_seed2_w32_x1")}
PREDS = os.path.join(DATA, "scaling_grid_preds.json")
LOGS = os.environ.get("SG_LOGS")      # directory of the runs' full logs (site logs -n 3000000 <run> > <id>.log)


def run_dir(w, f):
    return f"runs/catalog_v4_long/v4_het_default25_w{w}_x{f}_hp1262"


def pull():
    from size_figs import SIGNED
    dirs = [run_dir(w, f) for (w, f) in NEW]
    py = ("import json,sys,os\nimport numpy as np\n"
          "for r in sys.argv[1:]:\n"
          "    p=r+'/sigma_stage/preds_val.npz'\n"
          "    if not os.path.exists(p):\n"
          "        print('MISSING', r); continue\n"
          "    z=np.load(p); n=[str(x) for x in z['names']]; pid=z['process_id']\n"
          "    d=(np.log(np.abs(z['raw_pred'].ravel()))-np.log(np.abs(z['raw_truth'].ravel())))**2\n"
          "    out={}\n"
          "    for i in np.unique(pid):\n"
          "        m=pid==i; v=d[m]; f=np.isfinite(v)\n"
          "        out[n[i]]=[float(v[f].mean()) if f.any() else None, int(m.sum()), int((~f).sum())]\n"
          "    print('PREDS', r, json.dumps(out))\n")
    cmd = (f"cat > /tmp/zz_probe_sgpreds.py <<'PYEOF'\n{py}PYEOF\npython3 /tmp/zz_probe_sgpreds.py {' '.join(dirs)}; "
           "rm -f /tmp/zz_probe_sgpreds.py")
    out = subprocess.run(["site", "--timeout", "900", "run", "ccin2p3", "FA", "--", cmd], capture_output=True,
                         text=True, timeout=1000).stdout
    dec = json.JSONDecoder()
    per = {l.split()[1]: dec.raw_decode(l.split(" ", 2)[2])[0] for l in out.splitlines() if l.startswith("PREDS runs/")}
    missing = [l.split()[1] for l in out.splitlines() if l.startswith("MISSING ")]
    if not per:
        sys.exit("pull: nothing read\n" + out[-2000:])
    old = json.load(open(PREDS))["runs"] if os.path.exists(PREDS) else {}
    old.update(per)
    json.dump({"source": "sigma_stage/preds_val.npz of each new cell (scaling_grid.py pull, read-only site run on "
               "CC-IN2P3); per pool [MSE of ln|M|^2 over finite events, events, non-finite events]",
               "signed_pools_file": os.path.relpath(SIGNED, ROOT), "runs": old}, open(PREDS, "w"), indent=0)
    print(json.dumps({"read": sorted(per), "missing": missing}))


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def cell_mse(w, f):
    """(L, n_pools, non-finite events, source) of one cell, None when not read"""
    if (w, f) in REUSED:
        r = json.load(open(os.path.join(DATA, "size_mse_preds.json")))["runs"][REUSED[(w, f)][1]]
        return r["mse_logm2_gmean_all"], r["n_pools"], r["nonfinite_events"], "size_mse_preds.json"
    runs = json.load(open(PREDS))["runs"] if os.path.exists(PREDS) else {}
    p = runs.get(run_dir(w, f))
    if p is None:
        return None
    v = [x[0] for x in p.values() if x[0] is not None and x[0] > 0]
    return gmean(v), len(p), sum(x[2] for x in p.values()), "scaling_grid_preds.json"


def seed_sigma():
    """per width the relative seed error: |ln(MSE_42 / MSE_43)| of the 1x seed pair (size_mse_preds.json)"""
    r = json.load(open(os.path.join(DATA, "size_mse_preds.json")))["runs"]
    return {w: abs(math.log(r[a]["mse_logm2_gmean_all"] / r[b]["mse_logm2_gmean_all"])) for w, (a, b) in X1.items()}


def log_facts(run):
    """GPU and the main block's training minutes from the run's log (SG_LOGS/<id>.log)"""
    if not LOGS:
        return {}
    f = os.path.join(LOGS, run.split("-")[-1] + ".log")
    if not os.path.exists(f):
        return {}
    gpu = minutes = None
    for line in open(f, errors="replace"):
        if gpu is None and line.startswith("RUN_SUMMARY "):
            m = re.search(r'"gpu": "([^"]*)"', line)
            gpu = m.group(1) if m else None
        m = re.search(r"Finished training: (\d+) iterations .* in ([0-9.]+)min", line)
        if m and minutes is None and int(m.group(1)) > 1000:
            minutes = float(m.group(2))
    return {"gpu": gpu, "train_hours": None if minutes is None else round(minutes / 60, 3)}


def best_val(run):
    f = os.path.join(DATA, "reads", f"{run}_val.json")
    if not os.path.exists(f):
        return None
    v = [x for _, x in json.load(open(f))["blocks"][0]["vals"] if math.isfinite(x)]
    return min(v) if v else None


def read_r(num, den, cls, n_boot=10000):
    keys = sorted(set(num) & set(den))
    by = {}
    for k in keys:
        by.setdefault(cls(k), []).append(num[k] / den[k])
    v = np.array([gmean(x) for x in by.values()])
    rng = np.random.default_rng(0)
    meds = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return {"R": round(float(np.median(v)), 4), "boot95": [round(float(lo), 4), round(float(hi), 4)],
            "n_classes": len(v), "n_probes": len(keys)}


def fit_law(pts, shared_exponent=False):
    """Weighted least squares of L = A_g C^-alpha + L_inf over pts [(group, C, L, sigma_rel)], one A per group when
    shared_exponent (else one A): profiled over an (L_inf, alpha) grid with A in closed form, then refined (scipy
    least_squares); residuals (L - model) / (sigma_rel L). Returns {A, alpha, L_inf, chi2, dof, chi2_red, at_edge}"""
    from scipy.optimize import least_squares
    g = [p[0] if shared_exponent else 0 for p in pts]
    groups = sorted(set(g))
    c = np.array([p[1] for p in pts]); l = np.array([p[2] for p in pts]); s = np.array([p[3] for p in pts]) * l
    npar = len(groups) + 2
    dof = len(pts) - npar
    if dof < 0:
        return None
    lc = np.log(c / c.min())        # C scaled to the smallest cell, so A is the excess at the smallest C

    def amp(li, al):
        a = {}
        for gr in groups:
            m = np.array([x == gr for x in g])
            x = np.exp(-al * lc[m]); w = 1 / s[m] ** 2
            a[gr] = float(np.sum(w * x * (l[m] - li)) / np.sum(w * x * x))
        return a

    def res(li, al, a):
        return (l - (np.array([a[x] for x in g]) * np.exp(-al * lc) + li)) / s

    best = None
    for li in np.linspace(0, 0.999 * l.min(), 400):
        for al in np.linspace(0.01, 3.0, 300):
            a = amp(li, al)
            if min(a.values()) <= 0:
                continue
            q = float(np.sum(res(li, al, a) ** 2))
            if best is None or q < best[0]:
                best = (q, li, al)
    if best is None:
        return None
    _, li0, al0 = best
    a0 = amp(li0, al0)
    x0 = [li0, al0] + [math.log(a0[gr]) for gr in groups]

    def f(x):
        a = {gr: math.exp(x[2 + i]) for i, gr in enumerate(groups)}
        return res(x[0], x[1], a)
    sol = least_squares(f, x0, bounds=([0, 0] + [-np.inf] * len(groups), [l.min(), 5] + [np.inf] * len(groups)))
    li, al = float(sol.x[0]), float(sol.x[1])
    chi2 = float(np.sum(sol.fun ** 2))
    a = {str(gr): math.exp(sol.x[2 + i]) for i, gr in enumerate(groups)}
    edge = li < 1e-3 * l.min() or li > 0.99 * l.min()
    return {"A_at_Cmin": a if shared_exponent else a["0"], "C_min": float(c.min()), "alpha": round(al, 4),
            "L_inf": li, "chi2": round(chi2, 3), "dof": dof, "chi2_red": round(chi2 / dof, 3) if dof else None,
            "L_inf_at_edge": bool(edge)}


def held_default(run, ref, cls, ax_of):
    """{"held_default": R pooled, "held_default/<axis>": R per held-out axis} of run over ref on the default-sampled
    events of the held-out classes: PP4 and EE4 probes whose class is held out, plus HD4 and HD5 (keys (axis,
    probe)); {} when a read is missing"""
    import mixture_ab as mab
    num, den = {}, {}
    for ax in ("PP4", "EE4", "HD4", "HD5"):
        a, b = mab.read_of(run, ax), mab.read_of(ref, ax)
        if a is None or b is None:
            return {}
        num.update({(ax, p): x for p, x in a.items()}); den.update({(ax, p): x for p, x in b.items()})
    keys = [k for k in set(num) & set(den) if cls(k[1]) in ax_of]
    cl = lambda k: cls(k[1])
    out = {"held_default": read_r({k: num[k] for k in keys}, {k: den[k] for k in keys}, cl)}
    for hax in mab.HELD:
        ks = [k for k in keys if ax_of[cls(k[1])] == hax]
        if ks:
            out[f"held_default/{hax}"] = read_r({k: num[k] for k in ks}, {k: den[k] for k in ks}, cl)
    return out


def read():
    import mixture_ab as mab
    cls = mab.class_of()
    sig = seed_sigma()
    cells = {}
    for w in (8, 16, 32):
        for f in (2, 4, 8, 16):
            k = f"w{w}_x{f}"
            run = NEW.get((w, f)) or REUSED[(w, f)][0]
            train = run if (w, f) in NEW else REUSED[(w, f)][2]
            m = cell_mse(w, f)
            cells[k] = {"width": w, "factor": f, "steps": STEPS[f], "epochs": round(STEPS[f] / 6707, 3),
                        "lr": float(f"{7.82114e-3 * (STEPS[f] / 17056) ** -0.55:.7g}"),
                        "C": 6.0 * N_PARAMS[w] * STEPS[f] * BATCH, "run": run, "reused": (w, f) in REUSED,
                        "trained_by": train,
                        "mse_logm2_gmean": None if m is None else m[0],
                        "n_pools": None if m is None else m[1], "nonfinite_events": None if m is None else m[2],
                        "best_val_loss_no_reg": best_val(train), **log_facts(run)}
    ref = NEW[REF]
    ax_of = mab._held_axes(cls)
    reads = {}
    for k, c in cells.items():
        reads[k] = {}
        for ax in AXES:
            if c["run"] == ref:
                continue
            a, b = mab.read_of(c["run"], ax), mab.read_of(ref, ax)
            if a is None or b is None:
                continue
            reads[k][ax] = read_r(a, b, cls)
        if c["run"] != ref:
            reads[k].update(held_default(c["run"], ref, cls, ax_of))
    in_hand = [k for k, c in cells.items() if c["mse_logm2_gmean"] is not None]
    pt = lambda k, grp: (grp, cells[k]["C"], cells[k]["mse_logm2_gmean"], sig[cells[k]["width"]])
    fits = {"sigma_rel_per_width": {str(w): round(v, 4) for w, v in sig.items()}}
    per_w = {}
    for w in (8, 16, 32):
        ks = [f"w{w}_x{f}" for f in (2, 4, 8, 16)]
        per_w[str(w)] = (fit_law([pt(k, w) for k in ks]) if all(k in in_hand for k in ks)
                         else {"waits_on": [cells[k]["run"] for k in ks if k not in in_hand]})
        if all(k in in_hand for k in ks):
            x = np.log([cells[k]["C"] for k in ks]); y = np.log([cells[k]["mse_logm2_gmean"] for k in ks])
            per_w[str(w)]["bare_power_law_diagnostic_alpha"] = round(float(-np.polyfit(x, y, 1)[0]), 4)
    fits["ii_per_width"] = per_w
    per_l = {}
    for f in (2, 4, 8, 16):
        ks = [f"w{w}_x{f}" for w in (8, 16, 32)]
        if all(k in in_hand for k in ks):
            x = np.log([cells[k]["C"] for k in ks]); y = np.log([cells[k]["mse_logm2_gmean"] for k in ks])
            per_l[str(f)] = {"not_a_test": "3 points, 0 dof", "bare_power_law_diagnostic_alpha":
                             round(float(-np.polyfit(x, y, 1)[0]), 4)}
        else:
            per_l[str(f)] = {"waits_on": [cells[k]["run"] for k in ks if k not in in_hand]}
    fits["iii_per_length"] = per_l
    allk = list(cells)
    if all(k in in_hand for k in allk):
        fits["i_joint"] = fit_law([pt(k, 0) for k in allk])
        fits["iv_shared_exponent"] = fit_law([pt(k, cells[k]["width"]) for k in allk], shared_exponent=True)
    else:
        wait = [cells[k]["run"] for k in allk if k not in in_hand]
        fits["i_joint"] = {"waits_on": wait}
        fits["iv_shared_exponent"] = {"waits_on": wait}
        fits["INTERIM_not_a_verdict"] = {
            "cells": in_hand,
            "i_joint": fit_law([pt(k, 0) for k in in_hand]),
            "iv_shared_exponent": fit_law([pt(k, cells[k]["width"]) for k in in_hand], shared_exponent=True)}
    out = {"design": "plan catalog_coverage step 64/70, design passed 2026-11-06T22:46:58",
           "law": "L(C) = A C^-alpha + L_inf, C = 6 N S B (B = 16384), L = geometric mean over every training pool of "
                  "the validation MSE of ln|M|^2 (sigma stage's saved predictions)",
           "reference_for_R": f"w8_x2 ({ref})", "cells": cells, "reads_R_over_w8_x2": reads, "fits": fits,
           "blowups": mab.blowups([c["trained_by"] if not c["reused"] else c["run"] for c in cells.values()
                                   if c["mse_logm2_gmean"] is not None])}
    json.dump(out, open(os.path.join(DATA, "scaling_grid.json"), "w"), indent=1)
    print(json.dumps({"cells": {k: {x: c.get(x) for x in ("run", "mse_logm2_gmean", "best_val_loss_no_reg", "gpu",
                                                      "train_hours")} for k, c in cells.items()},
                      "fits": fits, "flagged": out["blowups"]["flagged"],
                      "spiked": out["blowups"]["spiked_recovered"]}, indent=1))


def fig():
    """figs/scaling_grid_compute, scaling_grid_hours, scaling_grid_reads (png + pdf) from data/scaling_grid.json"""
    sys.path.insert(0, ROOT)
    import plot_style as ps
    d = json.load(open(os.path.join(DATA, "scaling_grid.json")))
    cells, fits = d["cells"], d["fits"]
    x1 = json.load(open(os.path.join(DATA, "size_mse_preds.json")))["runs"]
    WCOL = {8: (ps.C.blue, "o"), 16: (ps.C.green, "D"), 32: (ps.C.vermillion, "^")}
    os.makedirs(FIGS, exist_ok=True)
    report = {}

    def series(w, xkey):
        ks = [f"w{w}_x{f}" for f in (2, 4, 8, 16) if cells[f"w{w}_x{f}"]["mse_logm2_gmean"] is not None]
        return ks, [xkey(cells[k]) for k in ks], [cells[k]["mse_logm2_gmean"] for k in ks]

    # 1: MSE against compute, per width; per-width fits where all four lengths are in
    f1, ax = ps.figure()
    for w, (col, mk) in WCOL.items():
        ks, xs, ys = series(w, lambda c: c["C"])
        ax.plot(xs, ys, color=col, lw=1.2)
        for k, x, y in zip(ks, xs, ys):
            ax.plot([x], [y], marker=mk, color=col, ls="none", mfc="none" if cells[k]["reused"] else col)
        ax.plot([], [], marker=mk, color=col, label=f"width {w}")
        fw = fits["ii_per_width"].get(str(w), {})
        if "alpha" in fw:
            cc = np.geomspace(min(xs), max(xs), 50)
            ax.plot(cc, fw["A_at_Cmin"] * (cc / fw["C_min"]) ** -fw["alpha"] + fw["L_inf"], color=col, ls="--", lw=0.9)
        report[f"w{w}"] = dict(zip(ks, [round(y, 5) for y in ys]))
        xb = [6.0 * N_PARAMS[w] * 4264 * BATCH] * 2
        yb = [x1[r]["mse_logm2_gmean_all"] for r in X1[w]]
        ax.plot(xb, yb, marker=mk, color=col, ls="none", alpha=0.35, ms=4)
    ax.plot([], [], marker="o", color="k", mfc="none", ls="none", label="reused run (H100)")
    ax.plot([], [], color="k", ls="--", lw=0.9, label="fit per width")
    ax.plot([], [], marker="o", color="k", alpha=0.35, ms=4, ls="none", label="0.64 epoch, untuned")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training compute $6NSB$ (FLOP)")
    ax.set_ylabel("validation MSE of $\\ln|M|^2$")
    ps.legend(ax, "upper right")
    ps.save(f1, os.path.join(FIGS, "scaling_grid_compute"))

    # 1b: best val_loss_no_reg (het NLL of the main block) against compute
    fl, ax = ps.figure()
    for w, (col, mk) in WCOL.items():
        ks = [f"w{w}_x{f}" for f in (2, 4, 8, 16) if cells[f"w{w}_x{f}"]["best_val_loss_no_reg"] is not None]
        xs = [cells[k]["C"] for k in ks]; ys = [cells[k]["best_val_loss_no_reg"] for k in ks]
        ax.plot(xs, ys, color=col, lw=1.2)
        for k, x, y in zip(ks, xs, ys):
            ax.plot([x], [y], marker=mk, color=col, ls="none", mfc="none" if cells[k]["reused"] else col)
        ax.plot([], [], marker=mk, color=col, label=f"width {w}")
        report[f"best_val_w{w}"] = dict(zip(ks, [round(y, 4) for y in ys]))
    ax.plot([], [], marker="o", color="k", mfc="none", ls="none", label="reused run (H100)")
    ax.set_xscale("log")
    ax.set_xlabel("training compute $6NSB$ (FLOP)")
    ax.set_ylabel("best validation het. loss")
    ps.legend(ax, "upper right")
    ps.save(fl, os.path.join(FIGS, "scaling_grid_loss"))

    # 2: MSE against training GPU-hours (the main block's training time; reused cells on H100)
    lb = json.load(open(os.path.join(DATA, "long_sweeps_best.json")))
    hrs = {"w8_x4": lb["17056"]["best"]["traintime_hours"], "w8_x16": lb["68224"]["best"]["traintime_hours"]}
    f2, ax = ps.figure()
    for w, (col, mk) in WCOL.items():
        ks, xs, ys = series(w, lambda c: c.get("train_hours"))
        xs = [hrs.get(k, x) for k, x in zip(ks, xs)]
        ax.plot(xs, ys, color=col, lw=1.2)
        for k, x, y in zip(ks, xs, ys):
            ax.plot([x], [y], marker=mk, color=col, ls="none", mfc="none" if cells[k]["reused"] else col)
        ax.plot([], [], marker=mk, color=col, label=f"width {w}")
        report[f"hours_w{w}"] = dict(zip(ks, [round(x, 2) for x in xs]))
    ax.plot([], [], marker="o", color="k", mfc="none", ls="none", label="reused run (H100)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training time (GPU-hours)")
    ax.set_ylabel("validation MSE of $\\ln|M|^2$")
    ps.legend(ax, "lower left")
    ps.save(f2, os.path.join(FIGS, "scaling_grid_hours"))

    # 3: R over the width-8 2x cell on the pp test pools (PP4) and the held-out classes' default-sampled events
    # pooled (held_default: their PP4/EE4 probes, HD4, HD5), against compute
    f3, ax = ps.figure()
    for w, (col, mk) in WCOL.items():
        for ax_name, ls, mfc in (("PP4", "-", None), ("held_default", ":", "none")):
            ks = [f"w{w}_x{f}" for f in (2, 4, 8, 16)
                  if f"w{w}_x{f}" == "w8_x2" or ax_name in d["reads_R_over_w8_x2"][f"w{w}_x{f}"]]
            xs = [cells[k]["C"] for k in ks]
            ys = [1.0 if k == "w8_x2" else d["reads_R_over_w8_x2"][k][ax_name]["R"] for k in ks]
            ax.plot(xs, ys, color=col, ls=ls, marker=mk, mfc=mfc or col, lw=1.1)
            report[f"R_{ax_name}_w{w}"] = dict(zip(ks, ys))
        ax.plot([], [], color=col, marker=mk, ls="none", label=f"width {w}")
    ax.plot([], [], color="k", ls="-", label="pp test")
    ax.plot([], [], color="k", ls=":", marker="o", mfc="none", label="held-out classes")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training compute $6NSB$ (FLOP)")
    ax.set_ylabel("MSE / MSE(width 8, 2$\\times$)")
    ps.legend(ax, "upper right")
    ps.save(f3, os.path.join(FIGS, "scaling_grid_reads"))
    print(json.dumps(report, indent=1))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "pull":
        pull()
    elif cmd == "read":
        read()
    elif cmd == "fig":
        fig()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
