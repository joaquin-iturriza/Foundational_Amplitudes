"""Compare run(s) to the solo scalings — one panel per process.

For each process draws, on log-log axes:
  * the solo scaling POINTS (best val_loss at each compute level of the
    scaling_solo_full_* sweeps — what the solo exponent is fit from), as markers;
  * the solo fit line EXTENDED across the union of all plotted compute ranges;
  * optionally a run's per-process loss vs its own compute (thinned to log-spaced
    points), and/or reference points from a JSON (e.g. a finished run's final
    per-process loss), as a star per process.

Reads <run_dir>/plots_<idx>/per_process_metrics.json and the solo sweeps directly.

UNITS (important — the fairness fix).  The losses stored on both sides are the
*standardised* val MSE: the amplitude target is preprocessed as (log|A| - mu)/sigma,
so the reported loss is MSE_log / sigma**2.  A joint (pretrained) run standardises
log|A| GLOBALLY (one mu/sigma pooled over all processes), while each solo run
standardises on its single process.  Plotting the raw stored losses therefore
compares two different vertical scales (sigma_global vs sigma_solo,p) and is unfair.
By default we undo this: every loss is multiplied back by the sigma**2 that
standardised it (global sigma for run/ref points, per-dataset sigma for solo
points), recovering the physical MSE on log|A|, which IS comparable across
processes and setups.  This only shifts curves vertically; the scaling slope is
unchanged.  Pass --raw-units to plot the as-reported standardised losses instead.
(Same method as sweep/plot_per_dataset_scaling.py.)

Usage (Jean Zay, project env — needs the data/ .npy files for the sigma rescale):
    python plot_vs_solo.py --ref data/orig_run_nh4.json          # original run vs solo, now
    python plot_vs_solo.py runs/pretrain_full_nh4_fresh/trial_0266            # the fresh run vs solo
    python plot_vs_solo.py runs/.../trial_0266 --ref data/orig_run_nh4.json   # both, overlaid
    options: --reg (with-reg losses)   --npts N (run points per panel, default 15)
             --raw-units (no sigma rescale)   --std-nrows N (rows for sigma, default 1e5)
A reference JSON is {process_name: [compute, loss]} in joint (global-sigma) units.
"""
import json, os, sys, glob, re, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEPS = os.path.join(ROOT, "sweeps")
DATA_DIR = os.path.join(ROOT, "data")
SOLO_BATCH = 16384
STAR_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]  # --star runs (≠ orig-run crimson)
short = lambda s: s.replace("ee_", "").replace("-1000GeV_amplitudes", "")
tag = lambda n: n.replace("_amplitudes", "").replace("_", "")


def solo_points(name):
    pts = []
    for d in glob.glob(os.path.join(SWEEPS, f"scaling_solo_full_{tag(name)}_t*")):
        best = None
        for f in glob.glob(os.path.join(d, "results", "*.json")):
            vl = json.load(open(f))["val_loss"]
            if best is None or vl < best:
                best = vl
        m = re.search(r"_t(\d+)$", d)
        if best and best > 0 and m:
            pts.append((int(m.group(1)) * SOLO_BATCH, best))
    return sorted(pts)


def logspace_thin(c, l, n):
    c = np.asarray(c, float); l = np.asarray(l, float)
    m = (c > 0) & (l > 0); c, l = c[m], l[m]
    if len(c) <= n:
        return c, l
    targets = np.geomspace(c.min(), c.max(), n)
    idx = sorted({int(np.abs(np.log(c) - np.log(t)).argmin()) for t in targets})
    return c[idx], l[idx]


def fit(c, l):
    b, a = np.polyfit(np.log(c), np.log(l), 1)
    return a, b


def _log_amp(name, n_rows):
    """log|A| (signed-log if any amplitude is non-positive, matching
    resolve_amp_trafos) over the first n_rows rows of data/<name>.npy."""
    path = os.path.join(DATA_DIR, f"{name}.npy")
    a = np.load(path, mmap_mode="r")
    amp = np.asarray(a[:n_rows, -1], dtype=np.float64)
    return np.log(amp) if amp.min() > 0 else np.sign(amp) * np.log1p(np.abs(amp))


def solo_std(name, n_rows, cache):
    """sigma of log|A| for one process (the solo standardisation scale)."""
    if name not in cache:
        try:
            cache[name] = float(_log_amp(name, n_rows).std())
        except FileNotFoundError:
            cache[name] = None
    return cache[name]


def global_std(datasets, n_rows):
    """sigma of log|A| pooled over all datasets (the joint standardisation scale)."""
    pooled = []
    for n in datasets:
        try:
            pooled.append(_log_amp(n, n_rows))
        except FileNotFoundError:
            print(f"  WARNING: missing data/{n}.npy, skipped for sigma_global")
    if not pooled:
        return None
    return float(np.concatenate(pooled).std())


def _config_datasets():
    """data.dataset from config/amplitudes.yaml (the joint pool), or None."""
    try:
        import yaml
        cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "amplitudes.yaml")))
        return list(cfg["data"]["dataset"])
    except Exception:
        return None


def main():
    args = sys.argv[1:]
    use_reg = "--reg" in args
    raw_units = "--raw-units" in args        # default: corrected (physical) units
    npts = int(args[args.index("--npts") + 1]) if "--npts" in args else 15
    std_nrows = int(args[args.index("--std-nrows") + 1]) if "--std-nrows" in args else 100_000
    ref_path = args[args.index("--ref") + 1] if "--ref" in args else None
    # --star [LABEL=]RUNDIR (repeatable): plot a run's FINAL per-process loss as a
    # labelled star (same style as --ref's "orig run"), so multiple runs can be
    # compared at their endpoint against the solo scaling.
    star_specs = [args[i + 1] for i, a in enumerate(args) if a == "--star" and i + 1 < len(args)]
    flags = {"--reg", "--raw-units"}
    skip = set()
    for i, a in enumerate(args):
        if a == "--star":
            skip.add(i); skip.add(i + 1)
    for f in ("--npts", "--ref", "--std-nrows"):
        if f in args:
            skip.add(args.index(f)); skip.add(args.index(f) + 1)
    positional = [a for i, a in enumerate(args) if a not in flags and i not in skip]
    run_dir = positional[0] if positional else None

    def load_star(spec):
        """[LABEL=]run_dir -> (label, {proc: [final_compute, final_loss]}, dataset_order)."""
        label, sep, path = spec.partition("=")
        if not sep:
            path = label
            par = os.path.basename(os.path.dirname(os.path.abspath(path)))
            label = (par or os.path.basename(os.path.abspath(path))).replace("pretrain_full_", "")
        mfiles = glob.glob(os.path.join(path, "plots_*", "per_process_metrics.json"))
        if not mfiles:
            print(f"--star: no per_process_metrics.json under {path}/plots_*/"); sys.exit(1)
        d = json.load(open(sorted(mfiles)[0]))
        key = "proc_val_losses" if use_reg else "proc_val_losses_no_reg"
        Ls = d.get(key) or d["proc_val_losses"]; Cs = d["proc_compute"]
        pts = {n: [Cs[n][-1], Ls[n][-1]] for n in Ls
               if n in Cs and Ls.get(n) and Cs.get(n)}
        return label, pts, list(d.get("dataset_order", list(pts.keys())))

    stars = [load_star(s) for s in star_specs]   # [(label, {proc:[c,l]}, datasets)]

    L = C = None; order = None; outdir = ROOT; run_datasets = None
    if run_dir:
        mfiles = glob.glob(os.path.join(run_dir, "plots_*", "per_process_metrics.json"))
        if not mfiles:
            print(f"No per_process_metrics.json under {run_dir}/plots_*/"); sys.exit(1)
        d = json.load(open(sorted(mfiles)[0]))
        key = "proc_val_losses" if use_reg else "proc_val_losses_no_reg"
        L = d.get(key) or d["proc_val_losses"]; C = d["proc_compute"]
        order = [n for n in d.get("dataset_order", list(L.keys())) if n in L and n in C]
        run_datasets = list(d.get("dataset_order", order))
        outdir = os.path.dirname(sorted(mfiles)[0])
    ref = json.load(open(ref_path)) if ref_path else None
    if order is None:
        if ref:
            order = list(ref.keys())
        elif stars:
            order = list(stars[0][2])
        else:
            order = []
    if run_datasets is None and stars:
        run_datasets = list(stars[0][2])
    if not order:
        print("Nothing to plot: give a run_dir, --ref, and/or --star."); sys.exit(1)

    # --- sigma rescaling (the fairness fix) -------------------------------------
    # Joint/ref losses use sigma_global (pooled over the joint dataset list); solo
    # losses use each process's own sigma. Multiply each by its sigma**2 to recover
    # the physical MSE on log|A|. Disabled with --raw-units.
    sigma_global = None; solo_cache = {}
    if not raw_units:
        global_datasets = run_datasets or _config_datasets() or order
        print(f"Computing sigma_global over {len(global_datasets)} datasets "
              f"(first {std_nrows} rows each)...")
        sigma_global = global_std(global_datasets, std_nrows)
        if sigma_global is None:
            print("  could not read any data/*.npy — falling back to --raw-units.")
            raw_units = True
        else:
            print(f"sigma_global = {sigma_global:.4f}\n")
    g2 = sigma_global ** 2 if sigma_global is not None else 1.0

    def s2_solo(n):
        """Per-process solo sigma**2 (1.0 in raw mode or if data missing)."""
        if raw_units:
            return 1.0
        ss = solo_std(n, std_nrows, solo_cache)
        return ss ** 2 if ss else None

    ylab = "val loss (standardised)" if raw_units else "val MSE on log|A| (physical)"

    ncol = 2; nrow = math.ceil(len(order) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.4 * nrow), squeeze=False)
    if raw_units:
        print(f"{'ds':10s} {'compute':>9s} {'loss':>9s} {'soloL@C':>9s} {'/solo':>6s} {'soloα':>6s}")
    else:
        print(f"{'ds':10s} {'σ_solo':>7s} {'σ_glob':>7s} {'factor':>7s} "
              f"{'rawx':>6s} {'corrx':>6s} {'soloα':>6s}")
    for k, n in enumerate(order):
        ax = axes[k // ncol][k % ncol]
        s2 = s2_solo(n)        # solo rescale; None => data missing, leave solo raw
        sp = solo_points(n)
        computes = []
        a = b = None
        if len(sp) >= 2 and s2 is not None:
            sc = np.array([p[0] for p in sp], float)
            sl = np.array([p[1] for p in sp], float) * s2
            a, b = fit(sc, sl)
            ax.plot(sc, sl, "s", color="0.25", ms=6, label="solo points")
            computes += [sc.min(), sc.max()]
        rc = rl = None
        if L is not None and n in L:
            rc, rl = logspace_thin(C[n], L[n], npts)
            rl = np.asarray(rl, float) * g2
            if len(rc):
                ax.plot(rc, rl, "o-", color="C0", ms=4, lw=1.1, label="this run")
                computes += [rc.min(), rc.max()]
        rp = ref.get(n) if ref else None
        if rp:
            ax.plot([rp[0]], [rp[1] * g2], "*", color="crimson", ms=15, label="orig run", zorder=5)
            computes += [rp[0]]
        star_pt = None
        for si, (slabel, spts, _) in enumerate(stars):
            spt = spts.get(n)
            if spt:
                col = STAR_COLORS[si % len(STAR_COLORS)]
                ax.plot([spt[0]], [spt[1] * g2], "*", color=col, ms=15, label=slabel, zorder=6)
                computes += [spt[0]]
                if star_pt is None:
                    star_pt = (spt[0], spt[1] * g2)
        if a is not None and computes:
            cg = np.geomspace(min(computes), max(computes), 80)
            ax.plot(cg, np.exp(a + b * np.log(cg)), ":", color="0.25", lw=1.6,
                    label=f"solo fit (α={-b:.2f})")
            # table row (prefer ref point, else run endpoint)
            pt = ((rp[0], rp[1] * g2) if rp else
                  (rc[-1], rl[-1]) if rc is not None and len(rc) else
                  star_pt)
            if pt:
                ls = math.exp(a + b * math.log(pt[0]))
                if raw_units:
                    print(f"{short(n):10s} {pt[0]:9.2e} {pt[1]:9.2e} {ls:9.2e} "
                          f"{pt[1]/ls:5.1f}x {-b:6.2f}")
                else:
                    corr = pt[1] / ls
                    raw = corr * (s2 / g2)   # undo rescale → original standardised ratio
                    print(f"{short(n):10s} {math.sqrt(s2):7.3f} {sigma_global:7.3f} "
                          f"{g2/s2:7.2f} {raw:5.1f}x {corr:5.1f}x {-b:6.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(short(n), fontsize=12)
        ax.grid(True, which="both", lw=0.4, alpha=0.4)
        ax.legend(fontsize=8, frameon=False)
        if k // ncol == nrow - 1: ax.set_xlabel("compute (samples seen)")
        if k % ncol == 0: ax.set_ylabel(ylab + ("" if use_reg else " (no reg)"))
    for k in range(len(order), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    title = ("Per-process vs extended solo scaling"
             + ("  [raw standardised units]" if raw_units
                else "  [σ-corrected: physical log|A| MSE]"))
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    out = os.path.join(outdir, "loss_vs_solo.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    if not raw_units:
        print(f"\nfactor = (σ_global/σ_solo)²   (raw ratio × factor = corrected ratio)")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
