"""Per-process census of catalog_v2 runs from their in-training validation lines.

    python analysis/catalog_v2/census.py runs/<exp>[/<run>] [runs/<exp2> ...] [--last] [--layers]

For each run: the per-process validation loss at the best combined validation (default) or
the last one (--last), and per group the median and 90th percentile of that loss with the group
size. Groups are the classes the figures use (tree 2->2, resonant 2->2, tree 2->3, tree 2->4,
positive and signed one-loop) and the families the notes discuss (the s-channel 2->2 family, its
M_Z-shifted copies, the flavour twins); --layers groups by catalog layer instead (base trees by
multiplicity, one-loop, loop-induced, the three scan families). No pass/fail threshold: a loss is
read against its group, not against a line. Values are MSEs on the standardized target; a run
made after the signed-log scale is on a different target for the signed pools, so those compare
across that change only in rank.
"""
import csv, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NEEDLE = ["ee_mumu", "ee_tautau", "ee_uu", "ee_ddbar", "ee_bbbar", "ee_numu", "ee_nnbar"]
TWINS = ["uu_uu", "ud_ud", "us_us", "dd_dd_nlo", "ds_ds_nlo", "cs_cs_nlo", "uu_uu_nlo",
         "ud_ud_nlo", "us_us_nlo"]

#: Figure labels for the per-class keys the plotting scripts use (mathtext arrows, no "->").
CLASS_LABEL = {"tree 2->2": r"tree $2\to2$", "resonant 2->2": r"resonant $2\to2$",
               "tree 2->3": r"tree $2\to3$", "tree 2->4": r"tree $2\to4$",
               "positive 1-loop": "positive one-loop", "signed 1-loop": "signed one-loop"}
CLASSES = ["tree 2->2", "resonant 2->2", "tree 2->3", "tree 2->4", "positive 1-loop", "signed 1-loop"]


def signed_classes():
    rows = list(csv.DictReader(open(os.path.join(HERE, "signed_pools.csv"))))
    name = lambda r: re.sub(r"_\d+-\d+GeV_train$", "", r["name"])
    all50 = {name(r) for r in rows}
    s27 = {name(r) for r in rows if float(r["neg_frac"]) > 0.05}
    return s27, all50


def read_run(path):
    logs = sorted(glob.glob(os.path.join(path, "**", "out_0.log"), recursive=True))
    if not logs:
        raise SystemExit(f"no out_0.log under {path}")
    log = logs[-1]
    vals = []   # (combined, {name: loss})
    for line in open(log, errors="replace"):
        m = re.search(r"Val loss \(combined\): ([0-9.eE+-]+) \| (.*)$", line)
        if not m:
            continue
        d = {}
        for tok in m.group(2).split(", "):
            k, _, v = tok.partition("=")
            try:
                d[k.strip()] = float(v)
            except ValueError:
                pass
        vals.append((float(m.group(1)), d))
    return os.path.relpath(os.path.dirname(log), ROOT), vals


def groups(d, all50, layers=False):
    """Ordered {group: [process names]} for a run's per-process losses `d`."""
    NP = json.load(open(os.path.join(HERE, "n_particles.json")))
    def cls(n):
        if n in all50: return "signed 1-loop"
        if n.endswith("_nlo") or n.endswith("_loop"): return "positive 1-loop"
        if n in NEEDLE or "__mz" in n: return "resonant 2->2"
        return f"tree 2->{NP[n] - 2}"
    names = [n for n in d if n in NP]
    out = {"all": names}
    if layers:
        base = [n for n in names if "__" not in n]
        for k in (4, 5, 6):
            out[f"base trees 2->{k - 2}"] = [n for n in base if NP[n] == k and not n.endswith(("_nlo", "_loop"))]
        out["one-loop QCD"] = [n for n in base if n.endswith("_nlo")]
        out["loop-induced"] = [n for n in base if n.endswith("_loop")]
        for tag, lab in (("__as", "alpha_s scan"), ("__mt", "m_t scan"), ("__mz", "M_Z scan")):
            out[lab] = [n for n in names if tag in n]
        return out
    for c in CLASSES:
        out[c] = [n for n in names if cls(n) == c]
    out["s-channel family"] = [n for n in NEEDLE if n in d]
    out["M_Z family"] = [n for n in names if "__mz" in n]
    out["flavour twins"] = [n for n in TWINS if n in d]
    return out


def stats(d, names):
    v = np.array([d[n] for n in names])
    return (len(v), float(np.median(v)), float(np.percentile(v, 90))) if len(v) else (0, np.nan, np.nan)


def main(argv):
    last = "--last" in argv; layers = "--layers" in argv
    paths = [a for a in argv if not a.startswith("--")]
    _, all50 = signed_classes()
    runs = []
    for p in paths:
        name, vals = read_run(p)
        if not vals:
            print(f"{name}: no validation lines"); continue
        idx = len(vals) - 1 if last else int(np.argmin([c for c, _ in vals]))
        runs.append((name, idx, len(vals), vals[idx][0], vals[idx][1]))
    if not runs:
        return
    keys = list(groups(runs[0][4], all50, layers))
    print(f"{'group':20s}" + "".join(f"{r[0][-36:]:>38s}" for r in runs))
    print(f"{'(validation used)':20s}" + "".join(f"{f'#{r[1]+1} of {r[2]}, combined {r[3]:.3g}':>38s}" for r in runs))
    for k in keys:
        cells = []
        for r in runs:
            n, med, p90 = stats(r[4], groups(r[4], all50, layers)[k])
            cells.append(f"{med:.2g} / {p90:.2g} ({n})")
        print(f"{k:20s}" + "".join(f"{c:>38s}" for c in cells))
    print("\n(median / 90th percentile of the per-process validation loss (group size))")


if __name__ == "__main__":
    main(sys.argv[1:])
