"""Per-process census of catalog_v2 runs from their in-training validation lines.

    python analysis/catalog_v2/census.py runs/<exp>[/<run>] [runs/<exp2> ...] [--last]

For each run: the per-process validation loss at the best combined validation (default) or
the last one (--last), the failing count (> 0.05) and median over all processes and per
class: the 27 signed one-loop pools with more than 5% negative events on the train pool (32; the census counted 27), all 50 signed
one-loop pools, the s-channel 2->2 family, its M_Z-shifted twins, and the flavour twins.
Values are MSEs on the standardized target; a run made after the signed-log scale is on a
different target for the signed pools, so those compare by failing status only.
"""
import csv, glob, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FAIL = 0.05
NEEDLE = ["ee_mumu", "ee_tautau", "ee_uu", "ee_ddbar", "ee_bbbar", "ee_numu", "ee_nnbar"]
TWINS = ["uu_uu", "ud_ud", "us_us", "dd_dd_nlo", "ds_ds_nlo", "cs_cs_nlo", "uu_uu_nlo",
         "ud_ud_nlo", "us_us_nlo"]


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


def census(d, s27, all50):
    def cls(names):
        v = np.array([d[n] for n in names if n in d])
        return (int((v > FAIL).sum()), len(v), float(np.median(v))) if len(v) else (0, 0, float("nan"))
    allv = np.array(list(d.values()))
    mz = [n for n in d if "__mz" in n]
    out = {"all": (int((allv > FAIL).sum()), len(allv), float(np.median(allv))),
           "signed >5% neg": cls(s27), "signed all (50)": cls(all50),
           "s-channel family": cls(NEEDLE), "M_Z family": cls(mz), "flavour twins": cls(TWINS)}
    return out


def main(argv):
    last = "--last" in argv
    paths = [a for a in argv if not a.startswith("--")]
    s27, all50 = signed_classes()
    runs = []
    for p in paths:
        name, vals = read_run(p)
        if not vals:
            print(f"{name}: no validation lines"); continue
        idx = len(vals) - 1 if last else int(np.argmin([c for c, _ in vals]))
        runs.append((name, idx, len(vals), vals[idx][1]))
    keys = ["all", "signed >5% neg", "signed all (50)", "s-channel family", "M_Z family", "flavour twins"]
    print(f"{'class':22s}" + "".join(f"{r[0][-38:]:>40s}" for r in runs))
    print(f"{'(validation used)':22s}" + "".join(f"{f'#{r[1]+1} of {r[2]}':>40s}" for r in runs))
    for k in keys:
        cells = []
        for r in runs:
            f, n, med = census(r[3], s27, all50)[k]
            cells.append(f"{f}/{n} failing, median {med:.3g}")
        print(f"{k:22s}" + "".join(f"{c:>40s}" for c in cells))
    print("\nper process (twins, s-channel family):")
    for n in TWINS + NEEDLE:
        print(f"  {n:12s}" + "".join(f"{r[3].get(n, float('nan')):>14.3g}" for r in runs))


if __name__ == "__main__":
    main(sys.argv[1:])
