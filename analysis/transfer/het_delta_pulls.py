"""Pulls of the het retrains of the Delta cells (D47): (mu - y)/sigma on the standardized test target, mu and sigma of
each cell's sigma stage (D70). Inputs analysis/transfer/het_delta_data/<site>.npz (collect_het_delta.py), the cells of
delta_audit.json's paper_8k grid.
  pull_het_ladder_<probe>   the pull density of scratch and rungs 0-9 on one probe, N(0,1) dashed
  pull_het_summary          per cell, the pull std (left) and the fraction with |pull| < 1 (right), against the probe
Printed: per cell the pull std, |pull|<1, and the cells missing.
    python analysis/transfer/het_delta_pulls.py
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402
from delta_plots import LAB, OUT, tag  # noqa: E402

BINS = np.linspace(-5, 5, 41)


def main():
    P = {}
    for f in sorted(glob.glob(os.path.join(HERE, "het_delta_data", "*.npz"))):
        with np.load(f) as z:
            P.update({k[: -len("__pull")]: z[k] for k in z.files if k.endswith("__pull")})
    rows = json.load(open(os.path.join(HERE, "delta_audit.json")))["rows"]
    cells, missing = {}, []
    for r in rows:
        p = r.get("paper_8k") or {}
        t = tag(p.get("run_dir", "?"))
        (cells.setdefault(r["probe"], {}).__setitem__(r["arm"], P[t]) if t in P else missing.append(f"{r['probe']} {r['arm']} {t}"))
    os.makedirs(OUT, exist_ok=True)
    seq = dict(zip([f"r{i}" for i in range(10)], ps.sequence(10)))
    g = np.linspace(-5, 5, 201)
    for probe, arms in cells.items():
        ladder = {a: v for a, v in arms.items() if a == "scratch" or a in seq}
        if not ladder:
            continue
        fig, ax = ps.figure()
        for a in sorted(ladder, key=lambda a: (a != "scratch", a)):
            h, e = np.histogram(ladder[a], bins=BINS, density=True)
            ax.stairs(h, e, color="k" if a == "scratch" else seq[a], label="from scratch" if a == "scratch" else f"rung {a[1:]}")
        ax.plot(g, np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), "--", color="grey", label=r"$\mathcal N(0,1)$")
        ax.set_xlabel(r"pull $(\mu-y)/\sigma$"); ax.set_ylabel("density")
        ps.process_label(ax, LAB.get(probe, probe))
        ps.legend(ax, "upper left", ncol=2)
        ps.save(fig, os.path.join(OUT, f"pull_het_ladder_{probe}"))
    fig, (a1, a2) = ps.figure(ncols=2)
    probes = sorted(cells)
    for i, probe in enumerate(probes):
        v = list(cells[probe].values())
        x = np.full(len(v), i) + np.linspace(-0.25, 0.25, len(v))
        a1.plot(x, [p.std() for p in v], "o", color=ps.C.blue, label="cell" if i == 0 else None)
        a2.plot(x, [np.mean(np.abs(p) < 1) for p in v], "o", color=ps.C.blue)
        for p_, a in zip(v, cells[probe]):
            print(f"{probe:18s} {a:28s} pull std {p_.std():.2f}  |pull|<1 {np.mean(np.abs(p_) < 1):.2f}")
    a1.axhline(1, linestyle="--", color="grey", label="Gaussian")
    a2.axhline(0.683, linestyle="--", color="grey")
    for ax in (a1, a2):
        ax.set_xticks(range(len(probes))); ax.set_xticklabels([str(i + 1) for i in range(len(probes))])
        ax.set_xlabel("probe")
    a1.set_ylabel("pull std"); a2.set_ylabel(r"fraction $|{\rm pull}|<1$")
    ps.legend(a1, "upper left")
    ps.save(fig, os.path.join(OUT, "pull_het_summary"))
    print("probes:", ", ".join(f"{i + 1} {p}" for i, p in enumerate(probes)))
    print(f"{len(missing)} cells missing")
    for m in missing:
        print("  missing", m)


if __name__ == "__main__":
    main()
