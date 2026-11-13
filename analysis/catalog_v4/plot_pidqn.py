"""Plan optimizations_recap step 1 (pidqn): per-class MSE of ln|M|^2 of the one-hot PDG arm against the quantum-number
arm (1x width-8 catalog_v4 cell, seed 42 each; reads extracted by mixture_ab.py extract), one point per class (the
geometric mean over its probes), on PP4, EE4, the catalog-sampled held-out classes (H3 apart) and the default-sampled
held-out classes (HD4, HD5) -> figs/pidqn_classes.{png,pdf}"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import plot_style as ps
import mixture_ab as m


def per_class(a, b, keys, cls):
    by = {}
    for k in keys:
        by.setdefault(cls(m.probe_name(k)), []).append((a[k], b[k]))
    return {c: (m.gmean([x for x, _ in v]), m.gmean([y for _, y in v])) for c, v in by.items()}


def main():
    cls = m.class_of()
    pid, qn = m.PIDQN["pid"], m.PIDQN["qn"]
    panels = []
    for name, reads in ((r"$pp$ test", ("PP4",)), (r"$e^+e^-$ test", ("EE4",)), ("held out", ("held",)),
                        ("held out, MadEvent", ("HD4", "HD5"))):
        a, b = {}, {}
        for r in reads:
            ra, rb = m.read_of(pid, r), m.read_of(qn, r)
            a.update({(r, k): v for k, v in ra.items()}); b.update({(r, k): v for k, v in rb.items()})
        keys = sorted(set(a) & set(b), key=str)
        h3 = [k for k in keys if isinstance(k[1], tuple) and k[1][0] == "H3"]
        rest = [k for k in keys if k not in h3]
        un = lambda k: k[1] if isinstance(k[1], tuple) else k
        cl = lambda p: cls(p)
        mk = lambda ks: per_class({un(k): a[k] for k in ks}, {un(k): b[k] for k in ks}, [un(k) for k in ks], cl)
        panels.append((name, mk(rest), mk(h3) if h3 else None))
    fig, axes = ps.figure(ncols=2, nrows=2)
    for ax, (name, d, d3) in zip(axes.flat, panels):
        x, y = np.array([v[1] for v in d.values()]), np.array([v[0] for v in d.values()])
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        if d3:
            x3, y3 = np.array([v[1] for v in d3.values()]), np.array([v[0] for v in d3.values()])
            lo, hi = min(lo, x3.min(), y3.min()), max(hi, x3.max(), y3.max())
        ax.plot([lo, hi], [lo, hi], color=ps.C.grey, ls="--", label=r"$y=x$")
        ax.scatter(x, y, color=ps.C.blue, alpha=0.5, label="class" if d3 is None else "other class")
        if d3:
            ax.scatter(x3, y3, color=ps.C.vermillion, alpha=0.6, label="physical top")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"MSE of $\ln|\mathcal{M}|^2$, quantum numbers")
        ax.set_ylabel(r"MSE, one-hot token")
        ps.process_label(ax, name)
        if d3:
            ps.legend(ax, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "pidqn_classes"))


if __name__ == "__main__":
    main()
