"""Figure of plan finetune_mechanism step 4 (parent_rerun_check.json): per process, the W parent's final-model val MSE of
the standardised log|M|^2 for the original (CC, V100) and its rerun (lxplus, H100, TF32 on), ordered by the original's,
and their ratio rerun / original beside it.

    python3 analysis/finetune_mechanism/plot_parent_rerun.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import numpy as np  # noqa: E402
import plot_style as ps  # noqa: E402


def main():
    r = json.load(open(os.path.join(HERE, "data", "parent_rerun_check.json")))
    procs = r["summary"]["procs"]          # ordered by the original's MSE
    a = np.array([r["cc"]["mse"][n] for n in procs])
    b = np.array([r["snap"]["mse"][n] for n in procs])
    x = np.arange(1, len(procs) + 1)
    fig, (l, rt) = ps.figure(ncols=2)
    l.plot(x, a, "o", color=ps.C.blue, label="original, V100")
    l.plot(x, b, "s", color=ps.C.vermillion, label="rerun, H100,\nreduced-precision matmuls")
    l.set_yscale("log")
    l.set_xlabel("process, by the original's MSE")
    l.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)/\sigma^2$")
    ps.legend(l, "upper left")
    rt.plot(x, b / a, "s", color=ps.C.vermillion, label="rerun / original")
    rt.axhline(1.0, color="black", ls=":", label="equal")
    rt.set_yscale("log")
    rt.set_xlabel("process, by the original's MSE")
    rt.set_ylabel("MSE ratio, rerun / original")
    ps.legend(rt, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "parent_rerun_check"))


if __name__ == "__main__":
    main()
