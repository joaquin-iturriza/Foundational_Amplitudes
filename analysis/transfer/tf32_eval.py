"""Evaluation precision on Jean Zay's A100 (decision D42, the Delta plots' audit): every one of the 107 runs drawn from
Jean Zay was rebuilt twice by tools/eval_delta.py, TF32 matmuls on (FA-jeanzay-e99a690d) and off (FA-jeanzay-5c91508e),
and each rebuilt validation loss is divided by the run's recorded one (analysis/transfer/delta_audit.json, mse_logM2;
rebuilt val_loss times prepd_std^2). Input analysis/transfer/tf32_jz.json; figure analysis/transfer/figs/tf32_eval.
    python analysis/transfer/tf32_eval.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402


def main():
    d = json.load(open(os.path.join(HERE, "tf32_jz.json")))
    on = np.array([v["on_over_rec"] for v in d.values()])
    off = np.array([v["off_over_rec"] for v in d.values()])
    fig, ax = ps.figure()
    ax.scatter(off, on, color=ps.C.vermillion, label=f"{len(on)} runs")
    lim = [min(on.min(), off.min()) / 1.5, max(on.max(), off.max()) * 1.5]
    ax.plot(lim, [1, 1], "--", color="k", label="recorded")
    ax.plot([1, 1], lim, "--", color="k")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("TF32 off / recorded"); ax.set_ylabel("TF32 on / recorded")
    ps.legend(ax, "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "tf32_eval"))
    m_on, m_off = np.abs(np.log10(on)) < 0.02, np.abs(np.log10(off)) < 0.02
    print(f"both match {np.sum(m_on & m_off)}, on only {np.sum(m_on & ~m_off)}, off only {np.sum(~m_on & m_off)}, "
          f"neither {np.sum(~m_on & ~m_off)}; worst on {on.max():.3g}, worst off {off.min():.3g}..{off.max():.3g}")


if __name__ == "__main__":
    main()
