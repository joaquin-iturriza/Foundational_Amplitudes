"""Jonas's causal mask (decision D55, the draft's Sec. 5): real zero-shot of three rung-9 pretrainings (hp73, 64k steps)
on probes past what they trained on: u dbar -> W+ 4g (rung 9 stops at W+3g) and u ubar -> Z + n g, n = 1..5 (no Z+ng in
rung 9). tools/zero_shot_parent.py scores each parent's own prediction with no data of the probe (the weights it
validated, the EMA shadow where it had EMA on).
  tp3_jcaus_r9     causal mask, prefix frames, p.P order, affine-in-N target statistics, coupling powers dropped
  tp3_jcausbnn_r9  the same with the Bayesian last layer (posterior-mean weights)
  tp3_gstd_r9      one shared standardization, no mask (the earlier real zero-shot parent)
Left: validation MSE of log|M|^2 divided by the probe's variance (1 = the probe's exact mean, which no parent is given).
Right: the same after the best affine map a + b h fitted on the probe (what is left once the scale and offset are free).
Input: analysis/transfer/zero_shot_jcaus.txt (ZERO_SHOT lines of scripts/job_zero_shot_parent.sh).
    python analysis/transfer/zero_shot_jcaus.py    -> figs/zero_shot_jcaus
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402

ARMS = (("tp3_jcaus_r9", "causal", ps.C.vermillion), ("tp3_jcausbnn_r9", "causal + BNN", ps.C.blue),
        ("tp3_gstd_r9", "shared std., no mask", "k"))


def main():
    rows = []
    for line in open(os.path.join(HERE, "zero_shot_jcaus.txt")):
        if line.startswith("ZERO_SHOT {"):
            d = json.loads(line.split(" ", 1)[1])
            if "error" in d:          # listed, not dropped (zero_shot_parent.py writes them for exactly this)
                print(f"ERROR {d['parent']} {d['recipe']}: {d['error']}")
            else:
                rows.append(d)
    fig, (a1, a2) = ps.figure(ncols=2)
    for key, lab, c in ARMS:
        rs = [r for r in rows if f"/{key}/" in r["parent"]]
        z = sorted((r for r in rs if r["probe"].startswith("uubar_Z")), key=lambda r: r["n_particles"])
        w = [r for r in rs if r["probe"] == "udbar_Wgggg"]
        if not z:
            continue
        n = [r["n_particles"] - 3 for r in z]                       # gluons: Z+ng has 3 + n particles
        a1.plot(n, [r["mse_logm2_real"] / r["var_logm2"] for r in z], "o-", color=c, label=lab)
        a2.plot(n, [r["val_loss_affine"] for r in z], "o-", color=c)
        for r in w:
            a1.plot([4], [r["mse_logm2_real"] / r["var_logm2"]], "D", color=c, mfc="none")
            a2.plot([4], [r["val_loss_affine"]], "D", color=c, mfc="none")
        for r in z + w:
            print(f"{key:16s} {r['probe']:13s} ema={r.get('ema_applied')}  MSE/var {r['mse_logm2_real'] / r['var_logm2']:.3f}"
                  f"  affine {r['val_loss_affine']:.4f}  corr {r['corr']:.3f}")
    a1.plot([], [], "D", color="grey", mfc="none", label=r"$W{+}4g$")
    for ax in (a1, a2):
        ax.set_yscale("log"); ax.set_xlabel(r"gluons $n$ in $u\bar u\to Z+ng$")
    a1.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)/\mathrm{Var}$")
    a2.set_ylabel(r"MSE$/\mathrm{Var}$ after affine map")
    ps.legend(a1, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "zero_shot_jcaus"))


if __name__ == "__main__":
    main()
