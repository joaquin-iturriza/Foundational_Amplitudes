"""Where the real zero-shot error of the rung-9 pretrainings comes from (Fig. zero_shot_jcaus's arms): the MSE of
log|M|^2 split into the level and the spread of the prediction.
Left: the squared mean offset over the probe's variance, bias^2/Var (the part a shift of the overall level fixes).
Right: the spread ratio, std of the prediction over std of the truth (1 = the right dynamic range; below 1 the
prediction is compressed towards its mean).
Input: analysis/transfer/zero_shot_jcaus.txt (ZERO_SHOT lines of scripts/job_zero_shot_parent.sh: bias, spread_ratio).
    python analysis/transfer/zero_shot_parts.py    -> figs/zero_shot_parts
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402
from zero_shot_jcaus import ARMS  # noqa: E402


def main():
    rows = [json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(HERE, "zero_shot_jcaus.txt"))
            if l.startswith("ZERO_SHOT {")]
    rows = [r for r in rows if "error" not in r]
    fig, (a1, a2) = ps.figure(ncols=2)
    for key, lab, c in ARMS:
        rs = [r for r in rows if f"/{key}/" in r["parent"]]
        z = sorted((r for r in rs if r["probe"].startswith("uubar_Z")), key=lambda r: r["n_particles"])
        w = [r for r in rs if r["probe"] == "udbar_Wgggg"]
        if not z:
            continue
        n = [r["n_particles"] - 3 for r in z]
        a1.plot(n, [r["bias"] ** 2 / r["var_logm2"] for r in z], "o-", color=c, label=lab)
        a2.plot(n, [r["spread_ratio"] for r in z], "o-", color=c)
        for r in w:
            a1.plot([4], [r["bias"] ** 2 / r["var_logm2"]], "D", color=c, mfc="none")
            a2.plot([4], [r["spread_ratio"]], "D", color=c, mfc="none")
    a1.plot([], [], "D", color="grey", mfc="none", label=r"$W{+}4g$")
    a2.axhline(1.0, color="grey", linestyle="--")
    a1.plot([], [], "--", color="grey", label="true spread (right)")
    for ax in (a1, a2):
        ax.set_xlabel(r"gluons $n$ in $u\bar u\to Z+ng$")
    a1.set_yscale("log")
    a1.set_ylabel(r"$\mathrm{bias}^2/\mathrm{Var}$")
    a2.set_ylabel(r"$\mathrm{std}(\hat y)/\mathrm{std}(y)$")
    ps.shared_legend(fig, a1, ncol=3)   # both panels draw the same series
    ps.save(fig, os.path.join(HERE, "figs", "zero_shot_parts"))


if __name__ == "__main__":
    main()
