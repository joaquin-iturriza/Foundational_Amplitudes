"""Plan finetune_mechanism step 1, arm (2c) on q q~ -> Z+4g (D515): Sg/F8, the sigma-head reset search's best over the
template fine-tune search's first 8 trials' best (same 3,3,2 waves and DyHPO seed), per batch; the design's bar 1.41.

Best val MSE(log|M|^2) = a trial's best val mu-MSE (standardised; the driven trial's val_loss) x std^2, std the D507
scratch cell's target std (analysis/zng_uq/data/d507_read.json): every arm standardises the same 1e4 events with its own
stats (fine_tune.target_stats own), so the std is shared and Sg/F8 does not depend on it. Diverged trials (no val)
are listed apart. Figure: each trial's best val MSE per arm and batch beside the D507 scratch seeds 42-44.

    python3 analysis/finetune_mechanism/zng4g_sigreset_read.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

BAR = 1.41


def trials(name):
    fs = glob.glob(os.path.join(ROOT, "sweeps_local", name, "results", "*.json"))
    return sorted((json.load(open(f)) for f in fs), key=lambda t: t["trial_idx"])


def main():
    d507 = json.load(open(os.path.join(ROOT, "analysis", "zng_uq", "data", "d507_read.json")))
    out = {}
    for b in ("2048", "128"):
        std = d507[f"muon_b{b}"]["std"]
        ft, sg = trials(f"ftmech_zng4g_ft_b{b}"), trials(f"ftmech_zng4g_sigreset_b{b}")
        f8 = [t for t in ft if t["trial_idx"] < 8]
        assert len(f8) == 8 and len(sg) == 8, (b, len(f8), len(sg))
        val = lambda ts: [t["val_loss"] * std ** 2 for t in ts if t["val_loss"] is not None]   # noqa: E731
        fv, sv = val(f8), val(sg)
        out[b] = {"std": std, "F8": fv, "Sg": sv, "F_all": val(ft), "n_ft_done": len(ft),
                  "diverged": {"F8": [t["hp_idx"] for t in f8 if t["val_loss"] is None],
                               "Sg": [t["hp_idx"] for t in sg if t["val_loss"] is None]},
                  "S_seeds": list(d507[f"muon_b{b}"]["seeds"].values()),
                  "Sg_over_F8": min(sv) / min(fv)}
        r = out[b]["Sg_over_F8"]
        out[b]["verdict"] = "within" if 1 / BAR <= r <= BAR else ("Sg higher" if r > BAR else "Sg lower")
        print(f"b{b}: F8 best {min(fv):.4g}, Sg best {min(sv):.4g}, Sg/F8 {r:.3f} -> {out[b]['verdict']} (bar {BAR}); "
              f"ft so far ({len(ft)} trials) best {min(out[b]['F_all']):.4g}; scratch seeds {out[b]['S_seeds']}")
    json.dump(out, open(os.path.join(HERE, "data", "zng4g_sigreset_read.json"), "w"), indent=1)
    plot(out)


def plot(out):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    arms = [("S_seeds", "from scratch, seeds 42-44", ps.C.blue, "o"),
            ("F8", "fine-tuned, first 8 trials", ps.C.vermillion, "s"),
            ("Sg", "fine-tuned, sigma head reset", ps.C.green, "^")]
    for ax, b in zip(axs, ("2048", "128")):
        for i, (k, lab, col, m) in enumerate(arms):
            v = out[b][k]
            ax.plot(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, m, color=col, label=lab)
        ax.set_yscale("log")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["scratch", "fine-tuned", "reset"])
        ax.set_xlim(-0.6, 2.6)
        ax.set_xlabel(f"arm, batch {b}")
        ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(axs[0], r"$q\bar q\to Z+4g$")
    ps.legend(axs[1], "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "zng4g_sigreset"))


if __name__ == "__main__":
    main()
