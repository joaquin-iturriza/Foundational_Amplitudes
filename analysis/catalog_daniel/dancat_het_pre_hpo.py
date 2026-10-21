"""Every trial of the catalog pretraining sweep (D59, sweep/sweep_config_dancat_het_pre.yaml) against each searched HP:
the best validation NLL (best checkpoint) against lr, L2 lambda, warmup fraction, cosine eta_min and EMA decay, one
point per trial, the x axis spanning exactly the searched range. Colour: the wave the trial was suggested in (3+3+2);
filled marker EMA on, open EMA off (the decay is searched either way, used only with EMA on); a cross for hp1466, which
blew up after step 15.4k (drawn at its best checkpoint). Input: sweeps_local/dancat_het_pre/driver_state.json.
    python analysis/catalog_daniel/dancat_het_pre_hpo.py   -> figs/dancat_het_pre_hpo_{a..e}
"""
import json, os, sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

BLOWUP = {1466}
AXES = (("training.lr", "learning rate"), ("training.regularization_lambda", r"L2 $\lambda$"),
        ("training.cosanneal_warmup_frac", "warmup fraction"), ("training.cosanneal_eta_min", r"cosine $\eta_{\min}$"),
        ("training.ema_decay", "EMA decay"))


def main():
    tr = json.load(open(os.path.join(REPO, "sweeps_local", "dancat_het_pre", "driver_state.json")))["trials"]
    tr = sorted(tr.values() if isinstance(tr, dict) else tr, key=lambda t: t["idx"])
    space = {s["name"]: s for s in yaml.safe_load(open(os.path.join(REPO, "sweep", "sweep_config_dancat_het_pre.yaml")))["search_space"]}
    bounds, wave = [3, 6, 8], {}
    for t in tr:
        wave[t["idx"]] = next(i for i, b in enumerate(bounds) if t["idx"] < b)
    cols = [ps.C.blue, ps.C.vermillion, ps.C.green]
    figs = ps.panels(len(AXES))
    for (fig, ax), (key, lab) in zip(figs, AXES):
        for t in tr:
            c, on = cols[wave[t["idx"]]], str(t["hp"]["ema"]).lower() == "true"
            if t["hp_idx"] in BLOWUP:
                ax.plot([t["hp"][key]], [t["val_loss"]], "x", color=c, clip_on=False)
            else:
                ax.plot([t["hp"][key]], [t["val_loss"]], "o", color=c, mfc=c if on else "none", clip_on=False)
        s = space[key]
        if s["type"].endswith("log"):
            ax.set_xscale("log")
        ax.set_xlim(s["low"], s["high"])           # exactly the searched range; markers on its edge are not clipped
        if key == "training.lr":
            ax.set_xticks([1e-3, 3e-3, 9e-3], [r"$10^{-3}$", r"$3\times10^{-3}$", r"$9\times10^{-3}$"])
            ax.set_xticks([], minor=True)
        ax.set_xlabel(lab); ax.set_ylabel("validation NLL")
    ax = figs[0][1]
    for i, c in enumerate(cols):
        ax.plot([], [], "o", color=c, label=f"wave {i + 1}")
    ax.plot([], [], "o", color="grey", mfc="none", label="EMA off")
    ax.plot([], [], "x", color="grey", label="blew up")
    ps.legend(ax, "upper left")
    ps.save_panels(figs, os.path.join(HERE, "figs", "dancat_het_pre_hpo"))


if __name__ == "__main__":
    main()
