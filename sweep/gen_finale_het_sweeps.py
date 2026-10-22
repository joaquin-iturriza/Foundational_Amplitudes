"""Stage 1 of the het fine-tune benchmark (the user, 2026-10-22: "sweep HPs on a few processes cells training on het loss
for the finale model, check if the optimal HPs are relatively close, then based on that either sweep the rest, and the
big catalog, or just run with the best HPs the rest and the big catalog"). Each sweep is the finale's own MSE fine-tune
cell (tp3_finfte_<probe>_d<k>: same search space, 5 trials, horizon (2816 steps at D = 10^2, 8000 at 10^4), data and
parent tp3_finale/trial_0073) with the
loss switched to the het loss as gen_het23_sweeps.py switches it, and the checkpoint selected on the validation mu MSE
(D85, fig:het_mumse: lower mu error than the NLL selection in 23 of 24 cells), so DyHPO observes the mu MSE; every
trial ends with the sigma-only stage (D70). Cells: four probes that have no twin in the big catalog (the stage-2
comparison, docs/results.tex), one per kind (2->2 massless, 2->2 massive with a t-channel, one loop, 2->4), at
D = 10^2 and 10^4 (k = 4, 8). Writes sweep/sweep_config_<cell>_het.yaml.
    python sweep/gen_finale_het_sweeps.py
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES = ("ee_ddbar", "ee_WW", "ee_dd_nlo", "uubar_Zggg")
KS = (4, 8)
HET = {"training.loss": "HETEROSC", "training.heterosc_beta": 0.0, "training.heterosc_select": "mu_mse",
       "model.net.sigma_after_pool": "true", "model.net.detach_sigma_backbone": "false",
       "training.sign_head": "false", "training.track_train_eval": "true", "evaluation.train_subsample": 2000}


def main():
    for p in PROBES:
        for k in KS:
            s = f"tp3_finfte_{p}_d{k}"
            c = yaml.safe_load(open(os.path.join(HERE, f"sweep_config_{s}.yaml")))
            c["sweep_name"] = s + "_het"
            c["fixed_params"].update(HET)
            n = int(c["n_trials"])
            c.setdefault("dyhpo", {}).update({"waves": [3, 3, 2] if n == 8 else [3, n - 3]})
            out = os.path.join(HERE, f"sweep_config_{s}_het.yaml")
            yaml.safe_dump(c, open(out, "w"), sort_keys=False)
            print(out, n, "trials")


if __name__ == "__main__":
    main()
