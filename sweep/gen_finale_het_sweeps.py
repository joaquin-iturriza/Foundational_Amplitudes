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

--stage 2 (D94, the user 2026-10-23: "Yes, launch stage 2"): stage 1's optima are not close enough to share one point
(docs/results.tex fig:het_finale_s1), so the rest is swept, with the learning-rate window widened downwards to
[7e-4, 6e-3] (three of stage 1's optima at its lower edge, 1e-3): (a) the finale's other three no-twin probes
(e+e- -> nu nubar, t tbar, b bbar one-loop) at D = 10^2, 10^4; (b) the big-catalog run hp58 (runs/dancat_het_pre/trial_0058,
het, the arXiv:2606.23791 diagram encoder at width 64) fine-tuned on all seven no-twin probes at D = 10^1, 10^2, 10^3, 10^4
(k = 2, 4, 6, 8): the finale cells with the parent and its diagram encoder swapped in, everything else (data, horizon,
search space, trials) the finale cell's.
    python sweep/gen_finale_het_sweeps.py --stage 2
"""
import argparse, os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES = ("ee_ddbar", "ee_WW", "ee_dd_nlo", "uubar_Zggg")
KS = (4, 8)
HET = {"training.loss": "HETEROSC", "training.heterosc_beta": 0.0, "training.heterosc_select": "mu_mse",
       "model.net.sigma_after_pool": "true", "model.net.detach_sigma_backbone": "false",
       "training.sign_head": "false", "training.track_train_eval": "true", "evaluation.train_subsample": 2000}


PROBES2 = ("ee_nnbar", "ee_ttbar", "ee_bb_nlo")
LR2 = (7e-4, 6e-3)
# hp58's own best checkpoint (models/model_run0_best.pt.gz) was deleted on 2026-10-22 by sweep/prune.py after a step-time
# profile run overwrote its sweep result; its sigma stage survives: trunk and mu are hp58's EMA weights, frozen, with the
# calibrated sigma head (data_stats.json byte-identical). The fine-tunes start from it; their sigma head retrains anyway.
HP58 = "${PROJECT_DIR}/runs/dancat_het_pre/trial_0058/sigma_stage/models/model_run0_best.pt"
# hp58's diagram encoder (its config.yaml): the only architectural difference from the finale parent
HP58_MODEL = {"model.use_diagrams": "true", "model.diagram_encoder_impl": "llm4lhc",
              "model.diagram_encoder_llm4lhc.hidden_size": 64, "model.diagram_encoder_llm4lhc.num_layers": 3,
              "model.diagram_encoder_llm4lhc.num_heads": 4, "model.diagram_encoder_llm4lhc.laplacian_pe_dim": 8,
              "model.diagram_encoder_llm4lhc.dropout": 0.0, "model.diagram_encoder_llm4lhc.mode": "pooled",
              "model.diagram_encoder_llm4lhc.edge_feature_mode": "bias_kv"}


def het_cfg(src, name, lr=None, extra=None):
    c = yaml.safe_load(open(os.path.join(HERE, f"sweep_config_{src}.yaml")))
    c["sweep_name"] = name
    c["fixed_params"].update(HET)
    c["fixed_params"].update(extra or {})
    n = int(c["n_trials"])
    c.setdefault("dyhpo", {}).update({"waves": [3, 3, 2] if n == 8 else [3, n - 3]})
    if lr is not None:
        for e in c["search_space"]:
            if e["name"] == "training.lr":
                e["low"], e["high"] = float(lr[0]), float(lr[1])
    out = os.path.join(HERE, f"sweep_config_{name}.yaml")
    yaml.safe_dump(c, open(out, "w"), sort_keys=False)
    print(out, n, "trials")


def stage2():
    for p in PROBES2:
        for k in KS:
            het_cfg(f"tp3_finfte_{p}_d{k}", f"tp3_finfte_{p}_d{k}_het", lr=LR2)
    for p in PROBES + PROBES2:
        for k in (2, 4, 6, 8):
            het_cfg(f"tp3_finfte_{p}_d{k}", f"tp3_hp58fte_{p}_d{k}_het", lr=LR2,
                    extra=dict(HP58_MODEL, **{"fine_tune.pretrained_path": HP58}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)  # stage-2 trials run on CC only: the parent exists there alone
    if ap.parse_args().stage == 2:
        return stage2()
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
