"""Het-loss DyHPO sweeps on a few of the Delta cells with more than two final-state particles (D75, the user: "let's sweep a few, see if the HPs are more or
less consistent to see if we can choose a set of HPs for all, or if we will have to sweep everything"). Each sweep is
the cell's own MSE sweep config (same search space, trial count, horizon, data and parent) with the loss switched to
the het loss as scripts/job_het_delta.sh switches it, so best-vs-best compares with the MSE cell (CLAUDE.md A/B step 4):
its surrogate and candidate pool too (same seed), only dyhpo.objective_scale linear because the NLL is signed (commit
8d31b58), waves 3+3+2 (3+2 for a 5-trial fine-tune). Every trial ends with the sigma-only stage (D70). Cells: scratch and
rung 1 of each probe with more than two final-state particles, the agent's pick for D75's "a few" (u ubar -> Zggg is
2->4, the other three 2->3). Writes sweep/sweep_config_<sweep>_het.yaml, one per cell.
    python sweep/gen_het23_sweeps.py
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# the four probes with more than two final-state particles (2->3; uubar_Zggg 2->4), scratch and the rung-1 fine-tune
# of each (the one fine-tune arm all four have)
CELLS = ("tp3_scr_ee_ddbarg_d8", "tp3_r1fte_ee_ddbarg_d8", "tp3_scr_ee_ttbarg_d8", "tp3_r1fte_ee_ttbarg_d8",
         "tp_scr_uubar_Zgg_d8", "tp3_r1fte_uubar_Zgg_d8", "tp3_scrh_uubar_Zggg_d8", "tp3_r1fte_uubar_Zggg_d8")
HET = {"training.loss": "HETEROSC", "training.heterosc_beta": 0.0, "training.heterosc_select": "nll",
       "model.net.sigma_after_pool": "true", "model.net.detach_sigma_backbone": "false",
       "training.sign_head": "false", "training.track_train_eval": "true", "evaluation.train_subsample": 2000}


def main():
    for s in CELLS:
        c = yaml.safe_load(open(os.path.join(HERE, f"sweep_config_{s}.yaml")))
        c["sweep_name"] = s + "_het"
        c["fixed_params"].update(HET)
        n = int(c["n_trials"])
        c.setdefault("dyhpo", {}).update({"objective_scale": "linear",
                                          "waves": [3, 3, 2] if n == 8 else [3, n - 3]})
        out = os.path.join(HERE, f"sweep_config_{s}_het.yaml")
        yaml.safe_dump(c, open(out, "w"), sort_keys=False)
        print(out, n, "trials")


if __name__ == "__main__":
    main()
