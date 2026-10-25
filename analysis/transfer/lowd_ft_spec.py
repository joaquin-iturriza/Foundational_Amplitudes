"""The small-D fine-tuning study, first round (D120; docs/results.tex hand-off "Fine-tuning at small D"): the cells and
the hyperparameters every method runs at. A cell is (probe, rung, D) with D = 10, 10^1.5, 10^2, 10^2.5 (the grid's k = 2
to 5, steps 176, 704, 2816, 4576 from the cell's own sweep config); its HPs are the cell's best fine-tune trial in the
grid (cells.best of tp3_r<rung>fte_<probe>_d<k>, the best-checkpoint MSE of log|M|^2), the same for every method
(CLAUDE.md, A/B protocol step 3). Probes: e+e- -> Z gamma, e+e- -> d dbar and u d -> u d (the ones named in the design)
and the two with the largest mean gain over rungs at these D besides them (analysis/transfer/data/nogain.json:
e+e- -> b bbar at one loop +0.66 decades, u ubar -> g g +0.57). Rungs 1, 7, 9. Also each cell's scratch value
(cells.scratch) and the grid's recorded fine-tune value.
    python3 analysis/transfer/lowd_ft_spec.py   -> analysis/transfer/data/lowd_ft_spec.json
"""
import json, os, sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import cells  # noqa: E402

PROBES = ("ee_Za", "ee_ddbar", "ud_ud", "ee_bb_nlo", "uubar_gg")
RUNGS = (1, 7, 9)
KS = (2, 3, 4, 5)
HP_KEYS = {"lambda": "training.regularization_lambda", "warmup": "training.cosanneal_warmup_frac",
           "eta_min": "training.cosanneal_eta_min", "ema": "ema", "ema_decay": "training.ema_decay"}


def main():
    out = []
    for p in PROBES:
        for r in RUNGS:
            for k in KS:
                name = f"tp3_r{r}fte_{p}_d{k}"
                cfgf = os.path.join(ROOT, "sweep", f"sweep_config_{name}.yaml")
                if not os.path.exists(cfgf):
                    cfgf = os.path.join(ROOT, "sweep", f"sweep_config_{name}_002.yaml")
                cfg = yaml.safe_load(open(cfgf))
                val, t = cells.best(name)
                sval, _ = cells.scratch(p, k)
                assert t is not None, f"{name}: no best trial in scratch_sweeps.json"
                hp = {"training.lr": t["lr"]}
                hp.update({HP_KEYS[a]: v for a, v in t["hps"].items() if a in HP_KEYS})
                hp["ema"] = str(hp.get("ema", False)).lower()
                out.append({"probe": p, "rung": r, "k": k, "D": int(cfg["fixed_params"]["data.train_subsample"]),
                            "steps": int(cfg["fidelity_schedule"]["t_steps"][-1]), "config": os.path.relpath(cfgf, ROOT),
                            "best_hp": t["hp"], "hp": hp, "recorded_ft": val, "scratch": sval,
                            "prepd_std": t["prepd_std"]})
    json.dump(out, open(os.path.join(HERE, "data", "lowd_ft_spec.json"), "w"), indent=1)
    print(f"{len(out)} cells; steps {sorted({c['steps'] for c in out})}; D {sorted({c['D'] for c in out})}")


if __name__ == "__main__":
    main()
