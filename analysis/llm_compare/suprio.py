"""Suprio's LLM, pID embedding only (his baseline), from his own per-process test metrics: data/suprio_cond/
pid_only_seed{0,1,2}.json (= his cond_figures bundle's amplitude_qwen2_onehot/onehot_pf_last_as_rot_species_seed{s}/
amplitude_metrics.json, sent to Joaquin 2026-11-25) and test_split.json. Per process: rmse, level = mean(pred - true),
shape = std(pred - true) of ln|M|^2 (natural log, the base of our log|M|^2: his affine fit b = -6.713, sigma = 5.303
against ours -6.701, 5.316), 16384 test events, best-validation checkpoint; rmse^2 = level^2 + shape^2.
Names are ours ("dc_" + his folder). Seed 0's rmse^2 equals the values read off his figure before
(talks/progress_oct2026b/data/draft_pid_by_process.json) on all 139 held-out processes.
"""
import json, os

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "suprio_cond")
SEEDS = (0, 1, 2)
STAT = {"mse": lambda q: q["rmse"] ** 2, "bias": lambda q: q["level"], "bias2": lambda q: q["level"] ** 2,
        "shape": lambda q: q["shape"] ** 2}


def llm(stat="mse"):
    """{process: array over seeds 0-2} of mse (rmse^2), bias (signed level), bias2 (level^2) or shape (shape^2)."""
    runs = [json.load(open(os.path.join(D, f"pid_only_seed{s}.json")))["per_process"] for s in SEEDS]
    return {"dc_" + n: np.array([STAT[stat](r[n]) for r in runs]) for n in runs[0]}


def split():
    """{"interp": 96 relabels, "extrap": 43 new} from his test_split.json: the 24 same-sign charm-bottom processes count
    as interpolation (his README: relabels of training processes with a massless charm), the 2->1 group has no
    metrics and is left out (as in his figures)."""
    out = {"interp": [], "extrap": []}
    t = json.load(open(os.path.join(D, "test_split.json")))
    have = llm()
    for sp, gs in t.items():
        for g in gs:
            names = ["dc_" + f for f in g["folders"] if "dc_" + f in have]
            cb = g["group"].startswith("Same-sign charm-bottom")
            out["interp" if sp == "interpolation" or cb else "extrap"] += names
    assert (len(out["interp"]), len(out["extrap"])) == (96, 43), {k: len(v) for k, v in out.items()}
    return out
