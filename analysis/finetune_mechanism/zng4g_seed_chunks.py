"""Plan finetune_mechanism step 1 (D515 design, seeds): chunk files for scripts/job_seed_batch.sh re-running a driven
search's chosen trial (its lowest val_loss in sweeps_local/<sweep>/results) at other seeds, one run per chunk. The HPs
are rebuilt from the sweep config's candidate pool as sweep/seed_list.py does (dyhpo_sampler._sample_candidates).

    python3 analysis/finetune_mechanism/zng4g_seed_chunks.py ftmech_zng4g_ft_b2048 --seeds 43 44
"""
import argparse
import glob
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "sweep"))
from dyhpo_sampler import _sample_candidates  # noqa: E402


def hps(cfg_path, idx):
    c = yaml.safe_load(open(os.path.join(ROOT, cfg_path)))
    d = c.get("dyhpo", {})
    cand = _sample_candidates(c["search_space"], d.get("n_candidates", 300), d.get("seed", 42))
    assert idx < len(cand), f"hp{idx} is past the pre-sampled pool"
    fmt = lambda v: str(v).lower() if isinstance(v, bool) else v   # noqa: E731
    return c, {k: fmt(v) for k, v in cand[idx].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    a = ap.parse_args()
    # self-check: the b2048 scratch cell's chosen trial (hp1333) rebuilds to the lr its scrflags chunk carries (b128's
    # hp0950 does not: its config's lr window was widened after the draw, 16f200fc)
    _, h = hps("sweep/sweep_config_tp_zng_d1e4_muon_b2048_zgggg.yaml", 1333)
    assert abs(h["training.lr"] / 0.02029637 - 1) < 1e-6, h
    ts = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "sweeps_local", a.sweep, "results", "*.json"))]
    best = min((t for t in ts if t["val_loss"] is not None), key=lambda t: t["val_loss"])
    cfg = f"sweep/sweep_config_{a.sweep}.yaml"
    c, h = hps(cfg, best["hp_idx"])
    T = c["fidelity_schedule"]["t_steps"][-1]
    assert best["t_steps"] == T
    for s in a.seeds:
        out = os.path.join(HERE, "seed_chunks", f"{a.sweep.replace('ftmech_', '')}_s{s}.txt")
        open(out, "w").write(f"{cfg} {a.sweep}_s{s} {T} " + " ".join(f"{k}={v}" for k, v in h.items()) + f" seed={s}\n")
        print(out)
    print(f"{a.sweep}: chosen hp{best['hp_idx']} (trial {best['trial_idx']}, val {best['val_loss']:.5g}) of {len(ts)} trials")


if __name__ == "__main__":
    main()
