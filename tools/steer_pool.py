"""Build a sigma-steered pool once, from a reference model, for a recipe whose sampling mode is
`steered` (docs/results.tex, sec:ladder; the steering rule of sec:div-sampling, generated once
instead of on the fly, so every run of a data-scaling grid trains on the same fixed pool). Two stages:

  candidates  (CPU)  oversample x (n_train + n_val + n_test) events from the flat proposal (uniform
                     sqrt(s), uniform angles for a 2->2, RAMBO otherwise), labelled with the process's
                     compiled backend: datagen.ensure_dataset under $SCRATCH/steer_candidates, cached.
  select      (GPU)  the reference run (HETEROSC, detached sigma backbone, beta = 1: the sigma whose
                     ranking the steering study validated) is rebuilt with the candidates as its
                     validation split (tools/rebuild_run.py: its own preprocessing, checked against its
                     frozen stats); its best checkpoint gives sigma per candidate, mapped back to
                     candidate rows through the split's permutation. n_train + n_val + n_test candidates
                     are kept without replacement with probability prop. to sigma^gamma
                     (Efraimidis-Spirakis keys), shuffled and split into the three roles, so train, val
                     and test sample the same distribution, and each role is written at the path and
                     recipe id the loader asks for (tag _ssteer) with a <pool>.steer.json sidecar.

The recipe's `sampling` block carries the steering parameters, so they are part of the pools' identity:
{mode: steered, steer_ref: <reference run name>, steer_gamma: g, steer_oversample: k}.
    python tools/steer_pool.py --recipe recipes/<steered recipe>.yaml --stage candidates|select
"""
import argparse, json, os, sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import siteconf  # noqa: E402
import datagen  # noqa: E402
import mg5_pipeline_final as mg  # noqa: E402

CAND_SEED = 777          # the candidates' own draw, disjoint from the pools' seed-42 draws


def candidates(name, lo, hi, n):
    """Path of the labelled flat candidate pool (generated here if absent)."""
    base = mg.PROCESSES[name].get("sampling")
    mg.PROCESSES[name]["sampling"] = {"mode": "flat"}
    try:
        d = os.path.join(os.environ["SCRATCH"], "steer_candidates")
        os.makedirs(d, exist_ok=True)
        return datagen.ensure_dataset(name, lo, hi, n, role="train", seed=CAND_SEED, dest_dir=d)
    finally:
        if base is None:
            mg.PROCESSES[name].pop("sampling", None)
        else:
            mg.PROCESSES[name]["sampling"] = base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--stage", required=True, choices=["candidates", "select"])
    a = ap.parse_args()
    rec = yaml.safe_load(open(a.recipe))
    samp = rec["sampling"]
    assert samp.get("mode") == "steered", "the recipe's sampling mode must be `steered`"
    (spec,) = rec["processes"]
    name, (lo, hi) = spec["name"], spec["sqrts"]
    n_role = {"train": int(spec["n_train"]), "val": int(spec["n_val"]), "test": int(spec["n_test"])}
    n_keep = sum(n_role.values())
    gamma, k = float(samp["steer_gamma"]), int(samp["steer_oversample"])
    cand = candidates(name, lo, hi, k * n_keep)
    print(f"[steer] candidates: {cand}", flush=True)
    if a.stage == "candidates":
        return

    import torch
    from omegaconf import OmegaConf
    from rebuild_run import rebuild
    ref = os.path.join(siteconf.PROJECT_DIR, "runs", samp["steer_ref"])
    rc = OmegaConf.load(os.path.join(ref, "config.yaml"))
    assert rc.training.loss == "HETEROSC", f"{ref} is not a HETEROSC run (no sigma head)"
    assert float(rc.training.heterosc_beta) == 1.0 and bool(rc.model.net.detach_sigma_backbone), \
        f"{ref}: the steering rule was validated with a detached sigma backbone at beta = 1"
    C = np.load(cand)
    exp, loader = rebuild(ref, "val", cand, len(C))
    with torch.no_grad():
        _, _, sig = exp._collect_predictions(loader)
    sig = np.asarray(sig, np.float64).reshape(-1)
    perm = exp._role_perm["val"]
    assert len(sig) == len(C) == len(perm), (len(sig), len(C), len(perm))
    sigma = np.empty(len(C)); sigma[perm] = sig                       # candidate row -> sigma
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0), "non-finite or non-positive sigma"

    rng = np.random.default_rng(CAND_SEED + 1)
    key = np.log(-np.log(rng.uniform(size=len(C)))) - gamma * np.log(sigma)    # smallest keys win
    keep = np.argpartition(key, n_keep)[:n_keep]
    keep = keep[rng.permutation(n_keep)]
    parts, start = {}, 0
    for role in ("train", "val", "test"):
        parts[role] = keep[start:start + n_role[role]]; start += n_role[role]

    mg.register_recipe_processes([{"name": name, "base": spec.get("base", name), "physics": spec.get("physics"),
                                   "sampling": None}], default_sampling=samp)
    q = lambda x: {f"q{p}": float(np.quantile(x, p / 100)) for p in (1, 50, 90, 99, 99.9)}
    top = sigma >= np.quantile(sigma, 0.999)
    summary = dict(recipe=os.path.abspath(a.recipe), reference=ref, candidates=cand, n_candidates=int(len(C)),
                   gamma=gamma, sigma_candidates=q(sigma), sigma_kept=q(sigma[keep]),
                   kept_share_of_top01pct_sigma=float(top[keep].mean()), pools={})
    seed = int(rc.data.get("seed", 42))
    for role, rows in parts.items():
        r = mg.variable_energy_recipe(name, lo, hi, n_role[role], role=role, seed=seed)
        out = mg.recipe_output_path(r, datagen.dest_for_role(role))
        tmp = out + ".tmp.npy"
        np.save(tmp, C[rows]); os.replace(tmp, out)      # random order: a prefix is a random subset
        mg.write_recipe(out, r)
        summary["pools"][role] = dict(path=out, recipe_id=mg.recipe_id(r), n=int(len(rows)))
        print(f"[steer] {role}: {len(rows)} events -> {out}", flush=True)
    for role in parts:
        json.dump(summary, open(summary["pools"][role]["path"] + ".steer.json", "w"), indent=1)
    print("STEER_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
