"""Build a sigma-steered pool once, from a reference model, for a recipe whose sampling mode is
`steered` (docs/results.tex, sec:ladder; the steering rule of sec:div-sampling, generated once
instead of on the fly, so every run of a data-scaling grid trains on the same fixed pool).

  1. candidates  oversample x (n_train + n_val + n_test) events from the flat proposal (uniform
                 sqrt(s), uniform angles for a 2->2, RAMBO otherwise), labelled with the process's
                 compiled backend: datagen.ensure_dataset under $SCRATCH/steer_candidates, cached.
  2. score       the reference run (a HETEROSC run with a detached sigma head) is rebuilt from its
                 own config with the candidates in place of its validation split, so the inputs go
                 through exactly its preprocessing (momentum scale, target factors, off-shellness);
                 its best checkpoint gives sigma per candidate, mapped back to candidate rows through
                 the split's permutation.
  3. select      n_train + n_val + n_test candidates without replacement with probability
                 proportional to sigma^gamma (Efraimidis-Spirakis keys), shuffled and split into the
                 three roles, so train, val and test sample the same distribution.
  4. write       each role at the path and recipe id the loader asks for (mg5_pipeline_final
                 recipe_output_path, tag _ssteer), with a <pool>.steer.json sidecar (reference run,
                 candidate pool, sigma quantiles, selection counts).

The recipe's `sampling` block carries the steering parameters, so they are part of the pools'
identity: {mode: steered, steer_ref: <reference run name>, steer_gamma: g, steer_oversample: k}.
Needs a GPU (the model's attention is CUDA-only).
    python tools/steer_pool.py --recipe recipes/<steered recipe>.yaml
"""
import argparse, json, os, sys, tempfile

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import siteconf  # noqa: E402
import datagen  # noqa: E402
import mg5_pipeline_final as mg  # noqa: E402

CAND_SEED = 777          # the candidates' own draw, disjoint from the pools' seed-42 draws


def load_state(run_dir):
    import gzip, io
    p = os.path.join(run_dir, "models", "model_run0_best.pt")
    if not os.path.exists(p):
        with gzip.open(p + ".gz", "rb") as f:
            return torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)["model"]
    return torch.load(p, map_location="cpu", weights_only=False)["model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    a = ap.parse_args()
    rec = yaml.safe_load(open(a.recipe))
    samp = rec["sampling"]
    assert samp.get("mode") == "steered", "the recipe's sampling mode must be `steered`"
    (spec,) = rec["processes"]
    name, (lo, hi) = spec["name"], spec["sqrts"]
    n_role = {"train": int(spec["n_train"]), "val": int(spec["n_val"]), "test": int(spec["n_test"])}
    n_keep = sum(n_role.values())
    gamma, k = float(samp["steer_gamma"]), int(samp["steer_oversample"])
    ref = os.path.join(siteconf.PROJECT_DIR, "runs", samp["steer_ref"])

    # 1. candidates: the flat proposal, labelled
    base_sampling = mg.PROCESSES[name].get("sampling")
    mg.PROCESSES[name]["sampling"] = {"mode": "flat"}
    cand_dir = os.path.join(os.environ["SCRATCH"], "steer_candidates")
    os.makedirs(cand_dir, exist_ok=True)
    cand = datagen.ensure_dataset(name, lo, hi, k * n_keep, role="train", seed=CAND_SEED, dest_dir=cand_dir)
    if base_sampling is None:
        mg.PROCESSES[name].pop("sampling", None)
    else:
        mg.PROCESSES[name]["sampling"] = base_sampling
    C = np.load(cand)
    print(f"[steer] {len(C)} candidates at {cand}", flush=True)

    # 2. score: the reference run rebuilt with the candidates as its validation split
    cfg = OmegaConf.load(os.path.join(ref, "config.yaml"))
    assert cfg.training.loss == "HETEROSC", f"{ref} is not a HETEROSC run (no sigma head)"
    ref_rec = yaml.safe_load(open(os.path.expandvars(str(cfg.data.processes_file))))
    (ref_spec,) = ref_rec["processes"]
    assert ref_spec["name"] == name, (ref_spec["name"], name)
    ref_spec["n_val"] = len(C)
    tmp_rec = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(ref_rec, tmp_rec); tmp_rec.close()
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.count_flops = False
        cfg.warm_start_idx = None; cfg.fine_tune.pretrained_path = None
        cfg.data.processes_file = tmp_rec.name
        cfg.run_dir = tempfile.mkdtemp(prefix="steer_score_", dir=os.environ["SCRATCH"])
    real_split = datagen.ensure_split_set

    def split_with_candidates(specs, role, seed, dest_dir=None, require_cache=False):
        if role == "val":
            return {name: cand}
        return real_split(specs, role, seed, dest_dir=dest_dir, require_cache=require_cache)
    datagen.ensure_split_set = split_with_candidates
    torch.set_default_dtype(torch.float32)
    from experiment import AmplitudeExperiment
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra()
    exp.init_data(); exp._init_dataloader(); exp.init_model()
    datagen.ensure_split_set = real_split
    exp.model.load_state_dict(load_state(ref))
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    with torch.no_grad():
        _, _, sig = exp._collect_predictions(exp.val_loader)
    sig = np.asarray(sig, np.float64).reshape(-1)
    perm = exp._role_perm["val"]
    assert len(sig) == len(C) == len(perm), (len(sig), len(C), len(perm))
    sigma = np.empty(len(C)); sigma[perm] = sig                       # candidate row -> sigma
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0), "non-finite or non-positive sigma"

    # 3. select prop. to sigma^gamma without replacement, then split into the roles
    rng = np.random.default_rng(CAND_SEED + 1)
    key = np.log(-np.log(rng.uniform(size=len(C)))) - gamma * np.log(sigma)    # smallest keys win
    keep = np.argpartition(key, n_keep)[:n_keep]
    keep = keep[rng.permutation(n_keep)]
    parts, start = {}, 0
    for role in ("train", "val", "test"):
        parts[role] = keep[start:start + n_role[role]]; start += n_role[role]

    # 4. write each role where the loader looks for it
    mg.register_recipe_processes([{"name": name, "base": spec.get("base", name), "sqrts_min": float(lo),
                                   "sqrts_max": float(hi), "n_train": n_role["train"], "n_val": n_role["val"],
                                   "n_test": n_role["test"], "physics": spec.get("physics"),
                                   "sampling": None}], default_sampling=samp)
    q = lambda x: {f"q{p}": float(np.quantile(x, p / 100)) for p in (1, 50, 90, 99, 99.9)}
    top = sigma >= np.quantile(sigma, 0.999)
    summary = dict(recipe=os.path.abspath(a.recipe), reference=ref, candidates=cand, n_candidates=len(C),
                   gamma=gamma, sigma_candidates=q(sigma), sigma_kept=q(sigma[keep]),
                   top01pct_sigma_share_candidates=0.001, top01pct_sigma_share_kept=float(top[keep].mean()),
                   pools={})
    for role, rows in parts.items():
        r = mg.variable_energy_recipe(name, lo, hi, n_role[role], role=role, seed=int(cfg.data.get("seed", 42)))
        out = mg.recipe_output_path(r, datagen.dest_for_role(role))
        tmp = out + ".tmp.npy"
        np.save(tmp, C[rows]); os.replace(tmp, out)      # random order: a prefix is a random subset
        mg.write_recipe(out, r)
        summary["pools"][role] = dict(path=out, recipe_id=mg.recipe_id(r), n=int(len(rows)))
        json.dump(summary, open(out + ".steer.json", "w"), indent=1)
        print(f"[steer] {role}: {len(rows)} events -> {out}", flush=True)
    print("STEER_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
