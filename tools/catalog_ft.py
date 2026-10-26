"""Small-D fine-tunes of held-out catalog_v3 processes from a catalog_v3 short pretraining (the Muon against AdamW test,
D144/D146), modelled on tools/lowd_ft.py: the small-D cells (D = 10, 10^1.5, 10^2, 10^2.5 training events at 176, 704,
2816, 4576 steps), full retraining from the parent's best checkpoint, lr_scale 1 (the fine-tune lr is the parent's),
the parent's remaining HPs (run.py on the parent's own config.yaml, only the recipe, the steps and the fine-tune block
changed), the target's statistics fitted on the probe's own train split and the off-shellness on the parent's scale
(as the small-D round), validation on the probe's first 2000 validation events (the zero-shot read's), and the
fine-tune's optimizer chosen apart from the parent's (AdamW or MuonAdamW: the 2x2 of parent and fine-tune optimizers).
--scratch trains the same cells from a fresh initialisation instead (no pretrained weights, the target's and the
off-shellness statistics the probe's own), with the parent's config and HPs otherwise: whether fine-tuning a current
pretraining beats training from scratch at small D (the user, 2026-10-26).
--keep-parent-stats fine-tunes the same cells keeping the parent's target standardization (fine_tune.target_stats=
parent_shared: the probe's own transform, the parent's pooled mean and sigma), so the loss at step 0 is the parent's
zero-shot on the probe instead of its output read in the probe's own units (the refit fine-tunes threw that level away).
The four probes are the dimensionless arm's median-MSE probe of each axis (H1 new combinations, H2 2->4, H3 the top,
H5 one loop), fixed before any fine-tune ran.
One CATFT JSON line per cell: the best-checkpoint val_loss_no_reg (the run's result file) and that times prepd_std^2,
the MSE in the target's log units, as tools/lowd_ft.py reports a cell; a failed cell is listed with its error.
    python tools/catalog_ft.py --parent runs/catalog_v3_short/dimensionless_muon --optimizer AdamW --out $SCRATCH/zz_catft
        [--probes v3_gg_attx] [--only d2] [--steps-cap 20]
"""
import argparse, glob, json, os, subprocess, sys, time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = ["v3_dg_dgwpwm", "v3_usx_cxuwpz", "v3_gg_attx", "v3_udx_agwp_nlo"]
# the small-D round's cells (analysis/transfer/data/lowd_ft_spec.json): tag d<k>, D = 10^(k/2) events, steps
CELLS = {"d2": (10, 176), "d3": (32, 704), "d4": (100, 2816), "d5": (316, 4576)}
EVAL_EVENTS = 2000


def probe_recipe(probe, out):
    """The probe's entry from its held-out axis file, with the file's top-level keys (sampling: mixture)."""
    for f in sorted(glob.glob(os.path.join(ROOT, "recipes", "catalog_v3_heldout_*.yaml"))):
        r = yaml.safe_load(open(f))
        hit = [p for p in r["processes"] if p["name"] == probe]
        if hit:
            path = os.path.join(out, f"{probe}.recipe.yaml")
            yaml.safe_dump({**{k: v for k, v in r.items() if k != "processes"}, "processes": hit[:1]}, open(path, "w"))
            return path
    raise KeyError(f"{probe} is in no recipes/catalog_v3_heldout_*.yaml")


def command(parent, optimizer, recipe, D, T, run_dir, result_path, exp_name, scratch=False, target_stats="own"):
    ft = [] if scratch else [
        f"fine_tune.pretrained_path={os.path.abspath(parent)}/models/model_run0_best.pt",
        "fine_tune.lr_scale=1.0", "fine_tune.layer_decay=1.0", f"fine_tune.target_stats={target_stats}",
        "fine_tune.offshell_stats=parent"]
    return [sys.executable, os.path.join(ROOT, "run.py"), "--config-path", os.path.abspath(parent),
            "--config-name", "config", f"run_dir={run_dir}", "run_name=null", "warm_start_idx=null",
            f"exp_name={exp_name}", f"base_dir={ROOT}", "training.is_dyhpo_run=false", "training.increment_steps=null",
            f"training.iterations={T}", "training.validate_frac=0.01", "training.es_load_best_model=true",
            f"training.optimizer={optimizer}", f"training.result_path={result_path}",
            f"data.processes_file={recipe}", f"++data.train_subsample={D}", f"++data.eval_subsample={EVAL_EVENTS}",
            "evaluation.train_subsample=2000", "plot=true"] + ft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="the pretraining's run dir (config.yaml, models/)")
    ap.add_argument("--optimizer", required=True, choices=["AdamW", "MuonAdamW"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--probes", nargs="*", default=PROBES)
    ap.add_argument("--only", nargs="*", help="cells by tag (d2 d3 d4 d5)")
    ap.add_argument("--scratch", action="store_true", help="fresh initialisation, no pretrained weights")
    ap.add_argument("--keep-parent-stats", action="store_true",
                    help="the parent's target standardization on the probe (fine_tune.target_stats=parent_shared)")
    ap.add_argument("--steps-cap", type=int, default=None, help="cap every cell's steps (the smoke test)")
    a = ap.parse_args()
    pname = os.path.basename(os.path.normpath(a.parent))
    assert not (a.scratch and a.keep_parent_stats), "scratch has no parent statistics"
    kind = "scratch" if a.scratch else ("catft_parentstats" if a.keep_parent_stats else "catft")
    target_stats = "parent_shared" if a.keep_parent_stats else "own"
    base = os.path.join(a.out, f"{kind}_{pname}_{a.optimizer}")
    os.makedirs(base, exist_ok=True)
    failed = 0
    for probe in a.probes:
        recipe = probe_recipe(probe, base)
        for tag, (D, T) in CELLS.items():
            if a.only and tag not in a.only:
                continue
            T = min(T, a.steps_cap) if a.steps_cap else T
            run_dir = os.path.join(base, f"{probe}_{tag}")
            res = run_dir + "_result.json"
            rec = dict(kind=kind, target_stats=None if a.scratch else target_stats, parent=pname, optimizer=a.optimizer, probe=probe, tag=tag, D=D, steps=T)
            t0 = time.time()
            try:
                p = subprocess.run(command(a.parent, a.optimizer, recipe, D, T, run_dir, res,
                                           f"{kind}_{pname}_{a.optimizer}", a.scratch, target_stats), cwd=ROOT)
                if p.returncode:
                    raise RuntimeError(f"run.py exit {p.returncode}")
                r = json.load(open(res))
                std2 = float(json.load(open(os.path.join(run_dir, "data_stats.json")))["prepd_std"][0]) ** 2
                rec.update(val_loss_no_reg=float(r["val_loss"]), mse_log=float(r["val_loss"]) * std2,
                           prepd_std=std2 ** 0.5, best_step=r.get("best_step"), seconds=round(time.time() - t0))
                if r.get("val_loss_step0") is not None:     # the validation before any update (the parent weights)
                    rec.update(val_loss_step0=float(r["val_loss_step0"]), mse_log_step0=float(r["val_loss_step0"]) * std2)
            except Exception as e:                    # one failed cell is listed, never silently dropped
                rec["error"] = repr(e)[:300]
                failed += 1
            print("CATFT " + json.dumps(rec), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
