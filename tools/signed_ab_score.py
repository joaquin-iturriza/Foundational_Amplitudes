"""The signed pools' A/B (D99; docs/results.tex, the candidate targets), scored on one footing. Two catalog runs that
differ only in the signed pools' target (scripts/job_catalog_norm.sh): SgnHead, log|x| with a sign head (the 'abslog'
transform, floor s_B the pool's 1% quantile of |x| on its train split, plus a sign logit), and SgnR0, the signed log
sgn(x) log(1 + |x|/r0) with r0 three nats below the pool's 0.5% quantile (data.signedlog_scale_shift=-3). x is the
stored value times the run's per-event target factor F (data.target_physics_factor: s_hat and the coupling orders),
F > 0, so a difference of log|x| is one of log|V|.

Each pool is built from the run's own config with its recipe swapped in (as tools/zero_shot_parent.py: the run's
weights, the pool's own transform and statistics fitted on its train split, which only represent its truth and are
inverted exactly). The run's prediction is mu_c + sigma_c h (h its standardized output, (mu_c, sigma_c) the run's
shared standardization). Per pool on its validation events, both arms on the same quantity m(v) = log max(|v|, s_B)
(s_B from the SgnHead scoring of the same pool, passed to SgnR0 by --ref):
  mse_mag, bias, mse_after_offset   of m(prediction) - m(truth)
  wrong_sign                        the share of events whose predicted sign is wrong
  near_share, mse_mag_near, wrong_sign_near   on the events within a factor 10 of zero (|x| below a tenth of the
                                    pool's median |x|)
SgnR0 inverts its prediction with r0 = the pool's own scale (oracle: the level known; for a training pool that is
the scale the run trained with) and, with --ref, also with r0 moved by SgnHead's level offset on that pool
(predlevel: the level a model predicts, as a held-out pool would need), reported as <metric>_predlevel.
Prints one SIGNED_AB JSON line per pool (an "error" line for a pool that cannot be built: listed, not dropped).
Needs a GPU.
    python tools/signed_ab_score.py --run <run dir> --arm head|r0 [--ref <SgnHead SIGNED_AB lines>] <recipe.yaml> [...]
"""
import argparse, json, os, sys, tempfile

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import siteconf  # noqa: E402
from preprocessing import signedlog_scale  # noqa: E402
from rebuild_run import _build, load_best_weights  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="the A/B run dir (config.yaml, data_stats.json, models/)")
ap.add_argument("--arm", required=True, choices=("head", "r0"))
ap.add_argument("--ref", default=None, help="a file of the SgnHead scoring's SIGNED_AB lines (floor s_B, level offset)")
ap.add_argument("--train-subsample", type=int, default=10000,
                help="pool train events read for its own transform and statistics (the short run's 1e4)")
ap.add_argument("--eval-subsample", type=int, default=2000, help="validation events per pool")
ap.add_argument("recipes", nargs="+")
a = ap.parse_args()

run = os.path.abspath(a.run)
rstats = json.load(open(os.path.join(run, "data_stats.json")))
assert rstats.get("shared_standardization"), "the A/B runs use one shared standardization (job_catalog_norm.sh)"
mu_c, sd_c = float(rstats["prepd_mean"][0]), float(rstats["prepd_std"][0])
ref = {}
if a.ref:
    for line in open(a.ref):
        if line.startswith("SIGNED_AB {"):
            d = json.loads(line[len("SIGNED_AB "):])
            if "error" not in d:
                ref[d["pool"]] = d
assert a.arm == "head" or ref, "--arm r0 needs --ref (the SgnHead scoring: the common floor s_B)"


def metrics(xt, xp, floor, tag=""):
    """m(v) = log max(|v|, floor) of prediction against truth; signs; the events within a factor 10 of zero"""
    m = lambda v: np.log(np.maximum(np.abs(v), floor))
    d = m(xp) - m(xt)
    near = np.maximum(np.abs(xt), floor) < np.median(np.maximum(np.abs(xt), floor)) / 10
    wrong = np.sign(xp) != np.sign(xt)
    out = {"mse_mag": float(np.mean(d ** 2)), "bias": float(np.mean(d)), "mse_after_offset": float(np.var(d)),
           "wrong_sign": float(np.mean(wrong)), "near_share": float(np.mean(near)),
           "mse_mag_near": float(np.mean(d[near] ** 2)) if near.any() else None,
           "wrong_sign_near": float(np.mean(wrong[near])) if near.any() else None}
    return {k + tag: v for k, v in out.items()}


for recipe in a.recipes:
    recipe = os.path.abspath(siteconf._expand_str(recipe))
    import yaml
    spec = yaml.safe_load(open(recipe))["processes"]
    assert len(spec) == 1, f"{recipe}: one pool per recipe"
    pool = f"{spec[0]['name']}_{spec[0]['sqrts'][0]}-{spec[0]['sqrts'][1]}GeV"
    cfg = OmegaConf.load(os.path.join(run, "config.yaml"))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.count_flops = False; cfg.ema = False
        cfg.warm_start_idx = None
        cfg.training.validate_frac = 0; cfg.training.validate_every_n_steps = 1
        cfg.run_dir = tempfile.mkdtemp(prefix="signed_ab_", dir=os.environ["SCRATCH"])
        cfg.data.source = "recipes"
        cfg.data.processes_file = recipe
        cfg.data.train_subsample = a.train_subsample
        cfg.data.eval_subsample = a.eval_subsample
        cfg.data.memmap = False
        # the pool's own transform and statistics (a representation of its truth, inverted exactly)
        cfg.data.shared_standardization = False
        cfg.data.affine_multiplicity_std = False
        cfg.fine_tune.pretrained_path = os.path.join(run, "models", "model_run0_best.pt")
        cfg.fine_tune.target_stats = "own"
        cfg.fine_tune.offshell_stats = "parent"
    try:
        exp = _build(cfg, None, None, None)
    except Exception as e:                            # a missing pool, sidecar or alpha_s file: listed, not dropped
        print("SIGNED_AB " + json.dumps({"pool": pool, "arm": a.arm, "run": run, "error": repr(e)[:300]}), flush=True)
        continue
    load_best_weights(exp.model, run)
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    exp.ema, exp.train_sampler = None, None
    exp._init_regularization()
    hs, zs, logits, signs = [], [], [], []
    with torch.no_grad():
        for data in exp.val_loader:
            y_pred, y, _, _, _ = exp._forward_lloca(data)
            hs.append(y_pred.reshape(-1).double().cpu()); zs.append(y.reshape(-1).double().cpu())
            if exp._last_sign is not None:
                lg, sg = exp._last_sign
                logits.append(lg.reshape(-1).double().cpu()); signs.append(sg.reshape(-1).double().cpu())
    h, z = torch.cat(hs).numpy(), torch.cat(zs).numpy()
    tr = str((exp._amp_trafos_pp or [exp.cfg.data.amp_trafos])[0][0])
    mu_p, sd_p = float(exp.prepd_mean[0]), float(exp.prepd_std[0])
    t = mu_p + sd_p * z                               # the pool's own transformed truth
    q = mu_c + sd_c * h                               # the run's prediction in the same transform
    rec = {"pool": pool, "arm": a.arm, "run": run, "transform": tr, "n_val": int(len(z))}
    if a.arm == "head":
        assert tr.startswith("abslog:"), f"{pool}: SgnHead scores a sign-changing pool, transform {tr}"
        floor = signedlog_scale(tr)
        sgn_t = np.where(np.concatenate(signs) > 0.5, 1.0, -1.0)
        sgn_p = np.where(np.concatenate(logits) > 0, 1.0, -1.0)
        # the truth is known only down to the floor (abslog), which is where the metric clips it anyway
        xt, xp = sgn_t * np.exp(t), sgn_p * np.exp(q)
        rec.update({"floor": floor, "frac_at_floor": float(np.mean(t <= np.log(floor) + 1e-6))})
        rec.update(metrics(xt, xp, floor))
    else:
        assert tr.startswith("signedlog:"), f"{pool}: SgnR0 scores a sign-changing pool, transform {tr}"
        assert pool in ref, f"{pool}: no SgnHead scoring in --ref (the common floor)"
        floor = float(ref[pool]["floor"])
        r0 = signedlog_scale(tr)
        inv = lambda y, s: s * np.sign(y) * np.expm1(np.abs(y))
        xt = inv(t, r0)
        rec.update({"floor": floor, "r0": r0, "r0_over_floor": r0 / floor})
        rec.update(metrics(xt, inv(q, r0), floor))
        # the level a model predicts in place of the pool's own: r0 moved by SgnHead's level offset on this pool
        off = float(ref[pool]["bias"])
        rec.update({"level_offset": off})
        rec.update(metrics(xt, inv(q, r0 * np.exp(off)), floor, "_predlevel"))
    print("SIGNED_AB " + json.dumps(rec), flush=True)
