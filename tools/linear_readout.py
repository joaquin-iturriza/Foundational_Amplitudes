"""D108, the no-gain probe of par:nogain (rung 6 against rung 7 on u ubar -> Z g): does a pretraining's frozen trunk
already carry the probe's target? For a fine-tune cell (its sweep config: the parent checkpoint, the probe recipe, D,
the target statistics of the fine-tune), the experiment is built as the cell's fine-tune builds it at step 0 (the same
fixed parameters, init_physics -> init_data -> init_model, so the parent's weights, momentum scale and tokenizer, and
the probe's own target standardization), then the trunk is run frozen and the per-event features are read where the
readout sees them: the input of the network's linear_out (per particle, hidden width), mean-pooled per event as the
wrapper pools the output (the readout is linear, so a readout on the pooled features spans exactly what a retrained
head on a frozen trunk can reach). A ridge regression with intercept is fitted on the cell's D training events (the
penalty chosen by leave-one-out on those events alone, closed form through the SVD, never on validation) and scored on
the validation split; also the parent's own head output with a fitted affine map (one feature) as the reference.
Units: the MSE of the standardized target times prepd_std^2, i.e. the MSE of log|M|^2, as cells.py reports a cell.
    python tools/linear_readout.py --out $SCRATCH/zz_linear_readout sweep/sweep_config_tp3_r6fte_uubar_Zg_d4.yaml [...]
One READOUT JSON line per config, and <out>/<sweep_name>.json. Needs a GPU (xformers attention).
"""
import argparse, json, os, sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import siteconf  # noqa: E402

LAMBDAS = np.logspace(-6, 4, 41)


def build(cfg_path, run_dir, n_train_eval):
    from hydra import compose, initialize_config_dir
    sw = siteconf.resolve(yaml.safe_load(open(cfg_path)))
    fixed = dict(sw["fixed_params"])
    T = sw["fidelity_schedule"]["t_steps"][-1]
    ov = ["local=none"] + [f"{k}={v}" for k, v in fixed.items() if k != "training.iterations"]
    ov += [f"training.iterations={T}", f"training.increment_steps={T}", f"base_dir={sw['paths']['project_dir']}",
           f"run_dir={run_dir}", f"exp_name=zz_linear_readout", "warm_start_idx=null", "use_mlflow=false",
           "plot=false", f"evaluation.train_subsample={n_train_eval}", "ema=false"]
    with initialize_config_dir(config_dir=os.path.join(ROOT, "config"), version_base=None):
        cfg = compose(config_name="amplitudes", overrides=ov)
    torch.set_default_dtype(torch.float32)
    from experiment import AmplitudeExperiment
    exp = AmplitudeExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_geometric_algebra()
    exp.init_data()
    exp._init_dataloader()
    exp.init_model()                  # the fine-tune path: the parent's weights (fine_tune.pretrained_path)
    exp.model.to(exp.device, dtype=exp.dtype).eval()
    return exp, sw, fixed


def features(exp, loader):
    """per-event mean of linear_out's input, the head's output (channel 0) and the standardized target"""
    # the wrapper's net holds the LLoCa net, whose backbone holds the head (module name net.net.linear_out)
    lins = [m for n, m in exp.model.named_modules() if n.split(".")[-1] == "linear_out"]
    assert len(lins) == 1, f"expected one linear_out (the LLoCa muP head), found {len(lins)}"
    lin = lins[0]
    store = {}
    h = lin.register_forward_hook(lambda m, i, o: store.__setitem__("h", i[0].detach()))
    F, Y, Z = [], [], []
    with torch.no_grad():
        for particles, y, tokens, order_labels, ptr, process_ids in loader:
            ptr = ptr.to(exp.device)
            out = exp.model(particles.to(exp.device), tokens.to(exp.device), mean=exp.mom_mean[0],
                            std=exp.mom_std[0], ptr=ptr, order_labels=order_labels.to(exp.device),
                            process_ids=process_ids.to(exp.device))
            x = store["h"].reshape(-1, store["h"].shape[-1])
            assert x.shape[0] == int(ptr[-1]), (x.shape, int(ptr[-1]))
            seg = torch.repeat_interleave(torch.arange(len(ptr) - 1, device=exp.device), ptr[1:] - ptr[:-1])
            pooled = torch.zeros(len(ptr) - 1, x.shape[1], device=exp.device, dtype=x.dtype).index_add_(0, seg, x)
            pooled /= (ptr[1:] - ptr[:-1]).unsqueeze(1).to(x.dtype)
            F.append(pooled.double().cpu().numpy()); Z.append(out[:, 0].double().cpu().numpy())
            Y.append(y[:, 0].double().numpy())
    h.remove()
    return np.concatenate(F), np.concatenate(Y), np.concatenate(Z)


def ridge_loo(X, y):
    """ridge with intercept, the penalty by closed-form leave-one-out on (X, y); returns (w, b, lam, loo_mse)"""
    mx, my = X.mean(0), y.mean()
    Xc, yc = X - mx, y - my
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    uy = U.T @ yc
    n = len(y)
    best = None
    for lam in LAMBDAS * (s[0] ** 2 if len(s) else 1.0):
        d = s ** 2 / (s ** 2 + lam)
        fit = U @ (d * uy)
        hdiag = (U ** 2) @ d + 1.0 / n             # the intercept's leverage on top
        loo = np.mean(((yc - fit) / np.clip(1 - hdiag, 1e-9, None)) ** 2)
        if best is None or loo < best[3]:
            w = Vt.T @ (s / (s ** 2 + lam) * uy)
            best = (w, my - mx @ w, float(lam), float(loo))
    return best


def one(cfg_path, out_dir, n_train_eval):
    name = yaml.safe_load(open(cfg_path))["sweep_name"]
    exp, sw, fixed = build(cfg_path, os.path.join(out_dir, "runs", f"{name}_{os.environ.get('SLURM_JOB_ID', os.getpid())}"), n_train_eval)   # one fresh run dir per job
    std2 = float(np.asarray(exp.prepd_std[0]).ravel()[0]) ** 2
    Ft, yt, zt = features(exp, exp.train_eval_loader)
    Fv, yv, zv = features(exp, exp.val_loader)
    D = int(fixed["data.train_subsample"])
    assert len(yt) == D, f"{name}: {len(yt)} training events read, the cell trains on {D}"
    w, b, lam, loo = ridge_loo(Ft, yt)
    a1, b1 = np.polyfit(zt, yt, 1)
    rec = {"sweep": name, "parent": fixed["fine_tune.pretrained_path"], "D": D, "n_val": int(len(yv)),
           "feat_dim": int(Ft.shape[1]), "prepd_std": float(np.sqrt(std2)),
           "readout_val_mse": float(np.mean((Fv @ w + b - yv) ** 2) * std2),
           "readout_train_mse": float(np.mean((Ft @ w + b - yt) ** 2) * std2), "readout_loo_mse": loo * std2,
           "ridge_lambda": lam, "head_affine_val_mse": float(np.mean((a1 * zv + b1 - yv) ** 2) * std2),
           "head_raw_val_mse": float(np.mean((zv - yv) ** 2) * std2), "val_var": float(np.var(yv) * std2)}
    json.dump(rec, open(os.path.join(out_dir, name + ".json"), "w"), indent=1)
    print("READOUT " + json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="fine-tune cell sweep configs (sweep/sweep_config_<fam>_<probe>_d<k>.yaml)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train-eval", type=int, default=100000, help="cap of the train-split read (above every D)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for c in a.configs:
        try:
            one(c, a.out, a.n_train_eval)
        except Exception as e:                    # one failed cell is listed, never silently dropped
            print("READOUT " + json.dumps({"config": c, "error": repr(e)[:300]}), flush=True)


if __name__ == "__main__":
    main()
