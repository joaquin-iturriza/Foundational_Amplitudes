#!/usr/bin/env python3
"""Plan entry warm_start, step (a) (D291): does the het loss decide which trial of the k=533 warm-start sweep
ws_c1_k533_hpo8 wins? Each trial's validation NLL at its best checkpoint is split into its two terms,

    NLL_p = W_p + S_p,   W_p = mean over the process's validation events of mean_out (y - mu)^2 / (2 sigma^2),
                         S_p = mean of mean_out log sigma,
    MSE_p = mean of mean_out (y - mu)^2              (all in the standardized target, as the training loss),

and the sweep's objective is val_loss_no_reg = mean_p NLL_p (HETEROSC aggregates by the arithmetic mean). So
NLL = W + S with W = mean_p W_p, S = mean_p S_p, and the logged val_mse = mean_p MSE_p.

    site submit ccin2p3 FA scripts/job_ws_het_split.sh --item warm_start   (GPU: the run part below)
    python3 analysis/catalog_v4/ws_het_split.py fetch                 -> data/ws_hpo8/trial_curves.json (login node)
    python3 analysis/catalog_v4/ws_het_split.py summary --log <job log> -> data/ws_hpo8/het_split.json, figs/ws_het_split.{png,pdf}

run (on a site, GPU): the experiment is built once from the first trial's saved config (the trials differ only in the
searched fine-tune HPs, checked: their data blocks are equal), each trial's best checkpoint (the EMA weights it was
validated with, rebuild_run.load_best_weights) loaded in turn, and its validation split read event by event; also the
untuned k=533 warm start (the sweep's base run) and the parent (the model every trial starts from). One WS_SPLIT line
per model: per process n, NLL_p, W_p, S_p, MSE_p. The reread NLL is checked against the trial's logged val_loss.
fetch (laptop, site_python): each trial's logged val_loss, best step, val_curve, val_mse_curve and its searched HPs.
summary (laptop): per model W, S, NLL, MSE; across the 8 trials the share of the NLL spread carried by each term
(var and the ranking: Spearman of W, S and MSE against NLL), and whether the trial that is best on MSE is the
trial that is best on NLL.
"""
import argparse, glob, json, os, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data", "ws_hpo8")
FIGS = os.path.join(HERE, "figs")
SWEEP = "ws_c1_k533_hpo8"
UNTUNED = "runs/catalog_v4_short/v4_het_warm25_k533"
HPS = ("fine_tune.lr_scale", "fine_tune.layer_decay", "training.muon_weight_decay", "training.regularization_lambda",
       "training.cosanneal_warmup_frac", "training.cosanneal_eta_min", "ema", "training.ema_decay")


def run(a):
    import torch
    from omegaconf import OmegaConf, open_dict
    sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
    from rebuild_run import _build, load_best_weights
    trials = sorted(glob.glob(os.path.join(ROOT, "runs", SWEEP, "trial_*")))
    models = [(os.path.basename(t), t) for t in trials] + [("untuned_k533", os.path.join(ROOT, UNTUNED))]
    cfgs = {n: OmegaConf.load(os.path.join(d, "config.yaml")) for n, d in models}
    first = cfgs[models[0][0]]
    for n, c in cfgs.items():       # one data build serves every model only if every data block is the same
        assert OmegaConf.to_container(c.data) == OmegaConf.to_container(first.data), f"{n}: data block differs"
        assert c.seed == first.seed, f"{n}: seed differs"
    parent = str(first.fine_tune.pretrained_path)
    pdir = os.path.dirname(os.path.dirname(parent))
    assert os.path.basename(parent).startswith("model_run0_best.pt"), parent
    models.append(("parent", pdir))
    cfg = OmegaConf.create(OmegaConf.to_container(first))
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.count_flops = False; cfg.ema = False; cfg.log_module_names = False
        cfg.warm_start_idx = None
        cfg.run_dir = tempfile.mkdtemp(prefix="zz_ws_het_split_", dir=os.environ["SCRATCH"])
    exp = _build(cfg, None, None, None)
    exp.ema, exp.train_sampler = None, None
    names = [str(x) for x in exp.cfg.data.dataset]
    P = len(names)
    for tag, d in models:
        ema = load_best_weights(exp.model, d)
        exp.model.to(exp.device, dtype=exp.dtype).eval()
        acc = {k: torch.zeros(P, dtype=torch.float64, device=exp.device) for k in ("n", "w", "s", "mse")}
        with torch.no_grad():
            for data in exp.val_loader:
                y_pred, y, pids, sig, _ = exp._forward_lloca(data)
                sc = torch.clamp(sig, min=1e-15, max=1e5)            # as losses.heteroscedastic_loss
                r2 = (y - y_pred) ** 2
                acc["w"].index_add_(0, pids, (r2 / (2 * sc ** 2)).flatten(1).mean(1).double())
                acc["s"].index_add_(0, pids, torch.log(sc).flatten(1).mean(1).double())
                acc["mse"].index_add_(0, pids, r2.flatten(1).mean(1).double())
                acc["n"].index_add_(0, pids, torch.ones(len(pids), dtype=torch.float64, device=exp.device))
        acc = {k: v.cpu().numpy() for k, v in acc.items()}
        n = np.maximum(acc["n"], 1)
        rec = {"model": tag, "run_dir": d, "ema": bool(ema), "names": names, "n": acc["n"].astype(int).tolist(),
               "W": (acc["w"] / n).tolist(), "S": (acc["s"] / n).tolist(), "MSE": (acc["mse"] / n).tolist()}
        print("WS_SPLIT " + json.dumps(rec), flush=True)
    import shutil
    shutil.rmtree(cfg.run_dir, ignore_errors=True)


def fetch(a):
    sys.path.insert(0, ROOT)
    from analysis.common import site_python
    script = r'''
import glob, json, os, sys, yaml
out = {}
for f in sorted(glob.glob("sweeps/%s/results/*.json")):
    d = json.load(open(f)); hp = os.path.basename(f).split("_")[0]
    cfg = yaml.safe_load(open("runs/%s/trial_" + hp[2:] + "/config.yaml"))
    def get(k):
        x = cfg
        for p in k.split("."):
            x = x.get(p) if isinstance(x, dict) else None
        return x
    out["trial_" + hp[2:]] = {"val_loss": d["val_loss"], "best_step": d["best_step"], "validate_every": d["validate_every"],
        "val_curve": d["val_curve"], "val_mse_curve": d["val_mse_curve"], "val_loss_step0": d.get("val_loss_step0"),
        "hps": {k: get(k) for k in %r}}
print("TRIALS " + json.dumps(out))
''' % (SWEEP, SWEEP, HPS)
    out = site_python(script, [], "TRIALS")
    os.makedirs(DATA, exist_ok=True)
    json.dump(out, open(os.path.join(DATA, "trial_curves.json"), "w"), indent=1)
    print(f"{len(out)} trials -> {os.path.join(DATA, 'trial_curves.json')}")


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summary(a):
    tr = json.load(open(os.path.join(DATA, "trial_curves.json")))
    recs = {}
    for line in open(a.log):
        if line.startswith("WS_SPLIT "):
            r = json.loads(line.split(" ", 1)[1]); recs[r["model"]] = r
    rows = {}
    for tag, r in recs.items():
        n = np.array(r["n"]); keep = n > 0
        W, S, M = (np.array(r[k])[keep] for k in ("W", "S", "MSE"))
        rows[tag] = {"W": float(W.mean()), "S": float(S.mean()), "NLL": float((W + S).mean()), "MSE": float(M.mean()),
                     "n_proc": int(keep.sum())}
        if tag in tr:
            t = tr[tag]; i = t["best_step"] // t["validate_every"] - 1
            rows[tag].update(logged_nll=t["val_loss"], logged_mse_at_best=t["val_mse_curve"][i],
                             min_logged_mse=float(np.min(t["val_mse_curve"])), lr_scale=t["hps"]["fine_tune.lr_scale"])
    T = sorted(k for k in rows if k in tr)
    nll = np.array([rows[k]["NLL"] for k in T]); W = np.array([rows[k]["W"] for k in T])
    S = np.array([rows[k]["S"] for k in T]); M = np.array([rows[k]["MSE"] for k in T])
    lr = np.array([rows[k]["lr_scale"] for k in T])
    dn, dw, ds = nll - nll.mean(), W - W.mean(), S - S.mean()
    out = {"rows": rows, "trials": T,
           "share_var_W": float(np.dot(dw, dn) / np.dot(dn, dn)), "share_var_S": float(np.dot(ds, dn) / np.dot(dn, dn)),
           "spearman_vs_nll": {"W": spearman(W, nll), "S": spearman(S, nll), "MSE": spearman(M, nll)},
           "spearman_vs_lr": {"NLL": spearman(nll, lr), "MSE": spearman(M, lr), "W": spearman(W, lr), "S": spearman(S, lr)},
           "best_nll": T[int(np.argmin(nll))], "best_mse": T[int(np.argmin(M))],
           "max_abs_reread_minus_logged": float(max(abs(rows[k]["NLL"] - rows[k]["logged_nll"]) for k in T))}
    # the reread reproduces what was logged: each trial's best val_loss_no_reg, and the parent's the trials' step-0 one
    step0 = float(np.median([tr[k]["val_loss_step0"] for k in T]))
    out["parent_reread_minus_step0"] = rows["parent"]["NLL"] - step0 if "parent" in rows else None
    out["reproduced"] = bool(out["max_abs_reread_minus_logged"] < 2e-3
                             and (out["parent_reread_minus_step0"] is None or abs(out["parent_reread_minus_step0"]) < 2e-3))
    json.dump(out, open(os.path.join(DATA, "het_split.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for k in sorted(rows, key=lambda k: rows[k].get("lr_scale", 99)):
        r = rows[k]
        print(f"{k:14s} lr_scale {r.get('lr_scale', float('nan')):6.3f}  NLL {r['NLL']:.4f} (logged {r.get('logged_nll', float('nan')):.4f})"
              f"  W {r['W']:.4f}  S {r['S']:.4f}  MSE {r['MSE']:.5f}")
    # figure: per trial against lr_scale, NLL and its two terms (left), MSE of the mean (right); untuned and parent
    sys.path.insert(0, ROOT)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style  # noqa: F401  (the project's figure style)
    o = np.argsort(lr)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.plot(lr[o], dn[o], "o-", label="NLL (trial minus trial mean)")
    a1.plot(lr[o], dw[o], "s--", label=r"$W=\langle r^2/2\sigma^2\rangle$")
    a1.plot(lr[o], ds[o], "^--", label=r"$S=\langle\log\sigma\rangle$")
    a1.set_xscale("log"); a1.set_xlabel("fine_tune.lr_scale"); a1.set_ylabel("deviation from the 8-trial mean")
    a1.legend(fontsize=7)
    a2.plot(lr[o], M[o], "o-", label="trials")
    for tag in ("untuned_k533", "parent"):
        if tag in rows:
            a2.axhline(rows[tag]["MSE"], ls=":", label=f"{tag} (lr_scale 1)" if tag != "parent" else "parent (step 0)")
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlabel("fine_tune.lr_scale"); a2.set_ylabel("validation MSE of the mean")
    a2.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"ws_het_split.{ext}"), dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("run"); sp.add_parser("fetch")
    s = sp.add_parser("summary"); s.add_argument("--log", required=True)
    a = ap.parse_args()
    {"run": run, "fetch": fetch, "summary": summary}[a.cmd](a)
