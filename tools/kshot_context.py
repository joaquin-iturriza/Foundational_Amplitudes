"""K-shot fit of a held-out process's context vector (D105, the Level item's design (i); CAVIA, arXiv:1909.09157).
A pretraining run with model.context_dim (a per-process c_p added to every particle's encoded properties, zeroed with
probability model.context_dropout in training, so c = 0 is the zero-shot model) is built for each probe as
tools/zero_shot_parent.py builds it (the parent's own config, only the recipe swapped, the probe's own target
statistics as a representation inverted exactly, the parent's best weights). Everything is frozen; only one vector c
(the wrapper's context_override, started at 0) is fitted by Adam on K events drawn from the probe's validation split,
MSE in log(|M|^2 F) through the parent's statistics, and the probe is scored on its OTHER validation events.
Per probe and K in --ks, --draws seeded random draws: the MSE of log|M|^2 after the fit and, on the same draws, the
plain K-event offset (the mean error of the zero-shot prediction on the K events, subtracted). The shrunk offset of
fig:kshot_level is computed offline from the saved zero-shot q0 and the draws (kshot_level.py's estimator). Prints
one KSHOT JSON line per probe (with K = 0, the zero-shot read) and, with --save-dir, <probe>.npz: t, q0, the draws'
indices and their post-fit and plain-offset MSEs. A GPU for speed (runs on CPU too).
    python tools/kshot_context.py --parent <pretraining run dir> [--ks 1 4 16] [--draws 5] [--steps 300]
        [--lr 0.05] [--save-dir D] <probe recipe.yaml> [...]
"""
import argparse, json, os, sys, tempfile

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import siteconf  # noqa: E402
from rebuild_run import _build, load_best_weights  # noqa: E402


def subset(batch, idx):
    """The events idx (LongTensor) of one collated batch (particles, y, tokens, order_labels, ptr, process_ids)."""
    particles, y, tokens, order_labels, ptr, pids = batch
    counts = ptr[1:] - ptr[:-1]
    sel = torch.cat([torch.arange(int(ptr[i]), int(ptr[i + 1])) for i in idx.tolist()])
    nptr = torch.zeros(len(idx) + 1, dtype=torch.long)
    torch.cumsum(counts[idx], dim=0, out=nptr[1:])
    return particles[sel], y[idx], tokens[sel], order_labels[idx], nptr, pids[idx]


def cat_batches(batches):
    """One collated batch from several (a probe's whole validation split)."""
    parts, ys, toks, ords, ptrs, pids = zip(*batches)
    off, ptr = 0, [torch.zeros(1, dtype=torch.long)]
    for p in ptrs:
        ptr.append(p[1:] + off)
        off += int(p[-1])
    return torch.cat(parts), torch.cat(ys), torch.cat(toks), torch.cat(ords), torch.cat(ptr), torch.cat(pids)


def parent_frame(pstats, N):
    """(mu_c, sd_c) mapping the parent's standardized output to log(|M|^2 F), as zero_shot_parent.py reads it."""
    if pstats.get("affine_multiplicity"):
        am = pstats["affine_multiplicity"]
        return float(am["a"]) + float(am["b"]) * N, float(am["sigma"])
    if pstats.get("shared_standardization"):
        return float(pstats["prepd_mean"][0]), float(pstats["prepd_std"][0])
    raise SystemExit("a per-process parent has no zero-shot frame: kshot_context needs shared or affine statistics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="pretraining run dir with model.context_dim (config.yaml, "
                                                    "data_stats.json, context_names.json, models/)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--draws", type=int, default=5, help="random K-event draws per (probe, K)")
    ap.add_argument("--steps", type=int, default=300, help="Adam steps on c per draw")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-subsample", type=int, default=10000)
    ap.add_argument("--eval-subsample", type=int, default=2000)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("recipes", nargs="+")
    a = ap.parse_args()

    parent = os.path.abspath(a.parent)
    pstats = json.load(open(os.path.join(parent, "data_stats.json")))
    ckpt = os.path.join(parent, "models", "model_run0_best.pt")
    pcfg = OmegaConf.load(os.path.join(parent, "config.yaml"))
    assert int(pcfg.model.get("context_dim", 0) or 0), f"{parent} was not trained with model.context_dim"
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
    for recipe in a.recipes:
        recipe = os.path.abspath(siteconf._expand_str(recipe))
        cfg = OmegaConf.load(os.path.join(parent, "config.yaml"))
        with open_dict(cfg):
            cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
            cfg.use_mlflow = False; cfg.count_flops = False; cfg.ema = False
            cfg.warm_start_idx = None
            cfg.training.validate_frac = 0; cfg.training.validate_every_n_steps = 1
            cfg.run_dir = tempfile.mkdtemp(prefix="kshot_", dir=os.environ["SCRATCH"])
            cfg.data.source = "recipes"
            cfg.data.processes_file = recipe
            cfg.data.train_subsample = a.train_subsample
            cfg.data.memmap = False
            cfg.data.eval_subsample = a.eval_subsample
            cfg.data.shared_standardization = False
            cfg.data.affine_multiplicity_std = False
            cfg.fine_tune.pretrained_path = ckpt
            cfg.fine_tune.target_stats = "own"
            cfg.fine_tune.offshell_stats = "parent"
            cfg.model.context_names_file = os.path.join(parent, "context_names.json")
        try:
            exp = _build(cfg, None, None, None)
        except Exception as e:                          # a missing pool, or an entry the target cannot take: listed
            print("KSHOT " + json.dumps({"recipe": recipe, "parent": parent, "error": repr(e)[:300]}), flush=True)
            continue
        assert len(exp.cfg.data.dataset) == 1, f"{recipe}: one probe process per recipe"
        load_best_weights(exp.model, parent)
        probe = str(exp.cfg.data.dataset[0])
        N = int(exp._proc_npart[0])
        model = exp.model.to(exp.device, dtype=exp.dtype).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        exp.ema, exp.train_sampler = None, None
        exp._init_regularization()
        val = cat_batches(list(exp.val_loader))
        mu_p, sd_p = float(exp.prepd_mean[0]), float(exp.prepd_std[0])
        mu_c, sd_c = parent_frame(pstats, N)
        t = (mu_p + sd_p * val[1].reshape(-1).double()).numpy()            # log(|M|^2 F), the probe's truth
        n = len(t)
        tgt = ((torch.as_tensor(t) - mu_c) / sd_c).to(exp.dtype)            # the truth in the parent's output units

        def score_all():
            with torch.no_grad():
                return (mu_c + sd_c * exp._forward_lloca(val)[0].reshape(-1).double().cpu()).numpy()

        dim = int(model.context.shape[1])
        model.context_override = None
        q_table = score_all()                       # c from the table: zero unless the probe's name has a row
        model.context_override = torch.nn.Parameter(torch.zeros(dim, device=exp.device, dtype=exp.dtype))
        qz = score_all()                            # c = 0: the zero-shot model
        rec = {"probe": probe, "n_particles": N, "recipe": recipe, "parent": parent, "n_val": n,
               "var_logm2": float(np.var(t)), "mse_zero": float(np.mean((qz - t) ** 2)),
               "offset_zero": float(np.mean(qz - t)), "shape_zero": float(np.var(qz - t)),
               "has_row": bool(int(model._ctx_row[0]) >= 0), "mse_table": float(np.mean((q_table - t) ** 2)), "ks": {}}
        rng = np.random.default_rng(a.seed)
        saved = {}
        for K in a.ks:
            fit_mse, fit_off, fit_shape, off_mse, picks = [], [], [], [], []
            for _ in range(a.draws):
                pick = np.sort(rng.choice(n, size=K, replace=False))
                rest = np.setdiff1d(np.arange(n), pick)
                off = float(np.mean(qz[pick] - t[pick]))                  # the plain K-event offset, same draw
                off_mse.append(float(np.mean((qz[rest] - off - t[rest]) ** 2)))
                c = model.context_override
                with torch.no_grad():
                    c.zero_()
                opt = torch.optim.Adam([c], lr=a.lr)
                kb = subset(val, torch.as_tensor(pick))
                ky = tgt[torch.as_tensor(pick)].to(exp.device)
                with torch.enable_grad():
                    for _s in range(a.steps):
                        opt.zero_grad()
                        loss = torch.mean((exp._forward_lloca(kb)[0].reshape(-1) - ky) ** 2)
                        loss.backward()
                        opt.step()
                e = score_all()[rest] - t[rest]
                fit_mse.append(float(np.mean(e ** 2))); fit_off.append(float(np.mean(e)))
                fit_shape.append(float(np.var(e))); picks.append(pick)
            # the mean over draws: the expected error after K events, as kshot_level.py's estimator
            rec["ks"][str(K)] = {"mse_fit": float(np.mean(fit_mse)), "offset_fit": float(np.mean(fit_off)),
                                 "shape_fit": float(np.mean(fit_shape)), "mse_offset": float(np.mean(off_mse)),
                                 "draws": a.draws}
            saved.update({f"pick_k{K}": np.array(picks), f"msefit_k{K}": np.array(fit_mse),
                          f"mseoff_k{K}": np.array(off_mse)})
        if a.save_dir:
            np.savez_compressed(os.path.join(a.save_dir, probe + ".npz"), t=t.astype(np.float32),
                                q0=qz.astype(np.float32), **saved)
        model.context_override = None
        print("KSHOT " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
