"""The level item's design (ii) with the network in place of the exact parent (D99): the network's own 2->3 prediction
on each 2->4 probe's mapped events. tools/factorised_2to4.py maps a probe's first --n validation events onto its 2->3
parent's phase space (the Catani-Seymour initial-initial map) and labels them with the parent's exact |M|^2 (t3); here
the pretraining (--parent, a model.context_dim run: the parent process is a training process and keeps its own c_p from
the table) predicts the parent at those mapped events. The parent's recipe entry (recipes/catalog_daniel_train.yaml) is
built as tools/kshot_context.py builds a probe (the run's own config, only the recipe swapped, the parent's own target
statistics as a representation inverted exactly), with its validation pool replaced by the mapped events
(rows: mapped momenta, the parent's PDG ids, t3; rebuild_run._build's pool swap) with the probe events' alpha_s as
its sidecar. Saved per probe:
e3 = q - t, the network's error on log(|M|^2 F) at the mapped events in pool row order (F is a function of the momenta,
ids and alpha_s alone, all shared by q and t, so e3 is also its error on log|M|^2), and t. The 2->4 prediction of design (ii) is then
log t3 + e3 + h, h the correction (analysis/catalog_daniel/factorised_2to4.py). Needs a GPU.
    python tools/factorised_2to4_net.py --parent <run dir> --f24 <factorised_2to4 npz dir> [--n 2000] [--out DIR]
"""
import argparse, json, os, sys, tempfile

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import siteconf  # noqa: E402
import mg5_pipeline_final as mg  # noqa: E402
from rebuild_run import _build, load_best_weights  # noqa: E402
from kshot_context import cat_batches, parent_frame  # noqa: E402
from factorised_2to4 import beam_recoil_map, ii_map, pool, to_parent_order, v3_probes  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--f24", default=os.path.join(os.environ.get("SCRATCH", "."), "zz_factorised_2to4"))
    ap.add_argument("--recipe", default=os.path.join(ROOT, "recipes", "catalog_2to4_heldout_orders.yaml"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", "."), "zz_factorised_2to4_net"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--v3-nfinal", type=int, default=0,
                    help="catalog_v3: its probes of this multiplicity (factorised_2to4.py --v3 --nfinal), the parents' "
                         "entries from recipes/catalog_v3_train.yaml")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parent_run = os.path.abspath(a.parent)
    pstats = json.load(open(os.path.join(parent_run, "data_stats.json")))
    tr = yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_train.yaml" if a.v3_nfinal
                                          else "catalog_daniel_train.yaml")))
    top = {k: v for k, v in tr.items() if k != "processes"}     # catalog_v3's sampling: mixture
    train = {e["name"]: e for e in tr["processes"]}
    rdir = tempfile.mkdtemp(prefix="f24net_", dir=os.environ["SCRATCH"])
    if a.v3_nfinal:
        jobs = v3_probes(a.v3_nfinal)
    else:
        jobs = [(e["name"], mg.PROCESSES[e["name"]]["parent"], None) for e in yaml.safe_load(open(a.recipe))["processes"]]
    for p, par, extra in jobs:
        if a.only and p not in a.only:
            continue
        if par not in train:                               # a parent that is not a training process: listed
            print("F24NET " + json.dumps(dict(probe=p, parent=par, error="parent not a training process")), flush=True)
            continue
        cfg4 = mg.PROCESSES[p]
        ids4 = [int(i) for i in cfg4["pdg_ids"]]
        extra = ids4[-1] if extra is None else extra
        z = np.load(os.path.join(a.f24, p + ".npz"))
        X, al, _ = pool(p)
        X = X[: a.n]
        N = len(ids4)
        P = X[:, : 4 * N].reshape(-1, N, 4).astype(np.float64)
        cand = [j for j in range(2, N) if ids4[j] == extra]
        pt = np.sqrt(P[:, cand, 1] ** 2 + P[:, cand, 2] ** 2)
        k_slot = np.asarray(cand)[pt.argmin(1)]
        # the same map and row order as factorised_2to4.py, so t3 there labels these rows
        Pm, _, _ = (beam_recoil_map if any(abs(q) == 5 for q in ids4[:2]) else ii_map)(P, k_slot)
        if a.v3_nfinal:
            ids3l = [int(i) for i in mg.PROCESSES[par]["pdg_ids"]]
            Pm = np.stack([to_parent_order(Pm[i:i + 1], [ids4[j] for j in range(N) if j != int(k_slot[i])], ids3l)[0]
                           for i in range(len(Pm))])
        t3 = np.asarray(z["t3"], np.float64)
        n = len(Pm)
        ids3 = np.asarray(mg.PROCESSES[par]["pdg_ids"], np.float64)
        rows = np.concatenate([Pm.reshape(n, -1), np.broadcast_to(ids3, (n, len(ids3))), t3[:, None]], 1)
        alphas = None if al is None else np.asarray(al[: a.n], np.float64)
        Xp, _, _ = pool(par)
        assert Xp.shape[1] == rows.shape[1], (p, Xp.shape, rows.shape)
        vpath = os.path.join(rdir, f"{p}_mapped_val_amplitudes.npy")   # the parent's val pool, swapped in by _build
        np.save(vpath, rows)
        if alphas is not None:
            np.save(vpath[:-len(".npy")] + ".alphas.npy", alphas)
        rec_path = os.path.join(rdir, par + ".yaml")
        yaml.safe_dump({**top, "processes": [train[par]]}, open(rec_path, "w"))
        cfg = OmegaConf.load(os.path.join(parent_run, "config.yaml"))
        with open_dict(cfg):
            cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
            cfg.use_mlflow = False; cfg.count_flops = False; cfg.ema = False
            cfg.warm_start_idx = None
            cfg.training.validate_frac = 0; cfg.training.validate_every_n_steps = 1
            cfg.run_dir = tempfile.mkdtemp(prefix="f24net_run_", dir=os.environ["SCRATCH"])
            cfg.data.source = "recipes"; cfg.data.processes_file = rec_path
            cfg.data.train_subsample = 10000; cfg.data.memmap = False; cfg.data.eval_subsample = n
            cfg.data.shared_standardization = False; cfg.data.affine_multiplicity_std = False
            cfg.fine_tune.pretrained_path = os.path.join(parent_run, "models", "model_run0_best.pt")
            cfg.fine_tune.target_stats = "own"; cfg.fine_tune.offshell_stats = "parent"
            if int(cfg.model.get("context_dim", 0) or 0):
                cfg.model.context_names_file = os.path.join(parent_run, "context_names.json")
        exp = _build(cfg, "val", vpath, par)
        load_best_weights(exp.model, parent_run)
        model = exp.model.to(exp.device, dtype=exp.dtype).eval()
        if hasattr(model, "context_override"):
            model.context_override = None                  # the parent's own c_p from the table
        val = cat_batches(list(exp.val_loader))
        vi = np.asarray(exp._split_indices["val"])
        order = vi - vi.min()
        assert sorted(order.tolist()) == list(range(n)), f"{p}: the val split is not the {n} mapped rows"
        mu_p, sd_p = float(exp.prepd_mean[0]), float(exp.prepd_std[0])
        mu_c, sd_c = parent_frame(pstats, int(exp._proc_npart[0]))
        t = (mu_p + sd_p * val[1].reshape(-1).double()).numpy()
        with torch.no_grad():
            q = (mu_c + sd_c * exp._forward_lloca(val)[0].reshape(-1).double().cpu()).numpy()
        e3 = np.empty(n); tt = np.empty(n)
        e3[order] = q - t; tt[order] = t
        has_row = bool(int(model._ctx_row[0]) >= 0) if hasattr(model, "_ctx_row") else None
        np.savez_compressed(os.path.join(a.out, p + ".npz"), e3=e3, t=tt)
        print("F24NET " + json.dumps(dict(probe=p, parent=par, n=n, has_row=has_row, mse3=float(np.mean(e3 ** 2)),
                                          var3=float(np.var(e3)), mean3=float(np.mean(e3)))), flush=True)


if __name__ == "__main__":
    main()
