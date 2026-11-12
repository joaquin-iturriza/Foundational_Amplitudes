#!/usr/bin/env python3
"""Eval-only probe of a KIT transformer run: what the network sees on each stratum's events, against what it saw in
training (bulk_train), and the error it makes there. Per event (n per stratum, event-frame rebuild as kit_predict):
  lam      log10 of the largest |entry| of the event's local frames (a frame boosting by gamma has entries ~gamma;
           ill-conditioned frames read large)
  feat     the largest |standardized local-frame momentum component| the transformer reads (features_local)
  sij_err  the largest relative error of s_ij/s computed from the momenta as the network receives them (its dtype,
           after the rebuild's preprocessing) against float64 from the stored momenta (nan for a soft-rescaled run)
  res      |prediction - truth| in log10|M|^2
and the dtypes. Per stratum: medians and maxima, and feat over bulk_train's 99.5% quantile.

    python tools/kit_transformer_probe.py <run_dir> [--n 200] --out <json>      (CPU, in the site checkout)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import kit_hardps as kh  # noqa: E402
from tools.kit_predict import run_target  # noqa: E402


def sij_over_s(p):
    """(N, n, 4) -> (N, pairs) s_ij/s, s = (p0 + p1)^2, s_ij = (p_i + p_j)^2"""
    n = p.shape[1]
    iu = np.triu_indices(n, 1)
    q = p[:, iu[0]] + p[:, iu[1]]
    m2 = lambda v: v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1)
    return m2(q) / m2(p[:, 0] + p[:, 1])[:, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir"); ap.add_argument("--n", type=int, default=200); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from tools.rebuild_run import rebuild
    torch.set_num_threads(4)
    stem, target, _ = kh.parse(run_target(a.run_dir))
    blocks, rows = [], []
    for split in ("bulk_train", "bulk_test", "hard_test"):
        for st in kh.strata(stem, split):
            r = kh.stratum_rows(stem, target, split, st)
            r = r[:2000] if split == "bulk_train" else r[:a.n]
            blocks.append((f"{split}/{st}", len(r))); rows.append(r)
    rows = np.concatenate(rows)
    out_dir = os.path.dirname(os.path.abspath(a.out)); os.makedirs(out_dir, exist_ok=True)
    pool = os.path.join(out_dir, f"probe_{os.getpid()}.npy")
    np.save(pool, rows)
    import h5py
    with h5py.File(kh.h5_path(stem), "r") as f:
        np.save(pool[:-4] + ".alphas.npy", np.full(len(rows), float(json.loads(f.attrs["metadata"])["alpha_s"])))
    import yaml
    if yaml.safe_load(open(os.path.join(a.run_dir, "config.yaml")))["data"].get("target_over_born"):
        kh.write_born(np.concatenate([kh.stratum_rows(stem, "born", *b[0].split("/"))[:b[1]][:, -1]
                                      for b in blocks]) * kh.strip_factor(kh.metadata(stem), "born"), pool)
    exp, loader = rebuild(a.run_dir, "test", pool, len(rows), event_frame=True)
    net = exp.model.net
    cap = []

    def pre(mod, args, kwargs):
        cap.append({"fm": args[0].detach(), "mean": args[2], "std": args[3], "ptr": kwargs["ptr"].detach()})

    def post(mod, args, out):
        cap[-1]["frames"] = out
    h1 = net.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = net.framesnet.register_forward_hook(post)
    with torch.no_grad():
        pred, truth, _ = exp._collect_predictions(loader)
        lam, feat, sij, dt = [], [], [], set()
        for c in cap:
            fr = c["frames"]; M = fr.matrices if hasattr(fr, "matrices") else fr
            loc = (net.trafo_fourmomenta(c["fm"], fr) - c["mean"]) / c["std"]
            ptr = c["ptr"].cpu().numpy(); cnt = np.diff(ptr)
            ev = np.repeat(np.arange(len(cnt)), cnt)
            dt |= {str(c["fm"].dtype), str(loc.dtype), str(M.dtype)}
            lm = torch.log10(M.abs().flatten(1).max(1).values.clamp(min=1e-300)).cpu().numpy()
            fl = loc.abs().max(1).values.cpu().numpy()
            for arr, dst in ((lm, lam), (fl, feat)):
                v = np.full(len(cnt), -np.inf); np.maximum.at(v, ev, arr); dst.append(v)
            fm = c["fm"].cpu().numpy(); n = int(cnt.max())
            sij.append(np.stack([sij_over_s(fm[ptr[i]:ptr[i + 1]][None])[0] for i in range(len(cnt))]))
    h1.remove(); h2.remove()
    lam, feat, sij = np.concatenate(lam), np.concatenate(feat), np.concatenate(sij)
    perm = np.asarray(exp._role_perm["test"])[:len(lam)]
    res = np.abs(np.asarray(pred, np.float64).reshape(-1) - np.asarray(truth, np.float64).reshape(-1)) \
        * float(exp.prepd_std[0]) / np.log(10.0)
    o = np.empty(len(rows), int); o[perm] = np.arange(len(perm))     # row -> loader position
    npart = sij.shape[1]
    nfin = (rows.shape[1] - 1) // 5
    truth_s = sij_over_s(rows[:, :4 * nfin].reshape(-1, nfin, 4).astype(np.float64))
    err = np.abs(sij[o] / truth_s - 1).max(1)
    if getattr(exp.model, "input_soft_eps", 0):    # the net reads soft-rescaled vectors: s_ij/s is not theirs to keep
        err = np.full(len(err), np.nan)
    out, i = {"run": os.path.abspath(a.run_dir), "dtypes": sorted(dt), "strata": {}}, 0
    q995 = None
    for name, n in blocks:
        k = o[i:i + n]; i += n
        e = {"n": n, "lam_median": float(np.median(lam[k])), "lam_max": float(lam[k].max()),
             "feat_median": float(np.median(feat[k])), "feat_max": float(feat[k].max()),
             "sij_err_max": float(err[i - n:i].max()), "res_median": float(np.median(res[k])),
             "logmse": float(np.mean(res[k] ** 2)),
             "corr_res_lam": float(np.corrcoef(res[k], lam[k])[0, 1]) if lam[k].std() > 0 else None,
             "corr_res_feat": float(np.corrcoef(res[k], feat[k])[0, 1]) if feat[k].std() > 0 else None}
        if name.startswith("bulk_train"):
            q995 = float(np.quantile(feat[k], 0.995))
        e["feat_over_train_q995"] = float(np.median(feat[k]) / q995) if q995 else None
        out["strata"][name] = e
        print(f"{name:45s} lam med {e['lam_median']:6.2f} max {e['lam_max']:6.2f} | feat med {e['feat_median']:9.3g} "
              f"max {e['feat_max']:9.3g} (x{e['feat_over_train_q995'] or 0:.2g} train q995) | sij err {e['sij_err_max']:.1e} "
              f"| res med {e['res_median']:.3g} logmse {e['logmse']:.3g} | corr lam {e['corr_res_lam'] or 0:+.2f} "
              f"feat {e['corr_res_feat'] or 0:+.2f}")
    print("dtypes", sorted(dt))
    json.dump(out, open(a.out, "w"), indent=1)
    for f in (pool, pool[:-4] + ".alphas.npy"):
        os.remove(f)


if __name__ == "__main__":
    main()
