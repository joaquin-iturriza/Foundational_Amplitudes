#!/usr/bin/env python3
"""Eval-only probe of a KIT transformer run: what the network sees on each stratum's events, against what it saw in
training (bulk_train), and the error it makes there. Per event (n per stratum, event-frame rebuild as kit_predict):
  lam      log10 of the largest |entry| of the event's local frames (a frame boosting by gamma has entries ~gamma;
           ill-conditioned frames read large)
  feat     the largest |standardized local-frame momentum component| the transformer reads (features_local)
  sij_err  the largest relative error of s_ij/s computed from the momenta as the network receives them (its dtype,
           after the rebuild's preprocessing) against float64 from the stored momenta (nan for a soft-rescaled run)
  gam      log10 of the largest boost factor gamma = Lambda_00 of the event's local frames (D411 frames mechanism)
  gamrel   log10 of the largest gamma of the relative transforms Lambda_i Lambda_j^-1 over particle pairs: what the
           attention's vector messages from j to i are transformed by (the route model.input_route_off=frames cuts)
  res      |prediction - truth| in log10|M|^2
and the dtypes. Per stratum: medians and maxima, and feat over bulk_train's 99.5% quantile.

    python tools/kit_transformer_probe.py <run_dir> [--n 200] [--out <json>]    (CPU, in the site checkout; prints KITPROBE <json>)
    python tools/kit_transformer_probe.py --plot <uug json> <tt json> --figs <dir>   (local: the frames-boost figure)
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


def plot(paths, figs):
    """kit_frames_boost_{a,b}: per stratum median log10 gamma of the frames (solid) and of their relative transforms
    (dashed) against the stratum's distance variable, uug collinear y and soft x (a), tt virtual sqrt s (b); the grey band
    is bulk_train's median to maximum of gamma, the dotted line its maximum of the relative gamma; signed error beside"""
    import plot_style as ps
    from tools.kit_figures import edges
    ps.use()
    panels = ps.panels(2)
    for (fig, ax), path, fams in zip(panels, paths, ((("coll_y", ps.C.vermillion, "collinear $y$"),
                                                       ("soft_x", ps.C.blue, "soft $x$")),
                                                      (("sudakov_sqrts", ps.C.vermillion, r"$\sqrt{s}$ [GeV]"),))):
        d = json.load(open(path))["strata"]
        tr = d["bulk_train/bulk"]
        ax.axhspan(tr["gam_median"], tr["gam_max"], color=ps.C.grey, alpha=0.25, lw=0, label=r"bulk train $\gamma$")
        ax.axhline(tr["gamrel_max"], color=ps.C.grey, ls=":", lw=1, label=r"bulk train max rel. $\gamma$")
        for fam, c, lab in fams:
            ks = sorted((k for k in d if k.startswith("hard_test/" + fam)), key=lambda k: edges(k.split("/")[1])[0])
            x = [np.sqrt(np.prod(edges(k.split("/")[1]))) for k in ks]
            ax.plot(x, [d[k]["gam_median"] for k in ks], "o-", color=c, label=r"$\gamma$, " + lab)
            ax.plot(x, [d[k]["gamrel_median"] for k in ks], "s--", color=c, label=r"rel. $\gamma$, " + lab)
        ax.set_xscale("log")
        from matplotlib.ticker import NullFormatter
        ax.xaxis.set_minor_formatter(NullFormatter())      # the sqrt s axis spans ~1 decade: minor labels overlap
        ax.set_xlabel("distance variable of the stratum")
        ax.set_ylabel(r"median $\log_{10}\gamma$ of the frames")
        ps.legend(ax, "upper right" if "uug" in path else "upper left")
    ps.save_panels(panels, os.path.join(figs, "kit_frames_boost"))


def main():
    if "--plot" in sys.argv:
        i = sys.argv.index("--plot")
        return plot(sys.argv[i + 1:i + 3], sys.argv[sys.argv.index("--figs") + 1])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir"); ap.add_argument("--n", type=int, default=200); ap.add_argument("--out")
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
    import atexit, shutil, tempfile
    tmp = tempfile.mkdtemp()                       # never inside the checkout; removed however the probe ends
    atexit.register(shutil.rmtree, tmp, True)
    pool = os.path.join(tmp, f"probe_{os.getpid()}.npy")
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
        lam, feat, sij, gam, gamrel, dt = [], [], [], [], [], set()
        for c in cap:
            fr = c["frames"]; M = fr.matrices if hasattr(fr, "matrices") else fr
            loc = (net.trafo_fourmomenta(c["fm"], fr) - c["mean"]) / c["std"]
            ptr = c["ptr"].cpu().numpy(); cnt = np.diff(ptr)
            ev = np.repeat(np.arange(len(cnt)), cnt)
            dt |= {str(c["fm"].dtype), str(loc.dtype), str(M.dtype)}
            lm = torch.log10(M.abs().flatten(1).max(1).values.clamp(min=1e-300)).cpu().numpy()
            fl = loc.abs().max(1).values.cpu().numpy()
            g0 = torch.log10(M[:, 0, 0].abs()).cpu().numpy()
            for arr, dst in ((lm, lam), (fl, feat), (g0, gam)):
                v = np.full(len(cnt), -np.inf); np.maximum.at(v, ev, arr); dst.append(v)
            Mi = fr.inv if hasattr(fr, "inv") else torch.linalg.inv(M)
            gr = np.empty(len(cnt))
            for e in range(len(cnt)):                 # (Lambda_i Lambda_j^-1)_00 over all pairs of the event
                a_, b_ = ptr[e], ptr[e + 1]
                gr[e] = torch.einsum("ia,ja->ij", M[a_:b_, 0, :], Mi[a_:b_, :, 0]).abs().max().log10().item()
            gamrel.append(gr)
            fm = c["fm"].cpu().numpy(); n = int(cnt.max())
            sij.append(np.stack([sij_over_s(fm[ptr[i]:ptr[i + 1]][None])[0] for i in range(len(cnt))]))
    h1.remove(); h2.remove()
    lam, feat, sij = np.concatenate(lam), np.concatenate(feat), np.concatenate(sij)
    gam, gamrel = np.concatenate(gam), np.concatenate(gamrel)
    perm = np.asarray(exp._role_perm["test"])[:len(lam)]
    sgn = (np.asarray(pred, np.float64).reshape(-1) - np.asarray(truth, np.float64).reshape(-1)) \
        * float(exp.prepd_std[0]) / np.log(10.0)       # prediction - truth, log10|M|^2
    res = np.abs(sgn)
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
             "logmse": float(np.mean(res[k] ** 2)), "err_median": float(np.median(sgn[k])),
             **{f"{q}_{f}": float(fn(v[k])) for q, v in (("gam", gam), ("gamrel", gamrel))
                for f, fn in (("median", np.median), ("q90", lambda x: np.quantile(x, 0.9)), ("max", np.max))},
             "corr_res_lam": float(np.corrcoef(res[k], lam[k])[0, 1]) if lam[k].std() > 0 else None,
             "corr_res_feat": float(np.corrcoef(res[k], feat[k])[0, 1]) if feat[k].std() > 0 else None}
        if name.startswith("bulk_train"):
            q995 = float(np.quantile(feat[k], 0.995))
            tr = {q: v[k] for q, v in (("gam", gam), ("gamrel", gamrel), ("feat", feat))}
        for q, v in (("gam", gam), ("gamrel", gamrel), ("feat", feat)):   # share of bulk_train events above the median
            e[f"{q}_train_share_above"] = float(np.mean(tr[q] >= np.median(v[k])))
        e["feat_over_train_q995"] = float(np.median(feat[k]) / q995) if q995 else None
        out["strata"][name] = e
        print(f"{name:45s} lam med {e['lam_median']:6.2f} max {e['lam_max']:6.2f} | feat med {e['feat_median']:9.3g} "
              f"max {e['feat_max']:9.3g} (x{e['feat_over_train_q995'] or 0:.2g} train q995) | sij err {e['sij_err_max']:.1e} "
              f"| res med {e['res_median']:.3g} logmse {e['logmse']:.3g} | corr lam {e['corr_res_lam'] or 0:+.2f} "
              f"feat {e['corr_res_feat'] or 0:+.2f} | log10 gamma med {e['gam_median']:.2f} max {e['gam_max']:.2f} "
              f"rel med {e['gamrel_median']:.2f} max {e['gamrel_max']:.2f}")
    print("dtypes", sorted(dt))
    print("KITPROBE " + json.dumps(out))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
