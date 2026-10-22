"""D83 exploration, sixth pass: a CPU proxy for shape transfer under target preprocessings. Per event (norm_extract.py,
400 events per process from the train pools): inputs = log s_hat, the 9 sorted pair-invariant logs over s_hat, Lg,
Lmin, Lnn, E, plus the process descriptors (stats_net.features and the diagram descriptors of norm_explore3.py),
standardized on the training processes. A 3x128 MLP (torch, CPU, Adam, 2 seeds) is trained on the 1352 training
processes' events to predict the target t; its prediction is turned back into log|M|^2 and scored on the 139 held-out
processes: shape MSE = the variance over a process's events of (prediction - truth) (its level error removed), and on
its own MSE (the level error kept), median and 90% over the 43 new and the 96 relabels. Targets (all with one shared
standardization on top except P):
  B   y = log|M|^2
  P   (y - mu_p) / sd_p with the process's own mean and std (the old per-process preprocessing; needs the process's data)
  A   y + log F, log F = (4 - N)(log s0 - L) - log C_p (A as launched)
  S   y + (4 - N)(log s0 - L) (the s_hat factor alone)
  W   y - c_N L, c_N the within-process slope on L fitted per multiplicity on the training processes
  X   y - sum_k c_{N,k} X_k over (L, Lg, Lmin, Lnn, E), fitted per multiplicity as W (norm_explore2.py's best)
  K   y - E (an eikonal-like count: each massless final leg's log(s_hat / its nearest pair invariant), coefficient 1)
  M   y standardized per multiplicity (mean and std of the training events at each N; data-free for a new process)
  Q   the training events' quantile of y mapped to a standard normal (one shared monotone map, data-free)
  C   B with a few-shot context as an input: the mean and std of y over 10 random events of the process (scored
      on the other events; the network learns to use the level it is given)
Output data/norm_explore6.json.
    .venv-driver/bin/python analysis/catalog_daniel/norm_explore6.py [--epochs 6]
"""
import argparse, json, math, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import stats_net as sn  # noqa: E402

LS0 = math.log(289.0)


def proc_feats(meta, dc, n):
    f = list(sn.features(meta[n]))
    l = np.array(dc[n]["logc2"]); mi = dc[n]["massive_internal"]
    f += [float(l.max()), float(np.log(np.exp(l - l.max()).sum())), float(np.log(len(l))),
          float(mi[int(np.argmax(l))]), float(max(mi))]
    return np.array(f, np.float32)


def within_coefs(Y, F, N, tr, cols):
    """per multiplicity: the regression of y on F[:, cols] within processes (each process centred), training only"""
    out = {}
    for n in sorted(set(N)):
        idx = [i for i in tr if N[i] == n]
        X = np.concatenate([F[i][:, cols] - F[i][:, cols].mean(0) for i in idx])
        y = np.concatenate([Y[i] - Y[i].mean() for i in idx])
        out[n] = np.linalg.lstsq(X, y, rcond=None)[0]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--targets", default="B,P,A,S,W,X,K,M,Q,C")
    ap.add_argument("--out", default="norm_explore6.json")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    z = np.load(os.path.join(HERE, "data", "normx.npz"))
    names, split, N = list(z["names"]), list(z["split"]), z["npart"]
    Y, F = z["y"].astype(np.float64), z["feats"].astype(np.float64)
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    dc = json.load(open(os.path.join(HERE, "data", "diagram_couplings.json")))
    rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    tr = [i for i, s in enumerate(split) if s == "train"]; ho = [i for i, s in enumerate(split) if s == "heldout"]
    PF = np.stack([proc_feats(meta, dc, n) for n in names]).astype(np.float64)
    logc = np.array([max(dc[n]["logc2"]) for n in names])
    L = F[:, :, 0]
    cW = within_coefs(Y, F, N, tr, [0]); cX = within_coefs(Y, F, N, tr, [0, 10, 11, 12, 13])

    def offset(kind):
        """per event: t = y + off (P handled apart)"""
        if kind == "B":
            return np.zeros_like(Y)
        if kind == "A":
            return (4 - N)[:, None] * (LS0 - L) - logc[:, None]
        if kind == "S":
            return (4 - N)[:, None] * (LS0 - L)
        if kind == "W":
            return -np.stack([cW[n][0] * L[i] for i, n in enumerate(N)])
        if kind == "X":
            return -np.stack([F[i][:, [0, 10, 11, 12, 13]] @ cX[n] for i, n in enumerate(N)])
        if kind == "K":
            return -F[:, :, 13]
        raise ValueError(kind)

    ne = Y.shape[1]
    Xin = np.concatenate([F, np.repeat(PF[:, None, :], ne, 1)], 2)
    mu_x, sd_x = Xin[tr].reshape(-1, Xin.shape[2]).mean(0), Xin[tr].reshape(-1, Xin.shape[2]).std(0) + 1e-6
    Xs = ((Xin - mu_x) / sd_x).astype(np.float32)
    out = {"coefs_W": {int(k): v.tolist() for k, v in cW.items()}, "coefs_X": {int(k): v.tolist() for k, v in cX.items()}}
    for kind in a.targets.split(","):
        Xk = Xs
        if kind == "P":
            mp, sp = Y.mean(1, keepdims=True), Y.std(1, keepdims=True)
            T = (Y - mp) / sp; back = lambda pred, i: mp[i] + sp[i] * pred
        elif kind == "M":
            mN = {n: Y[[i for i in tr if N[i] == n]].mean() for n in set(N)}
            sN = {n: Y[[i for i in tr if N[i] == n]].std() for n in set(N)}
            mm = np.array([mN[n] for n in N])[:, None]; ss = np.array([sN[n] for n in N])[:, None]
            T = (Y - mm) / ss; back = lambda pred, i, mm=mm, ss=ss: mm[i] + ss[i] * pred
        elif kind == "Q":
            from scipy.special import ndtri
            srt = np.sort(Y[tr].reshape(-1)); qs = (np.arange(len(srt)) + 0.5) / len(srt)
            T = ndtri(np.interp(Y, srt, qs))
            zq = ndtri(qs)
            back = lambda pred, i, srt=srt, zq=zq: np.interp(pred, zq, srt)
        elif kind == "C":
            rng = np.random.default_rng(0)
            ctx = np.stack([rng.choice(ne, 10, replace=False) for _ in range(len(Y))])
            cm = np.array([Y[i, c].mean() for i, c in enumerate(ctx)]); cs = np.array([Y[i, c].std() for i, c in enumerate(ctx)])
            ext = np.stack([cm, cs], 1)
            ext = (ext - ext[tr].mean(0)) / ext[tr].std(0)
            Xk = np.concatenate([Xs, np.repeat(ext[:, None, :], ne, 1).astype(np.float32)], 2)
            m0, s0 = Y[tr].mean(), Y[tr].std()
            T = (Y - m0) / s0; back = lambda pred, i, m0=m0, s0=s0: m0 + s0 * pred
        else:
            off = offset(kind); T0 = Y + off
            m0, s0 = T0[tr].mean(), T0[tr].std()
            T = (T0 - m0) / s0; back = lambda pred, i, off=off, m0=m0, s0=s0: m0 + s0 * pred - off[i]
        preds = []
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            net = torch.nn.Sequential(torch.nn.Linear(Xk.shape[2], 128), torch.nn.SiLU(), torch.nn.Linear(128, 128),
                                      torch.nn.SiLU(), torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, 1))
            opt = torch.optim.Adam(net.parameters(), 2e-3)
            xt = torch.from_numpy(Xk[tr].reshape(-1, Xk.shape[2])); yt = torch.from_numpy(T[tr].reshape(-1).astype(np.float32))
            steps = a.epochs * (len(xt) // 1024)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
            g = torch.Generator().manual_seed(seed)
            for _ in range(a.epochs):
                perm = torch.randperm(len(xt), generator=g)
                for b in range(len(xt) // 1024):
                    j = perm[b * 1024:(b + 1) * 1024]
                    loss = ((net(xt[j]).squeeze(1) - yt[j]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                preds.append(net(torch.from_numpy(Xk[ho].reshape(-1, Xk.shape[2]))).squeeze(1).numpy().reshape(len(ho), ne))
        pr = np.mean(preds, 0)
        shape, own, sflat, slow, daff = {}, {}, {}, {}, {}
        for r, i in enumerate(ho):
            q_ = back(pr[r].astype(np.float64), i); e = q_ - Y[i]
            shape[names[i]] = float(e.var()); own[names[i]] = float((e ** 2).mean())
            # sampling sensitivity (item 3): the same events reweighted flat in log s_hat (20 bins), and with the 20% of
            # lowest Lmin (closest to a singular region) carrying half the weight; shape MSE under each weighting, and
            # how far the affine map fitted on the reweighted events moves from the uniform one (|delta a| + |delta b|
            # on the standardized truth)
            Li, Lm = L[i], F[i][:, 11]
            b = np.clip(((Li - Li.min()) / max(np.ptp(Li), 1e-9) * 20).astype(int), 0, 19)
            cnt = np.bincount(b, minlength=20); wf = 1.0 / cnt[b]; wf /= wf.sum()
            low = Lm <= np.quantile(Lm, 0.2); wl = np.where(low, 0.5 / low.sum(), 0.5 / (~low).sum())
            wv = lambda w: float(np.sum(w * (e - np.sum(w * e)) ** 2))
            sflat[names[i]] = wv(wf); slow[names[i]] = wv(wl)
            def aff(w):
                A = np.stack([np.ones_like(q_), q_], 1) * np.sqrt(w)[:, None]
                return np.linalg.lstsq(A, Y[i] * np.sqrt(w), rcond=None)[0]
            c0, c1, c2 = aff(np.full(len(q_), 1 / len(q_))), aff(wf), aff(wl)
            sdy = Y[i].std()
            daff[names[i]] = float(max(abs(c1[0] - c0[0]) / sdy + abs(c1[1] - c0[1]), abs(c2[0] - c0[0]) / sdy + abs(c2[1] - c0[1])))
        new = [names[i] for i in ho if names[i] not in rel]; rl = [names[i] for i in ho if names[i] in rel]
        q = lambda d, S, x: float(np.quantile([d[n] for n in S], x))
        out[kind] = dict(shape_new=q(shape, new, .5), shape_new90=q(shape, new, .9), shape_rel=q(shape, rl, .5),
                         own_new=q(own, new, .5), own_rel=q(own, rl, .5), per_process_shape=shape,
                         flat_new=q(sflat, new, .5), lowLmin_new=q(slow, new, .5), flat_rel=q(sflat, rl, .5),
                         lowLmin_rel=q(slow, rl, .5),
                         ratio_flat_new=float(np.median([sflat[n] / shape[n] for n in new])),
                         ratio_low_new=float(np.median([slow[n] / shape[n] for n in new])),
                         affine_shift_new=q(daff, new, .5), affine_shift_rel=q(daff, rl, .5))
        v = out[kind]
        print(f"{kind}: shape MSE 43 new {v['shape_new']:.3g} (90% {v['shape_new90']:.3g}), 96 rel {v['shape_rel']:.3g} | "
              f"on its own 43 new {v['own_new']:.3g}, 96 rel {v['own_rel']:.3g} | reweighted/uniform shape on the new: "
              f"flat {v['ratio_flat_new']:.2f}, low-Lmin {v['ratio_low_new']:.2f}; affine shift new {v['affine_shift_new']:.3g}, "
              f"rel {v['affine_shift_rel']:.3g}", flush=True)
    json.dump(out, open(os.path.join(HERE, "data", a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
