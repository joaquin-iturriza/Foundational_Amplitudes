#!/usr/bin/env python3
"""Bakar's reference MLP of the KIT benchmark (its src/baseline.py), ported to torch, and a ladder of one-change arms
from it towards our setup: which difference explains why his 3 x 128 MLP extrapolates into the limits and ours not.

    python tools/kit_mlp_ladder.py <file> <target> --arm <arm> [--seed S] [--n-train 100000] [--runs runs/kit_mlp]

baseline.py, read at the benchmark's src: features = log10 of every pair invariant s_ij/s0 (s0 the largest s_ij of the
training sample; 'logthr' adds log10((s_tt - 4 m^2)/s0)), standardized on bulk_train; target log10|M|^2 standardized;
3 hidden layers of 128, leaky ReLU (0.01), He-normal init, Adam (0.9, 0.999, 1e-8), lr 2e-3 cosine to 1e-6, 300
epochs of batch 2048 over the first n_train bulk_train events, MSE; float64. Arms (ARMS) change one thing each:
  mlp      his setup ('logthr' features for ee_tt, as his tt baseline)
  target   the catalog parent's emission-scale target T = (ln|M|^2 + sum_k ln q_k^2)/ln 10 (physics_scales.qperp_logs)
  mom      inputs: every particle's 4-momentum over sqrt(s) (the dimensionless momenta our network reads), standardized
  raw      inputs: s_ij/s0 without the log (his 'raw')
  gelu     activation GELU (our transformer's) for the leaky ReLU
  het      loss: Gaussian NLL with a learned per-event sigma (our HETEROSC), mu scored
  ours     mom + target + gelu + het together
The run is written as <runs>/kit_A_<file>_<target>_n<N>_t<steps>_mlp<arm>_s<seed>/kit_benchmark/{predictions.h5,
kit_result.json} (tools/kit_predict.evaluate: the benchmark's own evaluate.py), so tools/kit_figures.py collects it.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import kit_hardps as kh  # noqa: E402

ARMS = {"mlp": {}, "target": {"target": "T"}, "mom": {"features": "mom"}, "raw": {"features": "raw"},
        "gelu": {"act": "gelu"}, "het": {"loss": "het"},
        "ours": {"features": "mom", "target": "T", "act": "gelu", "loss": "het"}}
EPOCHS, BATCH, WIDTH = int(os.environ.get("KIT_MLP_EPOCHS", 300)), 2048, 128   # epochs: 1 for a dry run


def invariants(p):
    iu = np.triu_indices(p.shape[1], 1)
    q = p[:, iu[0]] + p[:, iu[1]]
    return q[..., 0] ** 2 - np.sum(q[..., 1:] ** 2, -1)


def featurize(p, s0, kind, mass):
    if kind == "mom":
        P = p[:, 0] + p[:, 1]
        return (p / np.sqrt(P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(-1))[:, None, None]).reshape(len(p), -1)
    inv = invariants(p) / s0
    if kind == "raw":
        return inv
    log = np.log10(np.maximum(np.abs(inv), 1e-300))
    if kind == "logthr":     # baseline.py: the t t~ pair is the last invariant (finals last)
        return np.column_stack([log, np.log10(inv[:, -1] - 4 * mass ** 2 / s0)])
    return log


def target_of(rows, n, kind):
    """(y, offset): the regressed target per event and log10|M|^2 - y (0 for the plain target)"""
    from physics_scales import qperp_logs
    l10 = np.log10(rows[:, -1])
    if kind == "plain":
        return l10, np.zeros_like(l10)
    off = -qperp_logs(rows[:, :4 * n].reshape(-1, n, 4)) / np.log(10.0)
    return l10 - off, off


def train(x, y, cfg, seed, device):
    import torch
    torch.manual_seed(seed)
    dt = torch.float64
    sizes = [x.shape[1], WIDTH, WIDTH, WIDTH, 2 if cfg["loss"] == "het" else 1]
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        lin = torch.nn.Linear(a, b, dtype=dt)
        torch.nn.init.normal_(lin.weight, std=np.sqrt(2 / a)); torch.nn.init.zeros_(lin.bias)
        layers += [lin, torch.nn.LeakyReLU(0.01) if cfg["act"] == "leaky" else torch.nn.GELU()]
    net = torch.nn.Sequential(*layers[:-1]).to(device)
    X, Y = torch.tensor(x, dtype=dt, device=device), torch.tensor(y, dtype=dt, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-8)
    nb = len(X) // BATCH
    steps = EPOCHS * max(1, nb)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (0.5 * (1 + np.cos(np.pi * k / steps)) + 5e-4))
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X), generator=g).to(device)
        for i in range(nb):
            b = perm[i * BATCH:(i + 1) * BATCH]
            out = net(X[b])
            if cfg["loss"] == "het":
                sig = torch.nn.functional.softplus(out[:, 1]) + 1e-6
                loss = (0.5 * ((Y[b] - out[:, 0]) / sig) ** 2 + torch.log(sig)).mean()
            else:
                loss = ((out[:, 0] - Y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return lambda z: np.concatenate([net(torch.tensor(z[i:i + 50000], dtype=dt, device=device))[:, 0]
                                     .detach().cpu().numpy() for i in range(0, len(z), 50000)]), steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file"); ap.add_argument("target")
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=100000)
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs", "kit_mlp"))
    a = ap.parse_args()
    import h5py
    import torch
    cfg = dict(dict(features="logthr" if a.file == "ee_tt" else "log", target="plain", act="leaky", loss="mse"),
               **ARMS[a.arm])
    if cfg["features"] == "log" and a.file == "ee_tt":
        cfg["features"] = "logthr"
    src = kh.h5_path(a.file)
    with h5py.File(src, "r") as f:
        mass = json.loads(f.attrs["metadata"])["particles"][2]["mass"]
    n = None
    tr = kh.stratum_rows(a.file, a.target, "bulk_train", "bulk", src)[:a.n_train]
    n = (tr.shape[1] - 1) // 5
    mom = lambda r: r[:, :4 * n].reshape(-1, n, 4)
    s0 = float(np.max(invariants(mom(tr))))
    xtr = featurize(mom(tr), s0, cfg["features"], mass)
    ytr, _ = target_of(tr, n, cfg["target"])
    mx, sx, my, sy = xtr.mean(0), xtr.std(0) + 1e-12, ytr.mean(), ytr.std()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f_pred, steps = train((xtr - mx) / sx, (ytr - my) / sy, cfg, a.seed, dev)
    rd = os.path.join(a.runs, f"kit_A_{a.file}_{a.target}_n{a.n_train}_t{steps}_mlp{a.arm}_s{a.seed}")
    out = os.path.join(rd, "kit_benchmark")
    os.makedirs(out, exist_ok=True)
    lname = kh.log10_name(a.file, a.target, src)
    path = os.path.join(out, "predictions.h5")
    with h5py.File(src, "r") as t, h5py.File(path, "w") as w:
        for split in ("hard_test", "bulk_test"):
            for st in kh.strata(a.file, split, src):
                r = kh.stratum_rows(a.file, a.target, split, st, src)
                y, off = target_of(r, n, cfg["target"])
                z = f_pred((featurize(mom(r), s0, cfg["features"], mass) - mx) / sx) * sy + my
                # log10 pred in the file's target: its truth plus the residual (exact for the stripped constant)
                w[f"{split}/{st}/{lname}"] = t[f"{split}/{st}"][lname][:] + (z + off - np.log10(r[:, -1]))
    json.dump(dict(cfg, arm=a.arm, seed=a.seed, n_train=a.n_train, steps=steps, s0=s0), open(os.path.join(rd, "mlp.json"), "w"))
    from tools.kit_predict import evaluate
    evaluate(a.file, path, out)
    print("MLP_DONE " + rd)


if __name__ == "__main__":
    main()
