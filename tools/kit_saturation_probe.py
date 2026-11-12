#!/usr/bin/env python3
"""Eval-only: where our KIT transformer stops following its inputs past their trained range (D411: why it does not
continue a trend where Bakar's ReLU MLP does). Two reads of one run, CPU, float64, the run's best weights:

strata  per event of bulk_train (2000) and each hard_test stratum (n), per transformer block b:
          x_rms      rms of the residual stream entering block b (what its pre-LayerNorm divides out)
          attn_max   the attention's largest weight per query, mean over heads and queries (1 = one-hot: saturated)
          attn_rms   rms of the attention branch's output, mlp_rms the MLP branch's
          gelu_off   share of the MLP's GELU pre-activations below -2 (GELU ~ 0 and flat there: an off unit)
          gelu_lin   share above +2 (GELU ~ identity: a linear unit)
        and h_rms (the residual entering the readout), mu and truth (the standardized prediction and target times
        std/ln 10: their differences, err and dmu, are in log10|M|^2; their levels carry no mean).
scan    bulk_train events with ONE input pushed past its range, everything else held: uug the event's most collinear
        final-state pair log (log(2 p_i.p_j/s)/5, shifted by k decades of s_ij), tt the event's log10 sqrt(s_hat)
        column (the dimensionless mass column's source, shifted by k decades of sqrt s); per k the same block stats
        and the readout's move d mu = mu(k) - mu(0) in log10|M|^2. A linear continuation keeps d mu / k constant.

    python tools/kit_saturation_probe.py <run_dir> [--n 200] [--scan-n 200] --out <json>     (CPU, site checkout)
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import kit_hardps as kh  # noqa: E402
from tools.kit_predict import run_target  # noqa: E402

UUG_K = [0, 1, 2, 3, 4, 5, 6, 7, 8]                # decades of s_ij below the event's value
TT_K = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]   # decades of sqrt(s_hat) above it


class Probe:
    """Forward hooks on the backbone; per-event block stats of the current batch (ptr from the wrapper's call)."""

    def __init__(self, model):
        import models.attention_lloca_mup as am
        import torch
        self.torch, self.am, self.model = torch, am, model
        self.blocks = list(model.net.net.blocks)
        self.rows, self.cur, self.ptr = [], None, None
        self.hooks = [model.register_forward_pre_hook(self._pre, with_kwargs=True)]
        for b, blk in enumerate(self.blocks):
            self.hooks += [blk.register_forward_pre_hook(lambda m, a, b=b: self._put(b, "x_rms", self._rms(a[0]))),
                           blk.attention.register_forward_hook(lambda m, a, o, b=b: self._put(b, "attn_rms", self._rms(o))),
                           blk.mlp[0].register_forward_hook(lambda m, a, o, b=b: self._gelu(b, o)),
                           blk.mlp.register_forward_hook(lambda m, a, o, b=b: self._put(b, "mlp_rms", self._rms(o)))]
        self.hooks.append(model.net.net.linear_out.register_forward_pre_hook(
            lambda m, a: self._put(-1, "h_rms", self._rms(a[0]))))
        self._orig_attn = am._pair_ctx_attention
        self._block_i = 0
        am._pair_ctx_attention = self._attn

    def _pre(self, mod, args, kwargs):
        self.ptr = kwargs["ptr"]
        self.cur = {}
        self.rows.append(self.cur)
        self._block_i = 0

    def _ev(self, v):
        """per-particle (N,) -> per-event mean (B,)"""
        cnt = (self.ptr[1:] - self.ptr[:-1])
        ev = self.torch.repeat_interleave(self.torch.arange(len(cnt)), cnt)
        return (self.torch.zeros(len(cnt), dtype=v.dtype).index_add_(0, ev, v) / cnt).numpy()

    def _rms(self, t):
        t = t.reshape(-1, t.shape[-1]) if t.shape[-2] == self.ptr[-1] else t.reshape(t.shape[-2], -1)
        return self._ev(t.pow(2).mean(-1).sqrt())

    def _put(self, b, k, v):
        self.cur.setdefault(f"{k}" if b < 0 else f"b{b}/{k}", v)

    def _gelu(self, b, o):
        o = o.reshape(-1, o.shape[-1])
        self._put(b, "gelu_off", self._ev((o < -2).double().mean(-1)))
        self._put(b, "gelu_lin", self._ev((o > 2).double().mean(-1)))

    def _attn(self, query, key, value, ctx, scale):
        """the float64 path's attention, its weights read: largest weight per valid query, mean over heads"""
        torch = self.torch
        *lead, H, N, C = query.shape
        B, n_max, pad2flat = ctx["B"], ctx["n_max"], ctx["pad2flat"]

        def pad(t):
            t = torch.cat([t.reshape(H, N, C), t.new_zeros(H, 1, C)], dim=1)
            return t[:, pad2flat].reshape(H, B, n_max, C).transpose(0, 1)
        qp, kp = pad(query), pad(key)
        m = ctx["mask_add"].to(qp.dtype) + (0 if ctx["bias"] is None else ctx["bias"].to(qp.dtype))
        sc = scale if scale is not None else 1 / math.sqrt(C)
        w = torch.softmax(qp @ kp.transpose(-1, -2) * sc + m, dim=-1)          # (B, H, n, n)
        valid = (pad2flat.reshape(B, n_max) < N)                                  # (B, n) real queries
        mx = w.max(-1).values.mean(1)                                             # (B, n)
        self._put(self._block_i, "attn_max", ((mx * valid).sum(1) / valid.sum(1)).numpy())
        self._block_i += 1
        return self._orig_attn(query, key, value, ctx, scale)

    def collect(self):
        keys = self.rows[0].keys()
        out = {k: np.concatenate([r[k] for r in self.rows]) for k in keys}
        self.rows = []
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir"); ap.add_argument("--n", type=int, default=200); ap.add_argument("--scan-n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    stem, target, _ = kh.parse(run_target(a.run_dir))
    blocks, rows = [], []
    for split in ("bulk_train", "hard_test"):
        for st in kh.strata(stem, split):
            r = kh.stratum_rows(stem, target, split, st)
            r = r[:2000] if split == "bulk_train" else r[:a.n]
            blocks.append((f"{split}/{st}", len(r))); rows.append(r)
    rows = np.concatenate(rows)
    out_dir = os.path.dirname(os.path.abspath(a.out)); os.makedirs(out_dir, exist_ok=True)
    pool = os.path.join(out_dir, f"satprobe_{os.getpid()}.npy")
    np.save(pool, rows)
    import h5py
    with h5py.File(kh.h5_path(stem), "r") as f:
        np.save(pool[:-4] + ".alphas.npy", np.full(len(rows), float(json.loads(f.attrs["metadata"])["alpha_s"])))
    try:
        _probe(a, pool, rows, blocks, stem)
    finally:
        for f in (pool, pool[:-4] + ".alphas.npy"):
            os.remove(f)


def _probe(a, pool, rows, blocks, stem):
    import torch
    import wrappers
    from tools.rebuild_run import rebuild
    exp, loader = rebuild(a.run_dir, "test", pool, len(rows), event_frame=True)
    model = exp.model
    to10 = float(exp.prepd_std[0]) / np.log(10.0)                     # prepd differences -> log10|M|^2 differences
    perm = np.asarray(exp._role_perm["test"])[:len(rows)]
    o = np.empty(len(rows), int); o[perm] = np.arange(len(perm))      # row -> loader position
    probe = Probe(model)

    def run():
        with torch.no_grad():
            pred, truth, _ = exp._collect_predictions(loader)
        st = probe.collect()
        st["mu"] = np.asarray(pred, np.float64).reshape(-1) * to10
        st["truth"] = np.asarray(truth, np.float64).reshape(-1) * to10
        return st

    def summ(st, idx):
        return {k: float(np.median(v[idx])) for k, v in st.items()}

    base = run()
    res = {"run": os.path.abspath(a.run_dir), "n_blocks": len(probe.blocks), "strata": {}, "scan": {}}
    i = 0
    for name, n in blocks:
        idx = o[i:i + n]; i += n
        res["strata"][name] = dict(summ(base, idx), err=float(np.median(base["mu"][idx] - base["truth"][idx])))
    sc_idx = o[:min(a.scan_n, blocks[0][1])]                          # bulk_train events (first block)

    tt = "tt" in stem
    orig_pl = wrappers._pair_log_inputs
    for k in (TT_K if tt else UUG_K):
        if tt:
            hook = model.register_forward_pre_hook(
                lambda m, args, kw, k=k: (args, dict(kw, order_labels=torch.cat(
                    [kw["order_labels"][:, :-1], kw["order_labels"][:, -1:] + k], dim=1))), with_kwargs=True)
        else:
            def shifted(p, ptr, n, threshold=False, k=k):
                f = orig_pl(p, ptr, n, threshold)
                fin = torch.tensor([i_ >= 2 and j_ >= 2 for i_ in range(n) for j_ in range(i_ + 1, n)])
                g = f.masked_fill(~fin | (f == 0), float("inf"))          # final-state pairs that exist
                c = g.argmin(-1)
                return f.scatter_add(1, c[:, None], torch.full_like(f[:, :1], -k * math.log(10) / 5))
            wrappers._pair_log_inputs = shifted
        try:
            st = run()
        finally:
            wrappers._pair_log_inputs = orig_pl
            if tt:
                hook.remove()
        s = summ(st, sc_idx)
        s["dmu"] = float(np.median(st["mu"][sc_idx] - base["mu"][sc_idx]))
        res["scan"][str(k)] = s
        print(f"scan k={k}: dmu {s['dmu']:+.3f}  " + "  ".join(
            f"b{b} off {s[f'b{b}/gelu_off']:.2f} att {s.get(f'b{b}/attn_max', float('nan')):.2f} x {s[f'b{b}/x_rms']:.3g}"
            for b in (0, len(probe.blocks) // 2, len(probe.blocks) - 1)), flush=True)
    for name, e in res["strata"].items():
        print(f"{name:45s} err {e['err']:+.3f} h_rms {e['h_rms']:.3g} " + "  ".join(
            f"b{b} off {e[f'b{b}/gelu_off']:.2f} lin {e[f'b{b}/gelu_lin']:.2f} att {e.get(f'b{b}/attn_max', float('nan')):.2f} "
            f"x {e[f'b{b}/x_rms']:.3g}" for b in (0, len(probe.blocks) - 1)))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
