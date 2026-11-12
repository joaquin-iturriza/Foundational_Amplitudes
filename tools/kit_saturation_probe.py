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

    python tools/kit_saturation_probe.py <run_dir> [--n 200] [--scan-n 200]     (CPU, site checkout; prints 'KITSAT <json>', taken home with
        site run ... | sed -n 's/^KITSAT //p' > analysis/kit_benchmark/data/kit_sat_<x>.json)
    python tools/kit_saturation_probe.py plot <uug json> <tt json> --figs <dir>
    python tools/kit_saturation_probe.py compare <uug ctrl> <uug arm> <tt ctrl> <tt arm> --figs <dir>
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
    ap.add_argument("--out", help="also write the JSON here (local runs only: a file written into a site checkout's "
                    "tracked tree blocks site sync once the same path is committed)")
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
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
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
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("KITSAT " + json.dumps(res))


def plot(uug, tt, figs):
    """kit_sat_resid_{uug,tt}: rms of the residual stream entering each block and the readout, per stratum of the
    singular family and bulk_train; kit_sat_scan_{uug,tt}: the readout's move dmu against the one-input shift k"""
    import plot_style as ps
    from tools.kit_figures import family, edges
    for tag, path, fam in (("uug", uug, "coll_y"), ("tt", tt, "sudakov_sqrts")):
        d = json.load(open(path))
        nb = d["n_blocks"]
        fig, ax = ps.figure()
        ks = sorted([k for k in d["strata"] if k.startswith("hard_test/") and family(k.split("/")[1]) == fam],
                    key=lambda k: edges(k.split("/")[1])[0], reverse=(fam == "coll_y"))
        cols = [ps.C.sky, ps.C.blue, ps.C.green, ps.C.orange, ps.C.vermillion]
        for k, c in zip(["bulk_train/bulk"] + ks, [ps.C.grey] + cols[-len(ks):]):
            e = d["strata"][k]
            y = [e[f"b{b}/x_rms"] for b in range(nb)] + [e["h_rms"]]
            if k.startswith("bulk"):
                lab = "bulk train"
            else:
                lo, hi = edges(k.split("/")[1])
                lab = (rf"$y\in[10^{{{np.log10(lo):.0f}}},10^{{{np.log10(hi):.0f}}}]$" if fam == "coll_y"
                       else rf"$\sqrt{{s}}\in[{lo / 1e3:.1f},{hi / 1e3:.0f}]$ TeV")
            ax.plot(range(nb + 1), y, "o-", color=c, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel(r"block $b$ ($8$ = readout input)")
        ax.set_ylabel(r"rms of residual stream $x_b$")
        ps.process_label(ax, {"uug": r"$e^+e^-\to u\bar u g$", "tt": r"$e^+e^-\to t\bar t$ virtual"}[tag])
        ps.legend(ax, "upper left" if tag == "uug" else "lower right")
        ps.make_room(ax)
        ps.save(fig, os.path.join(figs, f"kit_sat_resid_{tag}"))
        fig, ax = ps.figure()
        sk = sorted(d["scan"], key=float)
        ax.plot([float(k) for k in sk], [d["scan"][k]["dmu"] for k in sk], "o-", color=ps.C.vermillion,
                label=r"readout $\Delta\mu(k)$")
        ax.axhline(0, color=ps.C.grey, lw=0.8)
        ax.set_xlabel({"uug": r"$k$: shift of one $\log_{10} s_{ij}$ input", "tt": r"$k$: shift of $\log_{10}\sqrt{\hat s}$ input"}[tag])
        ax.set_ylabel(r"$\Delta\mu$, $\log_{10}|\mathcal{M}|^2$")
        ps.process_label(ax, {"uug": r"$e^+e^-\to u\bar u g$", "tt": r"$e^+e^-\to t\bar t$ virtual"}[tag])
        ps.legend(ax, "lower right" if tag == "tt" else "lower left")
        ps.make_room(ax)
        ps.save(fig, os.path.join(figs, f"kit_sat_scan_{tag}"))


def compare(pairs, figs):
    """kit_sat_cmp_<tag>_a: rms of the residual entering the readout per stratum of the singular family (bulk_train at
    the family's in-range end), _b: the one-input scan's readout move dmu(k); one curve per run (control, arm)"""
    import plot_style as ps
    from tools.kit_figures import family, edges, XLAB
    for tag, runs in pairs.items():
        fam = {"uug": "coll_y", "tt": "sudakov_sqrts"}[tag]
        panels = ps.panels(2)
        for (path, lab), c, mk in zip(runs, [ps.C.vermillion, ps.C.sky], ["o-", "P-"]):
            d = json.load(open(path))
            ks = sorted([k for k in d["strata"] if k.startswith("hard_test/") and family(k.split("/")[1]) == fam],
                        key=lambda k: edges(k.split("/")[1])[0])
            x = [float(np.sqrt(np.prod(edges(k.split("/")[1])))) for k in ks]
            panels[0][1].plot(x, [d["strata"][k]["h_rms"] for k in ks], mk, color=c, label=lab)
            sk = sorted(d["scan"], key=float)
            panels[1][1].plot([float(k) for k in sk], [d["scan"][k]["dmu"] for k in sk], mk, color=c, label=lab)
        ax = panels[0][1]
        from matplotlib.ticker import NullFormatter
        ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())   # the Sudakov axis spans one decade
        ax.set_xlabel(XLAB.get(fam, fam)); ax.set_ylabel(r"rms of readout input $h$")
        ax = panels[1][1]
        ax.axhline(0, color=ps.C.grey, lw=0.8)
        ax.set_xlabel({"uug": r"$k$: shift of one $\log_{10} s_{ij}$ input", "tt": r"$k$: shift of $\log_{10}\sqrt{\hat s}$ input"}[tag])
        ax.set_ylabel(r"$\Delta\mu$, $\log_{10}|\mathcal{M}|^2$")
        for _, ax in panels:
            ps.process_label(ax, {"uug": r"$e^+e^-\to u\bar u g$", "tt": r"$e^+e^-\to t\bar t$ virtual"}[tag])
            ps.legend(ax, "lower right" if tag == "tt" else "upper left" if ax is panels[0][1] else "lower left")
            ps.make_room(ax)
        ps.save_panels(panels, os.path.join(figs, f"kit_sat_cmp_{tag}"))


if __name__ == "__main__":
    if sys.argv[1:2] == ["compare"]:
        # compare <uug ctrl> <uug arm> <tt ctrl> <tt arm> --figs <dir> [--labels "ctrl;arm"]
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("paths", nargs=4); ap.add_argument("--figs", required=True)
        ap.add_argument("--labels", default="transformer;transformer, MLP inputs")
        a = ap.parse_args()
        L = a.labels.split(";")
        compare({"uug": list(zip(a.paths[:2], L)), "tt": list(zip(a.paths[2:], L))}, a.figs)
    elif sys.argv[1:2] == ["plot"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("uug"); ap.add_argument("tt"); ap.add_argument("--figs", required=True)
        a = ap.parse_args()
        plot(a.uug, a.tt, a.figs)
    else:
        main()
