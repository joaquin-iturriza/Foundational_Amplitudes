"""Continual add-W along the run (D30, D31; item addhoriz): the finale fine-tuned on W+4g, W+5g for 1k, 2k or 4k steps by
seven methods (naive, EWC, LoRA, frozen lower blocks, layer-wise lr decay, 1% and 10% replay buffers), 5-trial searches,
the finale's 28 processes scored at every validation as monitors (never trained on, outside the checkpoint-selecting
aggregate; the buffers train on 1% or 10% of their pools). W = the run's val_loss_no_reg (geometric mean over W+4g,
W+5g, preprocessed); forgetting = the geometric mean over the 28 monitors' validation MSE (preprocessed) divided by the
finale's own on the same pools (FA-ccin2p3-4e4e6153). Data: analysis/transfer/waddm_curves.json, built from
waddm_collect.py's output on each site by waddm_build.py (a trial re-run under the same HPs counts once).
(a) per method, the Pareto front over every validation point of all its runs and horizons;
(b) each run's best checkpoint on W (model_run0_best.pt), marker = horizon.
    python analysis/transfer/waddm_tradeoff.py   -> figs/waddm_tradeoff_a, _b, _legend; table on stdout
"""
import json, math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

D = json.load(open(os.path.join(ROOT, "analysis", "transfer", "waddm_curves.json")))
GM = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
F0 = GM(D["finale_old28"].values())
METHODS = [("naive", "naive", ps.C.grey), ("ewc", "EWC", ps.C.blue), ("lora", "LoRA", ps.C.sky),
           ("frz", "frozen lower blocks", ps.C.green), ("lld", "layer-wise lr decay", ps.C.orange),
           ("buf1pc", "replay 1%", ps.C.purple), ("buf10pc", "replay 10%", ps.C.vermillion)]
HOR = {"1k": "o", "2k": "s", "4k": "^"}


def split(sweep):
    tail = sweep[len("tp3_waddm_"):]
    return tail[:-2], tail[-2:]


pts = {}                     # method -> [(W, forgetting, sweep, trial, step)]
best = []                    # (method, horizon, W, forgetting, trial, step, lr)
for r in D["runs"]:
    m, h = split(r["sweep"])
    n = len(r["W"])
    f = [GM([c[i] for c in r["old28"].values()]) / F0 for i in range(n)]
    pts.setdefault(m, []).extend((r["W"][i], f[i], r["sweep"], r["trial"], (i + 1) * r["every"]) for i in range(n))
    i = int(np.argmin(r["W"]))
    best.append((m, h, r["W"][i], f[i], r["trial"], (i + 1) * r["every"], r["hp"]["lr"]))


def front(p):
    out, fmin = [], float("inf")
    for w, f, *rest in sorted(p):
        if f < fmin:
            out.append((w, f, *rest))
            fmin = f
    return out


figs = ps.panels(2)
(fa, axa), (fb, axb) = figs
for m, lab, col in METHODS:
    fr = front(pts.get(m, []))
    if fr:
        axa.plot([x[0] for x in fr], [x[1] for x in fr], color=col, label=lab, drawstyle="steps-post")
    for mm, h, w, f, *_ in best:
        if mm == m:
            axb.plot(w, f, HOR[h], color=col, linestyle="none")
for h, mk in HOR.items():
    axb.plot([], [], mk, color=ps.C.grey, linestyle="none", label=f"{h} steps")
for ax in (axa, axb):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"error on $W+4g$, $W+5g$ (MSE)")
    ax.set_ylabel(r"forgetting: $\mathrm{MSE}_{28}/\mathrm{MSE}_{28}^\mathrm{finale}$")
base = os.path.join(ROOT, "analysis", "transfer", "figs", "waddm_tradeoff")
ps.save_panels(figs, base)
# one legend for both panels: the methods' colours from (a), the horizons' markers from (b)
for hh, ll in zip(*axb.get_legend_handles_labels()):
    axa.plot([], [], hh.get_marker(), color=ps.C.grey, linestyle="none", label=ll)
ps.legend_strip(axa, base + "_legend", ncol=5)

print(f"finale, geometric mean over the 28: {F0:.3e}")
print("per sweep: n runs | best on W: trial, step, W, old28/finale, lr | best on all 30: trial, step, W, old28/finale")
for m, lab, _ in METHODS:
    for h in HOR:
        rs = [r for r in D["runs"] if split(r["sweep"]) == (m, h)]
        if not rs:
            print(f"  {m:8s} {h}: none")
            continue
        bw = min(best, key=lambda b: b[2] if (b[0], b[1]) == (m, h) else float("inf"))
        a = min(((r, i) for r in rs for i in range(len(r["all30"]))), key=lambda ri: ri[0]["all30"][ri[1]])
        r, i = a
        fa_ = GM([c[i] for c in r["old28"].values()]) / F0
        print(f"  {m:8s} {h}: n={len(rs)} | W {bw[4]} @{bw[5]} {bw[2]:.2e} {bw[3]:.3g} lr {bw[6]:.1e} | "
              f"all {r['trial']} @{(i + 1) * r['every']} {r['W'][i]:.2e} {fa_:.3g}")
for m, lab, _ in METHODS:
    p = pts.get(m, [])
    for thr in (3, 10, 30, 100):
        ok = [x for x in p if x[1] <= thr]
        b = min(ok) if ok else None
        print(f"  {m:8s} best W with old28/finale <= {thr:3d}: " +
              (f"{b[0]:.2e} ({b[2][10:]} {b[3]} @{b[4]})" if b else "none"))
