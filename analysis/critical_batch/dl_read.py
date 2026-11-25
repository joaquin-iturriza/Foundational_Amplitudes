"""critical_batch (D560 part B): the critical batch of the optimizer's own update, B_dL, from a run's
<run>/noise_scale.json (find_BS with BS_finding.update: chunk records carrying dL_b<b>).

    python analysis/critical_batch/dl_read.py <noise_scale.json> <name> --ref 87 261 [--window 200] [--from 0]

dL(b) = mean over the other chunks' events of L(w) - L(w - u(g_b)), the loss decrease of one step of the update made
from b events (base_experiment._noise_update). Per window of `window` steps the reads' dL(b) are averaged at each b and
fitted with McCandlish et al.'s per-step form dL(b) = dL_inf / (1 + B_dL / b), as the line 1/dL = 1/dL_inf + (B_dL /
dL_inf) / b (least squares over the b's). A window whose mean dL is <= 0 at any b is not fitted and is listed apart
(rule fixed in the design before the data). A fit with 1/dL_inf <= 0 (dL still growing linearly at the largest b) is
listed as above the range. B_dL's standard error is a jackknife over the window's reads. The read-to-read spread of
dL(b_max)/dL(b_min) gives the standard error of each window's ratio (the 200- against 400-step pooling choice).
--ref lo hi: the critical batches it is compared with; the band is [lo/2, 2 hi] (the design's factor 2).
Writes analysis/critical_batch/data/<name>.json and figs/<name>.png/.pdf."""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))


def reads(recs, start=0):
    rs = sorted((r for r in recs if r.get("kind") == "chunk" and r["step"] >= start
                 and any(k.startswith("dL_b") for k in r)), key=lambda r: r["step"])
    bs = sorted({int(k[4:]) for r in rs for k in r if k.startswith("dL_b")})
    return rs, bs


def fit(dl, bs):
    """(B_dL, dL_inf) of mean dL at each b; None when any dL <= 0; B = inf when 1/dL_inf <= 0."""
    dl, x = np.asarray(dl, float), 1.0 / np.asarray(bs, float)
    if (dl <= 0).any():
        return None
    c, a = np.polyfit(x, 1.0 / dl, 1)
    return (np.inf, np.inf) if a <= 0 else (max(c, 0.0) / a, 1.0 / a)


def window(rs, bs):
    D = np.array([[r[f"dL_b{b}"] for b in bs] for r in rs], float)
    m = D.mean(0)
    f = fit(m, bs)
    out = {"step_lo": rs[0]["step"], "step_hi": rs[-1]["step"], "n_reads": len(rs),
           "dL": {str(b): float(v) for b, v in zip(bs, m)},
           "A": {str(b): float(np.mean([sum(v for k, v in r.items() if k.endswith(f"_A_b{b}")) for r in rs])) for b in bs}}
    if f is None:
        return {**out, "status": "dL<=0"}
    out.update(status="above range" if np.isinf(f[0]) else "fitted", B_dL=float(f[0]), dL_inf=float(f[1]))
    if out["status"] == "fitted" and len(rs) > 2:
        jk = [fit(np.delete(D, i, 0).mean(0), bs) for i in range(len(rs))]
        jk = [j[0] for j in jk if j is not None and np.isfinite(j[0])]
        if len(jk) == len(rs):
            out["B_dL_se"] = float(np.sqrt((len(jk) - 1) / len(jk) * ((np.array(jk) - np.mean(jk)) ** 2).sum()))
    ok = D[:, 0] > 0
    if ok.sum() > 1:
        r = D[ok, -1] / D[ok, 0]
        out["ratio"] = float(m[-1] / m[0]) if m[0] > 0 else None
        out["ratio_se"] = float(r.std(ddof=1) / np.sqrt(ok.sum()))
    return out


def read(recs, win, ref, start=0):
    rs, bs = reads(recs, start)
    ws, lo = [], rs[0]["step"] if rs else 0
    while rs and lo <= rs[-1]["step"]:
        w = [r for r in rs if lo <= r["step"] < lo + win]
        if w:
            ws.append(window(w, bs))
        lo += win
    band = [ref[0] / 2, ref[1] * 2]
    fitted = [w["B_dL"] for w in ws if w["status"] == "fitted"]
    return {"b": bs, "window": win, "ref": ref, "band": band, "windows": ws,
            "summary": {"n_windows": len(ws), "n_fitted": len(fitted),
                        "n_dL_nonpositive": sum(w["status"] == "dL<=0" for w in ws),
                        "n_above_range": sum(w["status"] == "above range" for w in ws),
                        "B_dL_median": float(np.median(fitted)) if fitted else None,
                        "B_dL_q16_q84": [float(np.quantile(fitted, q)) for q in (0.16, 0.84)] if fitted else None,
                        "n_in_band": sum(band[0] <= x <= band[1] for x in fitted)}}


def figure(res, base):
    import plot_style as ps
    import matplotlib as mpl
    fig, (axB, axD) = ps.figure(ncols=2)
    mid = lambda w: 0.5 * (w["step_lo"] + w["step_hi"])
    fw = [w for w in res["windows"] if w["status"] == "fitted"]
    if fw:
        axB.errorbar([mid(w) for w in fw], [w["B_dL"] for w in fw], yerr=[w.get("B_dL_se", 0) for w in fw], fmt="o",
                     color=ps.C.vermillion, label="update's critical batch")
    lo, hi = res["ref"]
    axB.axhline(lo, color=ps.C.blue, ls="--", label="two-batch critical batch, range")
    axB.axhline(hi, color=ps.C.blue, ls="--")
    up = [w for w in res["windows"] if w["status"] == "above range"]
    top = max([w["B_dL"] for w in fw] + [hi]) * 2
    if up:
        axB.plot([mid(w) for w in up], [top] * len(up), "^", color=ps.C.grey, label="above the measured batches")
    neg = [w for w in res["windows"] if w["status"] == "dL<=0"]
    if neg:
        axB.plot([mid(w) for w in neg], [min([w["B_dL"] for w in fw] + [lo]) / 2] * len(neg), "v", color=ps.C.grey,
                 label="loss rises at some batch")
    axB.set_yscale("log")
    ys = [w["B_dL"] for w in fw] + [lo, hi]
    axB.set_ylim(min(ys) / 4, max(ys) * 4)   # the reference lines and the edge markers inside the frame
    axB.set_xlabel("training step")
    axB.set_ylabel(r"$B_{\rm dL}$ (events)")
    ps.legend(axB, "upper left")
    cmap = mpl.colormaps["viridis"]
    bs = np.array(res["b"], float)
    xx = np.geomspace(bs.min(), bs.max(), 50)
    allw = res["windows"]
    steps = [mid(w) for w in allw]
    norm = mpl.colors.Normalize(min(steps), max(steps)) if steps else None
    for w in allw:
        c = cmap(norm(mid(w)))
        axD.plot(bs, [w["dL"][str(int(b))] for b in bs], "o-", color=c)
        axD.plot(bs, [w["A"][str(int(b))] for b in bs], "s--", color=c, mfc="none")
        if w["status"] == "fitted":
            axD.plot(xx, w["dL_inf"] / (1 + w["B_dL"] / xx), ":", color=c)
    axD.plot([], [], "o-", color=ps.C.grey, label=r"loss decrease $\Delta L(b)$")
    axD.plot([], [], "s--", color=ps.C.grey, mfc="none", label=r"first-order part $A(b)$")
    if fw:
        axD.plot([], [], ":", color=ps.C.grey, label=r"fit $\Delta L_\infty/(1+B_{\rm dL}/b)$")
    axD.axhline(0, color="black", label="no change")
    axD.set_xscale("log")
    axD.set_xlabel(r"events per update $b$")
    axD.set_ylabel(r"loss change per step")
    ps.legend(axD, "upper left")
    if norm:
        ps.colorbar(axD, mpl.cm.ScalarMappable(norm=norm, cmap=cmap), "training step")
    ps.save(fig, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("name")
    ap.add_argument("--ref", type=float, nargs=2, required=True)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--from", dest="start", type=int, default=0)
    a = ap.parse_args()
    res = read(json.load(open(a.path)), a.window, a.ref, a.start)
    res["source"] = a.path
    json.dump(res, open(os.path.join(HERE, "data", f"{a.name}.json"), "w"), indent=1)
    print(json.dumps(res["summary"]))
    for w in res["windows"]:
        print(w["step_lo"], w["step_hi"], w["status"], w.get("B_dL"), w.get("B_dL_se"), w.get("ratio"), w.get("ratio_se"))
    figure(res, os.path.join(HERE, "figs", a.name))


if __name__ == "__main__":
    main()
