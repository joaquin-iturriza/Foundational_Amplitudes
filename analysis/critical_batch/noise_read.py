"""critical_batch (D532): the gradient noise scale along one run, from its <run>/noise_scale.json (find_BS).

    python analysis/critical_batch/noise_read.py <noise_scale.json> <name> [--window 20] [--batch 16384]

S = tr Sigma and |G|^2 per measurement from noise_scale.noise_scale_terms, summed over the parameter sets (Muon,
AdamW), averaged separately over windows of `window` consecutive chunk measurements, B_simple = mean S / mean |G|^2.
The brute-force checks: S and |G|^2 from 16 independent batch gradients at one step, beside the chunk estimate on the
same batches with its bootstrap standard error over them; the brute values' own standard error is a jackknife over
the batches from the record's Gram matrix g_i.g_j. Agreement: within 2 combined standard errors
sqrt(se_chunk^2 + se_brute^2). Writes
analysis/critical_batch/data/<name>.json and figs/<name>.png/.pdf."""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from noise_scale import noise_scale_terms  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def terms(r):
    """(S, |G|^2) of one record, summed over its parameter sets."""
    sets = sorted({k[:-len("_g2_small")] for k in r if k.endswith("_g2_small")})
    st = [noise_scale_terms(r[f"{s}_g2_small"], r[f"{s}_g2_big"], r["b_small"], r["b_big"]) for s in sets]
    return sum(a for a, _ in st), sum(b for _, b in st), {s: v for s, v in zip(sets, st)}


def gram_terms(r, idx):
    """(S, |G|^2, B) of a brute record restricted to the batches idx, from its Gram matrices (summed over sets)."""
    S = G2 = 0.0
    m = len(idx)
    for key in (k for k in r if k.endswith("_gram")):
        G = np.asarray(r[key])[np.ix_(idx, idx)]
        a, b = noise_scale_terms(np.trace(G) / m, G.sum() / m ** 2, r["b_small"], r["b_small"] * m)
        S, G2 = S + a, G2 + b
    return S, G2, S / G2 if G2 > 0 else np.nan


def jackknife_se(r):
    M = r["k"]
    th = np.array([gram_terms(r, [i for i in range(M) if i != j]) for j in range(M)])
    return np.sqrt((M - 1) / M * ((th - th.mean(0)) ** 2).sum(0))


def read(recs, window):
    ch = [r for r in recs if r["kind"] == "chunk"]
    out = {"windows": [], "checks": []}
    for i in range(0, len(ch) - window + 1, window):
        w = ch[i:i + window]
        ST = np.array([terms(r)[:2] for r in w])
        per = {}
        for s in terms(w[0])[2]:
            a = np.array([terms(r)[2][s] for r in w])
            per[s] = {"S": float(a[:, 0].mean()), "G2": float(a[:, 1].mean()),
                      "B_simple": float(a[:, 0].mean() / a[:, 1].mean()) if a[:, 1].mean() > 0 else None}
        S, G2 = ST.mean(0)
        out["windows"].append({"step_lo": w[0]["step"], "step_hi": w[-1]["step"], "S": float(S), "G2": float(G2),
                               "S_se": float(ST[:, 0].std(ddof=1) / np.sqrt(len(w))),
                               "G2_se": float(ST[:, 1].std(ddof=1) / np.sqrt(len(w))),
                               "B_simple": float(S / G2) if G2 > 0 else None, "sets": per,
                               # the largest single |S| over the window's summed |S| (S can be negative): near 1,
                               # one measurement sets the window's mean (a heavy-tailed gradient); a description only
                               "S_max_share": float(np.abs(ST[:, 0]).max() / np.abs(ST[:, 0]).sum()),
                               "S_max_step": int(w[int(np.abs(ST[:, 0]).argmax())]["step"])})
    for r in (r for r in recs if r["kind"] == "brute"):
        same = [terms(c)[:2] for c in recs if c["kind"] == "chunk_at_brute" and c["step"] == r["step"]]
        a = np.array(same)
        Sb, Gb, _ = terms(r)
        rng = np.random.default_rng(0)
        boot = [a[rng.integers(0, len(a), len(a))].mean(0) for _ in range(2000)]
        se = np.std(boot, axis=0)
        jse = jackknife_se(r)
        x = {"step": r["step"], "n_batches": r["k"], "S_brute": Sb, "G2_brute": Gb,
             "B_brute": Sb / Gb if Gb > 0 else None, "S_brute_se": float(jse[0]), "G2_brute_se": float(jse[1]),
             "B_brute_se": float(jse[2]), "S_chunk": float(a[:, 0].mean()), "G2_chunk": float(a[:, 1].mean()),
             "S_chunk_se": float(se[0]), "G2_chunk_se": float(se[1]),
             "B_chunk": float(a[:, 0].mean() / a[:, 1].mean()) if a[:, 1].mean() > 0 else None}
        for q in ("S", "G2"):
            x[f"{q}_agree"] = bool(abs(x[f"{q}_chunk"] - x[f"{q}_brute"])
                                   <= 2 * np.hypot(x[f"{q}_chunk_se"], x[f"{q}_brute_se"]))
        out["checks"].append(x)
    return out


def figure(res, batch, base, curves=None):
    """(a) B_simple along the run, all parameters and each set, with the brute checks; (b) its two terms; (c) S at
    the checks, chunk against brute with their errors; (d) the validation curves of `curves` {label: result.json}."""
    import plot_style as ps
    fig, ((axB, axT), (axS, axV)) = ps.figure(ncols=2, nrows=2)
    mid = lambda xs: [(x["step_lo"] + x["step_hi"]) / 2 for x in xs]
    c = [x for x in res["checks"] if x["B_brute"] is not None]
    # the y range is set by rule, never after a look: up to 3x the larger of the training batch and the 16-batch
    # checks' B (+ its error); a window above it (one heavy-tailed measurement, S up to 1e31) is drawn at that edge
    top = 3 * max([batch] + [x["B_brute"] + x["B_brute_se"] for x in c])
    one = lambda x: x["B_simple"] is not None and x["B_simple"] > top
    pos = lambda x: x["B_simple"] is not None and x["B_simple"] > 0
    w = [x for x in res["windows"] if pos(x) and not one(x)]
    axB.plot(mid(w), [x["B_simple"] for x in w], color=ps.C.blue, label="all, chunks")
    for s, col, lab in (("muon", ps.C.green, "Muon parameters"), ("adamw", ps.C.orange, "AdamW parameters")):
        ws = [x for x in w if x["sets"].get(s, {}).get("B_simple") is not None]
        if ws:
            axB.plot(mid(ws), [x["sets"][s]["B_simple"] for x in ws], color=col, ls=":", label=lab)
    M = res["checks"][0]["n_batches"] if res["checks"] else 0
    axB.errorbar([x["step"] for x in c], [x["B_brute"] for x in c], yerr=[x["B_brute_se"] for x in c], fmt="o",
                 color=ps.C.vermillion, label=f"all, {M} batches")
    neg = mid([x for x in res["windows"] if not pos(x)])
    if neg:   # windows with mean |G|^2 <= 0 (no B) or mean S < 0 (B < 0): off a log axis, marked at the bottom
        axB.plot(neg, [min([x["B_simple"] for x in w] + [batch]) / 2] * len(neg), "v", color=ps.C.blue,
                 label=r"window with no $B_{\rm simple}>0$")
    axB.axhline(batch, color="k", ls="--", label="training batch")
    big = [x for x in res["windows"] if one(x)]
    if big:
        axB.plot(mid(big), [top] * len(big), "^", color=ps.C.grey, label="window above the range")
    axB.set_yscale("log")
    axB.set_xlabel("training step")
    axB.set_ylabel(r"$B_{\rm simple}={\rm tr}\,\Sigma/|G|^2$ (events)")
    ps.legend(axB, "upper left", ncol=2)
    # same rule: up to 3x the larger check value of either term; a window above it at that edge
    tt = 3 * max(max(x["S_brute"] + x["S_brute_se"], (x["G2_brute"] + x["G2_brute_se"]) * batch) for x in res["checks"])
    big = [x for x in res["windows"] if max(x["S"], x["G2"] * batch) > tt]
    wt = [x for x in res["windows"] if x not in big and x["S"] > 0 and x["G2"] > 0]
    axT.plot(mid(wt), [x["S"] for x in wt], color=ps.C.blue, label=r"${\rm tr}\,\Sigma$")
    axT.plot(mid(wt), [x["G2"] * batch for x in wt], color=ps.C.vermillion, label=rf"$|G|^2\times {batch}$")
    if big:
        axT.plot(mid(big), [tt] * len(big), "^", color=ps.C.grey, label="window above the range")
    low = [x for x in res["windows"] if x not in big and x not in wt]
    if low:   # a term <= 0 is off the log axis: marked at the bottom, not dropped
        axT.plot(mid(low), [min(min(x["S"], x["G2"] * batch) for x in wt) / 3] * len(low), "v", color=ps.C.grey,
                 label="window with a term $\\leq 0$")
    axT.set_yscale("log")
    axT.set_xlabel("training step")
    axT.set_ylabel(r"squared gradient norm terms")
    ps.legend(axT, "upper left")
    st = [x["step"] for x in res["checks"]]
    axS.errorbar(st, [x["S_brute"] for x in res["checks"]], yerr=[2 * x["S_brute_se"] for x in res["checks"]],
                 fmt="o", color=ps.C.vermillion, label=f"{M} independent batches")
    axS.errorbar(st, [x["S_chunk"] for x in res["checks"]], yerr=[2 * x["S_chunk_se"] for x in res["checks"]],
                 fmt="s", color=ps.C.blue, label="chunks of the same batches")
    axS.set_yscale("log")
    axS.set_xlabel("training step")
    axS.set_ylabel(r"${\rm tr}\,\Sigma$ at the checks")
    ps.legend(axS, "upper left")
    for (lab, r), col in zip((curves or {}).items(), (ps.C.vermillion, ps.C.blue)):
        v = r["val_curve"]
        axV.plot([i * r["validate_every"] for i in range(len(v))], v, color=col, label=lab)
    axV.set_xlabel("training step")
    axV.set_yscale("symlog", linthresh=1.0)   # the het NLL is negative; its spikes reach 1e9
    axV.set_ylabel(r"validation loss (no reg.)")
    ps.legend(axV, "upper right")
    ps.save(fig, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("name")
    ap.add_argument("--window", type=int, default=20); ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--result", help="this run's result.json (its validation curve)")
    ap.add_argument("--control", help="the control run's result.json, same setup without the measurement")
    a = ap.parse_args()
    res = read(json.load(open(a.path)), a.window)
    res["source"] = os.path.abspath(a.path)
    os.makedirs(f"{HERE}/data", exist_ok=True)
    json.dump(res, open(f"{HERE}/data/{a.name}.json", "w"), indent=1)
    for x in res["checks"]:
        print(f"CHECK step {x['step']}: S brute {x['S_brute']:.4g} chunk {x['S_chunk']:.4g} +- {x['S_chunk_se']:.2g}; "
              f"|G|^2 brute {x['G2_brute']:.4g} chunk {x['G2_chunk']:.4g} +- {x['G2_chunk_se']:.2g}; "
              f"B brute {x['B_brute']} +- {x['B_brute_se']:.3g} chunk {x['B_chunk']}; S brute se {x['S_brute_se']:.2g}, "
              f"|G|^2 brute se {x['G2_brute_se']:.2g}; agree S {x['S_agree']} |G|^2 {x['G2_agree']}")
    for x in res["windows"][::max(1, len(res['windows']) // 10)]:
        print(f"WINDOW steps {x['step_lo']}-{x['step_hi']}: S {x['S']:.4g} |G|^2 {x['G2']:.4g} B_simple {x['B_simple']}")
    for x in res["windows"]:
        print(f"SETS steps {x['step_lo']}-{x['step_hi']}: " + " ".join(f"{s} B {v['B_simple']}" for s, v in x["sets"].items())
              + f"; largest S share {x['S_max_share']:.3f} at step {x['S_max_step']}")
    curves = {}
    if a.result:
        curves["with the noise measurement"] = json.load(open(a.result))
    if a.control:
        curves["control, no measurement"] = json.load(open(a.control))
    for lab, r in curves.items():
        print(f"VAL {lab}: final {r['val_curve'][-1]:.4f} best {min(r['val_curve']):.4f} at step "
              f"{int(np.argmin(r['val_curve'])) * r['validate_every']}")
    figure(res, a.batch, f"{HERE}/figs/{a.name}", curves)
    # the summary the notes' claims key into: each check by its step; B over ALL windows (median, which the
    # heavy-tailed windows cannot move far, and its range), the windows without B counted, the largest |S| shares
    # listed apart; the validation curves' last values and training hours
    r2 = lambda v: float(f"{v:.3g}") if v is not None else None
    late = res["windows"]
    Bs = [x["B_simple"] for x in late if x["B_simple"] is not None]
    med = lambda f: r2(float(np.median([v for v in map(f, late) if v is not None])))
    summ = {"n_windows": len(late), "n_no_B": len(late) - len(Bs),
            "share_top": [[x["step_lo"], round(x["S_max_share"], 3)] for x in
                          sorted(late, key=lambda x: -x["S_max_share"])[:8]],
            "S_max": r2(max(x["S"] for x in late)),
            "B_median": med(lambda x: x["B_simple"]), "B_min": r2(min(Bs)), "B_max": r2(max(Bs)),
            "B_q25": r2(float(np.percentile(Bs, 25))), "B_q75": r2(float(np.percentile(Bs, 75))),
            "B_muon_median": med(lambda x: x["sets"]["muon"]["B_simple"]),
            "B_adamw_median": med(lambda x: x["sets"]["adamw"]["B_simple"]),
            "B_muon_last": r2(late[-1]["sets"]["muon"]["B_simple"])}
    for x in res["checks"]:
        summ[f"check{x['step']}"] = {k: r2(x[k]) for k in ("S_brute", "S_brute_se", "S_chunk", "S_chunk_se", "G2_brute",
                                                           "G2_chunk", "B_brute", "B_brute_se", "B_chunk")}
        summ[f"check{x['step']}"].update(S_ratio=r2(x["S_chunk"] / x["S_brute"]), S_agree=x["S_agree"],
                                         G2_agree=x["G2_agree"])
    for lab, r in zip(("run", "control"), curves.values()):
        summ[lab] = {"val_last": round(r["val_curve"][-1], 3), "hours": round(r["traintime_hours"], 2)}
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
