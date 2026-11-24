"""critical_batch (D532): the gradient noise scale along one run, from its <run>/noise_scale.json (find_BS).

    python analysis/critical_batch/noise_read.py <noise_scale.json> <name> [--window 20] [--batch 16384]

S = tr Sigma and |G|^2 per measurement from base_experiment.noise_scale_terms, summed over the parameter sets (Muon,
AdamW), averaged separately over windows of `window` consecutive chunk measurements, B_simple = mean S / mean |G|^2.
The brute-force checks: S and |G|^2 from 16 independent batch gradients at one step, beside the chunk estimate on the
same 16 batches with its bootstrap standard error over those batches (the brute record keeps sums only, so it has no
error bar of its own). Writes
analysis/critical_batch/data/<name>.json and figs/<name>.png/.pdf."""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from base_experiment import noise_scale_terms  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def terms(r):
    """(S, |G|^2) of one record, summed over its parameter sets."""
    sets = sorted({k[:-len("_g2_small")] for k in r if k.endswith("_g2_small")})
    st = [noise_scale_terms(r[f"{s}_g2_small"], r[f"{s}_g2_big"], r["b_small"], r["b_big"]) for s in sets]
    return sum(a for a, _ in st), sum(b for _, b in st), {s: v for s, v in zip(sets, st)}


def read(recs, window):
    ch = [r for r in recs if r["kind"] == "chunk"]
    out = {"windows": [], "checks": []}
    for i in range(0, len(ch) - window + 1, window):
        w = ch[i:i + window]
        ST = np.array([terms(r)[:2] for r in w])
        per = {}
        for s in terms(w[0])[2]:
            a = np.array([terms(r)[2][s] for r in w])
            per[s] = {"S": float(a[:, 0].mean()), "G2": float(a[:, 1].mean())}
        S, G2 = ST.mean(0)
        out["windows"].append({"step_lo": w[0]["step"], "step_hi": w[-1]["step"], "S": float(S), "G2": float(G2),
                               "S_se": float(ST[:, 0].std(ddof=1) / np.sqrt(len(w))),
                               "G2_se": float(ST[:, 1].std(ddof=1) / np.sqrt(len(w))),
                               "B_simple": float(S / G2) if G2 > 0 else None, "sets": per})
    for r in (r for r in recs if r["kind"] == "brute"):
        same = [terms(c)[:2] for c in recs if c["kind"] == "chunk_at_brute" and c["step"] == r["step"]]
        a = np.array(same)
        Sb, Gb, _ = terms(r)
        rng = np.random.default_rng(0)
        boot = [a[rng.integers(0, len(a), len(a))].mean(0) for _ in range(2000)]
        se = np.std(boot, axis=0)
        out["checks"].append({"step": r["step"], "n_batches": r["k"], "S_brute": Sb, "G2_brute": Gb,
                              "B_brute": Sb / Gb if Gb > 0 else None, "S_chunk": float(a[:, 0].mean()),
                              "G2_chunk": float(a[:, 1].mean()), "S_chunk_se": float(se[0]),
                              "G2_chunk_se": float(se[1]),
                              "B_chunk": float(a[:, 0].mean() / a[:, 1].mean()) if a[:, 1].mean() > 0 else None})
    return out


def figure(res, batch, base):
    import plot_style as ps
    fig, (axB, axT) = ps.figure(ncols=2)
    w = [x for x in res["windows"] if x["B_simple"] is not None]
    mid = [(x["step_lo"] + x["step_hi"]) / 2 for x in w]
    axB.plot(mid, [x["B_simple"] for x in w], color=ps.C.blue, label=r"$B_{\rm simple}$, batch split in chunks")
    c = [x for x in res["checks"] if x["B_brute"] is not None]
    axB.plot([x["step"] for x in c], [x["B_brute"] for x in c], "o", color=ps.C.vermillion,
             label=r"$B_{\rm simple}$, 16 independent batches")
    axB.axhline(batch, color="k", ls="--", label="training batch")
    axB.set_yscale("log")
    axB.set_xlabel("training step")
    axB.set_ylabel(r"$B_{\rm simple}={\rm tr}\,\Sigma/|G|^2$ (events)")
    ps.legend(axB, "upper left")
    allw = res["windows"]
    m2 = [(x["step_lo"] + x["step_hi"]) / 2 for x in allw]
    axT.plot(m2, [x["S"] for x in allw], color=ps.C.blue, label=r"${\rm tr}\,\Sigma$")
    axT.plot(m2, [x["G2"] * batch for x in allw], color=ps.C.vermillion, label=r"$|G|^2\times 16384$")
    axT.set_yscale("symlog", linthresh=max(1e-12, min(abs(x["S"]) for x in allw) / 10))
    axT.set_xlabel("training step")
    axT.set_ylabel("gradient norm terms")
    ps.legend(axT, "upper left")
    ps.save(fig, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("name")
    ap.add_argument("--window", type=int, default=20); ap.add_argument("--batch", type=int, default=16384)
    a = ap.parse_args()
    res = read(json.load(open(a.path)), a.window)
    res["source"] = os.path.abspath(a.path)
    os.makedirs(f"{HERE}/data", exist_ok=True)
    json.dump(res, open(f"{HERE}/data/{a.name}.json", "w"), indent=1)
    for x in res["checks"]:
        print(f"CHECK step {x['step']}: S brute {x['S_brute']:.4g} chunk {x['S_chunk']:.4g} +- {x['S_chunk_se']:.2g}; "
              f"|G|^2 brute {x['G2_brute']:.4g} chunk {x['G2_chunk']:.4g} +- {x['G2_chunk_se']:.2g}; "
              f"B brute {x['B_brute']} chunk {x['B_chunk']}")
    for x in res["windows"][::max(1, len(res['windows']) // 10)]:
        print(f"WINDOW steps {x['step_lo']}-{x['step_hi']}: S {x['S']:.4g} |G|^2 {x['G2']:.4g} B_simple {x['B_simple']}")
    figure(res, a.batch, f"{HERE}/figs/{a.name}")


if __name__ == "__main__":
    main()
