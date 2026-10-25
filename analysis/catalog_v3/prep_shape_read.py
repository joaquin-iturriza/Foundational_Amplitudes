"""Reads the shape proxy on catalog_v3 (prep_shape.py, its json copied home into data/prep_shape_raw<tag>.json): per
set (200 training classes and each held-out tree axis) and target, the median over classes of the proxy's held-out MSE
(the shape a fixed small regressor leaves) and of that MSE over the target's variance; the per-class ratio of each
target's MSE to the raw target's; and, against the short pretrainings' zero-shot shape on the same classes
(data/short_reads.json, the variance of q - t), the rank correlation per arm and target, which says whether the proxy
tracks what the network leaves.
    python3 analysis/catalog_v3/prep_shape_read.py [--tag ""]   -> data/prep_shape<tag>.json, figs/prep_shape<tag>
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
TARGETS = ("raw", "dless", "qperp", "soft")
SETS = ("train", "H1", "H2", "H2p5", "H3", "H6", "H7c", "H7E")
ARM_OF = {"raw": "baseline", "qperp": "dimensionless"}     # the target each short pretraining's arm trained on


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); a = ap.parse_args()
    raw = json.load(open(os.path.join(HERE, "data", f"prep_shape_raw{a.tag}.json")))
    sr = json.load(open(os.path.join(HERE, "data", "short_reads.json")))
    out = {"skipped": raw.get("skipped", [])}
    for s in SETS:
        d = raw.get(s, {})
        if not d:
            out[s] = {"n": 0}
            continue
        r = {"n": len(d)}
        for t in TARGETS:
            r[t] = {"mse": float(np.median([v[t]["mse"] for v in d.values()])),
                    "frac": float(np.median([v[t]["frac"] for v in d.values()])),
                    "ratio_to_raw": float(np.median([v[t]["mse"] / v["raw"]["mse"] for v in d.values()])),
                    "below_raw": int(sum(v[t]["mse"] < v["raw"]["mse"] for v in d.values()))}
        if s != "train":
            r["vs_network"] = {}
            for t, arm in ARM_OF.items():
                pr = sr.get(arm, {}).get(s, {}).get("probes", {})
                common = sorted(set(pr) & set(d))
                if len(common) >= 5:
                    r["vs_network"][f"{t}~{arm}"] = {
                        "n": len(common), "spearman": spearman([d[p][t]["mse"] for p in common],
                                                               [pr[p]["shape"] for p in common]),
                        "proxy_median": float(np.median([d[p][t]["mse"] for p in common])),
                        "network_shape_median": float(np.median([pr[p]["shape"] for p in common]))}
        out[s] = r
        print(f"{s:5s} n={r['n']:3d} " + " ".join(f"{t} {r[t]['mse']:.3g}" for t in TARGETS)
              + "  " + json.dumps(r.get("vs_network", {})))
    json.dump(out, open(os.path.join(HERE, "data", f"prep_shape{a.tag}.json"), "w"), indent=1)
    plot(raw, sr, a.tag)


def plot(raw, sr, tag):
    """(a) per set, the median proxy MSE for each target; (b) per held-out class, the proxy's MSE on the dimensionless
    arm's target against that arm's zero-shot shape."""
    import plot_style as ps
    fig, (ax, bx) = ps.figure(ncols=2)
    cols = {"raw": ps.C.blue, "dless": ps.C.green, "qperp": ps.C.vermillion, "soft": ps.C.grey}
    labs = {"raw": r"$\log|\mathcal{M}|^2$", "dless": r"$+(N-4)\log\hat s$", "qperp": "emission scales",
            "soft": "emission scales, soft min"}
    mk = {"raw": "o", "dless": "s", "qperp": "^", "soft": "v"}
    sets = [s for s in SETS if raw.get(s)]
    x = np.arange(len(sets))
    for j, t in enumerate(TARGETS):
        ax.plot(x + (j - 1.5) * 0.12, [np.median([v[t]["mse"] for v in raw[s].values()]) for s in sets], mk[t],
                color=cols[t], label=labs[t])
    names = {"train": "tr", "H1": "1", "H2": "2", "H2p5": "2.5", "H3": "3", "H6": "6", "H7c": "7c", "H7E": "7E"}
    ax.set_xticks(x); ax.set_xticklabels([names[s] for s in sets])
    ax.set_yscale("log"); ax.set_xlabel("training (tr) or held-out axis H"); ax.set_ylabel("proxy shape MSE")
    ps.legend(ax, "upper left")
    xs, ys = [], []
    for s in sets:
        if s == "train":
            continue
        pr = sr.get("dimensionless", {}).get(s, {}).get("probes", {})
        for p in set(pr) & set(raw[s]):
            xs.append(raw[s][p]["qperp"]["mse"]); ys.append(pr[p]["shape"])
    if xs:
        bx.plot(xs, ys, "o", mfc="none", color=ps.C.vermillion, label="held-out class")
        lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
        bx.plot([lo, hi], [lo, hi], ls="--", color=ps.C.grey, label="equal")
        bx.set_xscale("log"); bx.set_yscale("log")
        bx.set_xlabel("proxy shape MSE, emission scales"); bx.set_ylabel("network zero-shot shape")
        ps.legend(bx, "upper left")
    for a_ in (ax, bx):
        ps.make_room(a_, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", f"prep_shape{tag}"))


if __name__ == "__main__":
    main()
