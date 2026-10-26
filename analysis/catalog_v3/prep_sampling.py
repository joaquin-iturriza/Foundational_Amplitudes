"""Reads the sampling sensitivity on catalog_v3 (prep_sampling_pools.py, run on CC; its SAMPLING lines copied home into
data/prep_sampling_raw.txt). Two reweightings of each pool's events (flat in log s_hat; the 20% of events nearest a
singular region carrying half the weight), defined in prep_sampling_pools.py.
Data part, per set (the 200 training classes; the held-out axes together) and target (raw log|M|^2, + (N - 4) log s_hat,
+ the emission scales, the same with tau = 0.3), over the positive pools: the median shift of the level in units of the
pool's std and the median ratio of the std, under each reweighting.
Network part, per short arm and held-out axis, over the probes every arm has numbers on: the median zero-shot MSE in
log|M|^2, its level (squared mean error) and shape (the rest) under uniform, flat and low weights; per axis whether the
arms' order by median MSE under a reweighting is the uniform one, and the best arm under each. Arms with no numbers on a
probe (not aligned, no prediction) are listed.
Figure: (a) per target, the median level shift under each reweighting, training and held out; (b) per arm, the median
zero-shot MSE over all held-out probes under each reweighting against the uniform one.
    python3 analysis/catalog_v3/prep_sampling.py   -> data/prep_sampling.json, figs/prep_sampling.{png,pdf}
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
AXES = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7c", "H7E")
TARGETS = ("raw", "dless", "qperp", "soft")
W = ("uni", "flat", "low")
ARMS = {"baseline": "base", "dimensionless": "qperp", "dimensionless_shat": "shat", "dimensionless_hybrid": "hyb",
        "dimensionless_blend025": "b25", "dimensionless_blend050": "b50", "dimensionless_context": "ctx",
        "dimensionless_twin": "twin"}


def load_raw():
    rows = []
    for l in open(os.path.join(HERE, "data", "prep_sampling_raw.txt")):
        if l.startswith("SAMPLING "):
            rows.append(json.loads(l[len("SAMPLING "):]))
    return rows


def main():
    rows = load_raw()
    out, flat = {}, {"n_classes": len(rows)}
    # data part
    for s, sel in (("train", lambda r: r["set"] == "train"), ("held", lambda r: r["set"] != "train")):
        d = [r for r in rows if sel(r) and not r["signed"]]
        o = {"n": len(d)}
        for t in TARGETS:
            for wn in ("flat", "low"):
                o[f"{t}_shift_{wn}"] = float(np.median([r[t][f"shift_{wn}"] for r in d]))
                o[f"{t}_sd_{wn}"] = float(np.median([r[t][f"sd_{wn}"] for r in d]))
        out[s] = o
        sh = {"train": "tr", "held": "ho"}[s]
        flat[f"{sh}_n"] = o["n"]
        for k, v in o.items():
            if k != "n":
                flat[f"{sh}_{k}"] = round(v, 3)
    # network part
    held = [r for r in rows if r["set"] != "train"]
    missing = {}
    for r in held:
        for a, v in r.get("net", {}).items():
            if "why" in v:
                missing.setdefault(a, []).append((r["set"], r["name"], v["why"]))
    out["missing"] = missing
    flat["n_missing"] = sum(len(v) for v in missing.values())
    full = lambda r: all(a in r.get("net", {}) and "why" not in r["net"][a] for a in ARMS)
    out["axes"] = {}
    n_same = 0
    for ax in AXES + ("all",):
        pr = [r for r in held if (ax == "all" or r["set"] == ax) and full(r)]
        o = {"n": len(pr)}
        if not pr:
            out["axes"][ax] = o
            continue
        for a in ARMS:
            for wn in W:
                m = np.array([r["net"][a][wn] for r in pr])           # mse, level, shape per probe
                o.setdefault(a, {})[wn] = [float(np.median(m[:, i])) for i in range(3)]
        order = {wn: sorted(ARMS, key=lambda a: o[a][wn][0]) for wn in W}
        o["order"] = order
        o["same_order"] = {wn: order[wn] == order["uni"] for wn in ("flat", "low")}
        o["best"] = {wn: order[wn][0] for wn in W}
        out["axes"][ax] = o
        if ax != "all":
            n_same += sum(o["same_order"].values())
        print(f"{ax:5s} n={o['n']:3d} best {o['best']} same order flat {o['same_order']['flat']} "
              f"low {o['same_order']['low']}")
        print("      " + "  ".join(f"{ARMS[a]}:" + "/".join(f"{o[a][wn][0]:.3g}" for wn in W) for a in order["uni"]))
        sh = "all" if ax == "all" else ax
        flat[f"{sh}_n_probes"] = o["n"]
        for a, ah in ARMS.items():
            for wn in W:
                flat[f"{sh}_{ah}_{wn}"] = round(o[a][wn][0], 4)
            flat[f"{sh}_{ah}_low_over_uni"] = round(o[a]["low"][0] / o[a]["uni"][0], 3)
            flat[f"{sh}_{ah}_flat_over_uni"] = round(o[a]["flat"][0] / o[a]["uni"][0], 3)
        for wn in W:
            flat[f"{sh}_best_{wn}"] = ARMS[o["best"][wn]]
    flat["n_axis_reweightings_same_order"] = n_same
    flat["n_axis_reweightings"] = 2 * sum(1 for ax in AXES if out["axes"][ax]["n"] > 0)
    json.dump(out, open(os.path.join(HERE, "data", "prep_sampling.json"), "w"), indent=1)
    plot(out)
    print(json.dumps({k: v for k, v in out.items() if k in ("train", "held")}))
    print(json.dumps(flat))


def plot(out):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    ax = axs[0]
    x = np.arange(len(TARGETS))
    for s, mk in (("train", "o"), ("held", "^")):
        for wn, col in (("flat", ps.C.blue), ("low", ps.C.vermillion)):
            lab = f"{'training' if s == 'train' else 'held out'}, {'flat in log s' if wn == 'flat' else 'singular region'}"
            ax.plot(x, [out[s][f"{t}_shift_{wn}"] for t in TARGETS], mk, mfc="none" if s == "held" else col,
                    color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(["raw", "s factor", "emission", "soft"])
    ax.set_ylabel("median level shift [std]")
    ps.legend(ax, "upper left")
    ax = axs[1]
    o = out["axes"]["all"]
    u = [o[a]["uni"][0] for a in ARMS]
    for wn, col, mk in (("flat", ps.C.blue, "o"), ("low", ps.C.vermillion, "^")):
        ax.plot(u, [o[a][wn][0] for a in ARMS], mk, mfc="none", color=col,
                label="flat in log s" if wn == "flat" else "singular region")
    lo, hi = min(u) / 1.5, max(max(o[a][wn][0] for a in ARMS for wn in W), max(u)) * 1.5
    ax.plot([lo, hi], [lo, hi], ls="--", color=ps.C.grey, label="equal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("zero-shot MSE, uniform"); ax.set_ylabel("zero-shot MSE, reweighted")
    ps.legend(ax, "upper left")
    for a in axs:
        ps.make_room(a, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "prep_sampling"))


if __name__ == "__main__":
    main()
