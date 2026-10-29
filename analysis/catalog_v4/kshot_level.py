"""What a few labelled events fix on catalog_v4, and where its zero-shot error sits (v3_prep_explore, CPU, no new run).
The catalog_v4 pretraining default, the emission-scale target with dimensionless inputs and Muon at the search's best,
seeds 42 and 43 (scripts/job_catalog_v3_short.sh: v4_dimensionless_muon, v4_dimensionless_muon_seed2; 4264 steps),
read zero-shot at each run's best checkpoint on 2000 validation events per held-out pool (tools/zero_shot_parent.py;
predictions copied home into analysis/catalog_v3/data/short_reads/<run>/<axis>/<probe>.npz), on every catalog_v4
held-out axis (recipes/catalog_v4_heldout_<axis>.yaml). Per probe, from its truth t and prediction q (log|M|^2 up to
the target factor, which cancels in d = q - t; a sign-changing one-loop pool on log max(|M|^2, floor), the sign head's
magnitude, as analysis/catalog_v3/short_reads.py):
  zero     mean d^2, the zero-shot MSE
  level    (mean d)^2, the part one number per process removes; floor = var d, what any such number reaches
  off<K>   K events drawn at random, their mean error subtracted, the MSE on the other events (200 draws, mean)
  shr<K>   that offset pulled toward the other probes' offsets on the same axis (leave-one-probe-out, by precision,
           analysis/catalog_daniel/kshot_context_cmp.shrunk); shr0 is the axis' mean offset alone
The same procedure as Daniel's catalog (analysis/catalog_daniel/kshot_level.py) and as catalog_v3's offsets
(analysis/catalog_v3/kshot_context.py). Per run and axis: medians over probes, the level share (median over probes
of level / zero), the axis' mean offset and its spread over probes, and every expected probe without a read listed.
Also the loop-induced pools' offsets (names ending in _loop), against log((16 pi^2)^2) = 10.1, the factor a squared
one-loop amplitude carries and no tree does (par. What a few labelled events fix, on Daniel's catalog).
    python3 analysis/catalog_v4/kshot_level.py   -> data/kshot_level.json, figs/kshot_level.{png,pdf}
The last line printed is the JSON of the numbers docs/results.tex quotes.
"""
import glob, json, os, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_daniel"))
import pool_physics  # noqa: E402
from kshot_context_cmp import shrunk  # noqa: E402

SR = os.path.join(ROOT, "analysis", "catalog_v3", "data", "short_reads")
RUNS = ("v4_dimensionless_muon", "v4_dimensionless_muon_seed2")
AXES = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")
KS = (1, 4, 16)
NDRAW = 200
LOOP_FACTOR = float(np.log((16 * np.pi ** 2) ** 2))


def load(run, axis):
    exp = [e["name"] for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v4_heldout_{axis}.yaml")))["processes"]]
    d, phys = {}, {"unchecked": 0, "allowed": 0}
    for n in exp:
        f = os.path.join(SR, run, axis, n + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        st, _ = pool_physics.require_npz(z, f)
        if st in phys:
            phys[st] += 1
        if "q" not in z.files or len(z["q"]) != len(z["t"]) or not len(z["t"]):
            continue
        d[n] = z["q"].astype(float) - z["t"].astype(float)
    return exp, d, phys


def per_probe(d, rng):
    offs = {p: float(v.mean()) for p, v in d.items()}
    out = {}
    for p, v in d.items():
        oth = np.array([x for q, x in offs.items() if q != p])
        r = {"zero": float(np.mean(v ** 2)), "level": offs[p] ** 2, "floor": float(v.var()), "offset": offs[p],
             "shr0": shrunk(v, oth, np.array([], dtype=int))}
        for K in KS:
            picks = [rng.choice(len(v), K, replace=False) for _ in range(NDRAW)]
            r[f"off{K}"] = float(np.mean([np.mean((np.delete(v, pk) - v[pk].mean()) ** 2) for pk in picks]))
            r[f"shr{K}"] = float(np.mean([shrunk(v, oth, pk) for pk in picks]))
        out[p] = r
    return out


def summary(rows):
    keys = ("zero", "level", "floor", "shr0") + tuple(f"{m}{K}" for m in ("off", "shr") for K in KS)
    s = {k: float(np.median([r[k] for r in rows.values()])) for k in keys}
    s["level_share"] = float(np.median([r["level"] / r["zero"] for r in rows.values()]))
    o = np.array([r["offset"] for r in rows.values()])
    s.update({"offset_mean": float(o.mean()), "offset_sd": float(o.std()), "n": len(rows),
              "shr16_below_off16": float(np.mean([r["shr16"] < r["off16"] for r in rows.values()]))})
    return s


def main():
    rng = np.random.default_rng(0)
    res, flat = {}, {}
    for run in RUNS:
        for ax in AXES:
            exp, d, phys = load(run, ax)
            rows = per_probe(d, rng)
            s = summary(rows) if rows else {"n": 0}
            s.update({"n_expected": len(exp), "missing": sorted(set(exp) - set(rows)), "physics": phys,
                      "loop_induced": {p: round(r["offset"], 3) for p, r in rows.items() if p.endswith("_loop") or "_loop_" in p}})
            res.setdefault(run, {})[ax] = {**s, "probes": rows}
            print(f"{run:30s} {ax:5s} {s['n']:3d}/{len(exp):3d} zero {s.get('zero', 0):.3g} level {s.get('level', 0):.3g} "
                  f"share {s.get('level_share', 0):.2f} floor {s.get('floor', 0):.3g} | off1 {s.get('off1', 0):.3g} "
                  f"off16 {s.get('off16', 0):.3g} shr0 {s.get('shr0', 0):.3g} shr1 {s.get('shr1', 0):.3g} "
                  f"shr16 {s.get('shr16', 0):.3g} | mean offset {s.get('offset_mean', 0):+.2f} sd {s.get('offset_sd', 0):.2f}"
                  + (f" | missing {len(s['missing'])}" if s["missing"] else "") + (f" | loop-induced {s['loop_induced']}" if s["loop_induced"] else ""))
            for k in ("n", "zero", "level", "level_share", "floor", "off1", "off16", "shr0", "shr1", "shr16",
                      "offset_mean", "offset_sd"):
                if k in s:
                    flat[f"{run}.{ax}.{k}"] = round(s[k], 3) if k != "n" else s[k]
            for p, o in s["loop_induced"].items():
                flat[f"{run}.{ax}.loop_offset.{p}"] = o
    flat["loop_factor"] = round(LOOP_FACTOR, 2)
    json.dump({"summary": {r: {a: {k: v for k, v in res[r][a].items() if k != "probes"} for a in AXES} for r in RUNS},
               "probes": {r: {a: res[r][a]["probes"] for a in AXES} for r in RUNS}},
              open(os.path.join(HERE, "data", "kshot_level.json"), "w"), indent=1)
    plot(res)
    print(json.dumps(flat))


def plot(res):
    import plot_style as ps
    fig, ax = ps.figure()
    x = np.arange(len(AXES))
    for run, mk, off in zip(RUNS, ("o", "s"), (-0.12, 0.12)):
        s = [res[run][a] for a in AXES]
        lab = run.endswith("seed2")
        ax.scatter(x + off, [r["zero"] for r in s], marker=mk, color=ps.C.blue, label=None if lab else "zero-shot")
        ax.scatter(x + off, [r["shr1"] for r in s], marker=mk, color=ps.C.green, label=None if lab else "shrunk offset, $K=1$")
        ax.scatter(x + off, [r["off16"] for r in s], marker=mk, color=ps.C.vermillion, label=None if lab else "offset, $K=16$")
        ax.scatter(x + off, [r["floor"] for r in s], marker=mk, facecolors="none", edgecolors=ps.C.grey,
                   label=None if lab else "floor")
    names = {"H1": "new comb.", "H2": r"$2\to4$", "H2p5": r"$2\to5$", "H3": "top", "H5": "one loop",
             "H6": r"$\alpha_s{=}0.13$", "H7E": "6-20 TeV", "H7c": r"$p_T{\geq}30$"}
    ax.set_xticks(x); ax.set_xticklabels([names[a] for a in AXES], rotation=35, ha="right")
    ax.set_yscale("log"); ax.set_ylabel(r"median MSE$(\log|\mathcal{M}|^2)$")
    ps.legend(ax, "upper left")
    ps.make_room(ax)
    ps.save(fig, os.path.join(HERE, "figs", "kshot_level"))


if __name__ == "__main__":
    main()
