"""Synthetic one-loop pools against the real one-loop probes' pools (D44; docs/results.tex sec:syn-loops): per pool the
fraction of negative events, the signed log sgn(x) log(1 + |x|/s) (s the 1% quantile of |x|, data.signedlog_quantile)
and the magnitude log|x| standardised (the sign-head target), its Spearman correlation with s and with t/s
(t = (p_a - p_3)^2, beam a against the first final particle), histograms and a subsample for scatter plots.
Runs where the pools are (a CPU job on a site; the synthetic pools are built here if missing, the real ones must be
in the cache). The born of a synthetic pool is re-evaluated (tools/synthetic_loops.evaluate_loop) for V / |M0|^2.
    python analysis/loops/loop_validate.py --out runs/loop_val/results_loop_val.json
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import yaml                                         # noqa: E402
import datagen                                      # noqa: E402
import mg5_pipeline_final as mg                     # noqa: E402
from tools import synthetic_amplitudes as syn       # noqa: E402
from tools import synthetic_loops as sl             # noqa: E402

PROBES = ["ee_dd_nlo", "ee_bb_nlo", "ee_dd_ew_nlo", "ee_ttbar_nlo_thr"]
EDGES_SL = np.linspace(-12, 12, 97)                 # signed log
EDGES_Z = np.linspace(-6, 6, 97)                    # standardised log|x|


def mink2(v):
    return v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1)


def pool(spec, sampling, role="train"):
    mg.register_recipe_processes([dict(spec)], default_sampling=sampling)
    real = not syn.is_synthetic(spec["name"])
    path = datagen.ensure_split_set([{"process": spec["name"], "sqrts_min": spec["sqrts"][0],
                                      "sqrts_max": spec["sqrts"][1], "n_events": spec["n_" + role]}],
                                    role=role, seed=42, require_cache=real)[spec["name"]]
    return np.load(path, mmap_mode="r")


def stats(name, a, n_parts, rng, n_sub=2500):
    P = np.asarray(a[:, :n_parts * 4], float).reshape(len(a), n_parts, 4)
    x = np.asarray(a[:, -1], float)
    s = mink2(P[:, 0] + P[:, 1])
    t = mink2(P[:, 0] - P[:, 2])
    ax = np.abs(x[x != 0])
    sc = float(np.quantile(ax, 0.01))
    sl_ = np.sign(x) * np.log1p(np.abs(x) / sc)
    lg = np.log(np.maximum(np.abs(x), sc))
    z = (lg - lg.mean()) / lg.std()
    neg = float(np.mean(x < 0))
    out = {"name": name, "n": int(len(x)), "neg_frac": neg, "scale": sc,
           "log_abs_std": float(lg.std()), "log_abs_mean": float(lg.mean()),
           "rho_sl_s": float(spearmanr(sl_, s).correlation), "rho_sl_t": float(spearmanr(sl_, t / s).correlation),
           "rho_abs_s": float(spearmanr(lg, s).correlation), "rho_abs_t": float(spearmanr(lg, t / s).correlation),
           "rho_neg_s": float(spearmanr(x < 0, s).correlation) if 0 < neg < 1 else None,
           "rho_neg_t": float(spearmanr(x < 0, t / s).correlation) if 0 < neg < 1 else None,
           "hist_sl": np.histogram(np.clip(sl_, EDGES_SL[0], EDGES_SL[-1]), EDGES_SL)[0].tolist(),
           "hist_z": np.histogram(np.clip(z, EDGES_Z[0], EDGES_Z[-1]), EDGES_Z)[0].tolist()}
    i = np.sort(rng.choice(len(x), size=min(n_sub, len(x)), replace=False))
    out["sub"] = {"sqrts": np.round(np.sqrt(s[i]), 3).tolist(), "t_over_s": np.round(t[i] / s[i], 5).tolist(),
                  "sl": np.round(sl_[i], 4).tolist()}
    return out, P, x, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default=os.path.join(ROOT, "recipes", "loop_synthetic.yaml"))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    res = {"edges_sl": EDGES_SL.tolist(), "edges_z": EDGES_Z.tolist(), "real": [], "synthetic": []}
    for p in PROBES:
        doc = yaml.safe_load(open(os.path.join(ROOT, "recipes", f"transfer_probe_{p}.yaml")))
        spec = doc["processes"][0]
        try:
            arr = pool(spec, doc.get("sampling"))
        except RuntimeError as e:
            print(f"{p}: real pool missing here ({e})", flush=True)
            continue
        n_parts = len(mg.PROCESSES[spec["name"]]["pdg_ids"])
        st, *_ = stats(p, arr, n_parts, rng)
        res["real"].append(st)
        print(f"real {p:18s} n={st['n']} neg={st['neg_frac']:.2f} std(log|x|)={st['log_abs_std']:.2f} "
              f"rho(sl,s)={st['rho_sl_s']:+.2f} rho(sl,t/s)={st['rho_sl_t']:+.2f}", flush=True)
    doc = yaml.safe_load(open(a.recipe))
    for spec in doc["processes"]:
        if not spec["name"].startswith("synl"):
            continue
        arr = pool(spec, doc.get("sampling"))
        stc = syn.structure(spec["name"])
        n_parts = len(stc["pdg_ids"])
        st, P, x, i = stats(spec["name"], arr, n_parts, rng)
        V, born, Vu = sl.evaluate_loop(stc, P[i], parts=True)
        assert np.allclose(V, x[i], rtol=1e-10, atol=0), f"{spec['name']}: pool labels differ from the labeller"
        st.update({"type": stc["loop"]["type"], "nfinal": stc["nfinal"], "order": stc["order"],
                   "v_over_b": np.round(V / born, 4).tolist(), "vu_over_b": np.round(Vu / born, 4).tolist()})
        res["synthetic"].append(st)
        print(f"syn  {spec['name']:18s} {st['type']} 2->{st['nfinal']} n={st['n']} neg={st['neg_frac']:.2f} "
              f"std(log|x|)={st['log_abs_std']:.2f} rho(sl,s)={st['rho_sl_s']:+.2f} rho(sl,t/s)={st['rho_sl_t']:+.2f}",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
