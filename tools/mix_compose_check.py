#!/usr/bin/env python3
"""The load-time default-share mixture (mix_compose.py) against the stored _smixd<NN> pool, on named classes, before the
stored pools go (catalog_coverage step 14; the user, 2026-10-31). Per class, recipe and role, as one JSON line:
  provenance  the catalog_v4 pool's rows told apart by replay (bulk, forward, flat), against the sampler's counts
              (provenance() raises unless every replayed event is found and the counts match)
  default     the composed default-sampled rows are the stored pool's own: share of them found in the stored pool by
              their momenta, max relative |M|^2 difference and max relative alpha_s difference on those rows
  same law    the composed and stored pools are two draws of one mixture: two-sample KS of log|M|^2 and of
              sqrt(s_hat) over the whole pool (statistic, p-value) and the difference of the means of log|M|^2 in units
              of its standard error; the stored pool's flat and forward rows come from another random stream (no bulk
              at f_default 0.50), so only the default rows can match row for row
--load: once the stored pools are gone, the loader's own path instead (datagen.ensure_dataset with require_cache, as a
run calls it): the pool it returns, whether it was composed at load, its rows, shares and the share of finite |M|^2
    site run --item catalog_coverage ccin2p3 FA -- python tools/mix_compose_check.py recipes/catalog_v4_default50_train.yaml v4_bb_abbh ...
"""
import argparse, json, os, sys, time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import datagen  # noqa: E402
import mg5_pipeline_final as mg  # noqa: E402
import mix_compose as mc  # noqa: E402


def ks(a, b):
    from scipy.stats import ks_2samp
    r = ks_2samp(a, b)
    return [round(float(r.statistic), 5), round(float(r.pvalue), 4)]


def sqrts(rows, k):
    P = rows[:, :4 * k].reshape(len(rows), k, 4)
    q = P[:, 0] + P[:, 1]
    return np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe"); ap.add_argument("names", nargs="+")
    ap.add_argument("--roles", nargs="+", default=["train", "test"]); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load", action="store_true")
    a = ap.parse_args()
    spec = yaml.safe_load(open(a.recipe))
    mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
                                   "sampling": p.get("sampling")} for p in spec["processes"]],
                                 default_sampling=spec.get("sampling"))
    ent = {p["name"]: p for p in spec["processes"]}
    for name in a.names:
        p = ent[name]; lo, hi = map(float, p["sqrts"]); k = int(mg.PROCESSES[name]["nfinal"]) + 2
        for role in a.roles:
            n = int(p["n_" + role]); t0 = time.time()
            if a.load:
                path = datagen.ensure_dataset(name, lo, hi, n, role, a.seed, require_cache=True)
                rows = np.load(path, mmap_mode="r"); side = json.load(open(path + ".recipe.json"))
                der = side.get("derived_from") or {}
                print(json.dumps({"class": name, "recipe": os.path.basename(a.recipe), "role": role, "n": n,
                                  "seconds": round(time.time() - t0, 1), "path": path,
                                  "composed_at_load": bool(der.get("composed_at_load")), "rows": int(len(rows)),
                                  "shares": der.get("shares"), "physics_block": bool(side.get("physics")),
                                  "finite_me2": float(np.isfinite(np.asarray(rows[:, -1])).mean())}), flush=True)
                continue
            rec = mg.variable_energy_recipe(name, lo, hi, n, role=role, seed=a.seed)
            stored_path = mg.recipe_output_path(rec, datagen.dest_for_role(role))
            stored = np.load(stored_path, mmap_mode="r")
            with mc._Swap(name, mc.control_config(name)):
                rec_c = mg.variable_energy_recipe(name, lo, hi, n, role=role, seed=a.seed)
                ctrl = mg.recipe_output_path(rec_c, datagen.dest_for_role(role))
            rows, kinds = mc.compose(name, lo, hi, n, role, a.seed, ctrl)
            dt = time.time() - t0
            st = np.asarray(stored, np.float64)
            d = np.where(kinds == mc.KIND["default"])[0]
            m = mc.match_rows(rows[d, :4 * k].reshape(len(d), k, 4), st[:, :4 * k].reshape(len(st), k, 4))
            found = [(j, int(i)) for j, i in zip(d, m) if i >= 0]
            rel = [abs(rows[j, -1] - st[i, -1]) / abs(st[i, -1]) for j, i in found if st[i, -1] != 0]
            a_st = np.load(stored_path[:-len(".npy")] + ".alphas.npy")
            a_co = mc.alphas_of_rows(rows, k, mg.PROCESSES[name])
            arel = [abs(a_co[j] - a_st[i]) / a_st[i] for j, i in found]
            with np.errstate(divide="ignore", invalid="ignore"):
                lc, ls = np.log(np.abs(rows[:, -1])), np.log(np.abs(st[:, -1]))
            lc, ls = lc[np.isfinite(lc)], ls[np.isfinite(ls)]
            se = np.sqrt(np.var(lc) / len(lc) + np.var(ls) / len(ls))
            print(json.dumps({
                "class": name, "recipe": os.path.basename(a.recipe), "role": role, "n": n, "seconds": round(dt, 1),
                "stored": stored_path, "control": ctrl,
                "shares": {kk: int((kinds == v).sum()) for kk, v in mc.KIND.items()},
                "default_rows": len(d), "default_found_in_stored": len(found),
                "default_max_rel_me2": float(max(rel)) if rel else None,
                "default_max_rel_alphas": float(max(arel)) if arel else None,
                "ks_logme2": ks(lc, ls), "ks_sqrts": ks(sqrts(rows, k), sqrts(st, k)),
                "mean_logme2_diff_in_se": round(float((lc.mean() - ls.mean()) / se), 3)}), flush=True)


if __name__ == "__main__":
    main()
