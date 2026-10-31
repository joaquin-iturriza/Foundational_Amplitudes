#!/usr/bin/env python3
"""MadEvent cost of every built lhe_pp pool, read from its sidecar (lhe_run: events, seconds, cores, events per run).

Run on the site where the pools are (through `site run`); prints one JSON object {entry: {...}} and, with --out,
writes it. Each entry: n_lhe (events MadEvent returned over its runs), madevent_s (wall seconds of the MadEvent runs),
mg_cores (cores MadEvent ran on), core_s_per_event = madevent_s * mg_cores / n_lhe (an upper bound of the one-core
cost: it assumes no loss to parallel inefficiency), events_per_run, n_req (events asked of each run), nfinal,
output_s (wall seconds of MadGraph's output and compile of the MadEvent and standalone directories, summed over the
runs' part jobs for a pool built in parts).
The default-share layout (recipes/gen_catalog_v4_default.py) sizes its per-run sub-entries from it.

    python tools/lhe_rates.py [--prefix pp13v4_ pp13_ ee4_] [--seed 42] [--out analysis/catalog_v4/data/lhe_rates.json]
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import datagen  # noqa: E402
import mg5_pipeline_final as mg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs="+", default=["pp13v4_", "pp13_", "ee4_"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    a = ap.parse_args()
    out, missing = {}, 0
    for name in sorted(n for n in mg.PROCESSES if n.startswith(tuple(a.prefix)) and mg.is_lhe_pp(n)):
        cfg = mg.PROCESSES[name]
        lo, hi = mg.lhe_layout(cfg)["train"]
        role = "train" if hi > lo else "test"
        lo, hi = mg.lhe_layout(cfg)[role]
        if hi <= lo:
            continue
        rec = mg.lhe_pp_recipe(name, 13000, 13000, hi - lo, role=role, seed=a.seed)
        path = mg.recipe_output_path(rec, datagen.dest_for_role(role))
        try:
            side = json.load(open(path + ".recipe.json"))
        except (OSError, json.JSONDecodeError):
            missing += 1
            continue
        if side.get("recipe_id") != mg.recipe_id(rec):
            missing += 1
            continue
        r = side.get("lhe_run") or {}
        s = (r.get("seconds") or {}).get("madevent")
        n = r.get("n_lhe_events")
        if not s or not n:
            continue
        c = int(r.get("mg_cores", 1))
        out[name] = {"n_lhe": int(n), "madevent_s": float(s), "mg_cores": c,
                     "core_s_per_event": float(s) * c / int(n), "events_per_run": r.get("events_per_run"),
                     "n_req": rec["lhe"]["n_requested"], "n_runs": int(cfg["lhe"].get("n_runs", 1)),
                     "nfinal": int(cfg["nfinal"]), "output_s": (r.get("seconds") or {}).get("output")}
    print(json.dumps({"n": len(out), "missing_or_stale": missing}), file=sys.stderr)
    txt = json.dumps(out, indent=1, sort_keys=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
