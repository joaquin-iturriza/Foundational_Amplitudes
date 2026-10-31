#!/usr/bin/env python3
"""The wall time a scripts/job_catalog_v3_short.sh job needs: its training steps plus every zero-shot read it makes,
probe by probe, so the time limit counts the reads (the step-11 pp arms and the step-14 mixture arms of plan entry
catalog_coverage were submitted at 2.5 h counting training only, and TIMEOUT cut their PP4 and EE4 reads).

    python tools/short_wall.py --train <recipe> --cat v4 --axes H1 H2 ... PP4 EE4 [--eval] [--limit 02:30:00]

Prints one JSON line {hours, slurm_time, probes per axis, ...}; with --limit, exits 1 when the limit is below it
(the job checks its own limit with it before training). Rates, measured on CC-IN2P3 V100s (2026-10-31, the step-14
arm FA-ccin2p3-6c246603 and its seed-2 twin 1d46b938): 1.01 s per training step at batch 16384 (0.94 and 1.01, the
slower kept), 1.6 s per zero-shot probe (6c246603's PP4 read: 187 probes in 5.0 min, 02:49:40 to 02:54:42), 15 min of start-up (data
load, model build) and 1 min per axis (recipe split, model reload); then a 1.25 margin. A HETEROSC arm (--sigma-stage)
adds its sigma-only stage (run.py: 250 steps, 25 validations, the evaluation with predictions saved): 20 min, an
estimate (250 x 1.01 s = 4.2 min of steps, the rest validations and the saved evaluation), to be replaced by the
smoke's measured time. A training recipe with default-share entries (sampling.f_default > 0, catalog_v4_default50/25)
composes each of them at load (mix_compose.ensure_composed, serially, no log line until data loading ends): 6.1 s per
entry for its train and val pools (measured 2026-10-31 on the CC-IN2P3 login node with tools/mix_compose_check.py --load:
v4_bb_abbh 4.9 + 0.1 s, v4_udx_aawpz 4.2 + 0.2 s, v4_gg_gggg 5.8 + 0.3 s, the slowest kept; an estimate for a GPU
node), about 93 min for the 912 of catalog_v4_default50_train.yaml; the step-15 het mixture arms were submitted at
3.75 h without it. Eval-only jobs (--eval) start in 1.5 min, not 15: the step-20 re-reads (catalog_coverage,
2026-10-31, 14 eval-only jobs, logs read whole) took 15-31 s (MSE) and 20-82 s (het) from job start to their first
probe, the slowest kept. A het arm (--het) reads at 1.45 s per probe, its sigma head scored beside the mean: the step-20
het re-reads ran 1.25-1.31 s per probe on the 142 HD4+HD5 probes and 1.33-1.45 s on PP4+EE4's 1612 (span over the
probes, start-up excluded), the slowest kept; the MSE re-reads 0.58-0.65 s, under S_PROBE."""
import argparse, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# S_PROBE: the zero-shot reader with the parent config parsed once and the parent checkpoint's tensors shared (1
# worker): 42 s for 62 H1 probes of the MSE control on a V100, start-up included (A/B FA-ccin2p3-30cc8928, path D),
# 0.68 s per probe; was 0.71 with the checkpoint deep-copied per probe (bfec5501), 1.6 with the per-probe load
S_ITER, S_PROBE, STARTUP_MIN, AXIS_MIN, MARGIN = 1.01, 0.68, 15.0, 1.0, 1.25
SIGMA_STAGE_MIN = 20.0
STARTUP_EVAL_MIN, S_PROBE_HET = 1.5, 1.45
S_COMPOSE = 6.1


def axis_recipe(axis, cat, train):
    """scripts/job_catalog_v3_short.sh's axis_recipe"""
    fixed = {"XV3": "recipes/catalog_v3_shared_heldout.yaml", "XV4": "recipes/catalog_v4_shared_heldout.yaml",
             "PP": "recipes/catalog_v3_pp_test.yaml", "PP4": "recipes/catalog_v4_pp_test.yaml",
             "EE4": "recipes/catalog_v4_ee_test.yaml",
             "HD4": "recipes/catalog_v4_heldout_default_test.yaml",
             "HD5": "recipes/catalog_v4_heldout_default_H5_test.yaml", "TR": train}
    return fixed.get(axis) or f"recipes/catalog_{cat}_heldout_{axis}.yaml"


def n_probes(path, axis):
    import yaml
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(p):
        raise SystemExit(f"short_wall: no recipe {path} for axis {axis}")
    d = yaml.load(open(p), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    procs = d.get("processes", d) if isinstance(d, dict) else d
    names = [x["name"] for x in procs]
    if axis == "EE4":
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from pool_physics import DROPPED
        names = [n for n in names if n not in DROPPED]
    return len(names)


def iterations(train):
    import yaml
    p = train if os.path.isabs(train) else os.path.join(ROOT, train)
    return math.ceil(len(yaml.safe_load(open(p))["processes"]) * 1e4 / 16384 * 5)


def composed_entries(train):
    """entries of the training recipe composed at load (sampling.f_default > 0, the entry's over the recipe's)"""
    import yaml
    p = train if os.path.isabs(train) else os.path.join(ROOT, train)
    r = yaml.load(open(p), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    top = r.get("sampling") or {}
    return sum(float({**top, **(e.get("sampling") or {})}.get("f_default", 0) or 0) > 0 for e in r["processes"])


def parse_limit(s):
    """slurm's time limit (D-HH:MM:SS, HH:MM:SS, MM:SS) in hours"""
    s = s.strip()
    d = 0
    if "-" in s:
        d, s = s.split("-", 1)
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(d) * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600


def plan(train, cat, axes, eval_only=False, iters=None, sigma_stage=False):
    """sigma_stage: a het arm (its sigma-only stage when it trains, the het per-probe rate in its reads)"""
    probes = {a: n_probes(axis_recipe(a, cat, train), a) for a in axes}
    it = 0 if eval_only else (iters if iters is not None else iterations(train))
    s_probe = S_PROBE_HET if sigma_stage else S_PROBE
    startup = STARTUP_EVAL_MIN if eval_only else STARTUP_MIN
    minutes = startup + it * S_ITER / 60 + sum(probes.values()) * s_probe / 60 + AXIS_MIN * len(axes)
    if sigma_stage and not eval_only:
        minutes += SIGMA_STAGE_MIN
    n_comp = 0 if eval_only else composed_entries(train)
    minutes += n_comp * S_COMPOSE / 60
    hours = minutes * MARGIN / 60
    whole_min = math.ceil(hours * 60 / 15) * 15
    return {"hours": round(hours, 2), "slurm_time": f"{whole_min // 60:02d}:{whole_min % 60:02d}:00",
            "iterations": it, "composed_at_load": n_comp, "probes": probes, "n_probes": sum(probes.values()),
            "rates": {"s_per_step": S_ITER, "s_per_probe": s_probe, "startup_min": startup, "margin": MARGIN, "s_per_composed_entry": S_COMPOSE,
                      "sigma_stage_min": SIGMA_STAGE_MIN if sigma_stage and not eval_only else 0.0}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--cat", default="v4")
    ap.add_argument("--axes", nargs="+", required=True)
    ap.add_argument("--eval", action="store_true", help="mode eval: reads only, no training")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--sigma-stage", "--het", dest="sigma_stage", action="store_true",
                    help="a HETEROSC arm: its sigma-only stage after training, and the het per-probe read rate")
    ap.add_argument("--limit", help="the job's time limit (squeue %%l): exit 1 when below the plan")
    a = ap.parse_args(argv)
    out = plan(a.train, a.cat, a.axes, a.eval, a.iters, a.sigma_stage)
    if a.limit:
        out["limit_hours"] = round(parse_limit(a.limit), 2)
        out["ok"] = out["limit_hours"] >= out["hours"]
    print(json.dumps(out))
    if a.limit and not out["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
