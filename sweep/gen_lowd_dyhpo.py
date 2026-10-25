"""The small-D fine-tuning study, second round (D120; docs/results.tex par:lowd_ft and the hand-off entry "Fine-tuning at
small D"): single-fidelity DyHPO for the three methods that did not win at the default's HPs, on the first round's 60
cells (analysis/transfer/data/lowd_ft_spec.json: e+e- -> Z gamma, e+e- -> d dbar, u d -> u d, e+e- -> b bbar at one loop,
u ubar -> g g; rungs 1, 7, 9; D = 10 to 10^2.5). Each sweep is its cell's grid config (sweep_config_tp3_r<rung>fte_<probe>
_d<k>.yaml: the cell's steps, its fixed parameters, its parent, 5 trials, the grid's fine-tune search space) with the
method's flags fixed and, for two methods, one more searched dimension:
  lpft     linear probe, then fine-tune (fine_tune.linear_probe=true); the grid's space as it is
  l2sp     the pull toward the start, (lambda/2)|theta - theta*|^2 on every weight (fine_tune.ewc.enabled, uniform);
           fine_tune.ewc.lambda log-uniform in [1e-8, 1]: from a penalty far below these cells' losses to 100 times the
           first round's untuned 1e-2 (which cost up to 0.87 decades at D = 10^2.5). The forgetting study's EWC range
           (1e2 to 1e7) multiplies a Fisher whose magnitude was not recorded, so it cannot be mapped onto a uniform weight
           exactly (not checked); this range brackets the first round's value on both sides by two decades and more.
  freeze   the lower blocks frozen, depth 2, 4 or 6 of 8 (fine_tune.freeze_blocks categorical)
The default's side of the comparison is the grid's own 5-trial search of each cell (analysis/transfer/cells.best).
Sweep names: tp3_lowd_<method>_r<rung>_<probe>_d<k> (the rung is in the name: three rungs share a probe and D).
    python3 sweep/gen_lowd_dyhpo.py              -> sweep/sweep_config_tp3_lowd_*.yaml (180 configs)
    python3 sweep/gen_lowd_dyhpo.py --generate   on the site: generate_sweep for each not yet generated (no submission;
                                                 FA_NEW_SWEEP_REASON=D120 for the cell guard)
"""
import argparse, json, os, sys, zlib

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
METHODS = {
    "lpft": ({"fine_tune.linear_probe": "true"}, []),
    "l2sp": ({"fine_tune.ewc.enabled": "true", "fine_tune.ewc.uniform": "true"},
             [{"name": "fine_tune.ewc.lambda", "type": "float_log", "low": 1.0e-8, "high": 1.0}]),
    "freeze": ({}, [{"name": "fine_tune.freeze_blocks", "type": "categorical",
                     "choices": ["[0,1]", "[0,1,2,3]", "[0,1,2,3,4,5]"]}]),
}


def configs():
    spec = json.load(open(os.path.join(ROOT, "analysis", "transfer", "data", "lowd_ft_spec.json")))
    out = []
    for c in spec:
        base = yaml.safe_load(open(os.path.join(ROOT, c["config"])))
        for m, (fixed, extra) in METHODS.items():
            name = f"tp3_lowd_{m}_r{c['rung']}_{c['probe']}_d{c['k']}"
            cfg = json.loads(json.dumps(base))
            cfg["sweep_name"] = name
            cfg["dyhpo"]["seed"] = zlib.crc32(name.encode()) % (2 ** 31)     # each sweep its own candidate seed
            cfg["fixed_params"].update(fixed)
            cfg["search_space"] = list(cfg["search_space"]) + extra
            assert cfg["n_trials"] == 5 and len(cfg["fidelity_schedule"]["t_steps"]) == 1, name
            out.append((name, cfg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="on the site: generate every sweep (no submission)")
    a = ap.parse_args()
    cf = configs()
    paths = []
    for name, cfg in cf:
        p = os.path.join(HERE, f"sweep_config_{name}.yaml")
        with open(p, "w") as f:
            f.write("# Written by sweep/gen_lowd_dyhpo.py (the small-D study's DyHPO round, D120).\n")
            yaml.safe_dump(cfg, f, sort_keys=False)
        paths.append(p)
    print(f"{len(cf)} configs: {len(METHODS)} methods x {len(cf) // len(METHODS)} cells")
    if a.generate:
        sys.path.insert(0, ROOT)
        import glob
        import siteconf
        from sweep.generate_sweep import run_generate
        os.environ.setdefault("FA_NEW_SWEEP_REASON", "D120")   # the cell guard: these differ from the grid cells only
                                                                # in the method (fixed flag or one searched dimension)
        for (name, _), p in zip(cf, paths):
            d = os.path.join(siteconf.SWEEP_DIR, name)
            if os.path.exists(os.path.join(d, "dyhpo_state.pkl")) and len(glob.glob(os.path.join(d, "jobs", "*.sh"))) >= 5:
                print(f"{name}: generated already, kept")
                continue
            run_generate(p, submit=False)


if __name__ == "__main__":
    main()
