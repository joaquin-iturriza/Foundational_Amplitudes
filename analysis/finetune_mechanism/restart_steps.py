#!/usr/bin/env python3
"""finetune_mechanism step 1 (evidence, no jobs): the first steps of each restart from a pretrained parent.
Per run the pre-clip gradient norm, training loss and lr of its first FIRST steps (train_steps.npz), its horizon and
warmup, for
  - the b b~ > Z Z tree fine-tune searches of data saving step 6 and its warmup test (R cosine parent + cosine
    fine-tune dsave_s2_ft, C cosine parent + WSD fine-tune dsave_s5cosparent, W WSD parent + WSD fine-tune
    dsave_s5wsdparent, L C with warmup searched in [0.05, 0.6] dsave_s6cosparentlongwu) and the scratch searches
    (dsave_s2_scr, a run's own warmup from random weights as the reference), every observed trial;
  - the restart test's searches (C', Oc, O, W', Bc, B and their WSD scratch, restart_test_readout.py);
  - the Z+ng fine-tunes (runs/kit/kit_A_ft_*: lr_scale 1, fresh optimizer, WSD, from trial 1262's best checkpoint).
The catalog warm-start restarts are read from analysis/catalog_v4/data/ws_{restart_study,fixes}_steps.json (steps 0-40).
lowd_ft (2026-10-25) predates train_steps.npz (0a616ed4, 2026-10-26): no per-step record.

    python analysis/finetune_mechanism/restart_steps.py [--cached]   -> data/restart_steps.json
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis", "data_saving"))
from analysis.common import site_python  # noqa: E402
from trial_overfit import driven_rows  # noqa: E402

FIRST = 400
CELLS = {100: (4, 4960), 316: (5, 7840), 3162: (7, 8500)}
ARMS = {"R": "s2_ft", "C": "s5cosparent_ft", "W": "s5wsdparent_ft", "L": "s6cosparentlongwu_ft", "S": "s2_scr",
        # the restart test's arms (restart_test_readout.py; Sw its WSD scratch)
        "C'": "s7cosrerun_ft", "Oc": "s7cosstep_ft", "O": "s7optstate_ft", "W'": "s7wsdrerun_ft", "Bc": "s7wsdstep_ft",
        "B": "s7wsdbranch_ft", "Sw": "s7wsd_scr"}

REMOTE = r'''
import glob, json, os, sys
import numpy as np, yaml
FIRST = %d
dirs = []
for a in sys.argv[1:]:
    dirs += sorted(glob.glob(a if a.startswith("runs/") else "runs/" + a + "/trial_*"))
out = []
for d in dirs:
    r = {"run_dir": d, "sweep": d.split("/")[1]}
    try:
        z = np.load(d + "/train_steps.npz")
        c = yaml.safe_load(open(d + "/config.yaml"))
        tr, ft = c["training"], (c.get("fine_tune") or {})
        r.update(step=z["step"][:FIRST].tolist(), grad_norm=z["grad_norm"][:FIRST].tolist(),
                 loss=z["loss"][:FIRST].tolist(), lr=z["lr"][:FIRST].tolist(), n=int(len(z["step"])),
                 iterations=tr.get("iterations"), warmup=tr.get("cosanneal_warmup_frac"), sched=tr.get("scheduler"),
                 lr_scale=ft.get("lr_scale"), opt_state=ft.get("load_optimizer_state"))
    except Exception as e:
        r["error"] = repr(e)[:160]
    out.append(r)
print("RSTEP " + json.dumps(out))
''' % FIRST


def name(arm, D):
    k, T = CELLS[D]
    return f"dsave_{ARMS[arm]}_bbx_zz_d{k}_t{T}"


def main(cached):
    cache = os.path.join(HERE, "data", "restart_steps.json")
    if cached:
        return json.load(open(cache))
    names = [name(a, D) for a in ARMS for D in CELLS]
    names = [n for n in names if os.path.isdir(os.path.join(REPO, "sweeps_local", n))]
    rows = driven_rows(names, REMOTE, "RSTEP")
    kit = []
    for site in ("ccin2p3", "lxplus"):
        try:
            kit += [dict(r, site=site) for r in site_python(REMOTE, ["'runs/kit/kit_A_ft_*'"], "RSTEP", site=site)]
        except SystemExit as e:
            print(f"kit A_ft on {site}: {e}")
    out = {"search": rows, "kit": kit}
    json.dump(out, open(cache, "w"))
    return out


if __name__ == "__main__":
    o = main("--cached" in sys.argv)
    for k, v in o.items():
        print(k, len(v), "with steps", sum("step" in r for r in v), "errors", sorted({r.get("error", "")[:60] for r in v} - {""})[:3])
