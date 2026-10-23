"""The `--check` of the active work items (site item ... --check): prints {"units": {name: {expected, done, active}},
"alerts": [...]} so `site supervise` raises READY when a comparison becomes possible and GAP when results are missing
with nothing queued (2026-10-23: no active item had a check, so finished work raised only per-run events and the
comparisons they made possible never reached the user).
    python tools/item_checks.py het_stage2|het_hp58_ext|het_mumse22|dimless_general|cut_test
Unit names match the item's --match globs.
  het_stage2       one unit per stage-2 sweep (sweeps_local/, drive.py): expected its n_trials, done its observed trials
  het_mumse22      one unit per cell of analysis/transfer/het_mumse22_jobs.txt: done once its eval_delta.json exists on CC
  dimless_general  one unit per (arm, held-out set) scoring: done once its prediction directory on CC holds the set
  cut_test         the same for the 30 GeV cut set (2to4c30)
"""
import glob, json, os, re, sqlite3, subprocess, sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.expanduser("~/.local/share/ccorch/runs.db")
ARMS = ("Adim", "Dless", "DlessIn", "DlessInVirt", "DlessInNoDiag", "Dpair_seed1")
# 2to4E100p: 36, four of its +gluon probes failed generation at 100 TeV (docs/results.tex, the dimensionless round)
SETS = {"catalog": None, "2to4": 40, "2to4E100": 40, "2to4E100p": 36}


def active_runs(item):
    c = sqlite3.connect(DB)
    return {r[0] for r in c.execute("SELECT id FROM runs WHERE project='FA' AND item=? AND state IN "
                                    "('SUBMITTED','PENDING','RUNNING','QUEUED','IDLE')", (item,))}


def on_cc(cmd):
    p = subprocess.run(["site", "--timeout", "600", "run", "ccin2p3", "FA", "--", cmd + "; echo __END__"],
                       capture_output=True, text=True, cwd=REPO)
    out = p.stdout
    if "__END__" not in out:
        return None
    return [l for l in out.splitlines() if l and not l.startswith("  run") and l != "__END__"]


def het_stage2():
    act = active_runs("het_stage2")
    units = {}
    for d in sorted(glob.glob(os.path.join(REPO, "sweeps_local", "tp3_*_het"))):
        name = os.path.basename(d)
        cfg = os.path.join(REPO, "sweep", f"sweep_config_{name}.yaml")
        if not (name.startswith("tp3_hp58fte_") or re.match(r"tp3_finfte_(ee_nnbar|ee_ttbar|ee_bb_nlo)_d\d_het$", name)):
            continue
        st = json.load(open(os.path.join(d, "driver_state.json")))["trials"].values()
        units[name] = {"expected": int(yaml.safe_load(open(cfg))["n_trials"]),
                       "done": sum(t.get("status") == "observed" for t in st),
                       "active": sum(t.get("rid") in act for t in st)}
    return units, []


def het_hp58_ext():
    """D97: the 13 hp58 het sweeps whose best was the highest lr tried, extended in place to lr 2e-2 and continued to 10
    trials (analysis/transfer/het_hp58_ext_cells.json)"""
    act = active_runs("het_hp58_ext")
    units = {}
    for name in json.load(open(os.path.join(REPO, "analysis", "transfer", "het_hp58_ext_cells.json"))):
        st = json.load(open(os.path.join(REPO, "sweeps_local", name, "driver_state.json")))["trials"].values()
        units[name] = {"expected": 10, "done": sum(t.get("status") == "observed" for t in st),
                       "active": sum(t.get("rid") in act for t in st)}
    return units, []


def het_mumse22():
    cells = []
    for l in open(os.path.join(REPO, "analysis", "transfer", "het_mumse22_jobs.txt")):
        if l.strip() and not l.startswith("#"):
            cells += l.split()[1:]
    lines = on_cc("ls -d $SCRATCH/zz_delta_het_mumse/*/eval_delta.json 2>/dev/null | xargs -n1 dirname | xargs -n1 basename")
    if lines is None:
        return {}, [], "ccin2p3 did not answer"
    have = set(lines)
    act = bool(active_runs("het_mumse22"))
    units = {}
    for c in cells:
        # a lad cell keeps its root in the name (runs__X_lad__sigma_stage), a trial cell drops it (X__trial_N__sigma_stage)
        tag = c.replace("/", "__") + "__sigma_stage"
        done = tag in have or tag.split("__", 1)[1] in have
        units["zz_delta_het_mumse/" + tag] = {"expected": 1, "done": int(done), "active": int(act and not done)}
    return units, []


def scorings(sets, item):
    lines = on_cc("for d in $SCRATCH/zz_catalog_norm_short_*; do echo \"$(basename $d) $(ls $d/*.npz 2>/dev/null | wc -l)\"; done")
    if lines is None:
        return {}, [], "ccin2p3 did not answer"
    n = dict(l.rsplit(" ", 1) for l in lines if " " in l)
    act = bool(active_runs(item))
    units = {}
    for arm in ARMS:
        for st, want in sets.items():
            suf = "" if st == "catalog" else "_" + st
            name = f"zz_catalog_norm_short_{arm}_sigstage{suf}"
            got = int(n.get(name, 0))
            done = int(got >= want if want else got > 0)
            units[name] = {"expected": 1, "done": done, "active": int(act and not done)}
    return units, []


# D93's parts after the round of fig:dimless_round (2026-10-23: they had no unit here, so when the round's jobs ended no
# event said the coupling-order factor and the particle-content and order sets were still to do, and the study stalled)
# _loop (recipes/catalog_loop_probes.yaml): the 99 positive one-loop pools (every window) and their trees; with the
# coupling-order factor 79 of them (the mixed-order entries are refused)
ORD_SETS = {"": 139, "_2to4": 40, "_2to4c30": 40, "_2to4E100": 40, "_2to4E100p": 36, "_top": 162, "_loop": 79, "_loopsigned": 61}
WRITEUPS = {"dimless_writeup_orders": "par:dimless_orders", "dimless_writeup_particle": "par:dimless_particle",
            "dimless_loop_probes": "par:dimless_loop"}
# the signed set's magnitude scoring (_loopsigned) is a diagnostic: the user (2026-10-23) asked for signed targets to be
# "dealt with ... in a proper way, not a patch", so its write-up waits on that design, which goes to the user first


def dimless_rest(item):
    lines = on_cc("cd $DATA_DIR; echo c30 $(ls d4c30_*_val_lhe_amplitudes.alphas.npy 2>/dev/null | wc -l); "
                  "echo d4r $(ls d4r_gdx_dxzh_g_*_val_lhe_amplitudes.alphas.npy 2>/dev/null | wc -l); "
                  "echo d4c30r $(ls d4c30r_gdx_dxzh_g_*_val_lhe_amplitudes.alphas.npy 2>/dev/null | wc -l); "
                  "for d in $SCRATCH/zz_catalog_norm_short_DlessIn*; do echo \"$(basename $d) $(ls $d/*.npz 2>/dev/null | wc -l)\"; done")
    if lines is None:
        return {}, [], "ccin2p3 did not answer"
    n = dict(l.rsplit(" ", 1) for l in lines if " " in l)
    act = bool(active_runs(item))
    u = lambda done: {"expected": 1, "done": int(done), "active": int(act and not done)}
    units = {"recover_alphas_2to4c30": u(int(n.get("c30", 0)) >= 39 and int(n.get("d4c30r", 0)) >= 1), "d4r_gdx_dxzh_g": u(int(n.get("d4r", 0)) >= 1)}
    for arm, sets in (("DlessInOrd", ORD_SETS), ("DlessInOrd_notop", {"_top": 162}), ("DlessIn", {"_loop": 99, "_loopsigned": 103})):
        for suf, want in sets.items():
            name = f"zz_catalog_norm_short_{arm}_sigstage{suf}"
            units[name] = u(int(n.get(name, 0)) >= want)
    tex = open(os.path.join(REPO, "docs", "results.tex")).read()
    pending = any(not v["done"] for v in units.values())
    for name, label in WRITEUPS.items():
        done = f"\\label{{{label}}}" in tex
        # a write-up waits on its scorings; with them done it is a GAP
        units[name] = {"expected": 1, "done": int(done), "active": int(pending and not done)}
    return units, []


# D99 (the user 2026-10-23: "you are allowed to do the gpu tests once something looks promising"): the first candidate
# through the CPU proxy (analysis/catalog_daniel/norm_targets.py), DlessInOrd with the emission scales (DlessInOrdQ), read
# on the sets DlessInOrd was read on that the A/B needs: the 139 held-out, the 2->4 probes at both cuts, the one-loop set
# (every set DlessInOrd was read on, and its no-top twin: CLAUDE.md 0b, the reviewer 2026-10-23)
QPERP_SETS = dict(ORD_SETS)


def dimless_kt(item):
    lines = on_cc("for d in $SCRATCH/zz_catalog_norm_short_DlessInOrdQ*; do echo \"$(basename $d) $(ls $d/*.npz 2>/dev/null | wc -l)\"; done; "
                  "ls runs/catalog_norm_short/DlessInOrdQ/sigma_stage/models/model_run0_best.pt* >/dev/null 2>&1 && echo model 1")
    if lines is None:
        return {}, [], "ccin2p3 did not answer"
    n = dict(l.rsplit(" ", 1) for l in lines if " " in l)
    act = bool(active_runs(item))
    u = lambda done: {"expected": 1, "done": int(done), "active": int(act and not done)}
    units = {"DlessInOrdQ_train": u(n.get("model") == "1")}
    for arm, sets in (("DlessInOrdQ", QPERP_SETS), ("DlessInOrdQ_notop", {"_top": 162})):
        for suf, want in sets.items():
            name = f"zz_catalog_norm_short_{arm}_sigstage{suf}"
            units[name] = u(int(n.get(name, 0)) >= want)
    pending = any(not v["done"] for v in units.values())
    done = "\\label{par:dimless_qperp}" in open(os.path.join(REPO, "docs", "results.tex")).read()
    units["dimless_writeup_qperp"] = {"expected": 1, "done": int(done), "active": int(pending and not done)}
    return units, []


def main():
    item = sys.argv[1]
    if item == "het_stage2":
        r = het_stage2()
    elif item == "het_hp58_ext":
        r = het_hp58_ext()
    elif item == "het_mumse22":
        r = het_mumse22()
    elif item == "dimless_general":
        r = scorings(SETS, item)
        if len(r) == 2:
            r2 = dimless_rest(item)
            r = (dict(r[0], **r2[0]), r[1] + r2[1]) + tuple(r2[2:])
    elif item == "dimless_kt":
        r = dimless_kt(item)
    elif item == "cut_test":
        r = scorings({"2to4c30": 40}, item)
    else:
        sys.exit(f"unknown item {item}")
    out = {"units": r[0], "alerts": r[1]}
    if len(r) > 2:
        out["partial"] = r[2]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
