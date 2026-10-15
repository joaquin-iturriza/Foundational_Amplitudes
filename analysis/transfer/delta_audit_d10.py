"""The D = 10^5 / 32k counterpart of delta_audit.json (decision D42, the |Delta| distributions): the same rows (group
ladder / finale_synth / star, probe, arm, figures), each with the run the 32k figures would draw at the largest D the
32k grid has for the cell: d10 (D = 10^5), d9 (10^4.5) for the one-loop probes ee_dd_nlo, ee_bb_nlo (their 32k cells
stop there). Selection = cells.best_chosen, the rule every 32k figure uses: the cell's value is the best of its two
chosen points (horizon32k_chosen.json, else expect_tp3.json for the cells added later; a sweep's `_002` continuation
counts as the same search), MSE of log|M|^2 = val_loss * prepd_std^2 at the best checkpoint. The finale and synthetic
pretraining at 32k are read the same way (rung_focus.draw_arm, synth32k.py). The star arms have no 32k cells.

Inputs: delta_audit_d10_sites.json, a read-only listing of every site (delta_audit_d10_probe.py; results are read on
the sites, not from scratch_sweeps.json, which predates part of the d9/d10 results; its value is recorded beside the
site's as `scratch_sweeps_value`).
Outputs: delta_audit_d10.json; delta_runs_d10_<site>.txt (one run dir per line, best model present on that site).
    python analysis/transfer/delta_audit_d10.py [--probe]     (--probe refreshes the listing of ccin2p3 and lxplus)
"""
import base64, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SITES_F = os.path.join(HERE, "delta_audit_d10_sites.json")
SITES = ("ccin2p3", "lxplus", "jeanzay")

if "--probe" in sys.argv:
    src = open(os.path.join(HERE, "delta_audit_d10_probe.py")).read()
    cmd = "python -c \"import base64;exec(base64.b64decode('" + base64.b64encode(src.encode()).decode() + "'))\""
    cur = json.load(open(SITES_F)) if os.path.exists(SITES_F) else {}
    for s in ("ccin2p3", "lxplus"):
        t = subprocess.run(["site", "run", s, "FA", "--", cmd], capture_output=True, text=True, timeout=900).stdout
        cur[s] = json.loads(t[t.index("__J__") + 5:])
        for n, e in cur[s].items():
            if not n.startswith("_"):
                e["sweepdir"] = {r: [x for x in v if x in ("HELD", "MOVED_TO", "ARRIVED", "summary.txt", "dyhpo_state.pkl")]
                                 for r, v in e["sweepdir"].items()}
    json.dump(cur, open(SITES_F, "w"), indent=0, sort_keys=True)

L = json.load(open(SITES_F))
A8 = json.load(open(os.path.join(HERE, "delta_audit.json")))
CHOSEN = json.load(open(os.path.join(HERE, "horizon32k_chosen.json")))
for _n, _v in json.load(open(os.path.join(HERE, "expect_tp3.json"))).items():   # as cells.py builds CHOSEN32
    if isinstance(_v, list) and "32k_" in _n:
        CHOSEN.setdefault(_n, _v)
S = json.load(open(os.path.join(HERE, "scratch_sweeps.json")))
ALL_TRIALS32 = {"tp3_scr32k_ee_nnbar_d8"}          # cells.ALL_TRIALS32 (d8 only: no d9/d10 cell scored on all trials)
ONE_LOOP = {"ee_dd_nlo", "ee_bb_nlo"}
KMAX = lambda p: 9 if p in ONE_LOOP else 10
FAM = {"scratch": "tp3_scr32k", "r0": "tp3_uu64fte32k", "synthetic": "tp3_synfte32k", "finale": "tp3_finfte32k",
       **{f"r{k}": f"tp3_r{k}fte32k" for k in range(1, 10)}}


def hp_of(fn):
    return int(re.match(r"hp(\d+)_t(\d+)_", fn).group(1))


def attempts(name):
    """Every (site, sweep, result file, val_loss, ...) of a search (name and its _002 continuation) on every site, with
    the run dir(s) whose config names that result file."""
    out = []
    for s in SITES:
        for nm in (name, name + "_002"):
            e = L[s].get(nm)
            if not e:
                continue
            for fn, r in e["results"].items():
                m = re.match(r"hp(\d+)_t(\d+)_", fn)
                if not m or r.get("val_loss") is None:
                    continue
                dirs = [(d, i) for d, i in e["dirs"].items() if i.get("rp") == fn]
                if not dirs:                  # no dir names it: the trial's own dir, if any
                    dirs = [(d, i) for d, i in e["dirs"].items() if d.endswith(f"/trial_{int(m.group(1)):04d}")]
                out.append(dict(site=s, sweep=nm, file=fn, hp=int(m.group(1)), T=int(m.group(2)), r=r, dirs=dirs))
    # a run dir copied to another site (runs_from_<site>/) counts as a copy of the same attempt
    for s in SITES:
        for nm in (name, name + "_002"):
            for d, i in (L[s].get(nm) or {}).get("dirs", {}).items():
                if d.startswith("runs_from_") and i.get("rp"):
                    for a in out:
                        if a["file"] == i["rp"] and a["sweep"] == nm:
                            a["dirs"].append((f"{s}:{d}", i))
    return out


def markers(name):
    m = {}
    for s in SITES:
        for nm in (name, name + "_002"):
            e = L[s].get(nm)
            if e is not None:
                m[f"{s}:{nm}"] = sorted({x for v in e["sweepdir"].values() for x in v} | ({"run_dirs"} if e["dirs"] else set()))
    return m


def figures_value(name):
    """cells.best_chosen on scratch_sweeps.json (what the figures read today)."""
    tr = [t for t in S.get(name, []) + S.get(name + "_002", [])
          if t.get("val_loss") is not None and t.get("prepd_std") and (name in ALL_TRIALS32 or t["hp"] in CHOSEN.get(name, []))]
    return min(t["val_loss"] * t["prepd_std"] ** 2 for t in tr) if tr else None


def cell(name):
    ch = CHOSEN.get(name)
    rec = dict(sweep=name, chosen_points=ch, T=32000, selection_rule="32k: best of the two chosen points (cells.best_chosen)",
               sweep_markers=markers(name), scratch_sweeps_value=figures_value(name))
    if ch is None:
        rec.update(status="missing", reason="no chosen points listed (horizon32k_chosen.json, expect_tp3.json): cell not in the plan")
        return rec
    at = attempts(name)
    sel = [a for a in at if a["hp"] in ch and a["T"] == 32000]
    for a in sel:                                  # prepd_std from the attempt's run dir, else any dir of the cell
        stds = [i["std"] for _, i in a["dirs"] if i.get("std")]
        a["std"] = stds[0] if stds else None
    cell_std = {round(i["std"], 6) for s in SITES for nm in (name, name + "_002")
                for i in (L[s].get(nm) or {}).get("dirs", {}).values() if i.get("std")}
    for a in sel:
        if a["std"] is None and len(cell_std) == 1:
            a["std"], a["std_from"] = next(iter(cell_std)), "another run dir of the cell"
    done = sorted({a["hp"] for a in sel})
    rec["chosen_finished"] = done
    rec["chosen_missing"] = [h for h in ch if h not in done]
    rec["other_trials_finished"] = sorted({a["hp"] for a in at if a["hp"] not in ch})
    if not sel:
        held = any("HELD" in v for v in rec["sweep_markers"].values())
        rec.update(status="missing", reason=("on hold (HELD), " if held else "") +
                   ("sweep never generated on any site" if not rec["sweep_markers"] else "no chosen point finished on any site"))
        return rec
    val = [a for a in sel if a["std"]]
    if not val:
        rec.update(status="missing", reason="chosen point(s) finished but no prepd_std anywhere (data_stats.json gone)")
        return rec
    for a in val:
        a["mse"] = a["r"]["val_loss"] * a["std"] ** 2
    b = min(val, key=lambda a: a["mse"])
    allv = [a["r"]["val_loss"] for a in at if a["T"] == 32000]

    def has_model(i):
        return i.get("pt") or i.get("ptgz")
    # the copy to evaluate: the run dir of the attempt with the best model present, on its own site first
    cands = [(a["site"], d, i) for a in val if a is b for d, i in a["dirs"]]
    cands += [(a["site"], d, i) for a in val if a is not b and a["hp"] == b["hp"] and a["file"] == b["file"] for d, i in a["dirs"]]
    cands.sort(key=lambda c: (not has_model(c[2]), ":" in c[1]))
    site, d, i = cands[0] if cands else (b["site"], None, {})
    if ":" in (d or ""):
        site, d = d.split(":", 1)
    rec.update(status="ok" if (d and has_model(i) and i.get("cfg") and i.get("ds")) else "incomplete",
               hp=b["hp"], fixed=False, mse_logM2=b["mse"], prepd_std=b["std"], site=site, run_dir=d,
               result_file=b["file"], result_site=b["site"], best_step=b["r"].get("best_step"),
               traintime_h=b["r"].get("traintime_hours"),
               model_best=("model_run0_best.pt" if i.get("pt") else "model_run0_best.pt.gz" if i.get("ptgz") else None),
               model_present=bool(has_model(i)), config=bool(i.get("cfg")), data_stats=bool(i.get("ds")),
               particle_tokenizer=bool(i.get("tok")),
               is_sweep_plain_val_best=bool(allv) and b["r"]["val_loss"] <= min(allv) * (1 + 1e-12),
               one_chosen_point_only=len(done) < len(ch),
               chosen_values={a["hp"]: a["mse"] for a in val},
               matches_scratch_sweeps=(rec["scratch_sweeps_value"] is not None
                                       and abs(rec["scratch_sweeps_value"] / b["mse"] - 1) < 1e-6),
               other_attempts_or_copies=[dict(site=a["site"], sweep=a["sweep"], hp=a["hp"], result_file=a["file"],
                                              dirs={dd: ("model_run0_best.pt" if ii.get("pt") else "model_run0_best.pt.gz"
                                                         if ii.get("ptgz") else None) for dd, ii in a["dirs"]})
                                         for a in sel])
    if rec["status"] == "incomplete":
        rec["reason"] = "best chosen point's run dir lacks " + ", ".join(
            x for x, ok in (("run dir", d), ("best model", has_model(i)), ("config.yaml", i.get("cfg")),
                            ("data_stats.json", i.get("ds"))) if not ok)
    return rec


rows = []
for r in A8["rows"]:
    p, arm, g = r["probe"], r["arm"], r["group"]
    k = KMAX(p)
    row = dict(group=g, probe=p, arm=arm, figures=r["figures"], k=k, D=10 ** (k / 2))
    if g == "star":
        row["d10"] = dict(status="missing", reason="star arms have no 32k cells (no tp3_s<arm>fte32k sweep on any site; "
                          "tp3_scr32k and tp3_r1fte32k cover the twelve ladder probes only, not the star probes); "
                          "the star figure is on the 8k grid, d8 at most")
        if p == "ee_ddbar" and arm.startswith("r1"):
            row["d10"]["note"] = ("rung 1 on ee_ddbar at 32k, d10 is the ladder row (tp3_r1fte32k_ee_ddbar_d10, its two "
                                  "chosen points, not r1_chosen.json's)")
        if p in ("ee_ddbarg", "ee_ttbarg", "udbar_enu", "uubar_Zg") or p == "ee_ddbar":
            fins = [n for n in (f"tp3_finfte32k_{p}_d10", f"tp3_finfte32k_{p}_d9") if n in CHOSEN]
            if fins and p != "ee_ddbar":
                row["d10"]["note"] = (f"the finale has 32k cells on this probe ({', '.join(fins)}), not a star-figure arm; "
                                      "not substituted")
    else:
        row["d10"] = cell(f"{FAM[arm.split()[0]]}_{p}_d{k}")
    rows.append(row)

# the finale's 32k cells on the star-arm probes (not rows: no star figure draws them; listed for the user's call)
star_finale = {}
for p in ("ee_ddbarg", "ee_ttbarg", "ee_ttbar_nlo_thr", "ee_dd_nlo_hi", "udbar_enu", "ee_dd_ew_nlo"):
    for k in (10, 9):
        n = f"tp3_finfte32k_{p}_d{k}"
        if n in CHOSEN:
            c = cell(n)
            star_finale[n] = {x: c.get(x) for x in ("status", "reason", "hp", "mse_logM2", "site", "run_dir", "model_present",
                                                     "chosen_finished", "chosen_missing")}
            break

meta = {p: dict(v, k=KMAX(p)) for p, v in A8["probe_meta"].items()}
out = dict(horizon="32k at the largest D of the 32k grid: d10 (D = 10^5), d9 (10^4.5) for ee_dd_nlo, ee_bb_nlo; selection "
                    "cells.best_chosen (the two chosen points); figures lists copied from delta_audit.json, the rung_focus "
                    "'best overall' tag ranked on the 8k grid as rung_focus.py does",
           probe_meta=meta, rows=rows, star_finale_32k=star_finale,
           sources=dict(listing="delta_audit_d10_sites.json", chosen="horizon32k_chosen.json + expect_tp3.json"))
json.dump(out, open(os.path.join(HERE, "delta_audit_d10.json"), "w"), indent=1)

per_site = {s: [] for s in SITES}
for r in rows:
    c = r["d10"]
    if c.get("status") == "ok" and c["run_dir"] not in per_site[c["site"]]:
        per_site[c["site"]].append(c["run_dir"])
for s, v in per_site.items():
    if v:
        open(os.path.join(HERE, f"delta_runs_d10_{s}.txt"), "w").write("\n".join(v) + "\n")
from collections import Counter
print("rows", len(rows), Counter(r["group"] for r in rows))
print("ok per site", {s: len(v) for s, v in per_site.items()})
print("status per group", Counter((r["group"], r["d10"]["status"]) for r in rows))
for r in rows:
    c = r["d10"]
    if c["status"] != "ok":
        print(f"  {r['group']:12s} {r['probe']:16s} {r['arm']:10s} {c['status']}: {c.get('reason')}")
    elif c.get("one_chosen_point_only") or not c.get("matches_scratch_sweeps"):
        print(f"  note {r['probe']} {r['arm']} {c['sweep']} hp{c['hp']} one_point={c['one_chosen_point_only']} "
              f"figs={c['scratch_sweeps_value']} site={c['mse_logM2']:.3g}")
