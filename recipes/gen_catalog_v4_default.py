"""catalog_v4's default-sampled share (plan catalog_coverage; the user's priority of 2026-10-29: a final catalog whose
sampling mixes the default generator sampling, how most users sample, with the smart choices, on every beam).

A tree pool with sampling.f_default = f takes the share f of each role from MadEvent unweighted events instead of the
RAMBO bulk (mg5_pipeline_final, "DEFAULT-SAMPLED SHARE"). This script writes the entries and recipes that needs:

  catalog_v4_ee_processes.yaml       ee4_<x>: every e+e- catalog_v4 tree class (training and held out) as an lhe entry,
                                     MadEvent at fixed collider energies (lhe.energies, equal shares among the energies
                                     past the class's threshold), recipes/catalog_v4_ee/run_card.dat, catalog_v4's
                                     parameter card, labels at alpha_s(sqrt(s_hat)) from 0.118; 2000 test events
                                     (the e+e- default-sampled test measure, as pp13v4's 2000 for the hadronic classes)
  catalog_v4_ee_test.yaml            the e+e- default-sampled test recipe (its counterpart of catalog_v4_pp_test.yaml)
  catalog_v4_default_processes.yaml  dsh_<x>: the default share's own MadEvent events of every training tree class,
                                     sized for the largest arm's f (F_MAX) times 1 + ACCEPT_MARGIN (the events outside
                                     the pool's sqrt(s_hat) window or partonic-frame cuts are dropped) less what the class's existing MadEvent pool
                                     of the same role already holds (`reuse`: its pp13v4 / pp13 / ee4 pool), its own
                                     iseeds (the name enters lhe_iseed), so no event repeats one of the reused pool's
  catalog_v4_default_lhe.yaml        the dsh pools to build first (a CPU MadEvent job, prebuild_recipes.py)
  catalog_v4_default<f>_train.yaml   catalog_v4_train.yaml with every tree class that has a default source at
                                     f_default = f (one per arm); everything else as catalog_v4_train.yaml

Open calls, waiting on the user (plan ask, 2026-10-29): the arms' fractions (ARMS), the e+e- energies (EE_ENERGIES),
the e+e- run-card cuts and ISR (the card below: the catalog's own e+e- acceptance, no ISR), and whether the one-loop,
heavy-quark, alpha_s and cut-point pools take a share (here: none, they keep catalog_v4's mixture). The values below are
the design's proposals, not decided; nothing is generated before the answers.
    python3 recipes/gen_catalog_v4_default.py
"""
import json
import math
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from gen_catalog_v3_pp import PARTONS

ARMS = (0.50, 0.25)                                    # the design's arms B and C (open: the user's call)
F_MAX = max(ARMS)
ACCEPT_MARGIN = 0.10      # extra default events per role for those outside the pool's window or cuts (open, unsourced)
EE_ENERGIES = (91.1876, 240.0, 365.0, 500.0, 1000.0, 3000.0)   # GeV (open: the user's call)
EE_CARDS = "recipes/catalog_v4_ee"
PP_CARDS = "recipes/catalog_v4_pp"
SIZES = {"n_train": 100000, "n_val": 10000, "n_test": 10000}   # catalog_v4's tree pools
EE_TEST = 2000
LEPTON_BEAMS = (11, 13)
# the one-run layout (catalog_coverage step 7: "just use more cpus", 2026-10-29): each dsh pool is split into MadEvent
# runs of at most RUN_CORE_S core-seconds each (lhe.n_runs x lhe.run_events), built as parallel one-run jobs
# (tools/lhe_parts.py) and concatenated in run order; the cost per event is the class's measured one
# (analysis/catalog_v4/data/lhe_rates.json, tools/lhe_rates.py: its reuse pool's MadEvent seconds x cores / events),
# else the median of its final-state multiplicity. A stalling class (its reuse pool needed n_runs > 1) asks at most
# STALL_SHARE of the fewest events one of its runs returned.
RATES = os.path.join(ROOT, "analysis", "catalog_v4", "data", "lhe_rates.json")
RUN_CORE_S = 1800.0       # half an hour of one core per run (open: sized so a job's runs end well inside an hour)
MIN_RUN_EVENTS = 1000     # below it a run spends more on its survey than on its events (open, unsourced)
STALL_SHARE = 0.8         # of a stalling class's fewest events per run (open, unsourced)
SEED = 42                 # the catalog's seed (every catalog_v4 pool)
# pools whose MadEvent runs came back short at random ("fail to reach target"; tools/lhe_parts.py yields over the
# queue's part files, 2026-10-29): their runs count whatever they return (lhe.accept_short) and are sized from the mean
# a run returned, STALL_SHARE of it; pp: more runs of the same size (the parts made are kept, lhe_parts _adopt); e+e-
# (one run per energy): a margin so the runs' sum covers the blocks. A pool whose runs returned nothing is left as is.
YIELDS = os.path.join(ROOT, "analysis", "catalog_v4", "data", "lhe_run_yields.json")
HDR = "# generated by recipes/gen_catalog_v4_default.py (catalog_v4's default-sampled share); do not edit\n"


# energies MadEvent's survey found zero cross section at with the catalog's cuts (generate_events exits 0 with no
# events): not past that process's own lower edge, which v4_sqrts_lo (masses and pT cut only) puts below them.
# v4_epem_ttxvevex at 365 GeV: t t~ at 345 GeV leaves 20 GeV to the pair and the cuts (queue run 1, 2026-10-30)
EE_ZERO_SURVEY = {"v4_epem_ttxvevex": (365.0,)}


def ee_energies(mg, e, n=None):
    """The collider energies past the class's lower edge (v4_sqrts_lo: its masses and the pT cut), less those its
    survey found empty (EE_ZERO_SURVEY)."""
    lo = mg.v4_sqrts_lo(e["pdg_ids"], e["m_finals"], e.get("fiducial_cuts"))
    return [E for E in EE_ENERGIES if E > lo and E not in EE_ZERO_SURVEY.get(n, ())]


def lhe_entry(mg, n, c, model, lhe=None):
    return {"kind": "lhe_pp", "folder": n, "mg5_generate": list(c["mg5_generate"]),
            "proc_str": c["mg5_generate"][0].replace("generate ", ""), "nfinal": int(c["nfinal"]),
            "pdg_ids": [int(p) for p in c["pdg_ids"]], "m_finals": [float(m) for m in c["m_finals"]],
            "order": [int(x) for x in mg.order_vector(c)], "alphas_power": int(c.get("alphas_power", 0)),
            **({"beam_masses": dict(c["beam_masses"])} if c.get("beam_masses") is not None else {}),
            **({"lhe": lhe} if lhe else {})}


def run_layout(mg, lhe, rate, stall_yield=None):
    """lhe.n_runs and lhe.run_events of a dsh pool (in place): one run when its events fit one run, else runs of
    run_events. pp: n_runs runs of run_events; e+e-: each energy n_runs runs of run_events."""
    total = int(lhe["n_val"]) + int(lhe["n_test"]) + int(lhe["n_train"])
    n_req = int(math.ceil(total * (1.0 + float(lhe.get("margin", mg.LHE_MARGIN)))))
    per = max(MIN_RUN_EVENTS, int(RUN_CORE_S / rate))
    if stall_yield:
        per = min(per, max(1, int(STALL_SHARE * stall_yield)))
    n_e = int(math.ceil(n_req / len(lhe["energies"]))) if lhe.get("energies") else n_req
    lhe.pop("n_runs", None); lhe.pop("run_events", None)
    if per < n_e:
        lhe["n_runs"] = int(math.ceil(n_e / per))
        lhe["run_events"] = per
    return lhe


def _yield_ratio(y):
    """Mean events a run returned over the events it asked, over the runs with a part; None when none returned any."""
    got = [(g, a) for g, a in zip(y["got"], y["asked"]) if g is not None]
    r = sum(g for g, _ in got) / sum(a for _, a in got) if got else 0.0
    return r if r > 0 else None


def short_margin(mg, lhe, y):
    """e+e- pool with short runs (in place, before run_layout): margin so STALL_SHARE of the measured yield covers."""
    r = _yield_ratio(y) if y else None
    if r:
        lhe["margin"] = round((1.0 + float(lhe.get("margin", mg.LHE_MARGIN))) / (STALL_SHARE * r) - 1.0, 3)
        lhe["accept_short"] = True
    return lhe


def short_runs(mg, lhe, y):
    """pp pool with short runs (in place, after run_layout): runs of the size asked before, as many as STALL_SHARE of
    the measured mean per run needs for the events asked, never fewer than before."""
    r = _yield_ratio(y) if y else None
    if not r:
        return lhe
    total = int(lhe["n_val"]) + int(lhe["n_test"]) + int(lhe["n_train"])
    n_req = int(math.ceil(total * (1.0 + float(lhe.get("margin", mg.LHE_MARGIN)))))
    per = int(lhe.get("run_events") or n_req)
    lhe["n_runs"] = max(int(lhe.get("n_runs", 1)), int(math.ceil(n_req / (STALL_SHARE * r * per))))
    if lhe["n_runs"] > 1:
        lhe["run_events"] = per
    lhe["accept_short"] = True
    return lhe


def main():
    import statistics
    import mg5_pipeline_final as mg
    rates = json.load(open(RATES))
    by_nf = {}
    for r in rates.values():
        by_nf.setdefault(int(r["nfinal"]), []).append(r["core_s_per_event"])
    med = {k: statistics.median(v) for k, v in by_nf.items()}
    yields = json.load(open(YIELDS))
    T = json.load(open(os.path.join(HERE, "catalog_v4_roles.json")))["trees"]
    P = mg.PROCESSES
    trees = sorted(n for n in T if n in P)
    is_had = lambda n: all(abs(int(p)) in PARTONS for p in P[n]["pdg_ids"][:2])
    is_ee = lambda n: all(abs(int(p)) in LEPTON_BEAMS for p in P[n]["pdg_ids"][:2])
    # the hadronic classes' existing MadEvent pools: catalog_v4_pp_test.yaml names pp13v4_<x> or the reused pp13_<x>
    pp_test = yaml.safe_load(open(os.path.join(HERE, "catalog_v4_pp_test.yaml")))["processes"]
    had = sorted(n for n in trees if is_had(n))
    assert len(pp_test) == len(had), (len(pp_test), len(had))
    pp_of = dict(zip(had, (e["name"] for e in pp_test)))
    assert all(p.endswith(n[len("v4_"):]) for n, p in pp_of.items()), "catalog_v4_pp_test.yaml out of class order"
    ee_lhe = {"cards_dir": EE_CARDS, "run_card": "run_card.dat", "param_card": "param_card.dat", "model": mg.V4_MODEL,
              "n_val": 0, "n_test": EE_TEST, "n_train": 0, "label_alphas": "running_sqrt_shat", "parts": True,
              "source": "catalog_v4's parameter card, an e+e- run card with catalog_v4's e+e- acceptance, no ISR"}
    ee, skipped = {}, []
    for n in trees:
        if not is_ee(n):
            continue
        Es = ee_energies(mg, P[n], n)
        if not Es:
            skipped.append(n); continue
        k4 = "ee4_" + n[len("v4_"):]
        ee[k4] = {**lhe_entry(mg, n, P[n], mg.V4_MODEL, short_margin(mg, {"energies": Es}, yields.get(k4))),
                                       "parent": n,
                                       "split": T[n]["role"], "test_group": f"e+e- v4: {T[n].get('axis') or 'train'}"}
    ee_of = {e["parent"]: k for k, e in ee.items()}
    for k, e in ee.items():          # a first run: the pipeline loaded no ee4 entries at import
        P.setdefault(k, dict(e, lhe={**ee_lhe, **e["lhe"]}))
    with open(os.path.join(HERE, "catalog_v4_ee_processes.yaml"), "w") as f:
        f.write(HDR)
        yaml.safe_dump({"lhe": ee_lhe, "processes": ee}, f, sort_keys=False)
    with open(os.path.join(HERE, "catalog_v4_ee_test.yaml"), "w") as f:
        f.write(HDR)
        yaml.safe_dump({"processes": [{"name": k, "sqrts": [13000, 13000], "n_train": 0, "n_val": 0,
                                       "n_test": EE_TEST} for k in ee]}, f, sort_keys=False)
    # the default share's own events: per role ceil(F_MAX x size) less the reused pool's block
    pp_lhe = yaml.safe_load(open(os.path.join(HERE, "catalog_v4_pp_processes.yaml")))["lhe"]
    dsh, src_of = {}, {}
    for n in trees:
        if T[n]["role"] != "train" or not (is_had(n) or n in ee_of):
            continue
        reuse = pp_of.get(n) or ee_of[n]
        rl = mg.lhe_layout(P[reuse])
        need = {r: int(math.ceil(F_MAX * SIZES[k] * (1.0 + ACCEPT_MARGIN))) for r, k in (("train", "n_train"), ("val", "n_val"),
                                                                 ("test", "n_test"))}
        new = {r: max(0, need[r] - (rl[r][1] - rl[r][0])) for r in need}
        r_meas = rates.get(reuse)
        rate = r_meas["core_s_per_event"] if r_meas else med.get(int(P[n]["nfinal"]), max(med.values()))
        stall = None
        if is_had(n):
            lhe = {**pp_lhe, "n_val": new["val"], "n_test": new["test"], "n_train": new["train"]}
            # a stalling MadEvent run (a fixed few thousand events however many are asked): each run asks less than
            # the fewest events one of the reused pool's runs returned
            r_lhe = P[reuse].get("lhe") or {}
            if int(r_lhe.get("n_runs", 1)) > 1:
                epr = (r_meas or {}).get("events_per_run")
                stall = min(epr) if epr else int(rl["train"][1] / int(r_lhe["n_runs"]))
            if "margin" in r_lhe:
                lhe["margin"] = float(r_lhe["margin"])
        else:
            lhe = {**ee_lhe, "energies": list(P[reuse]["lhe"]["energies"]),
                   "n_val": new["val"], "n_test": new["test"], "n_train": new["train"]}
        name = "dsh_" + n[len("v4_"):]
        if lhe.get("energies"):
            short_margin(mg, lhe, yields.get(name))
        elif name in yields and _yield_ratio(yields[name]) is None:
            # a pp pool whose run returned no event (dsh_ddx_awpwm): a new iseed, and runs sized like a stalling
            # class's, STALL_SHARE of the events the reused pool's run returned, each counting whatever it returns
            lhe["iseed_salt"] = 1
            lhe["accept_short"] = True
            stall = int(r_meas["n_lhe"] / int(r_meas.get("n_runs") or 1)) if r_meas else stall
        run_layout(mg, lhe, rate, stall)
        if not lhe.get("energies"):
            short_runs(mg, lhe, yields.get(name))
        lhe["parts"] = True        # one-run part jobs (tools/lhe_parts.py), in the pool identity
        e = lhe_entry(mg, n, P[n], mg.V4_MODEL, None)
        e.update(lhe=lhe, parent=n, reuse=reuse, split="train", test_group="catalog_v4 default share")
        # pp13v4 standalones carry the class's generate line, model and card: reuse their compiled backend
        if reuse.startswith(("pp13v4_", "ee4_")):
            e["standalone_of"] = reuse
        dsh[name] = e
        src_of[n] = name
        # the default share never repeats an event of the pool it follows: no MadEvent iseed in common (run seeds wrap
        # modulo 30000), at the catalog's seed
        P[name] = e
        seeds = lambda x: {r[0] for r in mg.lhe_runs(x, mg.lhe_iseed(x, SEED), mg.lhe_n_requested(P[x]))}
        common = seeds(name) & seeds(reuse)
        assert not common, f"{name} shares MadEvent iseeds {sorted(common)} with {reuse}"
    lhe_hdr = {"source": "per entry (catalog_v4_pp's or catalog_v4_ee's cards)"}
    with open(os.path.join(HERE, "catalog_v4_default_processes.yaml"), "w") as f:
        f.write(HDR)
        yaml.safe_dump({"lhe": lhe_hdr, "processes": dsh}, f, sort_keys=False)
    with open(os.path.join(HERE, "catalog_v4_default_lhe.yaml"), "w") as f:
        f.write(HDR)
        yaml.safe_dump({"processes": [{"name": k, "sqrts": [13000, 13000], **{s: e["lhe"][s] for s in SIZES}}
                                      for k, e in dsh.items()]}, f, sort_keys=False)
    base = yaml.safe_load(open(os.path.join(HERE, "catalog_v4_train.yaml")))
    samp = dict(base.get("sampling") or {})
    for f_d in ARMS:
        procs = []
        for e in base["processes"]:
            src = src_of.get(e["name"]) if not e.get("physics") and e.get("base", e["name"]) == e["name"] else None
            procs.append({**e, "sampling": {**samp, "f_default": f_d, "default_source": src}} if src else e)
        with open(os.path.join(HERE, f"catalog_v4_default{int(round(100 * f_d)):02d}_train.yaml"), "w") as f:
            f.write(HDR)
            yaml.safe_dump({**{k: v for k, v in base.items() if k != "processes"}, "processes": procs}, f,
                           sort_keys=False)
    n_shared = sum(1 for e in base["processes"] if e["name"] in src_of)
    print(json.dumps({"ee_classes": len(ee), "ee_below_every_energy": skipped, "dsh_entries": len(dsh),
                      "dsh_hadronic": sum(1 for e in dsh.values() if e["reuse"].startswith("pp13")),
                      "dsh_ee": sum(1 for e in dsh.values() if e["reuse"].startswith("ee4_")),
                      "train_entries_with_share": n_shared, "train_entries": len(base["processes"]),
                      "new_madevent_events": sum(sum(e["lhe"][s] for s in SIZES) for e in dsh.values()),
                      "dsh_runs": sum(int(e["lhe"].get("n_runs", 1)) * len(e["lhe"].get("energies") or [1])
                                      for e in dsh.values()),
                      "ee_test_events": EE_TEST * len(ee)}))


if __name__ == "__main__":
    main()
