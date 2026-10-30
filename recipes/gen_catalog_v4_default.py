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
# a class whose pool keeps less than 1 / (1 + ACCEPT_MARGIN) of its MadEvent events (e+e- with photons or leptons at the
# Z pole: the cuts on pairs of different kinds, which an e+e- run card cannot apply, remove up to 0.38 there) is sized
# from its measured kept fraction instead: KEPT_SAFETY / kept - 1 (analysis/catalog_v4/data/dsh_kept.json, tools/lhe_probe.py
# over one MadEvent run of 1000 events per energy; KEPT_SAFETY open, unsourced)
KEPT = os.path.join(ROOT, "analysis", "catalog_v4", "data", "dsh_kept.json")
KEPT_SAFETY = 1.03
ASSEMBLED = os.path.join(ROOT, "analysis", "catalog_v4", "data", "lhe_assembled.json")
MIXFAIL = os.path.join(ROOT, "analysis", "catalog_v4", "data", "lhe_mixcheck_failing.json")
PROBE_YIELDS = os.path.join(ROOT, "analysis", "catalog_v4", "data", "lhe_probe_yields.json")
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
# e+e- pools' events per energy in their part files against the equal share (tools/lhe_parts.py shares): a pool
# assembled with equal shares under short_margin keeps that layout; any other short e+e- pool takes energy_runs
EE_SHARES = os.path.join(ROOT, "analysis", "catalog_v4", "data", "ee_shares.json")
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


def accept_margin(kept, name, energies=None):
    """The default share's margin of a dsh pool: ACCEPT_MARGIN, or KEPT_SAFETY / kept - 1 where the pool keeps less.
    An e+e- pool takes equal shares of its energies, so it keeps the mean of each energy's fraction (an energy with no
    measurement counted at the lowest measured one, an assumption)."""
    k = kept.get(name)
    if not k:
        return ACCEPT_MARGIN
    by = k.get("by_energy") or {}
    if energies and by:
        lo = min(by.values())
        f = sum(by.get(f"{float(E):g}", lo) for E in energies) / len(energies)
    else:
        f = float(k["kept"])
    return max(ACCEPT_MARGIN, round(KEPT_SAFETY / f - 1.0, 3)) if f > 0 else ACCEPT_MARGIN


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
    """e+e- pool with short runs, as 0a7fb836 sized it (in place, before run_layout): one margin for every energy so
    STALL_SHARE of the measured mean yield covers. Kept only for the pools assembled with equal shares under it
    (EE_SHARES), so their pools keep their identity; every other short e+e- pool takes energy_runs."""
    r = _yield_ratio(y) if y else None
    if r:
        lhe["margin"] = round((1.0 + float(lhe.get("margin", mg.LHE_MARGIN))) / (STALL_SHARE * r) - 1.0, 3)
        lhe["accept_short"] = True
    return lhe


def energy_runs(lhe, y, energies_then):
    """e+e- pool with short runs (in place, after run_layout): energy j gets n_runs x ceil(1 / (STALL_SHARE x its
    measured yield)) runs of the same size (lhe.energy_runs), n_runs where every run returned what it asked, so each
    energy returns its equal share of the events (lhe_runs' design; the assembly refuses a pool where one does not).
    More runs rather than larger ones: a short run's yield is not shown to grow with the events asked, and runs of the
    layout's size each fit one part job. The yields' runs are read in lhe_runs' order over the energies they were run
    at (energies_then, n_runs each); an energy with no part takes the pool's lowest measured yield (an assumption: no
    measurement there)."""
    if not y:
        return lhe
    n = len(y["got"])
    if n % len(energies_then):
        raise SystemExit(f"yields of {len(y['got'])} runs do not divide into {len(energies_then)} energies")
    nr = n // len(energies_then)
    r = {}
    for j, E in enumerate(energies_then):
        got = [(g, q) for g, q in zip(y["got"][j * nr:(j + 1) * nr], y["asked"][j * nr:(j + 1) * nr]) if g is not None]
        r[E] = sum(g for g, _ in got) / sum(q for _, q in got) if got else None
    known = [v for v in r.values() if v]
    if not known or (all(v >= 1.0 for v in known) and None not in r.values()):
        return lhe
    worst = min(known)
    base = int(lhe.get("n_runs", 1))
    runs = [base if (r.get(E) or worst) >= 1.0 else base * int(math.ceil(1.0 / (STALL_SHARE * (r.get(E) or worst))))
            for E in lhe["energies"]]
    if any(x != base for x in runs):
        lhe["energy_runs"] = runs
        lhe["accept_short"] = True
    return lhe


# A pool whose runs return under Y_KEEP of the events they ask at its run size: MadEvent skips its second refine when the
# cross section is stable and then stops short of the target ("fail to reach target"), the more so the more events a
# run asks; at SMALL_RUN events a run returned all it asked in the probe (tools/lhe_probe.py, 2026-10-30: 166 of 169 pp
# default-share runs of 1000 events, pp13v4_uu_gguu 1000 of 1000 in 751 s against 137 of 14280 in 3684 s), so such a
# pool is laid out again in runs of SMALL_RUN events, its earlier parts not used. Both values open, unsourced.
SMALL_RUN = 1000
Y_KEEP = 0.3


# Pools laid out in runs that complete (lhe.complete_runs; catalog_coverage, 2026-10-30): a MadEvent run that stops short
# of its target returns a biased sample, one that completes does not, whatever its size
# (analysis/catalog_v4/short_run_bias.py). COMPLETE lists each pool rebuilt so, with its run size: the pools assembled
# from short runs (tools/lhe_parts.py shortbuilt), the three held out of the wave and dsh_ud_huuwm. The assembly uses
# only the runs that returned all they asked; COMPLETE_SHARE of the runs are expected to (166 of 169 probe runs of 1000
# events did; open).
COMPLETE = os.path.join(ROOT, "analysis", "catalog_v4", "data", "complete_runs.json")
COMPLETE_SHARE = 0.8
# A pool with "gridpack": true in COMPLETE draws its runs from one gridpack per energy (lhe.gridpack, tools/lhe_parts.py)
# integrated at the precision of one run asking the energy's whole need, the run that integrates on its own and
# completes being the reference (docs/results.tex par:v4_short_runs); never below the smallest gridpack the gridpack
# probe validated for its beams (catalog_coverage, 2026-10-30): pp 31774 (dsh_bb_bbgg), e+e- 13057 (dsh_epem_amupvmwm
# at 1000 GeV).
GRID_MIN = {"pp": 31774, "ee": 13057}


# A pool marked "size_from_probe" in COMPLETE takes its runs per energy from its gridpack probe's complete share there
# (tools/lhe_probe.py kept --by-energy-out), where that share is under COMPLETE_SHARE; an energy that would need more
# than MAX_ASK_FACTOR x the runs its events need is refused (CLAUDE.md: a margin above 2x is a finding to fix, never
# absorbed with more runs). "grid_events" in COMPLETE raises the pool's gridpack precision above the floor.
BY_ENERGY = os.path.join(ROOT, "analysis", "catalog_v4", "data", "lhe_probe_by_energy.json")
MAX_ASK_FACTOR = 2.0
RUN_SAFETY = 1.25           # fa-generation: events asked = needed / measured share, times at most 1.25


def complete_layout(lhe, run_events, gridpack=False, shares=None, grid_events=None, name="?"):
    """lhe (in place, a copy's): runs of run_events, as many as the blocks need at COMPLETE_SHARE of them complete (per
    energy for an e+e- pool), only complete runs assembled; the short-run layout's keys dropped. gridpack: the runs
    drawn from one gridpack per energy of max(the energy's need, GRID_MIN of its beams, grid_events) events' precision
    (lhe.gridpack). shares ({energy as f"{E:g}", or "all" for pp: complete share measured}): an energy measured under
    COMPLETE_SHARE x RUN_SAFETY takes ceil(need / (share / RUN_SAFETY x run_events)) runs (lhe.energy_runs for e+e-)."""
    need = sum(int(lhe[k]) for k in ("n_val", "n_test", "n_train"))
    for k in ("accept_short", "energy_runs", "margin", "n_runs", "run_events", "iseed_salt", "gridpack"):
        lhe.pop(k, None)
    per = int(math.ceil(need / len(lhe["energies"]))) if lhe.get("energies") else need
    n_runs = int(math.ceil(per / (COMPLETE_SHARE * int(run_events))))
    lhe.update(parts=True, complete_runs=True, run_events=int(run_events), n_runs=n_runs)
    if shares:
        keys = [f"{float(E):g}" for E in lhe["energies"]] if lhe.get("energies") else ["all"]
        sh = [float(shares[k]) if k in shares else 1.0 for k in keys]
        bad = {k: x for k, x in zip(keys, sh) if x <= 0 or 1.0 / x > MAX_ASK_FACTOR}
        if bad:
            raise SystemExit(f"{name}: complete share under 1/{MAX_ASK_FACTOR:g} at {bad}: fix the runs (a larger "
                             f"gridpack, grid_events in complete_runs.json), never more of them")
        runs = [int(math.ceil(per / (min(COMPLETE_SHARE, x / RUN_SAFETY) * int(run_events)))) for x in sh]
        if lhe.get("energies"):
            if any(r != n_runs for r in runs):
                lhe["energy_runs"] = runs
        else:
            lhe["n_runs"] = runs[0]
    if gridpack:
        lhe["gridpack"] = max(per, GRID_MIN["ee" if lhe.get("energies") else "pp"], int(grid_events or 0))
    return lhe


def complete_args(c, name):
    """complete_layout's keyword arguments for pool `name` of COMPLETE (c: its entry)."""
    by = json.load(open(BY_ENERGY)) if c.get("size_from_probe") and os.path.exists(BY_ENERGY) else {}
    sh = {E: v["complete"] for E, v in by.get(name, {}).items()} if c.get("size_from_probe") else None
    if c.get("size_from_probe") and not sh:
        raise SystemExit(f"{name}: size_from_probe set but no per-energy probe result in {BY_ENERGY}")
    return dict(gridpack=c.get("gridpack", False), shares=sh, grid_events=c.get("grid_events"), name=name)


def complete_set():
    return json.load(open(COMPLETE)) if os.path.exists(COMPLETE) else {}


# pools found with the same problem after the catalog_v4_w3 wave, waiting on the user's call to join the rebuild
# (analysis/catalog_v4/data/complete_runs_pending.json): both generators leave their layout as it is meanwhile
PENDING = os.path.join(ROOT, "analysis", "catalog_v4", "data", "complete_runs_pending.json")


def pending_set():
    return json.load(open(PENDING)) if os.path.exists(PENDING) else {}


def prefix_runs(y, need):
    """The fewest leading runs whose parts (all made) hold `need` events, or None."""
    s = 0
    for k, g in enumerate(y["got"]):
        if g is None:
            return None
        s += g
        if s >= need:
            return k + 1
    return None


def resize_short(lhe, y, need, y_small=1.0):
    """A pp pool not yet assembled whose runs came back short (in place; y: events asked and returned per run, in run
    order): its runs' events counted against the blocks' need. When the leading runs already made hold them, n_runs is
    cut to those (lhe_parts plan links their parts to the new names); else, at a mean yield of at least Y_KEEP, more runs
    of the same size for what is missing (STALL_SHARE of the mean per run), the parts made kept; below it, runs of
    SMALL_RUN events at the yield y_small measured at that size. Returns 'covered', 'more' or 'small'."""
    per = int(y["asked"][0])
    now = int(lhe.get("run_events") or 0)
    assert not now or now == per, f"yields measured at {per} events per run, the layout asks {now}"
    n = prefix_runs(y, need)
    lhe["accept_short"] = True
    if n:
        lhe["n_runs"] = n
        return "covered"
    r = _yield_ratio(y)
    if r and r >= Y_KEEP:
        k0 = next((k for k, g in enumerate(y["got"]) if g is None), len(y["got"]))
        have = sum(y["got"][:k0])
        lhe["n_runs"] = k0 + int(math.ceil((need - have) / (STALL_SHARE * r * per)))
        return "more"
    lhe["run_events"] = SMALL_RUN
    lhe["n_runs"] = int(math.ceil(need / (STALL_SHARE * y_small * SMALL_RUN)))
    return "small"


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
    shares = json.load(open(EE_SHARES)) if os.path.exists(EE_SHARES) else {}
    kept = json.load(open(KEPT)) if os.path.exists(KEPT) else {}
    # pools assembled (tools/lhe_parts.py status): their layout is never changed (ASSEMBLED, from the queues); the
    # yield at SMALL_RUN events per process from the probe (PROBE_YIELDS, tools/lhe_probe.py summary of its runs)
    assembled = set(json.load(open(ASSEMBLED))) if os.path.exists(ASSEMBLED) else set()
    probe_yield = json.load(open(PROBE_YIELDS)) if os.path.exists(PROBE_YIELDS) else {}
    mixfail = set(json.load(open(MIXFAIL))) if os.path.exists(MIXFAIL) else set()
    resized = {}
    complete = complete_set()
    pending = pending_set()
    _pf = os.path.join(HERE, "catalog_v4_default_processes.yaml")
    prev = (yaml.safe_load(open(_pf)) or {}).get("processes", {}) if os.path.exists(_pf) else {}

    def ee_short(mg, lhe, name, energies_then):
        """Equal shares kept: short_margin for a pool assembled with them (identity unchanged), else energy_runs."""
        s = shares.get(name)
        if s and s["assembled"] and s["ok"]:
            return short_margin(mg, lhe, yields.get(name))
        lhe["_then"] = energies_then           # energy_runs after run_layout (it needs the layout's n_runs)
        return lhe

    def ee_then(n, k4):
        """The energies an e+e- pool's yields (key k4) were run at: class n's energies now, with those EE_ZERO_SURVEY dropped since
        (only when its yields count their runs)."""
        lo = mg.v4_sqrts_lo(P[n]["pdg_ids"], P[n]["m_finals"], P[n].get("fiducial_cuts"))
        now = ee_energies(mg, P[n], n)
        y = yields.get(k4)
        if y and len(y["got"]) % len(now):
            return [E for E in EE_ENERGIES if E > lo]
        return now
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
        l4 = ee_short(mg, {"energies": Es}, k4, ee_then(n, k4))
        if "_then" in l4:
            energy_runs(l4, yields.get(k4), l4.pop("_then"))
        if k4 in complete:
            l4 = complete_layout({**ee_lhe, **l4}, complete[k4]["run_events"], **complete_args(complete[k4], k4))
            l4 = {k: v for k, v in l4.items() if k not in ee_lhe or ee_lhe[k] != v}
        ee[k4] = {**lhe_entry(mg, n, P[n], mg.V4_MODEL, l4 or None),
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
        name = "dsh_" + n[len("v4_"):]
        # an assembled pool keeps its sizes unless the mixture cannot read it (MIXFAIL: tools/lhe_probe.py mixcheck)
        am = (accept_margin(kept, name, P[reuse]["lhe"].get("energies"))
              if name not in assembled or name in mixfail else ACCEPT_MARGIN)
        need = {r: int(math.ceil(F_MAX * SIZES[k] * (1.0 + am))) for r, k in (("train", "n_train"), ("val", "n_val"),
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
        if lhe.get("energies"):
            ee_short(mg, lhe, name, ee_then(n, name))
        elif name in yields and _yield_ratio(yields[name]) is None:
            # a pp pool whose run returned no event (dsh_ddx_awpwm): a new iseed, and runs sized like a stalling
            # class's, STALL_SHARE of the events the reused pool's run returned, each counting whatever it returns
            lhe["iseed_salt"] = 1
            lhe["accept_short"] = True
            stall = int(r_meas["n_lhe"] / int(r_meas.get("n_runs") or 1)) if r_meas else stall
        run_layout(mg, lhe, rate, stall)
        if "_then" in lhe:
            energy_runs(lhe, yields.get(name), lhe.pop("_then"))
        # a pool laid out in complete runs (COMPLETE) takes complete_layout below, which drops the short-run layout's
        # keys: its short-run yields are not read (they were measured at another run size)
        if not lhe.get("energies") and name not in complete:
            if name in assembled or name not in yields or _yield_ratio(yields[name]) is None:
                short_runs(mg, lhe, yields.get(name))
            else:
                resized[name] = resize_short(lhe, yields[name], sum(int(lhe[k]) for k in SIZES),
                                             probe_yield.get(name, 1.0))
        lhe["parts"] = True        # one-run part jobs (tools/lhe_parts.py), in the pool identity
        e = lhe_entry(mg, n, P[n], mg.V4_MODEL, None)
        e.update(lhe=lhe, parent=n, reuse=reuse, split="train", test_group="catalog_v4 default share")
        # pp13v4 standalones carry the class's generate line, model and card: reuse their compiled backend
        if reuse.startswith(("pp13v4_", "ee4_")):
            e["standalone_of"] = reuse
        if name in assembled and name not in mixfail and name in prev:
            e = prev[name]                 # an assembled pool keeps the layout it was built with
        if name in pending and name in prev:
            e = prev[name]                 # waiting on the user's call to be rebuilt: its layout left as it is
        if name in complete:
            e = {**e, "lhe": complete_layout(dict(e["lhe"]), complete[name]["run_events"],
                                             **complete_args(complete[name], name))}
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
                      "dsh_runs": sum(sum(e["lhe"]["energy_runs"]) if e["lhe"].get("energy_runs") else
                                      int(e["lhe"].get("n_runs", 1)) * len(e["lhe"].get("energies") or [1])
                                      for e in dsh.values()),
                      "ee_runs": sum(sum(e["lhe"]["energy_runs"]) if e["lhe"].get("energy_runs") else
                                     len(e["lhe"]["energies"]) for e in ee.values()),
                      "ee_test_events": EE_TEST * len(ee),
                      "resized_short": {k: sorted(n for n, v in resized.items() if v == k)
                                        for k in ("covered", "more", "small")}}))


if __name__ == "__main__":
    main()
