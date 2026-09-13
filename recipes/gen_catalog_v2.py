#!/usr/bin/env python
"""catalog_v2: rule-based enumeration of the process catalog, validated by MadGraph.

The v1 catalog enumerated 2->2 exhaustively but hand-picked one representative per
structure at 2->3/2->4; this generator replaces the hand-picking by rules and lets
MadGraph decide existence, so the taxonomy falls out of the set instead of being
curated in.

Rules (beams: e+e-, u u~, u d~, u u, u d, u s; parton level, no PDFs):
  2->2   the v1 list (recipes/catalog_v1_train.yaml + hold-outs), unchanged.
  2->3   every 2->2 base + one leg: +g where a coloured leg exists, +a always,
         +z/+h on bosonic-final and heavy-quark bases, plus the explicit W/fusion
         processes of v1 (e- ve~ w+, ve ve~ h, e+ e- h, ...).
  2->4   every 2->3 of the rule above + g (coloured) / + a; four-fermion finals
         (pair x pair); >= 3-boson finals with the quartic vertices; nu nubar + XY fusion.
  NLO    [virt=QCD] of every tree with a coloured leg at 2->2 and 2->3 (2->4 loops
         only the v1 handful).
  loop   loop-induced candidates ([sqrvirt=...]); only certified ones enter recipes.
Dedup: massless-flavour relabels (u=c, d=s, nu_mu=nu_tau, mu=e-final w/o t-channel)
and pure crossings are not enumerated into training (v1 hold-outs cover them).

Stages:
  --enumerate   write recipes/catalog_v2_candidates.json (no MadGraph needed)
  --check       run every candidate through MG5 `display processes` (login node,
                parallel), record diagrams + WEIGHTED, drop the non-existent
  --write       write recipes/catalog_v2_processes.yaml (entries the pipeline loads
                into PROCESSES / VIRT_PROCESSES) and the train/hold-out recipes
"""
import argparse, json, os, re, subprocess, sys, itertools
from concurrent.futures import ThreadPoolExecutor
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg

# One-loop strings MadGraph accepts but whose MadLoop module fails the pole certification:
# the entry stays in the table (certified: false) and out of every recipe. Empty at present:
# u s > u s and d s > d s sat here while the checker's 2-body sampler pointed the first beam
# along -z, an orientation in which those two modules return wrong, history-dependent poles;
# in the pipeline's orientation (slot 1 along +z, the one the data use) they are exact.
VIRT_UNCERTIFIED = {}
CAND = os.path.join(ROOT, "recipes", "catalog_v2_candidates.json")
OUT_PROC = os.path.join(ROOT, "recipes", "catalog_v2_processes.yaml")
OUT_TRAIN = os.path.join(ROOT, "recipes", "catalog_v2_train.yaml")
OUT_HOLD = os.path.join(ROOT, "recipes", "catalog_v2_holdout.yaml")
VIRT_CHECK = os.path.join(ROOT, "recipes", "catalog_v2_virt_check.json")   # --check-virt record, merged by --write

MASS = {5: 4.7, 6: mg.LOCKED_MT, 15: 1.777, 23: 91.1880, 24: 80.419, 25: 125.0}
QUARKS = {"u", "d", "s", "c", "b", "t"}
COLOURED = QUARKS | {"g"}
BOSONS = {"a", "z", "w+", "w-", "h"}
CHARGE = {"e-": -1, "e+": 1, "mu-": -1, "mu+": 1, "ta-": -1, "ta+": 1, "ve": 0, "ve~": 0, "vm": 0, "vm~": 0,
          "vt": 0, "vt~": 0, "u": 2, "u~": -2, "d": -1, "d~": 1, "s": -1, "s~": 1, "c": 2, "c~": -2,
          "b": -1, "b~": 1, "t": 2, "t~": -2, "g": 0, "a": 0, "z": 0, "w+": 3, "w-": -3, "h": 0}   # in units of e/3
BEAM_NAME = {"e+ e-": "ee", "u u~": "uubar", "u d~": "udbar", "u u": "uu", "u d": "ud", "u s": "us"}
TOK_NAME = {"e-": "em", "e+": "ep", "mu-": "mum", "mu+": "mup", "ta-": "tam", "ta+": "tap", "ve": "ve", "ve~": "vebar",
            "vm": "vm", "vm~": "vmbar", "vt": "vt", "vt~": "vtbar", "u": "u", "u~": "ubar", "d": "d", "d~": "dbar",
            "s": "s", "s~": "sbar", "c": "c", "c~": "cbar", "b": "b", "b~": "bbar", "t": "t", "t~": "tbar",
            "g": "g", "a": "a", "z": "Z", "w+": "Wp", "w-": "Wm", "h": "H"}


# ---------------------------------------------------------------- helpers
def split(gen):
    """'generate u u~ > z g QED<=2' -> (beams 'u u~', finals [...], suffix 'QED<=2')."""
    s = gen.replace("generate", "", 1).strip()
    m = re.search(r"\s(QED\s*<=\s*\d+)$", s)
    suffix = m.group(1).replace(" ", "") if m else ""
    if m: s = s[:m.start()]
    beams, fin = s.split(">")
    return beams.strip(), fin.split(), suffix

def canon(gen):
    """Order-independent key of a process (finals sorted, suffix kept)."""
    b, f, suf = split(gen)
    return f"{b} > {' '.join(sorted(f))}" + (f" {suf}" if suf else "")

def name_of(beams, finals, suffix=""):
    return f"{BEAM_NAME[beams]}_{''.join(TOK_NAME[t] for t in finals)}"

def gen_of(beams, finals, suffix=""):
    return f"generate {beams} > {' '.join(finals)}" + (f" {suffix}" if suffix else "")

def has_colour(beams, finals):
    return any(t.rstrip('~') in COLOURED for t in beams.split() + finals)

def charge_ok(beams, finals):
    return sum(CHARGE[t] for t in beams.split()) == sum(CHARGE[t] for t in finals)

def existing_by_canon():
    """canon(generate) -> catalog name, for every tree entry already in PROCESSES."""
    out = {}
    for n, c in mg.PROCESSES.items():
        if c.get("_v2"):
            continue      # generated last time: not "existing" for the purpose of re-generation
        if "mg5_generate" in c and c.get("kind") != "virt" and "pdg_ids" in c and len(c["mg5_generate"]) == 1:
            out.setdefault(canon(c["mg5_generate"][0]), n)
    return out


# ---------------------------------------------------------------- enumeration
def v1_bases(role):
    """(name, generate) of the v1 2->2 entries in the given role file."""
    f = OUT_TRAIN.replace("v2_train", "v1_train") if role == "train" else OUT_HOLD.replace("v2_holdout", "v1_holdout")
    out = []
    for p in yaml.safe_load(open(f))["processes"]:
        c = mg.PROCESSES[p["name"]]
        if c.get("kind") != "virt" and c["nfinal"] == 2:
            out.append((p["name"], c["mg5_generate"][0]))
    return out

def add_leg(beams, finals, suffix, leg):
    return (beams, finals + [leg], suffix)

def enumerate_candidates():
    seen = {}
    def put(layer, beams, finals, suffix="", why=""):
        if not charge_ok(beams, finals):
            return
        g = gen_of(beams, finals, suffix); k = canon(g)
        if k in seen:
            return
        seen[k] = {"layer": layer, "beams": beams, "finals": finals, "suffix": suffix,
                   "generate": g, "name": name_of(beams, finals), "why": why}
    bases = v1_bases("train")
    # ---- 2->3 by rule
    for name, g in bases:
        b, f, suf = split(g)
        bosonic = all(t in BOSONS for t in f)
        heavy = any(t.rstrip("~") in ("t", "b") for t in f)
        if has_colour(b, f):
            put(3, b, f + ["g"], suf, f"{name}+g")
        put(3, b, f + ["a"], suf, f"{name}+a")
        if bosonic or heavy:
            put(3, b, f + ["z"], suf, f"{name}+z")
            put(3, b, f + ["h"], suf, f"{name}+h")
    # explicit 2->3 of v1 that no +leg rule produces (W emission with charge change, fusion)
    for g in ["generate e+ e- > e- ve~ w+", "generate e+ e- > ve ve~ h", "generate e+ e- > e+ e- h",
              "generate e+ e- > ve ve~ a", "generate u d~ > e+ ve g", "generate u d~ > w+ g g",
              "generate e+ e- > z h h", "generate e+ e- > t t~ h", "generate e+ e- > w+ w- h"]:
        b, f, suf = split(g); put(3, b, f, suf, "v1 explicit")
    layer3 = [c for c in seen.values() if c["layer"] == 3]
    # ---- 2->4 by rule
    for c in layer3:
        b, f, suf = c["beams"], c["finals"], c["suffix"]
        if has_colour(b, f):
            put(4, b, f + ["g"], suf, c["name"] + "+g")
        put(4, b, f + ["a"], suf, c["name"] + "+a")
    pairs = [["u", "u~"], ["d", "d~"], ["b", "b~"], ["t", "t~"], ["mu+", "mu-"], ["ta+", "ta-"], ["ve", "ve~"]]
    for p1, p2 in itertools.combinations_with_replacement(range(len(pairs)), 2):
        f = pairs[p1] + pairs[p2]
        put(4, "e+ e-", f, "", "four-fermion")          # e+e- beams: QED>=2 anyway, no suffix
        if sum(t.rstrip("~") in QUARKS for t in f) >= 2:
            put(4, "u u~", f, "QED<=2", "four-fermion, quark beams")
    for f in [["w+", "w-", "z", "z"], ["w+", "w-", "w+", "w-"], ["w+", "w-", "a", "a"], ["z", "z", "z", "z"],
              ["w+", "w-", "z", "a"], ["z", "z", "z", "a"], ["z", "z", "a", "a"], ["w+", "w-", "z", "h"],
              ["z", "z", "h", "h"], ["w+", "w-", "h", "h"]]:
        put(4, "e+ e-", f, "", "multi-boson"); put(4, "u u~", f, "", "multi-boson, quark beams")
    for xy in [["w+", "w-"], ["z", "z"], ["h", "h"], ["z", "h"], ["a", "a"], ["t", "t~"], ["b", "b~"]]:
        put(4, "e+ e-", ["ve", "ve~"] + xy, "", "fusion")
    return list(seen.values())


# ---------------------------------------------------------------- MG5 check
def mg5_check(cands, workers=8):
    """One MG5 session per candidate (an error would abort a batch); parse existence,
    diagram count and WEIGHTED. Runs where MG5_BIN exists (the login node)."""
    tmp = os.path.join(mg.WORK_DIR, "..", "tmp", "catv2"); os.makedirs(tmp, exist_ok=True)
    def run_mg5(tag, gen, model):
        f = os.path.join(tmp, tag + ".in")
        open(f, "w").write(f"import model {model}\n{gen}\ndisplay processes\nexit\n")
        r = subprocess.run([mg.MG5_BIN, f], capture_output=True, text=True, timeout=1800)
        out = r.stdout + r.stderr
        m = re.search(r"Process: (.*?)(?: WEIGHTED<=(\d+))?(?: @\d+)?\s*$", out, re.M)
        d = re.search(r"Total: 1 processes with (\d+) diagrams", out)
        ok = m is not None and d is not None
        err = "" if ok else (re.search(r"(NoDiagramException|InvalidCmd|Error)[^\n]{0,120}", out).group(0)
                             if re.search(r"NoDiagramException|InvalidCmd|Error", out) else "no process line")
        return ok, (int(d.group(1)) if d else 0), (int(m.group(2)) if (m and m.group(2)) else None), err
    def one(c):
        loop = c["layer"] in ("nlo", "loop")
        model = c.get("model", "loop_sm") if loop else "sm"
        ok, diag, W, err = run_mg5(c["name"], c["generate"], model)
        c2 = dict(c, exists=ok, diagrams=diag, weighted=W, error=err)
        if ok and c.get("suffix") and not loop:
            # the unsuffixed (MG5 default = minimal QED) string gives b_min via WEIGHTED, hence
            # the true alpha_s maximum n - b_min and whether the suffix binds at all
            ok0, diag0, W0, _ = run_mg5(c["name"] + "__default", c["generate"].replace(" " + c["suffix"], ""), model)
            c2.update(weighted_default=W0 if ok0 else None, diagrams_default=diag0)
        return c2
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, cands))


# ---------------------------------------------------------------- orders / entries
def masses_entry(e):
    """m_finals of a catalog entry (2->2 entries carry a scalar or pair `m_final`)."""
    if "m_finals" in e: return [float(m) for m in e["m_finals"]]
    m = e.get("m_final", 0.0)
    return [float(x) for x in m] if isinstance(m, (list, tuple)) else [float(m)] * int(e["nfinal"])


def orders_tree(c):
    """[L_QCD, L_EW, a_max, b_max] of a tree. MG5's default order is minimal QED, so its
    WEIGHTED line gives the minimal EW power b_min = W - n and the maximal alpha_s power
    a_max = n - b_min = 2n - W. With an explicit QED<=N the EW maximum is N (N >= b_min, or
    no diagrams would exist) and a_max is unchanged: it is set by the legs that cannot be
    QCD (leptons, gamma/Z/W/H, a Yukawa), not by the suffix."""
    n = len(c["finals"])
    if c["suffix"]:
        W0 = c.get("weighted_default")
        if W0 is None:
            raise ValueError(f"{c['name']}: suffixed candidate without an unsuffixed WEIGHTED (rerun --check)")
        return [0, 0, 2 * n - W0, int(re.search(r"\d+", c["suffix"]).group(0))]
    W = c["weighted"]; b = W - n; a = 2 * n - W
    return [0, 0, a, b]


def suffix_binds(c):
    """False when QED<=N does not enlarge the default (minimal-QED) diagram set, i.e. when
    N == b_min: then the candidate IS the default-order process and must dedup against it."""
    if not c.get("suffix"):
        return True
    n = len(c["finals"]); W0 = c.get("weighted_default")
    return W0 is not None and int(re.search(r"\d+", c["suffix"]).group(0)) > W0 - n

def entry_tree(c):
    b, f = c["beams"], c["finals"]
    pdg_b, pdg_f = mg.generate_slot_pdgs(c["generate"])
    if pdg_b[1] == -pdg_b[0] and pdg_b[0] < 0:
        pdg_b = [pdg_b[1], pdg_b[0]]     # stored convention: the particle (e-, quark) in row 0
    masses = [MASS.get(abs(p), 0.0) for p in pdg_f]
    a = orders_tree(c)[2]
    e = {"mg5_generate": [c["generate"]], "nfinal": len(f), "pdg_ids": pdg_b + pdg_f,
         "m_finals": masses, "run_card_patches": {"lpp1": "0", "lpp2": "0"},
         "param_card_patches": (dict(mg.TOP_PATCHES) if 6 in map(abs, pdg_f) else {}),
         "layer": c["layer"], "why": c["why"], "diagrams": c["diagrams"]}
    if a: e["alphas_power"] = a
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enumerate", action="store_true"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-virt", action="store_true", help="run the generated [virt=QCD] strings through MG5 (after --write); non-existent ones are removed from the recipes")
    ap.add_argument("--scan", action="store_true", help="write the sparse parameter-scan layer (catalog_v2_scan.yaml, catalog_v2_train_scan.yaml, catalog_v2_holdout_scan.yaml)")
    args = ap.parse_args()
    if args.enumerate:
        cands = enumerate_candidates()
        ex = existing_by_canon()
        for c in cands:
            c["existing"] = ex.get(canon(c["generate"]))
        json.dump(cands, open(CAND, "w"), indent=1)
        from collections import Counter
        print(f"{len(cands)} candidates:", dict(Counter(c['layer'] for c in cands)),
              f"({sum(1 for c in cands if c['existing'])} already in the catalog)")
    if args.check:
        cands = json.load(open(CAND))
        todo = [c for c in cands if not c.get("existing")]
        res = mg5_check(todo, args.workers)
        byname = {c["name"]: c for c in res}
        cands = [byname.get(c["name"], c) for c in cands]
        json.dump(cands, open(CAND, "w"), indent=1)
        ok = [c for c in cands if c.get("exists") or c.get("existing")]
        bad = [c for c in cands if not (c.get("exists") or c.get("existing"))]
        print(f"exist: {len(ok)}   non-existent/failed: {len(bad)}")
        for c in bad: print("   ", c["generate"], "->", c.get("error"))
    if args.write:
        cands = json.load(open(CAND))
        procs, virt = {}, {}
        ex = existing_by_canon()
        dropped = 0
        for c in cands:
            if c.get("existing") or not c.get("exists"):
                continue
            if not suffix_binds(c):
                # same diagram set as the default-order process: either hand-written already
                # (e+e- > u u~ u u~) or its own unsuffixed candidate; never a second entry
                c0 = dict(c, suffix="", generate=c["generate"].replace(" " + c["suffix"], ""),
                          weighted=c.get("weighted_default"), diagrams=c.get("diagrams_default", c["diagrams"]))
                if ex.get(canon(c0["generate"])) or any(canon(o["generate"]) == canon(c0["generate"]) and o is not c for o in cands if o.get("exists")):
                    dropped += 1; continue
                c = c0
            procs[c["name"]] = entry_tree(c)
        print(f"dropped {dropped} candidates whose QED<=N suffix does not bind (duplicates of default-order processes)")
        # ---- NLO rule: [virt=QCD] of every coloured tree at 2->2 and 2->3 that has no
        # one-loop entry yet (existing *_nlo entries and their VIRT bases are kept as is)
        from tools.nlo_virtual_pipeline import VIRT_PROCESSES
        have_virt = {canon(re.sub(r"\s*\[.*?\]\s*$", "", v["mg5"])) for v in VIRT_PROCESSES.values() if not v.get("_v2")}
        trees = {n: mg.PROCESSES[n] for n in mg.PROCESSES if mg.PROCESSES[n].get("kind") != "virt" and "mg5_generate" in mg.PROCESSES[n]
                 and "pdg_ids" in mg.PROCESSES[n] and not mg.PROCESSES[n].get("_v2")}
        trees.update({n: e for n, e in procs.items()})
        n_nlo = 0
        for n, e in trees.items():
            g = e["mg5_generate"][0]; bm, f, suf = split(g)
            if e["nfinal"] > 3 or not has_colour(bm, f) or canon(g) in have_virt:
                continue
            base = n                                              # VIRT table key = tree name
            if base in VIRT_PROCESSES and not VIRT_PROCESSES[base].get("_v2"):
                raise KeyError(f"one-loop key {base} already hand-written with a different string: {VIRT_PROCESSES[base]['mg5']}")
            virt[base] = {"mg5": g + " [virt=QCD]", "pdg_ids": list(e["pdg_ids"]), "m_finals": masses_entry(e)}
            procs[n + "_nlo"] = {"kind": "virt", "virt": True, "virt_base": base, "nfinal": e["nfinal"],
                                 "n_loops": 1, "alphas_power": int(e.get("alphas_power", 0)) + 1,
                                 "pdg_ids": list(e["pdg_ids"]), "m_finals": masses_entry(e),
                                 "param_card_patches": {}, "layer": "nlo", "why": f"one-loop QCD of {n}"}
            if base in VIRT_UNCERTIFIED:
                virt[base]["certified"] = False; virt[base]["why"] = VIRT_UNCERTIFIED[base]
                procs[n + "_nlo"]["certified"] = False; procs[n + "_nlo"]["why"] = "NOT certified: " + VIRT_UNCERTIFIED[base]
            n_nlo += 1
        # ---- loop-induced candidates (certification decides; uncertified never enter a recipe).
        # Both are pure-EW top loops in practice (probe: no alpha_s dependence), reached through
        # the mixed [sqrvirt=QCD QED] selection. uubar_HH certifies (poles = 0); uubar_Ha shows
        # the same spurious ~0.2 single pole as ee_aH (an H+gamma final pathology of MadLoop
        # 3.7.0) and stays uncertified.
        for name, g, model, order, pdg, m, cert in [
            ("uubar_Ha", "generate u u~ > h a [sqrvirt=QCD QED]", "loop_qcd_qed_sm", [0, 1, 0, 4], [2, -2, 25, 22], [125.0, 0.0], False),
            ("uubar_HH", "generate u u~ > h h [sqrvirt=QCD QED]", "loop_qcd_qed_sm", [0, 1, 0, 4], [2, -2, 25, 25], [125.0, 125.0], True),
        ]:
            virt[name] = {"mg5": g, "model": model, "loopind": True, "pdg_ids": pdg, "m_finals": m, "order": order, "certified": cert}
            procs[name + "_loop"] = {"kind": "virt", "virt": True, "virt_base": name, "nfinal": 2, "n_loops": 1,
                                     "alphas_power": order[2], "loopind": True, "order": order, "pdg_ids": pdg,
                                     "m_finals": m, "param_card_patches": {}, "layer": "loop", "certified": cert,
                                     "why": "loop-induced (pure-EW top loop)" if cert else "loop-induced, NOT certified (spurious MadLoop pole)"}
        if os.path.exists(VIRT_CHECK):      # keep the MG5 record of the one-loop strings across re-writes
            vc = json.load(open(VIRT_CHECK))
            for k, v in virt.items():
                if k in vc:
                    v["exists"] = vc[k]["exists"]; v["diagrams"] = vc[k]["diagrams"]
            for k in [k for k, v in virt.items() if vc.get(k, {}).get("exists") is False]:
                virt.pop(k); procs.pop(k + "_nlo", None)
        yaml.safe_dump({"processes": procs, "virt": virt}, open(OUT_PROC, "w"), sort_keys=False, width=160)
        print(f"wrote {OUT_PROC}: {len(procs) - n_nlo - 2} tree, {n_nlo} one-loop, 2 loop-induced candidates")
        write_recipes(procs, cands)
    if args.check_virt:
        check_virt(args.workers)
    if args.scan:
        write_scan()


# ---------------------------------------------------------------- sparse parameter scan
# A few representative, far-apart points per axis (not a dense grid), as decorated variants
# of the families the axis actually moves; the base point stays in the training set:
#   alpha_s(M_Z) in {0.09, 0.15} on the hand-reasoned trees with an alpha_s power, <= 2->3
#   m_t in {150, 195} GeV on the top family (threshold moves; the Yukawa follows)
#   M_Z in {80, 105} GeV on the s-channel fermion-pair families (the needle moves)
# Hold-outs on this axis are INTERPOLATION points between the two far points and the base.
SCAN_ALPHAS = [0.09, 0.15]
SCAN_MT = [150.0, 195.0]
SCAN_MZ = [80.0, 105.0]
SCAN_HOLDOUT = [("uubar_gg", {"alpha_s": 0.13}), ("ee_uug", {"alpha_s": 0.13}), ("ee_ttbar", {"masses": {6: 185.0}}),
                ("ee_uu", {"masses": {23: 95.0}}), ("ee_mumu", {"masses": {23: 95.0}})]
TOP_FAMILY = ["ee_ttbar", "uubar_ttbar", "udbar_tbbar", "ee_ttbarg", "ee_ttbarH", "uubar_ttbarg"]
ZPOLE_FAMILY = ["ee_mumu", "ee_tautau", "ee_uu", "ee_ddbar", "ee_bbbar", "ee_numu", "ee_nnbar",
                "uubar_uubar", "uubar_ddbar", "uubar_bbbar"]


def _tag(physics):
    if "alpha_s" in physics: return f"__as{int(round(physics['alpha_s'] * 1000)):03d}"
    (pdg, m), = physics["masses"].items()
    ms = f"{int(round(m)):03d}" if abs(m - round(m)) < 1e-9 else f"{m:.1f}".replace(".", "p")   # 172.5 -> 172p5, never collapsed
    return f"__{ {6: 'mt', 23: 'mz'}[int(pdg)] }{ms}"


_BASE_WINDOWS = None


def _base_window(base):
    """The base entry's window as written in the train recipe (a coupling-only variant must
    keep the phase space bit-identical to its base: the alpha_s axis is the only difference).
    The recipe is parsed once."""
    global _BASE_WINDOWS
    if _BASE_WINDOWS is None:
        _BASE_WINDOWS = {p["name"]: (int(p["sqrts"][0]), int(p["sqrts"][1]))
                         for p in yaml.safe_load(open(OUT_TRAIN))["processes"]}
    return _BASE_WINDOWS[base]


def _variant(base, physics, N):
    """Recipe line of a decorated variant: a mass variant's window floor follows the scanned
    masses, a coupling-only variant inherits the base window unchanged."""
    e = mg.PROCESSES[base]; m = masses_entry(e)
    if physics.get("masses"):
        for pdg, mv in physics["masses"].items():
            m = [float(mv) if abs(int(q)) == abs(int(pdg)) else mm for q, mm in zip(e["pdg_ids"][2:], m)]
        lo = int(round(max(1.05 * sum(m), 25.0)))
    else:
        lo = _base_window(base)[0]
    name = base + _tag(physics)
    phys = {k: (v if k != "masses" else {int(pp): float(vv) for pp, vv in v.items()}) for k, v in physics.items()}
    return name, (f"  - {{name: {name + ',':<26} base: {base + ',':<16} sqrts: [{lo:>4}, 1000], n_train: {N[0]}, n_val: {N[1]}, n_test: {N[2]}, "
                  f"physics: {yaml.safe_dump(phys, default_flow_style=True, width=200).strip()}}}")


def write_scan():
    T = (100000, 10000, 10000)
    v1 = [p["name"] for p in yaml.safe_load(open(OUT_TRAIN.replace("v2_train", "v1_train")))["processes"]]
    alphas_family = [n for n in v1 if mg.PROCESSES[n].get("kind") != "virt" and mg.PROCESSES[n]["nfinal"] <= 3 and mg.PROCESSES[n].get("alphas_power", 0) >= 1]
    lines, names = [], []
    def add(base, physics):
        n, l = _variant(base, physics, T); lines.append(l); names.append(n)
    lines.append("  # alpha_s(M_Z) in {0.09, 0.15}: trees with an alpha_s power (shared backend, alpha_s per event)")
    for b in alphas_family:
        for a in SCAN_ALPHAS: add(b, {"alpha_s": a})
    lines.append("  # m_t in {150, 195} GeV: the top family (own standalone per point; Yukawa follows)")
    for b in TOP_FAMILY:
        for m in SCAN_MT: add(b, {"masses": {6: m}})
    lines.append("  # M_Z in {80, 105} GeV: s-channel fermion pairs (own standalone per point)")
    for b in ZPOLE_FAMILY:
        for m in SCAN_MZ: add(b, {"masses": {23: m}})
    assert len(names) == len(set(names))
    hdr = ["# catalog_v2 parameter-scan layer: a few far-apart points per axis, decorated variants of",
           "# the base entries (the base point itself is in catalog_v2_train.yaml). The physics block is",
           "# the single source of truth for generation (card patches, masses) and the coupling feature.",
           "sampling: {mode: mixture}", "processes:"]
    open(OUT_TRAIN.replace("v2_train", "v2_scan"), "w").write("\n".join(hdr + lines) + "\n")
    tr = open(OUT_TRAIN).read().rstrip("\n")
    open(OUT_TRAIN.replace("v2_train", "v2_train_scan"), "w").write(tr + "\n  # --- parameter-scan layer (catalog_v2_scan.yaml) ---\n" + "\n".join(lines) + "\n")
    hl = ["# catalog_v2 scan hold-outs: interpolation points between the far scan points and the base.",
          "sampling: {mode: mixture}   # (val/test always flat; the train pool of a hold-out is for fine-tune curves)", "processes:"]
    for b, ph in SCAN_HOLDOUT:
        n, l = _variant(b, ph, T); hl.append(l + "   # interpolation hold-out")
    open(OUT_HOLD.replace("v2_holdout", "v2_holdout_scan"), "w").write("\n".join(hl) + "\n")
    print(f"scan layer: {len(names)} variants ({2*len(alphas_family)} alpha_s, {2*len(TOP_FAMILY)} m_t, {2*len(ZPOLE_FAMILY)} M_Z) + {len(SCAN_HOLDOUT)} interpolation hold-outs")


def write_recipes(procs, cands):
    """catalog_v2_train.yaml = v1 train + every validated v2 entry (loop-induced only if
    certified); catalog_v2_holdout.yaml = v1 hold-outs (relabels/crossings) unchanged."""
    masses_of = masses_entry
    def line(n, c, N):
        lo = int(round(max(1.05 * sum(masses_of(c)), 25.0)))
        return f"  - {{name: {n + ',':<22} sqrts: [{lo:>4}, 1000], n_train: {N[0]}, n_val: {N[1]}, n_test: {N[2]}}}"
    T, NLO = (100000, 10000, 10000), (50000, 5000, 5000)
    v1 = yaml.safe_load(open(OUT_TRAIN.replace("v2_train", "v1_train")))["processes"]
    out = ["# catalog_v2 -- rule-enumerated catalog (recipes/gen_catalog_v2.py), MadGraph-validated.",
           "# = catalog_v1_train + every 2->3/2->4 tree the rules produce, one-loop QCD of every",
           "# coloured tree <= 2->3, and the certified loop-induced entries. Entries beyond v1 are",
           "# defined in recipes/catalog_v2_processes.yaml (loaded by the pipeline at import).",
           "# Axis hold-outs are configs: drop every '+g' at 2->4 (multiplicity), drop the *_nlo of",
           "# one family (order), drop one loop-induced entry.", "sampling: {mode: mixture}   # pipeline DEFAULT_SAMPLING: (1-f) uniform-sqrt(s) bulk + f flat-in-log|M|^2, f=0.35",
           "processes:", "  # --- catalog_v1 ---"]
    seen = set()
    for p in v1:
        out.append(f"  - {{name: {p['name'] + ',':<22} sqrts: [{p['sqrts'][0]:>4}, {p['sqrts'][1]}], n_train: {p['n_train']}, n_val: {p['n_val']}, n_test: {p['n_test']}}}"); seen.add(p["name"])
    for layer, title in ((3, "2->3 by rule"), (4, "2->4 by rule"), ("nlo", "one-loop QCD by rule"), ("loop", "loop-induced candidates (certified only)")):
        out.append(f"  # --- {title} ---")
        for n, e in procs.items():
            if e.get("layer") != layer or n in seen: continue
            if layer == "loop" and not e.get("certified", False):
                out.append(f"  # {n}: not certified yet (tools/nlo_virtual_pipeline.py {e['virt_base']} --build --certify)"); continue
            if layer == "nlo" and e.get("certified") is False:
                out.append(f"  # {n}: {e['why']}"); continue
            out.append(line(n, e, NLO if layer in ("nlo", "loop") else T) + f"   # {e.get('why','')}")
    open(OUT_TRAIN, "w").write("\n".join(out) + "\n")
    hold = open(OUT_HOLD.replace("v2_holdout", "v1_holdout")).read().replace("catalog_v1 hold-outs", "catalog_v2 hold-outs (= v1)")
    open(OUT_HOLD, "w").write(hold)
    n_train = sum(1 for l in out if l.strip().startswith("- {name:"))
    print(f"wrote {OUT_TRAIN}: {n_train} processes; {OUT_HOLD}: v1 hold-outs")


def check_virt(workers):
    """Run every generated one-loop string through MG5 `display processes`; drop the failures
    from catalog_v2_processes.yaml and rewrite the recipes."""
    d = yaml.safe_load(open(OUT_PROC))
    todo = [(k, v) for k, v in d["virt"].items() if not v.get("loopind")]
    tmp = os.path.join(mg.WORK_DIR, "..", "tmp", "catv2"); os.makedirs(tmp, exist_ok=True)
    def one(kv):
        k, v = kv
        f = os.path.join(tmp, k + "__virt.in")
        open(f, "w").write(f"import model {v.get('model', 'loop_sm')}\n{v['mg5']}\ndisplay processes\nexit\n")
        r = subprocess.run([mg.MG5_BIN, f], capture_output=True, text=True, timeout=3600)
        out = r.stdout + r.stderr
        ok = re.search(r"^Process: ", out, re.M) is not None and not re.search(r"NoDiagramException|InvalidCmd|Error detected", out)
        loops = re.search(r"Total: 1 processes with (\d+) diagrams", out)
        return k, ok, (int(loops.group(1)) if loops else 0), ("" if ok else re.sub(r"\s+", " ", out[-300:]))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, todo))
    bad = [k for k, ok, _, _ in res if not ok]
    for k, ok, n, err in res:
        d["virt"][k]["exists"] = ok; d["virt"][k]["diagrams"] = n
        if not ok: print("   MG5 rejects", k, d["virt"][k]["mg5"], "::", err[-160:])
    for k in bad:
        d["virt"].pop(k); d["processes"].pop(k + "_nlo", None)
    json.dump({k: {"exists": ok, "diagrams": n} for k, ok, n, _ in res}, open(VIRT_CHECK, "w"), indent=1)
    yaml.safe_dump(d, open(OUT_PROC, "w"), sort_keys=False, width=160)
    print(f"one-loop strings: {len(res) - len(bad)} exist, {len(bad)} removed")
    cands = json.load(open(CAND)); write_recipes(d["processes"], cands)


if __name__ == "__main__":
    main()
