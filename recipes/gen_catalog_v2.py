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

CAND = os.path.join(ROOT, "recipes", "catalog_v2_candidates.json")
OUT_PROC = os.path.join(ROOT, "recipes", "catalog_v2_processes.yaml")
OUT_TRAIN = os.path.join(ROOT, "recipes", "catalog_v2_train.yaml")
OUT_HOLD = os.path.join(ROOT, "recipes", "catalog_v2_holdout.yaml")

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
        suf = "QED<=2" if sum(t.rstrip("~") in QUARKS for t in f) == 4 else ""
        put(4, "e+ e-", f, suf, "four-fermion")
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
    def one(c):
        f = os.path.join(tmp, c["name"] + ".in")
        loop = c["layer"] in ("nlo", "loop")
        model = c.get("model", "loop_sm") if loop else "sm"
        open(f, "w").write(f"import model {model}\n{c['generate']}\ndisplay processes\nexit\n")
        r = subprocess.run([mg.MG5_BIN, f], capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
        m = re.search(r"Process: (.*?)(?: WEIGHTED<=(\d+))?(?: @\d+)?\s*$", out, re.M)
        d = re.search(r"Total: 1 processes with (\d+) diagrams", out)
        ok = m is not None and d is not None
        c2 = dict(c, exists=ok, diagrams=int(d.group(1)) if d else 0,
                  weighted=int(m.group(2)) if (m and m.group(2)) else None,
                  error=("" if ok else (re.search(r"(NoDiagramException|InvalidCmd|Error)[^\n]{0,120}", out).group(0)
                                        if re.search(r"NoDiagramException|InvalidCmd|Error", out) else "no process line")))
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
    """[L_QCD, L_EW, a_max, b_max] of a tree from MG5's WEIGHTED (a+2b=W, a+b=n) or,
    for an explicit QED<=N, b=N and a=n (a pure-QCD contribution exists there)."""
    n = len(c["finals"])
    if c["suffix"]:
        return [0, 0, n, int(re.search(r"\d+", c["suffix"]).group(0))]
    W = c["weighted"]; b = W - n; a = 2 * n - W
    return [0, 0, a, b]

def entry_tree(c):
    b, f = c["beams"], c["finals"]
    pdg_b, pdg_f = mg.generate_slot_pdgs(c["generate"])
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
        for c in cands:
            if c.get("existing") or not c.get("exists"):
                continue
            procs[c["name"]] = entry_tree(c)
        # ---- NLO rule: [virt=QCD] of every coloured tree at 2->2 and 2->3 that has no
        # one-loop entry yet (existing *_nlo entries and their VIRT bases are kept as is)
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from nlo_virtual_pipeline import VIRT_PROCESSES
        have_virt = {canon(re.sub(r"\s*\[.*?\]\s*$", "", v["mg5"])) for v in VIRT_PROCESSES.values() if not v.get("_v2")}
        trees = {n: mg.PROCESSES[n] for n in mg.PROCESSES if mg.PROCESSES[n].get("kind") != "virt" and "mg5_generate" in mg.PROCESSES[n]
                 and "pdg_ids" in mg.PROCESSES[n] and not mg.PROCESSES[n].get("_v2")}
        trees.update({n: e for n, e in procs.items()})
        n_nlo = 0
        for n, e in trees.items():
            g = e["mg5_generate"][0]; bm, f, suf = split(g)
            if e["nfinal"] > 3 or not has_colour(bm, f) or canon(g) in have_virt:
                continue
            base = n if not n.startswith("ee_") else n           # VIRT table key = tree name
            virt[base] = {"mg5": g + " [virt=QCD]", "pdg_ids": list(e["pdg_ids"]), "m_finals": masses_entry(e)}
            procs[n + "_nlo"] = {"kind": "virt", "virt": True, "virt_base": base, "nfinal": e["nfinal"],
                                 "n_loops": 1, "alphas_power": int(e.get("alphas_power", 0)) + 1,
                                 "pdg_ids": list(e["pdg_ids"]), "m_finals": masses_entry(e),
                                 "param_card_patches": {}, "layer": "nlo", "why": f"one-loop QCD of {n}"}
            n_nlo += 1
        # ---- loop-induced candidates (certification decides; uncertified never enter a recipe).
        # Pure-QED loops give no diagrams here (the light-quark line has no Yukawa): the top
        # loop is reached through a gluon, so these are mixed QCD x QED loops; the order vector
        # [1,1,2,2] is a placeholder until the probe pins the alpha_s scaling.
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
        yaml.safe_dump({"processes": procs, "virt": virt}, open(OUT_PROC, "w"), sort_keys=False, width=160)
        print(f"wrote {OUT_PROC}: {len(procs) - n_nlo - 2} tree, {n_nlo} one-loop, 2 loop-induced candidates")
        write_recipes(procs, cands)


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
           "# one family (order), drop one loop-induced entry.", "processes:", "  # --- catalog_v1 ---"]
    seen = set()
    for p in v1:
        out.append(f"  - {{name: {p['name'] + ',':<22} sqrts: [{p['sqrts'][0]:>4}, {p['sqrts'][1]}], n_train: {p['n_train']}, n_val: {p['n_val']}, n_test: {p['n_test']}}}"); seen.add(p["name"])
    for layer, title in ((3, "2->3 by rule"), (4, "2->4 by rule"), ("nlo", "one-loop QCD by rule"), ("loop", "loop-induced candidates (certified only)")):
        out.append(f"  # --- {title} ---")
        for n, e in procs.items():
            if e.get("layer") != layer or n in seen: continue
            if layer == "loop" and not e.get("certified", False):
                out.append(f"  # {n}: not certified yet (tools/nlo_virtual_pipeline.py {e['virt_base']} --build --certify)"); continue
            out.append(line(n, e, NLO if layer in ("nlo", "loop") else T) + f"   # {e.get('why','')}")
    open(OUT_TRAIN, "w").write("\n".join(out) + "\n")
    hold = open(OUT_HOLD.replace("v2_holdout", "v1_holdout")).read().replace("catalog_v1 hold-outs", "catalog_v2 hold-outs (= v1)")
    open(OUT_HOLD, "w").write(hold)
    n_train = sum(1 for l in out if l.strip().startswith("- {name:"))
    print(f"wrote {OUT_TRAIN}: {n_train} processes; {OUT_HOLD}: v1 hold-outs")


if __name__ == "__main__":
    main()
