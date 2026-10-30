"""Do the 13 TeV pp pools carry catalog_v4's physics? (catalog_coverage; the user, 2026-10-26: "You told me you were
specifically making the b quark massless to match pp beams, and now you are telling me that you generated the pp beams
with the same configuration as catalog 3? is this a joke?")
The pp pools (pp13_*, recipes/gen_catalog_v3_pp.py, 2026-10-25) were labelled with catalog_v3's physics: model sm (b at
4.7 GeV in beams and finals, b Yukawa on), the minimal QED order of the generate string, MT = ymt = 173 unless a top is
external (recipes/catalog_v3_pp/param_card.dat). catalog_v4 (recipes/gen_catalog_v4.py) has a massless b
(sm-no_b_mass), QED<=b on every tree (every tree diagram: VBF, VBS, the EW-QCD interference) and MT = ymt = 172.5
everywhere. Per pp class:
  classify  b      an external b (beam or final): a different |M|^2 by construction (mass, Yukawa)
            ew     catalog_v4's string has more diagrams than catalog_v3's (data/checked.json, MadGraph's count of both)
            itop   an internal top line (an external b and W and no external top: diagonal CKM, so a tree's top line
                   ends on a b-W vertex or an external top): MT 173 against 172.5; a subset of b
            none   none of these: the same function, up to card values checked below
            dropped the b-Higgs trees catalog_v4 has no diagram for (no counterpart)
  eval (CC) the class's catalog_v4 standalone (mg5amcnlo/v4_<x>_standalone, compiled for the v4 pools) on the first
            N rows of the pp validation pool (the rows the zero-shot reads score first), at the pool's own alpha_s
            (<pool>.alphas.npy, alpha_s(sqrt(s_hat)) at one loop), against the stored label; and the pp standalone
            (pp13_<x>_standalone) on the same rows, which must give the label back (the pipeline's check). A b class's
            momenta are first put on the massless b shell: in the partonic rest frame the beams keep sqrt(s_hat) along
            z, the finals' 3-momenta are scaled by one factor so the energies sum to sqrt(s_hat) again, masses other than
            the b's unchanged (the nearest massless-b event; the shift is recorded).
            Per class: the natural-log ratio r = ln(|M|^2_v4 / label): median, 10/90% quantiles, and mean r^2, the
            MSE a model that had learned catalog_v4's |M|^2 exactly would score on these labels (the reads' MSE is in
            the same units, natural log of |M|^2). The card values (MT, ymt, MB, ymb, WT, WZ, WW, WH) of both standalones.
    site run --item catalog_coverage ccin2p3 FA -- python analysis/catalog_v4/pp_physics_check.py eval --n 200 \
        --part K --parts P  > data/pp_physics_eval_K.json
    python3 analysis/catalog_v4/pp_physics_check.py classify            (counts)
    site run ... -- python analysis/catalog_v4/pp_physics_check.py cost > data/pp_physics_cost.json
    python3 analysis/catalog_v4/pp_physics_check.py summary             -> data/pp_physics_check.json (classes, evals)
The last line printed by summary is the JSON of its numbers.
"""
import glob, json, os, sys
from collections import Counter

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
CARD_KEYS = {("MASS", 5): "MB", ("MASS", 6): "MT", ("YUKAWA", 5): "ymb", ("YUKAWA", 6): "ymt",
             ("DECAY", 6): "WT", ("DECAY", 23): "WZ", ("DECAY", 24): "WW", ("DECAY", 25): "WH"}


SET = "v3"     # --set v4: the pp13v4_* pools (recipes/catalog_v4_pp_test.yaml), built with catalog_v4's physics
PREFIX = {"v3": "pp13_", "v4": "pp13v4_"}
RECIPE = {"v3": "catalog_v3_pp_test.yaml", "v4": "catalog_v4_pp_test.yaml"}


def pp_names():
    return [e["name"] for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", RECIPE[SET])))["processes"]]


def built():
    """--set v4: per pp entry, which of its train, val and test pools are on disk (with their recipe sidecar)."""
    import mg5_pipeline_final as mg
    import datagen
    out = {}
    for n in pp_names():
        cfg = mg.PROCESSES.get(n)
        r = {}
        for role in ("train", "val", "test"):
            try:
                lo, hi = mg.lhe_layout(cfg)[role]
                path = mg.recipe_output_path(mg.lhe_pp_recipe(n, 13000, 13000, hi - lo, role=role, seed=42),
                                             datagen.dest_for_role(role))
                r[role] = os.path.exists(path) and os.path.exists(path + ".recipe.json")
            except Exception as ex:
                r[role] = repr(ex)[:120]
        out[n] = r
    missing = {n: r for n, r in out.items() if not all(v is True for v in r.values())}
    json.dump({"entries": len(out), "all_roles": len(out) - len(missing), "missing": missing}, sys.stdout)


def classify():
    import mg5_pipeline_final as mg
    ck = {c["name"]: c for c in json.load(open(os.path.join(HERE, "data", "checked.json")))}
    out = {}
    for n in pp_names():
        x = n[len(PREFIX[SET]):]
        v4 = "v4_" + x
        pdg = [abs(int(p)) for p in mg.PROCESSES[n]["pdg_ids"]]
        if v4 not in mg.PROCESSES:
            out[n] = {"v4": None, "cls": "dropped", "b": 5 in pdg}
            continue
        if SET == "v4":         # same physics by construction: no reshelling (the pools' b is massless), no v3 count
            b = 5 in pdg
            out[n] = {"v4": v4, "b": b, "b_beam": 5 in pdg[:2], "ew": False, "itop": False, "reshell": False,
                      "nfinal": len(pdg) - 2, "cls": "b" if b else "none"}
            continue
        k = ck[v4]
        b, ew, itop = 5 in pdg, k["diagrams"] != k["diagrams_v3"], 5 in pdg and 24 in pdg and 6 not in pdg
        out[n] = {"v4": v4, "b": b, "b_beam": 5 in pdg[:2], "ew": ew, "itop": itop,
                  "diagrams_v3": k["diagrams_v3"], "diagrams_v4": k["diagrams"], "nfinal": len(pdg) - 2,
                  "cls": "b" if b else ("ew" if ew else "none")}
    return out


def card(path):
    v, block = {}, None
    for line in open(path):
        s = line.split("#")[0].split()
        if not s:
            continue
        if s[0].upper() == "BLOCK":
            block = s[1].upper()
        elif s[0].upper() == "DECAY":
            block = None
            if ("DECAY", int(s[1])) in CARD_KEYS:
                v[CARD_KEYS[("DECAY", int(s[1]))]] = float(s[2])
        elif block in ("MASS", "YUKAWA") and len(s) >= 2 and (block, int(s[0])) in CARD_KEYS:
            v[CARD_KEYS[(block, int(s[0]))]] = float(s[1])
    return v


def massless_b(P, pdg):
    """(N, k, 4) lab momenta -> the same events in the partonic rest frame with the b legs massless: beams
    (sqrt(s_hat)/2, 0, 0, +-sqrt(s_hat)/2), finals' 3-momenta scaled by one factor xi per event so the energies sum to
    sqrt(s_hat), other masses unchanged. Returns (momenta, xi)."""
    P = np.array(P, dtype=np.float64)
    tot = P[:, 0] + P[:, 1]
    bz = tot[:, 3] / tot[:, 0]
    g = 1.0 / np.sqrt(1.0 - bz ** 2)
    E, pz = P[..., 0].copy(), P[..., 3].copy()
    P[..., 0] = g[:, None] * (E - bz[:, None] * pz)
    P[..., 3] = g[:, None] * (pz - bz[:, None] * E)
    rs = (P[:, 0, 0] + P[:, 1, 0])
    s0 = np.sign(P[:, 0, 3])
    Q = np.zeros_like(P)
    Q[:, 0, 0] = Q[:, 1, 0] = rs / 2
    Q[:, 0, 3], Q[:, 1, 3] = s0 * rs / 2, -s0 * rs / 2
    F = P[:, 2:]
    m2 = np.clip(F[..., 0] ** 2 - (F[..., 1:] ** 2).sum(-1), 0, None)
    m2[:, [abs(int(p)) == 5 for p in pdg[2:]]] = 0.0
    p2 = (F[..., 1:] ** 2).sum(-1)
    lo, hi = np.full(len(P), 0.5), np.full(len(P), 2.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = np.sqrt(m2 + mid[:, None] ** 2 * p2).sum(-1) - rs
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    xi = 0.5 * (lo + hi)
    Q[:, 2:, 1:] = xi[:, None, None] * F[..., 1:]
    Q[:, 2:, 0] = np.sqrt(m2 + xi[:, None] ** 2 * p2)
    return Q, xi


def evaluate(n_ev, part, parts, only=None):
    import mg5_pipeline_final as mg
    C = classify()
    todo = [n for n in sorted(C) if C[n]["v4"] and (not only or n in only)]
    todo = todo[part::parts]
    out = {}
    for n in todo:
        e = C[n]
        v4 = e["v4"]
        base = os.path.join(os.environ["DATA_DIR"], f"{n}_13000-13000GeV_val_lhe_amplitudes")
        try:
            a = np.array(np.load(base + ".npy", mmap_mode="r")[:n_ev], dtype=np.float64)
            al = np.array(np.load(base + ".alphas.npy", mmap_mode="r")[:n_ev], dtype=np.float64)
            k = len(mg.PROCESSES[n]["pdg_ids"])
            P = a[:, :4 * k].reshape(-1, k, 4)
            lab = a[:, -1]
            r = {}
            for tag, proc, sa in (("pp", n, mg.lhe_standalone_dir(n)), ("v4", v4, f"{mg.WORK_DIR}/{mg.standalone_name(v4)}_standalone")):
                cfg = mg.PROCESSES[proc]
                _, _, drv, _ = mg.detect_compiled_backend(sa)
                if drv is None:
                    r[tag] = None
                    continue
                M, xi = (massless_b(P, cfg["pdg_ids"]) if tag == "v4" and e.get("reshell", e["b"]) else (P, None))
                perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
                with mg.CppDriverPipe(drv, sa) as pipe:
                    amp = pipe.compute([(m, None) for m in M], perm=perm, alphas=al)
                r[tag] = {"amp": amp, "card": card(f"{sa}/Cards/param_card.dat"),
                          "xi": None if xi is None else [float(np.median(np.abs(xi - 1))), float(np.max(np.abs(xi - 1)))]}
            o = {"n": len(lab), "beam_mass": [float(np.median(np.sqrt(np.clip(P[:, i, 0] ** 2 - (P[:, i, 1:] ** 2).sum(-1), 0, None)))) for i in (0, 1)]}
            for tag in ("pp", "v4"):
                if r[tag] is None:
                    o[tag] = None
                    continue
                amp = r[tag]["amp"]
                ok = np.isfinite(amp) & (amp > 0)
                lr = np.log(amp[ok] / lab[ok])
                o[tag] = {"n_ok": int(ok.sum()), "card": r[tag]["card"], "xi": r[tag]["xi"]}
                if ok.any():
                    o[tag].update(med=float(np.median(lr)), q10=float(np.quantile(lr, 0.1)), q90=float(np.quantile(lr, 0.9)),
                                  ms=float(np.mean(lr ** 2)), maxabs=float(np.max(np.abs(lr))))
            out[n] = o
        except Exception as ex:                       # recorded, never dropped
            out[n] = {"error": repr(ex)[:300]}
        print(f"[ppcheck] {n} {json.dumps(out[n])[:200]}", file=sys.stderr, flush=True)
    json.dump(out, sys.stdout)


def cost():
    """per pp class, the MadEvent run behind its pools, from the val pool's sidecar (<pool>.npy.recipe.json, lhe_run):
    seconds per stage, cores, events MadEvent returned and the runs it took; a run that came back short and was redone
    leaves only the redo's record"""
    out = {}
    for n in pp_names():
        f = os.path.join(os.environ["DATA_DIR"], f"{n}_13000-13000GeV_val_lhe_amplitudes.npy.recipe.json")
        try:
            r = json.load(open(f))["lhe_run"]
            out[n] = {"s": r["seconds"], "cores": r["mg_cores"], "n_lhe": r["n_lhe_events"], "runs": len(r.get("iseeds") or [0])}
        except Exception as ex:
            out[n] = {"error": repr(ex)[:200]}
    json.dump(out, sys.stdout)


def _last_json(path):
    """The eval's JSON object: the last line of the file that parses (the C++ driver's own lines can precede it on
    stdout, and the object can follow one of them on the same line)."""
    for line in reversed(open(path).read().splitlines()):
        i = line.find('{"')
        if i >= 0:
            try:
                return json.loads(line[i:])
            except ValueError:
                continue
    raise ValueError(f"{path}: no eval JSON")


def summary(evals=None, out=None):
    C = classify()
    E = {}
    for f in sorted(glob.glob(evals or os.path.join(HERE, "data", f"{'pp13v4' if SET == 'v4' else 'pp'}_physics_eval_*.json"))):
        E.update(_last_json(f))
    res = {"counts": dict(Counter(c["cls"] for c in C.values())),
           "counts_detail": dict(Counter(f"{c['cls']}{'/ew' if c.get('ew') and c['cls'] == 'b' else ''}"
                                         f"{'/itop' if c.get('itop') else ''}" for c in C.values())),
           "n_b_beam": sum(bool(c.get("b_beam")) for c in C.values())}
    err = {n: e["error"] for n, e in E.items() if "error" in e}
    res["n_eval"], res["errors"] = len(E), err
    # the pipeline check: the pp standalone gives the label back
    pr = [abs(e["pp"]["maxabs"]) for e in E.values() if e.get("pp") and "maxabs" in e["pp"]]
    res["pp_reproduces_label_maxabs_lnratio"] = float(max(pr)) if pr else None
    cards = {t: Counter(json.dumps(e[t]["card"], sort_keys=True) for e in E.values() if e.get(t)) for t in ("pp", "v4")}
    res["cards"] = {t: {k: v for k, v in c.most_common(6)} for t, c in cards.items()}
    for g, sel in (("none", lambda c: c["cls"] == "none"), ("ew", lambda c: c["cls"] == "ew"),
                   ("b_no_ew", lambda c: c["cls"] == "b" and not c["ew"]), ("b_ew", lambda c: c["cls"] == "b" and c["ew"]),
                   ("b_itop", lambda c: c["cls"] == "b" and c["itop"]), ("b_beam", lambda c: c["cls"] == "b" and c["b_beam"]),
                   ("b_final_only", lambda c: c["cls"] == "b" and not c["b_beam"])):
        ps_ = [n for n, c in C.items() if c["v4"] and sel(c) and E.get(n, {}).get("v4") and "med" in E[n]["v4"]]
        if not ps_:
            res[g] = {"n_classes": sum(1 for c in C.values() if c["v4"] and sel(c)), "n_eval": 0}
            continue
        med = np.array([E[n]["v4"]["med"] for n in ps_])
        ms = np.array([E[n]["v4"]["ms"] for n in ps_])
        spr = np.array([E[n]["v4"]["q90"] - E[n]["v4"]["q10"] for n in ps_])
        res[g] = {"n_classes": sum(1 for c in C.values() if c["v4"] and sel(c)), "n_eval": len(ps_),
                  "class_median_lnratio": {"median": float(np.median(med)), "q10": float(np.quantile(med, 0.1)),
                                           "q90": float(np.quantile(med, 0.9))},
                  "class_q10_q90_width_median": float(np.median(spr)),
                  "class_mean_sq_lnratio": {"median": float(np.median(ms)), "q90": float(np.quantile(ms, 0.9)),
                                            "max": float(np.max(ms))},
                  "share_ms_above": {str(t): float(np.mean(ms > t)) for t in (1e-6, 1e-3, 1e-2, 0.1, 1.0)},
                  "worst": sorted(((round(float(E[n]["v4"]["ms"]), 4), n) for n in ps_), reverse=True)[:5]}
    agree = {n: (E[n]["v4"]["ms"] if E.get(n, {}).get("v4") and "ms" in E[n]["v4"] else None) for n in C if C[n]["v4"]}
    json.dump({"classes": C, "eval": E, "ms_v4_vs_label": agree},
              open(out or os.path.join(HERE, "data", "pp13v4_physics_check.json" if SET == "v4" else "pp_physics_check.json"), "w"),
              indent=0)
    if SET == "v4":
        ev = [n for n in C if C[n]["v4"]]
        # the same function: catalog_v4's standalone gives the label back as exactly as the pool's own standalone does
        # (its round trip, the pipeline's precision), on every row
        ok = lambda e: (e.get("v4") and e.get("pp") and e["v4"].get("maxabs") is not None and e["pp"].get("maxabs")
                        is not None and e["v4"]["maxabs"] <= e["pp"]["maxabs"] and e["v4"]["n_ok"] == e["n"])
        off = sorted(n for n in ev if not ok(E.get(n, {})))
        res["v4_same_function"] = {"rule": "max|ln(v4/label)| <= max|ln(own standalone/label)|, every row",
                                   "classes": len(ev), "pass": len(ev) - len(off),
                                   "v4_maxabs_max": max((E[n]["v4"]["maxabs"] for n in ev if ok(E.get(n, {}))), default=None),
                                   "fail": {n: (E.get(n, {}).get("v4") or E.get(n, {}).get("error")) for n in off}}
    print(json.dumps(res))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("classify", "eval", "summary", "cost", "built"))
    ap.add_argument("--set", choices=("v3", "v4"), default="v3")
    ap.add_argument("--evals", default=None, help="summary: glob of the eval outputs (default data/<set>_physics_eval_*.json)")
    ap.add_argument("--out", default=None, help="summary: where the full JSON goes (default data/)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    SET = a.set
    if a.mode == "built":
        built()
    elif a.mode == "classify":
        C = classify()
        print(json.dumps(Counter(c["cls"] for c in C.values())))
    elif a.mode == "eval":
        evaluate(a.n, a.part, a.parts, a.only)
    elif a.mode == "cost":
        cost()
    else:
        summary(a.evals, a.out)
