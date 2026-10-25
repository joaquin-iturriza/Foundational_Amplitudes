"""catalog_v4's CPU validation (D130): the generator's catalog_v4 configuration (mg5_pipeline_final V4_*,
recipes/gen_catalog_v4.py) checked on a sample of every class family before any pool is regenerated.

The sample (seeded, SEED): every (layer, beam) cell of catalog_v3's tree classes (layers 2, 3, 4; beams e+e-, q q,
q g, g g) gives PER_CELL class representatives on their v4 entries, plus the three electroweak test processes named in
the plan (u u~ > d d~, u u > u u z, u d > u d h), a class with an internal top only (b b~ > w+ w-), two 2->5 probes, one
point of each other kind (heavy-quark layer at 50 and 300 GeV, an alpha_s point, a harder-cut probe, a BSM entry).
Per entry:
  diagrams   MadGraph on the v4 string (sm-no_b_mass, QED<=b), the v3 string under the same model (the five-flavour
             change alone), QED<=b-1 under it (b is attained: fewer diagrams), the v3 string under sm (must equal
             catalog_v3's recorded count: the tool's own check)
  backend    MadGraph output and compile through datagen.ensure_backend (the regeneration's path), wall seconds; the
             standalone's cards: model sm-no_b_mass in the proc card, MT = ymt = 172.5 (m_Q for a heavy-quark point),
             WT = the LO width at that mass, no b mass or Yukawa in the card
  pool       N events through datagen.gen_chunk (the pipeline's generation path) on the v4 window and mixture, written
             to $SCRATCH/zz_catalog_v4_val (never $DATA_DIR); seconds per kept event
  cuts       every event passes the v4 acceptance, recomputed here independently of fiducial_pass_mask (pT, |eta| < 5
             or |cos theta| < 0.99, Delta R, the pair mass only on e+e-, the b and tau legs included); the share
             outside catalog_v3's cuts (its rules and masses)
  kinematics beams and b finals massless, tops on 172.5 (m_Q), every other final on its mass; four-momentum conserved;
             sqrt(s_hat) inside [lo, 6 TeV], its share per decade against the log-uniform measure's
  sqrt(s)    KS distance of NKS draws of the pipeline's log sampler on the class's window against log-uniform
  forward    NKS raw draws turned by mg5_pipeline_final.rotate_forward: the chosen particle's u = log(1/(1-|cos|))
             against the drawn one (replayed from the seed), KS distance of u against uniform on [0, u_edge],
             conservation and mass shells after the turn; the pool's u histogram of its most forward final (U_EDGES)
             and the share beyond catalog_v3's edge (|cos| > 0.9)
  labels     share of finite, positive |M|^2
The one-loop layer: every v4 MadLoop entry's configuration (model, QED bound, width, cuts, measure, massless b), the
loop models' import, and MadGraph's generation of three loop strings (no MadLoop build here).
Output: $SCRATCH/zz_catalog_v4_val/validation.json and V4VAL lines in the log.
    python analysis/catalog_v4/validate_v4.py run [--n 2000] [--workers 16]      (CPU job on CC)
"""
import argparse, glob, json, os, random, re, subprocess, sys, time
from multiprocessing import Pool

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
SEED, PER_CELL, NKS = 0, 2, 20000
NAMED = ["v4_uux_ddx", "v4_uu_uuz", "v4_ud_dhu", "v4_bbx_wpwm"]
U_EDGES = np.linspace(0.0, 10.0, 21)
OUT = lambda: os.path.join(os.environ["SCRATCH"], "zz_catalog_v4_val")
NEUTRINOS = {12, 14, 16}


def load(p):
    return yaml.load(open(p), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def sample():
    """name -> recipe spec (name, base, physics, sqrts) of the entries validated"""
    import mg5_pipeline_final as mg
    R = json.load(open(os.path.join(ROOT, "recipes", "catalog_v3_roles.json")))["trees"]
    S1, SAMP = {}, {}
    for f in ("gen_stage1", "gen_stage3", "gen_stage4"):
        rec = load(os.path.join(ROOT, "recipes", f"catalog_v4_{f}.yaml"))
        for e in rec["processes"]:
            # an entry's sampling is its own block, else its recipe's (register_recipe_processes' default_sampling)
            S1[e["name"]] = e; SAMP[e["name"]] = e.get("sampling") or rec.get("sampling")
    default_sampling = load(os.path.join(ROOT, "recipes", "catalog_v4_gen_stage1.yaml")).get("sampling")
    beam = lambda p: ("ee" if abs(p[0]) == 11 else "gg" if p[:2] == [21, 21] else "qg" if 21 in p[:2] else "qq")
    cells = {}
    for n3 in sorted(R):
        n = "v4_" + n3[3:]
        # the cells as the validation drew them: a tree dropped since (the 34 b-Higgs trees with no diagram, D136)
        # keeps its catalog_v3 entry's layer and legs here, so the seeded pick is unchanged
        e = mg.PROCESSES.get(n) or mg.PROCESSES[n3]
        cells.setdefault((e["layer"], beam(list(e["pdg_ids"]))), []).append(n)
    rng = random.Random(SEED)
    pick = [n for c in sorted(cells) for n in rng.sample(cells[c], min(PER_CELL, len(cells[c])))]
    sys.path.insert(0, os.path.join(ROOT, "recipes"))
    from gen_catalog_v4 import vanished
    v3_stage = {e["name"] for f in ("gen_stage1", "gen_stage3", "gen_stage4")
                for e in load(os.path.join(ROOT, "recipes", f"catalog_v3_{f}.yaml"))["processes"]}
    gone = {n for n in vanished() if "v3_" + n[3:] in v3_stage and mg.PROCESSES["v3_" + n[3:]]["layer"] == 5}
    five = sorted({n for n in S1 if mg.PROCESSES.get(n, {}).get("layer") == 5} | gone)   # as drawn before the drop
    pick += rng.sample(five, 2)
    for tag in ("__mt50", "__mt300", "__as090", "__c30"):
        pick.append(sorted(n for n in S1 if n.endswith(tag))[0])
    pick.append("v4_zp_ee_mumu")
    pick += NAMED
    sys.path.insert(0, os.path.join(ROOT, "recipes"))
    from gen_catalog_v4 import window
    out = {}
    for n in dict.fromkeys(pick):
        # a named process that is no class representative (u u~ > d d~) is validated on its own v4 entry and window
        e = S1.get(n) or {"name": n, "sqrts": window(mg.PROCESSES[n])}
        out[n] = {"name": n, "base": e.get("base", n), "physics": e.get("physics") or {}, "sqrts": list(e["sqrts"]),
                  "sampling": SAMP.get(n) or default_sampling,
                  "cell": next((f"{c[0]}:{c[1]}" for c, ns in cells.items() if n in ns), "other")}
    return out


def register(spec):
    """the entry registered exactly as the regeneration does (scan and sampling policy): without the sampling block
    the generator takes the flat path, which refuses the log-uniform sqrt(s_hat)"""
    import mg5_pipeline_final as mg
    mg.register_recipe_processes([{k: spec[k] for k in ("name", "base", "physics", "sampling")}])
    assert spec["sampling"] is None or mg.PROCESSES[spec["name"]].get("sampling") == spec["sampling"], spec["name"]


LOOP_RE = re.compile(r"Contributing diagrams generated:\s*(\d+) Born,\s*(\d+)(?:\(\+(\d+)\))? loops,"
                     r"\s*(\d+) R2,\s*(\d+) UV")


def mg_run(gen, model, timeout=3600):
    """(exists, diagrams, the lines naming diagrams or the error) of one MadGraph generate; a loop string's diagrams
    are {born, loops, loops_discarded, R2, UV} summed over its subprocesses (MadLoop's 'Contributing diagrams' lines)"""
    import mg5_pipeline_final as mg
    r = subprocess.run([mg.MG5_BIN], input=f"import model {model}\n{gen}\ndisplay processes\nexit\n",
                       capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr
    d = re.search(r"Total: 1 processes with (\d+) diagrams", out)
    lines = [l.strip() for l in out.splitlines() if "diagram" in l.lower()][-4:]
    err = re.search(r"(NoDiagramException|InvalidCmd|InvalidModel|Error)[^\n]{0,160}", out)
    loop = LOOP_RE.findall(out)
    if d is None and loop:
        n = [sum(int(x[i] or 0) for x in loop) for i in range(5)]
        dd = {"born": n[0], "loops": n[1], "loops_discarded": n[2], "R2": n[3], "UV": n[4], "subprocesses": len(loop)}
        return err is None, dd, (err.group(0) if err else lines)
    return (d is not None and err is None), (int(d.group(1)) if d else None), (err.group(0) if err else lines)


def diagrams(name):
    import mg5_pipeline_final as mg
    e = mg.PROCESSES[name]
    gen = e["mg5_generate"][0]
    if e.get("v3_name") is None or "v3_key" not in e:              # a BSM entry: its own string only
        ok, d, _ = mg_run(gen, e["model"])
        return {"v4": d, "exists": ok}
    b = int(re.search(r"QED<=(\d+)", gen).group(1))
    v3gen = re.sub(r"\s*QED<=\d+", "", gen)
    res = {"string": gen, "qed_bound": b, "qed_v3": e["qed_v3"], "diagrams_v3_recorded": e["diagrams_v3"]}
    for k, (g, m) in {"v4": (gen, e["model"]), "v3_string_5fs": (v3gen, e["model"]), "v3_string_sm": (v3gen, "sm"),
                      "qed_b_minus_1": (re.sub(r"QED<=\d+", f"QED<={b - 1}", gen), e["model"])}.items():
        if k == "qed_b_minus_1" and b - 1 < e["qed_v3"]:
            continue
        ok, d, info = mg_run(g, m)
        res[k] = d if ok else None
        if not ok:
            res[k + "_error"] = info
    res["exists"] = res.get("v4") is not None
    res["bound_attained"] = (res.get("qed_b_minus_1") is None or (res.get("v4") or 0) > (res.get("qed_b_minus_1") or 0)) \
        if b > e["qed_v3"] else None
    res["tool_matches_v3"] = res.get("v3_string_sm") == e["diagrams_v3"]
    return res


def card(sa):
    """MT, ymt, WT, MB, ymb and the model of a standalone. A standalone_cpp output has no proc card (its Cards/ holds
    the param card only): the model is read from its src/Parameters_<model>.h, MadGraph's name with '-' as '_'"""
    pc = open(os.path.join(sa, "Cards", "param_card.dat")).read()
    val = lambda name: (float(m.group(1)) if (m := re.search(r"^\s*(?:DECAY\s+)?\d+\s+([-+0-9.eE]+)\s*#\s*" + name + r"\s*$",
                                                              pc, re.M)) else None)
    hs = sorted(glob.glob(os.path.join(sa, "src", "Parameters_*.h")))
    return {"MT": val("MT"), "ymt": val("ymt"), "WT": val("WT"), "MB": val("MB"), "ymb": val("ymb"),
            "model": os.path.basename(hs[0])[len("Parameters_"):-2] if len(hs) == 1 else None}


def v4_mask(P, pdg, mf, cuts):
    """the v4 acceptance, written out here (not fiducial_pass_mask)"""
    legs = [2 + i for i, (p, m) in enumerate(zip(pdg[2:], mf))
            if abs(int(p)) not in NEUTRINOS and (m == 0.0 or abs(int(p)) in (5, 15))]
    ok = np.ones(len(P), bool)
    pt = lambda q: np.hypot(q[:, 1], q[:, 2])
    for i in legs:
        q = P[:, i]
        ok &= (pt(q) > cuts["pt_min"]) & (np.abs(q[:, 3]) / np.linalg.norm(q[:, 1:], axis=1) < cuts["cos_max"])
    for a in range(len(legs)):
        for b in range(a + 1, len(legs)):
            qa, qb = P[:, legs[a]], P[:, legs[b]]
            eta = lambda q: np.arcsinh(q[:, 3] / pt(q))
            dphi = np.abs(np.angle(np.exp(1j * (np.arctan2(qa[:, 2], qa[:, 1]) - np.arctan2(qb[:, 2], qb[:, 1])))))
            ok &= np.hypot(eta(qa) - eta(qb), dphi) > cuts["dr_min"]
            if cuts["m_min"] > 0:
                s = qa + qb
                ok &= np.sqrt(np.clip(s[:, 0] ** 2 - (s[:, 1:] ** 2).sum(1), 0, None)) > cuts["m_min"]
    return ok


def ks_uniform(x, lo, hi):
    x = np.sort((np.asarray(x) - lo) / (hi - lo)); n = len(x)
    # the critical distance at 0.1% (about 70 such tests run, so a 5% level would fail one or more by chance)
    return float(max((np.arange(1, n + 1) / n - x).max(), (x - np.arange(n) / n).max())), float(1.95 / np.sqrt(n))


def u_of(P):
    c = np.abs(P[..., 3]) / np.clip(np.linalg.norm(P[..., 1:], axis=-1), 1e-300, None)
    return np.log(1.0 / np.clip(1.0 - c, 1e-300, None))


def one(args):
    name, spec, n, seed = args
    import datagen
    import mg5_pipeline_final as mg
    register(spec)
    row = {"name": name, "cell": spec["cell"], "sqrts": spec["sqrts"]}
    try:
        cfg = mg.PROCESSES[name]
        pdg, mf, cuts = list(cfg["pdg_ids"]), [float(m) for m in cfg["m_finals"]], cfg["fiducial_cuts"]
        lo, hi = spec["sqrts"]
        row["diagrams"] = diagrams(spec["base"])
        t = time.time(); datagen.ensure_backend(name); row["t_build"] = time.time() - t
        sa = f"{mg.WORK_DIR}/{mg.standalone_name(name)}_standalone"
        m_t = float((spec["physics"].get("masses") or {}).get(6, mg.V4_MT))
        row["card"] = c = card(sa)
        want_model = cfg.get("model") or "sm"
        row["card_ok"] = bool(c["MT"] == m_t and c["ymt"] == m_t and c["WT"] is not None
                              and abs(c["WT"] - mg.top_width_lo(m_t)) < 1e-5 * max(1, c["WT"])
                              and (want_model != mg.V4_MODEL or (c["MB"] in (None, 0.0) and c["ymb"] in (None, 0.0)))
                              and c["model"] == want_model.replace("-", "_"))
        task = {"process": name, "count": n, "sqrts_min": float(lo), "sqrts_max": float(hi), "role": "val",
                "seed": seed, "out_dir": os.path.join(OUT(), "pools", name), "idx": 0}
        t = time.time(); _, path = datagen.gen_chunk(task); row["t_gen"] = time.time() - t
        row["sec_per_event"] = row["t_gen"] / n
        a = np.load(path)
        k = len(pdg)
        P = a[:, :4 * k].reshape(-1, k, 4); me = a[:, -1]
        sq = np.sqrt(np.clip(((P[:, 0] + P[:, 1]) ** 2 * [1, -1, -1, -1]).sum(-1), 0, None))
        m2 = P[..., 0] ** 2 - (P[..., 1:] ** 2).sum(-1)
        mass = np.array([0.0, 0.0] + [m_t if abs(p) == 6 else m for p, m in zip(pdg[2:], mf)])
        row["mass_dev_rel"] = float((np.abs(m2 - mass ** 2) / sq[:, None] ** 2).max())
        row["b_masses_zero"] = bool(all(m == 0.0 for p, m in zip(pdg[2:], mf) if abs(p) == 5))
        row["beams_massless"] = bool(np.abs(m2[:, :2]).max() / (sq ** 2).min() < 1e-9)
        row["conservation_rel"] = float((np.abs(P[:, 0] + P[:, 1] - P[:, 2:].sum(1)).max(1) / sq).max())
        row["sqrts_min"], row["sqrts_max"] = float(sq.min()), float(sq.max())
        row["in_window"] = bool(sq.min() >= lo * (1 - 1e-9) and sq.max() <= hi * (1 + 1e-9))
        dec = np.unique(np.concatenate([[np.log10(lo)], np.arange(np.ceil(np.log10(lo)), np.log10(hi)), [np.log10(hi)]]))
        h = np.histogram(np.log10(sq), dec)[0]
        row["decades"] = {"edges": [float(10 ** x) for x in dec], "share": [float(x) for x in h / h.sum()],
                          "log_uniform": [float(x) for x in np.diff(dec) / (dec[-1] - dec[0])]}
        row["pass_v4"] = float(v4_mask(P, pdg, mf, cuts).mean())
        cfg3 = mg.PROCESSES.get(cfg.get("v3_name") or mg.PROCESSES[spec["base"]].get("v3_name") or "", {})
        if cfg3:
            mf3 = cfg3.get("m_finals") or (list(cfg3["m_final"]) if isinstance(cfg3.get("m_final"), (list, tuple))
                                           else [float(cfg3.get("m_final", 0.0))] * int(cfg3["nfinal"]))
            row["outside_v3_cuts"] = float(1 - mg.fiducial_pass_mask(P, mf3, pdg, mg.FIDUCIAL_CUTS).mean())
        uf = u_of(P[:, 2:]).max(1)
        row["u_hist"] = [int(x) for x in np.histogram(np.clip(uf, 0, 9.999), U_EDGES)[0]]
        row["forward_share_v3_edge"] = float((uf > np.log(10.0)).mean())
        row["labels_positive"] = float((np.isfinite(me) & (me > 0)).mean())
        # the log sampler and the forward turn, on raw draws of this class
        r = np.random.default_rng(seed + 1)
        x = mg._sqrts_sampler(lo, hi, "log", r)(NKS)
        row["ks_log_sqrts"] = ks_uniform(np.log(x), np.log(lo), np.log(hi))
        ev, sqr = mg._candidates(NKS, mg._sqrts_sampler(lo, hi, "log", r), mf, pdg, r, None, 0.5, 1e-6, 0.0, {})
        R0 = np.stack([e[0] for e in ev])
        cap = float(cuts["cos_max"]); u_edge = float(np.log(1.0 / (1.0 - cap)))
        Q = mg.rotate_forward(R0, cap, np.random.default_rng(seed + 2))
        rr = np.random.default_rng(seed + 2)
        j = rr.integers(2, k, NKS); u_drawn = rr.uniform(0.0, u_edge, NKS)
        u_meas = u_of(Q[np.arange(NKS), j])
        row["forward"] = {"u_edge": u_edge, "u_replay_maxdev": float(np.abs(u_meas - u_drawn).max()),
                          "ks_u": ks_uniform(u_meas, 0.0, u_edge),
                          "conservation_rel": float((np.abs(Q[:, 0] + Q[:, 1] - Q[:, 2:].sum(1)).max(1) / sqr).max()),
                          "mass_dev_rel": float((np.abs((Q[..., 0] ** 2 - (Q[..., 1:] ** 2).sum(-1))
                                                        - (R0[..., 0] ** 2 - (R0[..., 1:] ** 2).sum(-1))) / sqr[:, None] ** 2).max())}
        row["ok"] = bool(row["card_ok"] and row["pass_v4"] == 1.0 and row["in_window"] and row["beams_massless"]
                         and row["b_masses_zero"] and row["mass_dev_rel"] < 1e-6 and row["conservation_rel"] < 1e-9
                         and row["ks_log_sqrts"][0] < row["ks_log_sqrts"][1]
                         and row["forward"]["ks_u"][0] < row["forward"]["ks_u"][1]
                         and row["forward"]["u_replay_maxdev"] < 1e-6 and row["labels_positive"] > 0.99
                         and row["diagrams"].get("exists", False))
    except Exception as ex:                           # noqa: BLE001 (a row per entry, the failure recorded)
        import traceback
        row["error"] = f"{type(ex).__name__}: {ex}"; row["trace"] = traceback.format_exc()[-2000:]
    return row


def loops():
    import mg5_pipeline_final as mg
    from tools import nlo_virtual_pipeline as V
    out = {"entries": 0, "bad": []}
    for k, v in V.VIRT_PROCESSES.items():
        if not v.get("_v4"):
            continue
        out["entries"] += 1
        why = []
        if v["model"] not in ("loop_sm-no_b_mass", "loop_qcd_qed_sm-no_b_mass"):
            why.append("model " + v["model"])
        if not v.get("loopind") and "QED<=" not in v["mg5"]:
            why.append("no QED bound")
        if v.get("param_card_patches", {}).get("WT") != mg.V4_WT:
            why.append("WT")
        if v.get("sqrts_measure") != "log" or v.get("beam_masses") != {} or not v.get("fiducial_cuts"):
            why.append("cuts/measure/beams")
        if any(abs(p) == 5 and m != 0.0 for p, m in zip(v["pdg_ids"][2:], v["m_finals"])):
            why.append("b mass")
        if why:
            out["bad"].append([k, why])
    out["models"] = {m: mg_run("generate u u~ > d d~", m)[0] for m in ("sm-no_b_mass", "loop_sm-no_b_mass",
                                                                        "loop_qcd_qed_sm-no_b_mass")}
    # MadGraph's generation (no output) of three loop strings: two QCD virtuals whose QED bound adds EW born
    # diagrams, one loop-induced
    mixed = [k for k in sorted(V.VIRT_PROCESSES) if V.VIRT_PROCESSES[k].get("_v4") and not V.VIRT_PROCESSES[k].get("loopind")
             and k in mg.PROCESSES and int(re.search(r"QED<=(\d+)", V.VIRT_PROCESSES[k]["mg5"]).group(1)) > mg.PROCESSES[k]["qed_v3"]]
    induced = [k for k in sorted(V.VIRT_PROCESSES) if V.VIRT_PROCESSES[k].get("_v4") and V.VIRT_PROCESSES[k].get("loopind")]
    # and one entry of each loop model (loop_qcd_qed_sm-no_b_mass needs its restriction card, sites/setup.sh)
    by_model = [sorted(k for k in V.VIRT_PROCESSES if V.VIRT_PROCESSES[k].get("_v4") and V.VIRT_PROCESSES[k]["model"] == m)[:1]
                for m in ("loop_qcd_qed_sm-no_b_mass",)]
    for k in dict.fromkeys(mixed[:2] + induced[:1] + sum(by_model, [])):
        v = V.VIRT_PROCESSES[k]
        ok, d, info = mg_run(v["mg5"], v["model"], timeout=7200)
        out.setdefault("generate", {})[k] = {"string": v["mg5"], "model": v["model"], "ok": ok, "diagrams": d, "info": info}
    return out


def run(a):
    import datagen  # noqa: F401  (the site's environment)
    os.makedirs(OUT(), exist_ok=True)
    S = sample()
    print(f"V4VAL sample {len(S)} entries: {sorted(S)}", flush=True)
    for spec in S.values():
        register(spec)
    tasks = [(n, S[n], a.n, 7000 + i) for i, n in enumerate(sorted(S))]
    rows = []
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(one, tasks)):
            rows.append(row)
            d = row.get("diagrams", {})
            print(f"V4VAL [{i + 1}/{len(tasks)} {time.time() - t0:6.0f}s] {row['name']:30s} "
                  + (row["error"] if "error" in row else
                     f"ok={row['ok']} diagrams v3 {d.get('diagrams_v3_recorded')} -> v4 {d.get('v4')} "
                     f"(5fs minimal {d.get('v3_string_5fs')}, QED<=b-1 {d.get('qed_b_minus_1')}) pass={row['pass_v4']:.3f} "
                     f"outside_v3={row.get('outside_v3_cuts', float('nan')):.3f} fwd>0.9={row['forward_share_v3_edge']:.3f} "
                     f"build {row['t_build']:.0f}s gen {row['sec_per_event'] * 1e3:.2f} ms/ev"), flush=True)
    L = loops()
    print("V4VAL loops", json.dumps(L), flush=True)
    res = {"n": a.n, "rows": sorted(rows, key=lambda r: r["name"]), "loops": L,
           "n_ok": sum(bool(r.get("ok")) for r in rows), "n_rows": len(rows)}
    json.dump(res, open(os.path.join(OUT(), "validation.json"), "w"), indent=1)
    print(f"V4VAL_SUMMARY ok {res['n_ok']} of {res['n_rows']}; not ok "
          f"{[r['name'] for r in rows if not r.get('ok')]} -> {os.path.join(OUT(), 'validation.json')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sample"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "sample":
        S = sample()
        print(len(S), json.dumps({n: (s["cell"], s["sqrts"]) for n, s in S.items()}))
    else:
        run(a)


if __name__ == "__main__":
    main()
