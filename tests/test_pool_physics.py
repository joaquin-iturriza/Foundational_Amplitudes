"""The pool physics guard (tools/pool_physics.py; plan catalog_coverage): the 13 TeV pp test pools were labelled with
catalog_v3's physics (model sm: b at 4.7 GeV, the b Yukawa; the minimal QED order; MT 173 where the top is internal)
and catalog_v4's models, trained on a massless b with every tree diagram, were scored on them unnoticed. CPU only: the
MadGraph backends are stub directories (a param_card, a proc_card, the compiled defaults of src/Parameters_*.cc) under
a temporary WORK_DIR; no MadGraph, no torch, nothing on a site.
    python3 tests/test_pool_physics.py"""
import io, json, os, shutil, sys, tempfile, contextlib

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
TMP = tempfile.mkdtemp(prefix="pool_physics_test_")
os.environ["AMP_FROZEN_DIR"] = os.path.join(TMP, "datasets")
os.environ["AMP_TRAIN_CACHE_DIR"] = os.path.join(TMP, "cache")
os.makedirs(os.environ["AMP_FROZEN_DIR"]); os.makedirs(os.environ["AMP_TRAIN_CACHE_DIR"])

import mg5_pipeline_final as mg  # noqa: E402
import pool_physics as pp  # noqa: E402

mg.WORK_DIR = os.path.join(TMP, "mg5amcnlo")
mg.OUTPUT_DIR = os.environ["AMP_FROZEN_DIR"]

V3_CARD = dict(MB=4.7, ymb=4.7, MT=173.0, ymt=173.0, WT=1.4915, WZ=2.441404, WW=2.0476, WH=6.382339e-3)
V4_CARD = dict(MT=172.5, ymt=172.5, WT=1.4915, WZ=2.441404, WW=2.0476, WH=6.382339e-3)     # sm-no_b_mass: no MB, ymb


def backend(name, model, gens, card, indented=None, defaults=None, suffix="_standalone"):
    """a stub backend: Cards/param_card.dat (DECAY lines in column 1; `indented` ones with a leading blank, which
    MadGraph skips), Cards/proc_card_mg5.dat, src/Parameters_<model>.cc with the compiled defaults"""
    d = os.path.join(mg.WORK_DIR, name + suffix)
    os.makedirs(os.path.join(d, "Cards"), exist_ok=True)
    os.makedirs(os.path.join(d, "src"), exist_ok=True)
    L = ["BLOCK MASS #", *[f"    {p} {card[k]:.6e} # {k}" for p, k in ((5, "MB"), (6, "MT")) if k in card],
         "BLOCK YUKAWA #", *[f"    {p} {card[k]:.6e} # {k}" for p, k in ((5, "ymb"), (6, "ymt")) if k in card]]
    for p, k in ((6, "WT"), (23, "WZ"), (24, "WW"), (25, "WH")):
        if k in card:
            L.append(f"DECAY {p} {card[k]:.6e} # {k}")
    for p, (k, v) in (indented or {}).items():
        L.append(f" DECAY {p} {v:.6e} # {k}")
    open(os.path.join(d, "Cards", "param_card.dat"), "w").write("\n".join(L) + "\n")
    open(os.path.join(d, "Cards", "proc_card_mg5.dat"), "w").write(
        f"import model {model}\n" + "\n".join(gens) + f"\noutput standalone_cpp {d}\n")
    dl = defaults or {("decay", 6): 1.4915, ("decay", 23): 2.441404, ("decay", 24): 2.0476, ("decay", 25): 6.382339e-3,
                      ("mass", 6): 172.0, ("yukawa", 6): 172.0, **({("mass", 5): 4.7, ("yukawa", 5): 4.7}
                                                                    if "no_b_mass" not in model else {})}
    open(os.path.join(d, "src", f"Parameters_{model.replace('-', '_')}.cc"), "w").write(
        "\n".join(f'  x = slha.get_block_entry("{b}", {p}, {v:e});' for (b, p), v in dl.items()) + "\n")
    return d


def setup():
    P = mg.PROCESSES
    # catalog_v4's trees (their real entries) and catalog_v3's pp versions of them
    for n in ("v4_bbx_bbxgg", "v4_ddx_aag", "v4_gg_ttx", "v4_bbx_wpwm"):
        backend(n, "sm-no_b_mass", P[n]["mg5_generate"], V4_CARD)
    for n in ("pp13_bbx_bbxgg", "pp13_ddx_aag", "pp13_gg_ttx", "v3_bbx_bbxgg", "pp13_bbx_wpwm"):
        card = dict(V3_CARD, **({"MT": 172.5, "ymt": 172.5} if n.endswith("ttx") else {}))
        backend(n, "sm", P[n]["mg5_generate"], card)
    # the heavy-quark layer's top at 300 GeV (a physics scan: its own backend, declared in its recipe entry)
    mg.register_recipe_processes([{"name": "v4_gg_ttx__mt300", "base": "v4_gg_ttx",
                                   "physics": {"masses": {6: 300.0}, "widths": {6: 8.729709}}}])
    backend("v4_gg_ttx__mt300", "sm-no_b_mass", P["v4_gg_ttx"]["mg5_generate"],
            dict(V4_CARD, MT=300.0, ymt=300.0, WT=8.729709))
    rec = os.path.join(TMP, "train.yaml")
    yaml.safe_dump({"processes": [
        {"name": n, "sqrts": [100, 6000], "n_train": 10, "n_val": 10, "n_test": 10} for n in
        ("v4_bbx_bbxgg", "v4_ddx_aag", "v4_gg_ttx", "v4_bbx_wpwm")] + [
        {"name": "v4_gg_ttx__mt300", "base": "v4_gg_ttx", "physics": {"masses": {6: 300.0}, "widths": {6: 8.729709}},
         "sqrts": [700, 6000], "n_train": 10, "n_val": 10, "n_test": 10}]}, open(rec, "w"))
    return rec


def test_card_reader():
    """a DECAY line counts only in column 1; an indented one keeps the compiled default; a parameter the model's
    restriction removed is 0"""
    d = backend("zz_card", "sm-no_b_mass", ["generate g g > g g"], {k: v for k, v in V4_CARD.items() if k != "WZ"},
                indented={23: ("WZ", 9.9)})
    vals, notes = pp.read_card(os.path.join(d, "Cards", "param_card.dat"), d)
    assert vals["WZ"] == 2.441404 and notes["indented_decay_ignored"] == {"WZ": 9.9}, (vals, notes)
    assert vals["MB"] == 0.0 and vals["ymb"] == 0.0 and set(notes["restricted"]) == {"MB", "ymb"}, notes
    assert vals["MT"] == 172.5 and vals["WT"] == 1.4915
    assert pp.proc_card(d) == ("sm-no_b_mass", ["generate g g > g g"])
    print("ok card reader")


def test_blocks():
    v4 = pp.physics_block("v4_bbx_bbxgg")
    assert v4["model"] == "sm-no_b_mass" and v4["MB"] == 0.0 and v4["MT"] == 172.5, v4
    assert v4["order_clause"] == "QED<=2" and v4["b_row"] == {"code": 5000000, "mass": 0.0}, v4
    assert v4["alpha_s"] == {"rule": "1loop_nf5_mu_sqrt_shat", "alphas_mz": 0.118}
    p3 = pp.physics_block("pp13_bbx_bbxgg")
    assert p3["model"] == "sm" and p3["MB"] == 4.7 and p3["MT"] == 173.0 and p3["order_clause"] == "", p3
    assert p3["b_row"]["code"] == 5 and abs(p3["b_row"]["mass"] - 4.7) < 1e-6, p3["b_row"]
    assert p3["alpha_s"]["rule"] == v4["alpha_s"]["rule"], "the pp pools' alpha_s(sqrt(s_hat)) is the window pools' rule"
    # no backend: the entry's fields, the card's listed missing, nothing guessed
    nb = pp.physics_block("v4_gg_gg") if "v4_gg_gg" in mg.PROCESSES else pp.physics_block("v4_uux_gg")
    assert nb["card"] is None and "MT" in nb["missing"] and nb["model"] == "sm-no_b_mass", nb
    print("ok blocks")


def test_write_recipe():
    """generation writes the block into the sidecar; the block does not move the recipe_id (no pool regenerates)"""
    r = mg.variable_energy_recipe("v4_ddx_aag", 100, 6000, 10, role="val", seed=1)
    out = os.path.join(TMP, "datasets", "zz_v4_ddx_aag_val.npy")
    np.save(out, np.zeros((1, 1)))
    with contextlib.redirect_stdout(io.StringIO()):
        rid = mg.write_recipe(out, r)
    side = json.load(open(out + ".recipe.json"))
    assert rid == mg.recipe_id(r) == side["recipe_id"] == mg.recipe_id(side), "the physics block moved the recipe_id"
    assert side["physics"]["model"] == "sm-no_b_mass" and side["physics"]["MT"] == 172.5, side["physics"]
    print("ok write_recipe")


def check(ref, name, physics=None):
    mg.register_recipe_processes([{"name": name, "base": name if name in mg.PROCESSES else physics.pop("_base"),
                                   "physics": physics}])
    return pp.check_probe(name, physics, None, ref)


def test_scenario():
    """the case that happened: a catalog_v4 parent read on catalog_v3's pp pools"""
    ref = pp.TrainingReference(TRAIN)
    # a b class: the massive b (card, model, encoding row) and the electroweak diagrams catalog_v4 added
    c = pp.check_probe("pp13_bbx_bbxgg", None, None, ref)
    f = {m["field"] for m in c["mismatches"]}
    assert c["status"] == "mismatch" and c["reference"]["kind"] == "same process", c
    assert f == {"model", "MB", "ymb", "order_vector", "b_row"}, f       # no W: no top line, MT 173 cannot matter
    assert not pp.apply_allowance(c, None) and c["status"] == "mismatch"
    assert pp.apply_allowance(c, "the pp blend's audit, read knowing the b differs") and c["status"] == "allowed"
    assert pp.record(c)["reason"].startswith("the pp blend")
    # an internal top (b b~ > W+ W-, the t-channel top): catalog_v3's MT 173 against catalog_v4's 172.5 counts too
    c = pp.check_probe("pp13_bbx_wpwm", None, None, ref)
    assert {"MT", "ymt", "MB", "model", "b_row"} <= {m["field"] for m in c["mismatches"]}, c
    # a class with no b, no top and no added diagram (d d~ > a a g: QED<=2 is its minimal order): the same |M|^2
    # function (pp_physics_check's 544), so it passes although its card has MT 173 and its string no clause
    c = pp.check_probe("pp13_ddx_aag", None, None, ref)
    assert c["status"] == "ok", c
    # an external top without a W: no b line, so the model's b restriction cannot matter; MT 172.5 in both
    c = pp.check_probe("pp13_gg_ttx", None, None, ref)
    assert c["status"] == "ok", c
    # a declared scan is exempt: the top at 300 GeV is the heavy-quark layer's, set in its recipe entry
    c = pp.check_probe("v4_gg_ttx__mt300", {"masses": {6: 300.0}, "widths": {6: 8.729709}}, None, ref)
    assert c["status"] == "ok" and {"MT", "ymt", "WT"} <= set(c["exempt"]), c
    # the same probe without its declaration is refused (the same process's plain entry has MT 172.5)
    c = pp.check_probe("v4_gg_ttx__mt300", None, None, ref)
    assert c["status"] == "mismatch" and "MT" in {m["field"] for m in c["mismatches"]}, c
    # a process the catalog has only scans of (the alpha_s points 0.09 and 0.15): their alpha_s is no reference for
    # a probe at 0.118, the rest of their physics is
    mg.register_recipe_processes([{"name": "v4_ddx_aag__as150", "base": "v4_ddx_aag", "physics": {"alpha_s": 0.15}}])
    r2 = os.path.join(TMP, "train_scans.yaml")
    yaml.safe_dump({"processes": [{"name": "v4_ddx_aag__as150", "base": "v4_ddx_aag", "physics": {"alpha_s": 0.15},
                                   "sqrts": [100, 6000], "n_train": 10, "n_val": 10, "n_test": 10}]}, open(r2, "w"))
    c = pp.check_probe("pp13_ddx_aag", None, None, pp.TrainingReference(r2))
    assert c["status"] == "ok" and "alphas_mz" in c["not_compared"], c
    # a process the catalog has no entry of: the catalog-wide values (a catalog_v3 pp b class held out)
    mg.PROCESSES["pp13_zz_heldout"] = dict(mg.PROCESSES["pp13_bbx_bbxgg"])
    mg.PROCESSES["pp13_zz_heldout"]["pdg_ids"] = [5, -5, 5, -5, 21, 22]
    mg.PROCESSES["pp13_zz_heldout"]["mg5_generate"] = ["generate b b~ > b b~ g a"]
    backend("pp13_zz_heldout", "sm", ["generate b b~ > b b~ g a"], V3_CARD)
    c = pp.check_probe("pp13_zz_heldout", None, None, ref)
    assert c["reference"]["kind"] == "catalog-wide" and c["status"] == "mismatch", c
    assert {"model", "MB", "b_row"} <= {m["field"] for m in c["mismatches"]}, c
    # the b row against the probe's own card: a pool labelled with a massless b read as the 4.7 GeV row
    mg.PROCESSES["pp13_zz_rowlie"] = dict(mg.PROCESSES["pp13_bbx_bbxgg"])
    backend("pp13_zz_rowlie", "sm-no_b_mass", ["generate b b~ > b b~ g g QED<=2"], V4_CARD)
    c = pp.check_probe("pp13_zz_rowlie", None, None, ref)
    assert "b_row vs the probe's own MB" in {m["field"] for m in c["mismatches"]}, c
    # a parent without a training recipe: nothing to compare with, refused
    assert pp.check_probe("pp13_ddx_aag", None, None, None)["status"] == "mismatch"
    print("ok scenario")


def test_sidecar_block_used():
    """the probe's block comes from its pool's sidecar when there is one (what the backend did at generation), the b
    row from the particle table now"""
    ref = pp.TrainingReference(TRAIN)
    pool = os.path.join(TMP, "datasets", "zz_pp13_ddx_aag_val.npy")
    blk = pp.physics_block("pp13_ddx_aag")
    blk["WZ"] = 2.4952                  # a pool generated at another Z width than its backend's card has now
    json.dump({"process": "pp13_ddx_aag", "physics": blk}, open(pool + ".recipe.json", "w"))
    c = pp.check_probe("pp13_ddx_aag", None, pool, ref)
    assert c["block_source"] == "sidecar" and [m["field"] for m in c["mismatches"]] == ["WZ"], c
    print("ok sidecar block")


def test_lhe_parent_check():
    """generation of a pp pool refuses a standalone whose physics is not its parent's, before MadEvent"""
    with contextlib.redirect_stdout(io.StringIO()):
        assert mg.lhe_parent_physics_check("pp13_bbx_bbxgg")["status"] == "ok"        # catalog_v3's pp of catalog_v3
    saved = dict(mg.PROCESSES["pp13_bbx_bbxgg"])
    try:
        mg.PROCESSES["pp13_bbx_bbxgg"]["parent"] = "v4_bbx_bbxgg"                    # a v4 pp catalog on v3's card
        try:
            mg.lhe_parent_physics_check("pp13_bbx_bbxgg")
            raise AssertionError("not refused")
        except RuntimeError as e:
            assert "v4_bbx_bbxgg" in str(e) and "MB" in str(e), e
        mg.PROCESSES["pp13_bbx_bbxgg"]["physics_mismatch_reason"] = "test"
        with contextlib.redirect_stdout(io.StringIO()):
            assert mg.lhe_parent_physics_check("pp13_bbx_bbxgg")["status"] == "allowed"
    finally:
        mg.PROCESSES["pp13_bbx_bbxgg"] = saved
    print("ok lhe parent check")


def test_backfill():
    """only the new key is written; the recipe_id is unchanged; a sidecar in the catalog_v3 deletion manifest, one of
    an unknown process and one already filled are left as they are and counted"""
    import backfill_pool_physics as bf
    root = os.path.join(TMP, "bf")
    os.makedirs(root)
    r = mg.variable_energy_recipe("v4_ddx_aag", 100, 6000, 10, role="val", seed=1)
    with contextlib.redirect_stdout(io.StringIO()):
        for nm in ("a", "b"):
            p = os.path.join(root, f"{nm}_val_amplitudes.npy")
            np.save(p, np.zeros((1, 1)))
            mg.write_recipe(p, r)
    for nm in ("a", "b"):                       # as made before the block existed
        side = os.path.join(root, f"{nm}_val_amplitudes.npy.recipe.json")
        d = json.load(open(side)); d.pop("physics")
        json.dump(d, open(side, "w"), indent=2, sort_keys=True)
    json.dump({"process": "zz_unknown", "recipe_id": "x"}, open(os.path.join(root, "c.npy.recipe.json"), "w"))
    before = {f: open(os.path.join(root, f)).read() for f in os.listdir(root) if f.endswith(".json")}
    man = os.path.join(TMP, "manifest.json")
    json.dump({"groups": {"pools": {root: [["b_val_amplitudes.npy.recipe.json", 1, 1, 0]]}}}, open(man, "w"))
    bf.MANIFEST = man
    bf.register_all_recipes = lambda mg_: ({}, set())
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        sys.argv = ["backfill", "--roots", root]
        bf.main()
    rep = json.loads(out.getvalue().strip().splitlines()[-1])
    assert rep["counts"] == {"filled": 1, "skipped: in the catalog_v3 deletion manifest": 1, "unknown process": 1}, rep
    a0, a1 = json.loads(before["a_val_amplitudes.npy.recipe.json"]), json.load(open(os.path.join(root, "a_val_amplitudes.npy.recipe.json")))
    assert set(a1) - set(a0) == {"physics"} and all(a1[k] == a0[k] for k in a0), "the backfill changed another key"
    assert mg.recipe_id(a1) == a0["recipe_id"] and a1["physics"]["model"] == "sm-no_b_mass"
    for f in ("b_val_amplitudes.npy.recipe.json", "c.npy.recipe.json"):
        assert open(os.path.join(root, f)).read() == before[f], f + " was touched"
    # a second pass finds it filled
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        bf.main()
    assert json.loads(out.getvalue().strip().splitlines()[-1])["counts"]["already has a block"] == 1
    print("ok backfill")


def test_reads():
    """what reads see: the ZERO_SHOT record's status, the .npz's, and collect_run's verdict"""
    assert pp.zs_status({"probe": "x", "mse_logm2_real": 1.0}) == "unchecked"
    assert pp.zs_status({"probe": "x", "physics": {"status": "ok"}}) == "ok"
    assert pp.zs_status({"probe": "x", "physics": {"status": "allowed", "reason": "r"}}) == "allowed"
    assert pp.zs_status({"probe": "x", "error": "physics mismatch: MB: probe 4.7 vs [0.0]"}) == "mismatch"
    f = os.path.join(TMP, "p.npz")
    np.savez_compressed(f, t=np.zeros(2), physics_check=np.array(json.dumps({"status": "allowed", "reason": "why"})),
                        physics_reason=np.array("why"))
    assert pp.require_npz(np.load(f), f) == ("allowed", "why")
    np.savez_compressed(f, t=np.zeros(2))
    assert pp.npz_status(np.load(f)) == ("unchecked", None)
    np.savez_compressed(f, t=np.zeros(2), physics_check=np.array(json.dumps({"status": "mismatch"})))
    try:
        pp.require_npz(np.load(f), f)
        raise AssertionError("a mismatch without a reason was read")
    except RuntimeError:
        pass
    import collect_run as cr
    log = "\n".join([
        'PP ZERO_SHOT ' + json.dumps({"probe": "pp13_a", "parent": "/r", "mse_logm2_real": 1.0}),
        'PP ZERO_SHOT ' + json.dumps({"probe": "pp13_b", "parent": "/r", "mse_logm2_real": 1.0,
                                      "physics": {"status": "allowed", "reason": "knowingly"}}),
        'PP ZERO_SHOT ' + json.dumps({"probe": "pp13_c", "parent": "/r", "error": "physics mismatch: MB"}),
        'PP ZERO_SHOT ' + json.dumps({"probe": "pp13_d", "parent": "/r", "mse_logm2_real": 1.0,
                                      "physics": {"status": "ok"}})])
    z = cr.parse_log(log)["zs"]["PP"]
    assert z["physics_unchecked"] == {"pp13_a"} and z["physics_allowed"] == {"pp13_b": "knowingly"}, z
    assert "pp13_c" in z["err"] and z["ok"] == {"pp13_a", "pp13_b", "pp13_d"}, z
    print("ok reads")


def test_reference_from_sidecars():
    """the check without the standalones [D208]: the training reference reads each entry's block from its stored pool's
    sidecar, the probe its own; with the backends moved away the scenario's verdicts are unchanged, a sidecar edited to
    a wrong top mass is refused, a pool with neither a sidecar block nor a backend card is refused, and
    prune_standalones deletes only the standalones no pool without a block needs, manifest first"""
    import datagen
    train = yaml.safe_load(open(TRAIN))["processes"]
    with contextlib.redirect_stdout(io.StringIO()):
        for p in train:
            for role in ("train", "val"):
                r = mg.variable_energy_recipe(p["name"], *p["sqrts"], p[f"n_{role}"], role=role, seed=42)
                out = mg.recipe_output_path(r, datagen.dest_for_role(role))
                np.save(out, np.zeros((1, 1)))
                mg.write_recipe(out, r)
        probes = {}
        for n in ("pp13_bbx_bbxgg", "pp13_ddx_aag", "pp13_gg_ttx", "v4_gg_ttx"):
            r = mg.variable_energy_recipe(n, 100, 6000, 10, role="val", seed=7)
            out = mg.recipe_output_path(r, datagen.dest_for_role("val"))
            np.save(out, np.zeros((1, 1)))
            mg.write_recipe(out, r)
            probes[n] = out
    nosc = os.path.join(TMP, "datasets", "zz_noblock_val.npy")       # an old pool: sidecar without a block
    np.save(nosc, np.zeros((1, 1)))
    json.dump({"process": "pp13_ddx_aag", "recipe_id": "x"}, open(nosc + ".recipe.json", "w"))
    n, bad = pp.sidecar_scan([os.environ["AMP_FROZEN_DIR"], os.environ["AMP_TRAIN_CACHE_DIR"]])
    assert [b["sidecar"] for b in bad] == [nosc + ".recipe.json"], bad
    keep = [b["backend"] for b in bad]
    assert keep == [os.path.join(mg.WORK_DIR, "pp13_ddx_aag_standalone")], keep
    with_be = {c["probe"]: c["status"] for c in
               (pp.check_probe(n, None, f, pp.TrainingReference(TRAIN)) for n, f in probes.items())}
    # sidecar-only (the check as it runs without the standalones, on a site that still has them): same verdicts
    so = pp.TrainingReference(TRAIN, backend=False)
    assert so.sources == {"sidecar": len(train)}, so.sources
    assert {n: pp.check_probe(n, None, f, so, backend=False)["status"] for n, f in probes.items()} == with_be
    assert pp.check_probe("pp13_gg_ttx", None, nosc, so, backend=False)["mismatches"][0]["field"] == "physics block"
    # the dry run lists every standalone but the kept one and deletes nothing
    all_sa = pp.standalone_dirs(mg.WORK_DIR)
    r = pp.prune_standalones(mg.WORK_DIR, keep, apply=False)
    assert r["deletable"] == len(all_sa) - 1 and pp.standalone_dirs(mg.WORK_DIR) == all_sa, r
    open(os.path.join(mg.WORK_DIR, "VERSION"), "w").write("3.7.0")      # the install's own files stay
    os.makedirs(os.path.join(mg.WORK_DIR, "madgraph"), exist_ok=True)
    saved = TMP + "_backends"
    shutil.copytree(mg.WORK_DIR, saved)
    try:
        r = pp.prune_standalones(mg.WORK_DIR, keep, apply=True)
        assert pp.standalone_dirs(mg.WORK_DIR) == keep and r["freed_gb"] >= 0, r
        assert os.path.exists(os.path.join(mg.WORK_DIR, "VERSION")) and os.path.isdir(os.path.join(mg.WORK_DIR, "madgraph"))
        assert json.load(open(r["manifest"]))["entries"], r
        pp._CARD_CACHE.clear()
        ref = pp.TrainingReference(TRAIN)
        assert ref.sources == {"sidecar": len(train)} and not ref.unreadable, (ref.sources, ref.unreadable)
        without = {n: pp.check_probe(n, None, f, ref) for n, f in probes.items()}
        assert {n: c["status"] for n, c in without.items()} == with_be, (with_be, without)
        assert all(c["block_source"] == "sidecar" for c in without.values())
        assert with_be["pp13_bbx_bbxgg"] == "mismatch" and with_be["pp13_ddx_aag"] == "ok" and with_be["v4_gg_ttx"] == "ok"
        # a sidecar edited to a wrong top mass is refused
        side = json.load(open(probes["v4_gg_ttx"] + ".recipe.json"))
        side["physics"]["MT"] = side["physics"]["ymt"] = 173.0
        json.dump(side, open(probes["v4_gg_ttx"] + ".recipe.json", "w"))
        c = pp.check_probe("v4_gg_ttx", None, probes["v4_gg_ttx"], ref)
        assert c["status"] == "mismatch" and {"MT", "ymt"} <= {m["field"] for m in c["mismatches"]}, c
        # a probe whose sidecar has no block and whose standalone is gone is refused, not passed unchecked
        c = pp.check_probe("pp13_gg_ttx", None, nosc, ref)
        assert c["status"] == "mismatch" and c["mismatches"][0]["field"] == "physics block", c
        # a training entry with neither: listed, never a reference
        os.remove(mg.recipe_output_path(mg.variable_energy_recipe("v4_ddx_aag", 100, 6000, 10, role="train", seed=42),
                                        datagen.dest_for_role("train")))
        os.remove(mg.recipe_output_path(mg.variable_energy_recipe("v4_ddx_aag", 100, 6000, 10, role="val", seed=42),
                                        datagen.dest_for_role("val")))
        ref = pp.TrainingReference(TRAIN)
        assert [u[0] for u in ref.unreadable] == ["v4_ddx_aag"], ref.unreadable
    finally:
        shutil.rmtree(mg.WORK_DIR)
        shutil.move(saved, mg.WORK_DIR)
        pp._CARD_CACHE.clear()
    print("ok reference from sidecars")


if __name__ == "__main__":
    try:
        setup_rec = setup()
        TRAIN = setup_rec
        for t in (test_card_reader, test_blocks, test_write_recipe, test_scenario, test_sidecar_block_used,
                  test_lhe_parent_check, test_backfill, test_reads, test_reference_from_sidecars):
            t()
        print("all pool physics tests passed")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
