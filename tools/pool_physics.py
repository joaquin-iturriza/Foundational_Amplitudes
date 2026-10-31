"""The physics a pool's |M|^2 labels were computed with, and the check that a probe pool carries the physics of the
catalog a model was trained on (plan catalog_coverage: the 13 TeV pp test pools were labelled with catalog_v3's physics,
a massive b, the minimal QED order of the generate string, MT 173 where the top is internal, and catalog_v4's models,
trained on a massless b with every tree diagram and MT 172.5 everywhere, were scored on them without anything noticing).

The block (`physics`, in every pool sidecar <pool>.npy.recipe.json; written by mg5_pipeline_final.write_recipe at
generation, by tools/backfill_pool_physics.py for the pools made before it):
  model          the model the matrix element was generated in, with its restriction (the last `import model` of the
                 backend's Cards/proc_card_mg5.dat; the entry's when that file is absent)
  MB ymb MT ymt WT WZ WW WH
                 as MadGraph read them from the backend's own Cards/param_card.dat: a DECAY line counts only when it
                 starts in column 1 (MadGraph's readers skip an indented one and keep the default compiled into the
                 backend at output time, src/Parameters_<model>.cc or Source/MODEL/param_read.inc, which is then the
                 value); a parameter the model's restriction removed (sm-no_b_mass's MB, ymb) is 0
  generate, order_clause   the generate string and its coupling-order clause ("" for MadGraph's minimal QED order)
  order_vector   mg5_pipeline_final.order_vector of the entry
  alpha_s        {rule, alphas_mz}: how each event's alpha_s was set
  b_mass_generated, b_row  the b mass the generator gave the external b legs, and the particle-table row the model's
                 encoding reads for that b (particle_ids.species_pdgs: 5, the 4.7 GeV row, or the massless b 5000000)
A field that could not be read is absent and listed under `missing`; nothing is guessed.

The check (check_probe; tools/zero_shot_parent.py runs it on every probe before scoring it): the probe pool's block (its
sidecar's, else computed from its entry and backend) against the training catalog's entries for the same process (same
external particles, tree or one-loop alike), else against the catalog-wide values (the training entries of the same
model family and loop kind, without a physics scan of their own), plus the b's particle-table row against both the
reference and the probe's own card. A field the probe's recipe entry sets on purpose (a `physics` scan: masses, widths,
alpha_s) is exempt. Any other difference refuses the probe, unless --allow-physics-mismatch "<reason>" is given; the
reason is saved into the probe's .npz and its ZERO_SHOT record.

Reads scripts and tools/collect_run.py use zs_status / npz_status: "ok", "allowed" (a mismatch read on purpose, with
its reason), "mismatch" (refused) or "unchecked" (read before this check existed).

This module imports only the standard library at load; the parts that need the pipeline import it when called.
"""
import glob, hashlib, json, math, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCHEMA = 1
CARD_FIELDS = ("MB", "ymb", "MT", "ymt", "WT", "WZ", "WW", "WH")
CARD_KEYS = {("MASS", 5): "MB", ("MASS", 6): "MT", ("YUKAWA", 5): "ymb", ("YUKAWA", 6): "ymt",
             ("DECAY", 6): "WT", ("DECAY", 23): "WZ", ("DECAY", 24): "WW", ("DECAY", 25): "WH"}
# compared against the same process's training entry. The coupling orders are compared through the order vector, whose
# electroweak maximum is what the clause admits (MadGraph's minimal QED order and QED<=b give the same diagrams when b is
# that minimum: catalog_v4's u u~ > g g g QED<=0 is catalog_v3's u u~ > g g g); the clause string itself is recorded.
# The catalog-wide comparison replaces the process-specific order vector by the clause's kind (coupling_kind).
SAME_FIELDS = ("model_family", "model") + CARD_FIELDS + ("order_vector", "alpha_s_rule", "alphas_mz", "b_row")
WIDE_FIELDS = ("model_family", "model") + CARD_FIELDS + ("coupling_kind", "alpha_s_rule", "alphas_mz", "b_row")
B_FIELDS = ("model", "MB", "ymb")      # reach a tree's |M|^2 only through a b line (relevant(); the model's restriction
T_FIELDS = ("MT", "ymt", "WT")         # is the b's: sm against sm-no_b_mass), the top's only through a top line
EW_WIDTHS = ("WZ", "WW", "WH")
ALLOW_ENV = "ZS_ALLOW_PHYSICS_MISMATCH"


# ---------------------------------------------------------------------------------------------------------- the card
def _num(s):
    return float(str(s).replace("D", "e").replace("d", "e"))


def compiled_defaults(backend):
    """{field: value} compiled into a backend at output time, and whether the file was found: C++
    (src/Parameters_*.cc, slha.get_block_entry("<block>", <pdg>, <default>)) or MadLoop (Source/MODEL/param_read.inc,
    LHA_GET_REAL(...,'mdl_<name>',MDL_<name>,<default>)); the readers of analysis/catalog_v4/card_width_audit.py"""
    out = {}
    cc = sorted(glob.glob(os.path.join(backend, "src", "Parameters_*.cc")))
    if cc:
        for blk, pdg, v in re.findall(r'get_block_entry\("(\w+)",\s*(-?\d+),\s*([^)\s]+)\s*\)', open(cc[0]).read()):
            f = CARD_KEYS.get((blk.upper(), abs(int(pdg))))
            if f:
                try:
                    out[f] = _num(v)
                except ValueError:
                    pass
        return out, True
    pr = os.path.join(backend, "Source", "MODEL", "param_read.inc")
    if os.path.exists(pr):
        txt = open(pr).read().replace("\n     $ ", "")
        names = {f.lower(): f for f in CARD_FIELDS}
        for n, v in re.findall(r"LHA_GET_REAL\(NPARA,PARAM,VALUE,'mdl_(\w+)',MDL_\w+,([0-9.EeDd+-]+)\)", txt):
            if n.lower() in names:
                out[names[n.lower()]] = _num(v)
        return out, True
    return out, False


def read_card(card, backend=None):
    """(values, notes) of CARD_FIELDS as MadGraph reads `card`: BLOCK MASS / YUKAWA entries, DECAY lines in column 1
    only. A field with only an indented DECAY line, or absent from the card, takes the backend's compiled default; one
    absent from both is a parameter the model's restriction removed (0). notes: {indented_decay_ignored: {field: the
    ignored value}, compiled_default: [...], restricted: [...]}"""
    vals, ignored, block = {}, {}, None
    for line in open(card):
        code = line.split("#", 1)[0]
        s = code.split()
        if not s:
            continue
        head = s[0].upper()
        if head == "BLOCK":
            block = s[1].upper() if len(s) > 1 else None
            continue
        if head == "DECAY":
            block = None
            if len(s) < 3:
                continue
            f = CARD_KEYS.get(("DECAY", abs(int(s[1]))))
            if f is None:
                continue
            if code[:5].upper() == "DECAY":
                vals[f] = _num(s[2])
            else:
                ignored[f] = _num(s[2])
            continue
        if block in ("MASS", "YUKAWA") and len(s) >= 2:
            try:
                f = CARD_KEYS.get((block, abs(int(s[0]))))
            except ValueError:
                continue
            if f:
                vals[f] = _num(s[1])
    notes = {}
    rest = [f for f in CARD_FIELDS if f not in vals]
    if rest:
        dflt, found = compiled_defaults(backend) if backend else ({}, False)
        for f in rest:
            if f in dflt:
                vals[f] = dflt[f]
                notes.setdefault("compiled_default", []).append(f)
            elif found and f not in ignored:
                vals[f] = 0.0
                notes.setdefault("restricted", []).append(f)
    if ignored:
        notes["indented_decay_ignored"] = ignored
    return vals, notes


def proc_card(backend):
    """(model or None, [generate lines]) from the backend's Cards/proc_card_mg5.dat"""
    p = os.path.join(backend, "Cards", "proc_card_mg5.dat")
    if not os.path.exists(p):
        return None, []
    txt = open(p).read()
    models = re.findall(r"^\s*import\s+model\s+(\S+)", txt, re.M)
    gens = re.findall(r"^\s*((?:generate|add process)\s.*?)\s*$", txt, re.M)
    return (models[-1] if models else "sm"), gens


# ----------------------------------------------------------------------------------------------- strings and rules
_CLAUSE = re.compile(r"\b(?:QED|QCD)(?:\^2)?\s*(?:<=|>=|==|=|<|>)\s*\d+")


def order_clause(gen):
    """the coupling-order clause of a generate string ("QED<=2", "QCD<=2 QED<=4"; "" for none), [virt=QCD] and the
    process ignored, spaces removed"""
    return " ".join(sorted(re.sub(r"\s+", "", m) for m in _CLAUSE.findall(str(gen or "").split("[")[0])))


def _qed_max(pdg_ids):
    """catalog_v4's full-order bound (recipes/gen_catalog_v4.qed_max)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_catalog_v4", os.path.join(ROOT, "recipes", "gen_catalog_v4.py"))
    m = sys.modules.get("_gen_catalog_v4")
    if m is None:
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        sys.modules["_gen_catalog_v4"] = m
    return m.qed_max(pdg_ids)


def clause_kind(clause, pdg_ids):
    """minimal (no clause: MadGraph's minimal QED order), full (QED<=n - n_g, catalog_v4's every tree diagram), or the
    explicit clause"""
    if not clause:
        return "minimal"
    if pdg_ids is not None and clause == f"QED<={_qed_max(pdg_ids)}":
        return "full"
    return "explicit:" + clause


def coupling_kind(blk, pdg_ids):
    """which tree diagrams the pool's string admits: every one (the order vector's electroweak maximum is catalog_v4's
    full bound, by the clause or because the minimal order already reaches it), the minimal QED order only, or an
    explicit clause; one-loop pools: the clause's kind"""
    ov, clause = blk.get("order_vector"), blk.get("order_clause", "")
    if not blk.get("loop") and ov is not None and pdg_ids is not None and int(ov[3]) == _qed_max(pdg_ids) \
            and clause in ("", f"QED<={_qed_max(pdg_ids)}"):
        return "every tree diagram"
    k = clause_kind(clause, pdg_ids)
    return {"minimal": "minimal QED order"}.get(k, "explicit clause")   # an explicit string is process-specific


def relevant(field, pdg_ids, blk):
    """whether a field can change the pool's |M|^2 (a field that cannot is not compared). With a diagonal CKM a tree's
    fermion line changes flavour only t <-> b, at a W (or charged-Higgs) vertex, and ends on external legs: a b line
    (the b mass, its Yukawa, the model's b restriction) needs an external b, or an external top whose line turns into
    a b by emitting a W that leaves the diagram, as an external W or H+- or into a second light-quark line (a W
    reabsorbed elsewhere on the same line closes a loop), and likewise a top line needs an external top, or an external
    b with such a W. The W, Z and Higgs widths need an electroweak power or a colourless leg. A one-loop pool: every
    field (quark loops)."""
    if blk.get("loop"):
        return True
    ext = [abs(int(p)) for p in pdg_ids]
    w_out = any(p in (24, 37) for p in ext) or sum(p in (1, 2, 3, 4) for p in ext) >= 4
    if field in B_FIELDS:
        return 5 in ext or (6 in ext and w_out)
    if field in T_FIELDS:
        return 6 in ext or (5 in ext and w_out)
    if field in EW_WIDTHS:
        ov = blk.get("order_vector")
        return bool(set(ext) - {1, 2, 3, 4, 5, 6, 21}) or ov is None or int(ov[3]) > 0
    return True


def model_family(model):
    """sm for sm, sm-no_b_mass, loop_sm, loop_qcd_qed_sm-no_b_mass; the UFO name for the others"""
    m = str(model or "sm").split("-", 1)[0]
    for pre in ("loop_qcd_qed_", "loop_"):
        if m.startswith(pre):
            m = m[len(pre):]
    return m


def is_loop(cfg):
    return cfg.get("kind") == "virt" or bool(cfg.get("virt"))


# -------------------------------------------------------------------------------------------------------- the block
def _mg():
    import mg5_pipeline_final as mg
    return mg


def backend_dir(process, cfg=None):
    """the directory whose Cards/param_card.dat the pool's labels were computed with (None: no MadGraph backend)"""
    mg = _mg()
    cfg = cfg if cfg is not None else mg.PROCESSES.get(process, {})
    if mg.is_synthetic(process) or cfg.get("kind") == "synthetic":
        return None
    if cfg.get("kind") == "lhe_pp":
        return mg.lhe_standalone_dir(process)
    if is_loop(cfg):
        return f"{mg.WORK_DIR}/{cfg['virt_base']}_virt_standalone"
    return f"{mg.WORK_DIR}/{mg.standalone_name(process)}_standalone"


def entry_model(cfg):
    if cfg.get("kind") == "lhe_pp":
        return (cfg.get("lhe") or {}).get("model", "sm")
    if is_loop(cfg):
        from tools.nlo_virtual_pipeline import VIRT_PROCESSES
        return VIRT_PROCESSES.get(cfg.get("virt_base"), {}).get("model", "loop_sm")
    return cfg.get("model") or "sm"


def entry_generate(cfg):
    if is_loop(cfg):
        from tools.nlo_virtual_pipeline import VIRT_PROCESSES
        return VIRT_PROCESSES.get(cfg.get("virt_base"), {}).get("mg5")
    g = cfg.get("mg5_generate")
    return " ; ".join(g) if isinstance(g, (list, tuple)) else g


def alpha_s_of(cfg, recipe=None):
    """{rule, alphas_mz} of a pool: the window pools and the pp pools labelled at alpha_s(sqrt(s_hat)) share one rule
    (mg5_pipeline_final.compute_alphas: one-loop running from alpha_s(M_Z), n_f = 5)"""
    recipe = recipe or {}
    if cfg.get("kind") == "synthetic":
        return {"rule": "none", "alphas_mz": None}
    if cfg.get("kind") == "lhe_pp":
        lab = (cfg.get("lhe") or {}).get("label_alphas", "aqcdup")
        if lab == "running_sqrt_shat":
            return {"rule": "1loop_nf5_mu_sqrt_shat", "alphas_mz": 0.118}
        return {"rule": "madevent_" + str(lab), "alphas_mz": None}
    amz = recipe.get("alphas_mz", cfg.get("alphas_mz", 0.118))
    return {"rule": "1loop_nf5_mu_sqrt_shat", "alphas_mz": float(amz)}


def b_row(pdg_ids, cfg, spec_physics=None):
    """the particle-table row the encoding reads for the pool's b: {code, mass}; None without an external b"""
    import numpy as np
    from particle_ids import species_pdgs, PARTICLE_PROPERTIES, PARTICLE_FEATURE_NAMES, _MASSLESS
    if not any(abs(int(p)) == 5 for p in pdg_ids):
        return None
    codes = species_pdgs(np.asarray([int(p) for p in pdg_ids]), spec_physics, cfg)
    code = next(abs(int(c)) for c, p in zip(codes, pdg_ids) if abs(int(p)) == 5)
    lm = PARTICLE_PROPERTIES[code][PARTICLE_FEATURE_NAMES.index("log10_mass_gev")]
    return {"code": code, "mass": 0.0 if abs(lm - _MASSLESS) < 1e-6 else round(float(10.0 ** lm), 6)}


_CARD_CACHE = {}


def _card_values(backend):
    card = os.path.join(backend, "Cards", "param_card.dat")
    if not os.path.exists(card):
        return None
    st = os.stat(card)
    key = (card, st.st_mtime_ns, st.st_size)
    if key not in _CARD_CACHE:
        vals, notes = read_card(card, backend)
        sha = hashlib.sha256(open(card, "rb").read()).hexdigest()
        model, gens = proc_card(backend)
        _CARD_CACHE[key] = (vals, notes, sha, model, gens)
    return (card,) + _CARD_CACHE[key]


def physics_block(process, recipe=None, spec_physics=None, cfg=None):
    """the `physics` block of a pool of `process` (its PROCESSES entry, a recipe scan registered), read from the
    backend that labelled it; fields that cannot be read are listed in `missing`"""
    mg = _mg()
    if cfg is None:
        cfg = mg.register_synthetic(process) if mg.is_synthetic(process) else mg.PROCESSES[process]
    pdg = [int(p) for p in cfg["pdg_ids"]]
    blk = {"schema": SCHEMA, "process": process, "loop": is_loop(cfg)}
    be = backend_dir(process, cfg)
    blk["backend"] = be
    got = _card_values(be) if be else None
    if got:
        card, vals, notes, sha, model, gens = got
        blk.update(card=card, card_sha256=sha, **{f: vals[f] for f in CARD_FIELDS if f in vals})
        if notes:
            blk["card_notes"] = notes
        blk["model"], blk["model_source"] = (model, "proc_card") if model else (entry_model(cfg), "entry")
        gen = entry_generate(cfg)
        if gens and not is_loop(cfg):
            gen = " ; ".join(gens)
            blk["generate_source"] = "proc_card"
    else:
        blk["card"] = None
        blk["model"], blk["model_source"] = entry_model(cfg), "entry"
        gen = entry_generate(cfg)
    blk["generate"] = gen
    blk["order_clause"] = order_clause(gen)
    try:
        blk["order_vector"] = [int(x) for x in mg.order_vector(cfg)]
    except Exception:
        pass
    blk["alpha_s"] = alpha_s_of(cfg, recipe)
    try:
        from particle_ids import b_mass_of_process
        blk["b_mass_generated"] = b_mass_of_process(cfg, spec_physics)
        blk["b_row"] = b_row(pdg, cfg, spec_physics)
    except Exception as e:                       # recorded, never dropped
        blk["b_row_error"] = repr(e)[:200]
    blk["missing"] = [f for f in CARD_FIELDS + ("model", "order_vector", "b_row") if f not in blk]
    return blk


def safe_block(process, recipe=None, spec_physics=None):
    """physics_block, or {schema, error} when it cannot be built: generation never fails on it"""
    try:
        return physics_block(process, recipe, spec_physics)
    except Exception as e:
        return {"schema": SCHEMA, "process": process, "error": repr(e)[:300]}


# --------------------------------------------------------------------------------------------------- the comparison
def _view(blk, pdg_ids):
    """the comparable values of a block"""
    v = {f: blk[f] for f in CARD_FIELDS if blk.get(f) is not None}
    if blk.get("model"):
        v["model"] = blk["model"]
        v["model_family"] = model_family(blk["model"])
    if "order_clause" in blk:
        v["order_clause"] = blk["order_clause"]
        v["coupling_kind"] = coupling_kind(blk, pdg_ids)
    if blk.get("order_vector") is not None:
        v["order_vector"] = tuple(blk["order_vector"])
    a = blk.get("alpha_s") or {}
    if a.get("rule"):
        v["alpha_s_rule"] = a["rule"]
    if a.get("alphas_mz") is not None:
        v["alphas_mz"] = float(a["alphas_mz"])
    if "b_row" in blk:
        v["b_row"] = None if blk["b_row"] is None else (int(blk["b_row"]["code"]), float(blk["b_row"]["mass"]))
    return v


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def _show(x):
    return list(x) if isinstance(x, tuple) else x


def declared_fields(spec_physics):
    """the fields a recipe entry's physics scan sets on purpose"""
    p = spec_physics or {}
    out = set()
    if not p:
        return out
    try:
        patches, mass_by_pdg, amz = _mg().physics_to_generation(p)
        out |= {k for k in patches if k in CARD_FIELDS}
        if 5 in {abs(int(k)) for k in mass_by_pdg}:
            out |= {"MB", "ymb", "b_row"}
        if amz is not None:
            out.add("alphas_mz")
    except Exception:
        pass
    if p.get("alpha_s") is not None:
        out.add("alphas_mz")
    return out & set(SAME_FIELDS + WIDE_FIELDS)


def _canon_key(pdg_ids, loop):
    p = [int(x) for x in pdg_ids]
    return (tuple(sorted(p[:2])), tuple(sorted(p[2:])), bool(loop))


# probes left out of every read by the user's decision (never "missing until built"): the read skips them and lists
# them as excluded; tools/collect_run.py does not expect them, scripts/job_catalog_v3_short.sh leaves them out
DROPPED = {"ee4_epem_cgsxwm": "its 3 TeV block dropped (D193)"}


def recipe_sampling(doc):
    """a recipe's sampling policy for every entry without its own: its top-level `sampling` key, as the experiment,
    prebuild_recipes and check_site_data read it (the pool's file name follows it: `_smix_` for the mixture)"""
    return doc.get("sampling") if isinstance(doc, dict) else None


def load_recipe_specs(path):
    import yaml
    d = yaml.safe_load(open(path))
    procs = d.get("processes", d) if isinstance(d, dict) else d
    if isinstance(procs, dict):
        procs = [dict(v, name=k) for k, v in procs.items()]
    return [{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
             **{k: p[k] for k in ("sqrts", "n_train", "n_val", "n_test") if k in p}} for p in procs], d


def stored_block(name, spec, seed=42):
    """(block, pool) of the entry's stored pool from its sidecar: the train, val and test pools in turn, the first built
    one (pool present, sidecar recipe id the entry's) whose sidecar carries a physics block without error; an entry with
    a default share (sampling.f_default > 0) reads its catalog_v4 control pool's, which mix_compose copies into the
    composed pool (the same backend labelled both). (None, None) when there is none."""
    mg = _mg()
    import datagen
    if "sqrts" not in spec:
        return None, None
    lo, hi = (float(x) for x in spec["sqrts"])
    cfg = mg.PROCESSES.get(name)
    swap = None
    if cfg is not None and float((cfg.get("sampling") or {}).get("f_default", 0) or 0) > 0:
        import mix_compose
        swap = mix_compose._Swap(name, mix_compose.control_config(name))
    if swap:
        swap.__enter__()
    try:
        for role in ("train", "val", "test"):
            n = int(spec.get(f"n_{role}", 0) or 0)
            if n <= 0:
                continue
            rec = mg.variable_energy_recipe(name, lo, hi, n, role=role, seed=seed)
            path = mg.recipe_output_path(rec, datagen.dest_for_role(role))
            if not (os.path.exists(path) and datagen._is_cached(path, mg.recipe_id(rec))):
                continue
            try:
                blk = json.load(open(path + ".recipe.json")).get("physics")
            except ValueError:
                continue
            if blk and "error" not in blk:
                return dict(blk), path
    finally:
        if swap:
            swap.__exit__()
    return None, None


class TrainingReference:
    """the training catalog's blocks, from its recipe (a parent run's data.processes_file): each entry's block from its
    stored pool's sidecar (stored_block), else computed from its entry and backend card while the standalone exists;
    an entry with neither is listed in `unreadable`, never a reference. `sources` counts where the blocks came from.
    Indexed by process and by (model family, loop kind)"""

    def __init__(self, recipe_path, blocks=None, seed=42, backend=True):
        mg = _mg()
        self.recipe = recipe_path
        self.sources = {}
        self.unreadable = []
        specs, d = load_recipe_specs(recipe_path)
        mg.register_recipe_processes(specs, default_sampling=recipe_sampling(d))
        self.entries = []
        for s in specs:
            name = s["name"]
            cfg = mg.PROCESSES.get(name) or mg.PROCESSES.get(s["base"])
            if cfg is None:
                continue
            proc = name if name in mg.PROCESSES else s["base"]
            try:
                blk, src = (blocks or {}).get(name), "given"
                if blk is None:
                    blk, _ = stored_block(proc, s, seed)
                    src = "sidecar"
                    if blk is not None:     # the b row the encoding reads is the particle table's now, as for a probe
                        blk["b_row"] = b_row([int(p) for p in cfg["pdg_ids"]], cfg, s["physics"])
                if blk is None:
                    be = backend_dir(proc, cfg) if backend else None
                    if not be or not os.path.exists(os.path.join(be, "Cards", "param_card.dat")):
                        raise RuntimeError("no sidecar physics block and no backend card")
                    blk, src = physics_block(proc, spec_physics=s["physics"], cfg=cfg), "backend card"
            except Exception as e:                   # listed, never a reference
                self.unreadable.append((name, repr(e)[:120]))
                continue
            self.sources[src] = self.sources.get(src, 0) + 1
            self.entries.append({"name": name, "scan": bool(s["physics"]), "declared": declared_fields(s["physics"]),
                                 "pdg": [int(p) for p in cfg["pdg_ids"]],
                                 "loop": is_loop(cfg), "family": model_family(blk.get("model")), "block": blk})
        self.by_key = {}
        for e in self.entries:
            self.by_key.setdefault(_canon_key(e["pdg"], e["loop"]), []).append(e)

    def same_process(self, pdg_ids, loop):
        es = self.by_key.get(_canon_key(pdg_ids, loop), [])
        plain = [e for e in es if not e["scan"]]
        return plain or es

    def catalog_wide(self, family, loop):
        for sel in (lambda e: e["family"] == family and e["loop"] == loop, lambda e: e["loop"] == loop, lambda e: True):
            es = [e for e in self.entries if sel(e)]
            if es:
                plain = [e for e in es if not e["scan"]]
                return plain or es
        return []


def compare(probe_blk, pdg_ids, refs, fields, exempt=()):
    """[(field, probe value, reference values)] where the probe's value is none of the references'; plus the fields
    not compared (absent on either side, or set by a reference's own scan) and the exempt ones present"""
    pv = _view(probe_blk, pdg_ids)
    # a reference entry's own scan (the alpha_s points, the heavy-quark layer) is no reference for the fields it sets
    rv = [{k: x for k, x in _view(r["block"], r["pdg"]).items() if k not in r.get("declared", ())} for r in refs]
    bad, skipped, ex = [], [], []
    for f in fields:
        if f in exempt:
            ex.append(f)
            continue
        if not relevant(f, pdg_ids, probe_blk):
            continue
        vals = []
        for v in rv:
            if f in v and not any(_same(v[f], w) for w in vals):
                vals.append(v[f])
        if f not in pv or not vals:
            skipped.append(f)
            continue
        if f == "b_row" and pv[f] is None:
            continue
        if not any(_same(pv[f], w) for w in vals):
            bad.append({"field": f, "probe": _show(pv[f]), "reference": [_show(w) for w in vals]})
    return bad, skipped, ex


def check_probe(name, spec_physics, pool_path, reference, cfg=None, backend=True):
    """the check of one probe pool: {status ok|mismatch, reference, mismatches, ...}. `reference` is a
    TrainingReference (None: the parent has no training recipe, which refuses). backend=False: the sidecar's block
    only, as the check runs once the standalones are gone"""
    mg = _mg()
    cfg = cfg if cfg is not None else mg.PROCESSES[name]
    pdg = [int(p) for p in cfg["pdg_ids"]]
    side = None
    if pool_path and os.path.exists(pool_path + ".recipe.json"):
        try:
            side = json.load(open(pool_path + ".recipe.json"))
        except ValueError:
            side = None
    blk = (side or {}).get("physics")
    if blk and "error" not in blk:
        src = "sidecar"
        blk = dict(blk)
    else:
        src = "computed at read" + ("" if side else (" (no sidecar)" if not pool_path or os.path.exists(pool_path)
                                                     else " (no pool at this path)"))
        blk = physics_block(name, recipe=side, spec_physics=spec_physics, cfg=cfg)
        if blk.get("backend") and (blk.get("card") is None or not backend):     # nothing to read the physics from: refused
            return {"probe": name, "pool": pool_path, "block_source": src, "status": "mismatch",
                    "reference": {"kind": "none"},
                    "mismatches": [{"field": "physics block", "probe": None,
                                    "reference": "the pool's sidecar has no physics block and its backend card "
                                                 f"{blk['backend']}/Cards/param_card.dat is absent"}]}
    # the b row the encoding reads is a property of this read (the particle table now), never of the sidecar
    try:
        blk["b_row"] = b_row(pdg, cfg, spec_physics)
    except Exception as e:
        blk["b_row_error"] = repr(e)[:200]
    out = {"probe": name, "pool": pool_path, "block_source": src, "mismatches": []}
    if reference is None:
        out.update(status="mismatch", reference={"kind": "none"},
                   mismatches=[{"field": "training_recipe", "probe": None,
                                "reference": "the parent has no training recipe (data.processes_file) to compare with"}])
        return out
    exempt = declared_fields(spec_physics)
    loop = is_loop(cfg)
    refs = reference.same_process(pdg, loop)
    if refs:
        out["reference"] = {"kind": "same process", "entries": [r["name"] for r in refs][:8]}
        bad, skipped, ex = compare(blk, pdg, refs, SAME_FIELDS, exempt)
    else:
        fam = model_family(blk.get("model"))
        refs = reference.catalog_wide(fam, loop)
        out["reference"] = {"kind": "catalog-wide", "family": fam, "loop": loop, "n_entries": len(refs)}
        bad, skipped, ex = compare(blk, pdg, refs, WIDE_FIELDS, exempt)
    # the b row against the probe's own card: the encoding must read the b the pool was labelled with
    br, mb = blk.get("b_row"), blk.get("MB")
    if br is not None and mb is not None and "b_row" not in exempt and not _same(float(br["mass"]), float(mb)):
        bad.append({"field": "b_row vs the probe's own MB", "probe": [br["code"], br["mass"]], "reference": [mb]})
    out.update(status="mismatch" if bad else "ok", mismatches=bad, not_compared=skipped, exempt=ex,
               probe_physics={k: blk.get(k) for k in ("model", "order_clause", "MB", "MT", "WT", "b_row")})
    return out


def apply_allowance(chk, reason):
    """a mismatch read on purpose: status allowed, the reason kept; returns whether the probe may be scored"""
    if chk["status"] == "ok":
        return True
    if reason:
        chk["status"], chk["reason"] = "allowed", str(reason)
        return True
    return False


def summary(chk):
    """one line naming the differing fields"""
    return "; ".join(f"{m['field']}: probe {m['probe']} vs {m['reference']}" for m in chk.get("mismatches", []))[:600]


def record(chk):
    """the compact form a ZERO_SHOT record carries"""
    out = {"status": chk["status"], "reference": chk.get("reference"),
           "fields": [m["field"] for m in chk.get("mismatches", [])]}
    if chk.get("reason"):
        out["reason"] = chk["reason"]
    return out


# ------------------------------------------------------------------------------------------------- what reads read
def zs_status(rec):
    """the physics status of one ZERO_SHOT record: ok, allowed, mismatch (a refused probe's error) or unchecked"""
    if "error" in rec:
        return "mismatch" if str(rec["error"]).startswith("physics mismatch") else "error"
    p = rec.get("physics")
    if not isinstance(p, dict) or "status" not in p:
        return "unchecked"
    return p["status"]


def npz_status(z):
    """(status, reason) of a saved probe prediction (np.load of tools/zero_shot_parent.py's <probe>.npz)"""
    files = getattr(z, "files", z)
    if "physics_check" not in files:
        return "unchecked", None
    chk = json.loads(str(z["physics_check"]))
    return chk.get("status", "unchecked"), chk.get("reason")


def require_npz(z, path=""):
    """a reads script's gate: refuses a prediction whose probe failed the check without a reason; returns the
    (status, reason) for the script to report"""
    st, why = npz_status(z)
    if st == "mismatch":
        raise RuntimeError(f"{path}: probe scored with a physics mismatch and no reason (tools/pool_physics.py)")
    return st, why


# ------------------------------------------------------------------------------------------------------- command line
def val_pool_path(name, spec, recipe_doc, seed=42, role="val"):
    """the pool of `role` (default val) a probe read scores (the experiment's recipe path: datagen.ensure_dataset's
    file name)"""
    mg = _mg()
    import datagen
    lo, hi = (float(x) for x in spec["sqrts"])
    rec = mg.variable_energy_recipe(name, lo, hi, int(spec[f"n_{role}"]), role=role, seed=seed)
    return mg.recipe_output_path(rec, datagen.dest_for_role(role))


def sidecar_scan(dirs):
    """Every pool sidecar (*.npy.recipe.json) under `dirs`: (n sidecars, [{sidecar, process, backend, reason}] of those
    without a usable physics block). `backend` is the standalone that labelled the pool (backend_dir of its process),
    None when the process is not in the pipeline's table; `backends` every standalone that may have: a name only
    recipes define (an alpha_s scan such as ee_uug__as130, base ee_uug) gets the backend of each distinct definition
    recipes/**/*.yaml give it (scan_template.yaml and catalog_v2_holdout_scan.yaml define ee_uug__as130 differently,
    with and without alpha_ew, which names ee_uug__as130_standalone or ee_uug_standalone), so the prune keeps them all."""
    mg = _mg()
    n, out = 0, []
    index = []

    def candidates(proc):
        if proc in mg.PROCESSES:
            return [backend_dir(proc)]
        if not index:
            import yaml
            defs = {}
            for f in sorted(glob.glob(os.path.join(ROOT, "recipes", "**", "*.yaml"), recursive=True)):
                try:
                    d = yaml.load(open(f), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
                except Exception:
                    continue
                procs = d.get("processes") if isinstance(d, dict) else d
                for e in procs if isinstance(procs, list) else []:
                    if isinstance(e, dict) and "name" in e:
                        k = json.dumps({"base": e.get("base", e["name"]), "physics": e.get("physics")}, sort_keys=True,
                                       default=str)
                        defs.setdefault(e["name"], {})[k] = {"name": e["name"], "base": e.get("base", e["name"]),
                                                            "physics": e.get("physics")}
            index.append(defs)
        bes = set()
        for spec in index[0].get(proc, {}).values():
            try:
                mg.register_recipe_processes([spec])
                bes.add(backend_dir(proc))
            except Exception:
                pass
            finally:
                mg.PROCESSES.pop(proc, None)
        return sorted(b for b in bes if b)
    for d in dirs:
        for dp, _, fs in os.walk(d):
            for f in fs:
                if not f.endswith(".npy.recipe.json"):
                    continue
                n += 1
                fp = os.path.join(dp, f)
                try:
                    side = json.load(open(fp))
                except ValueError:
                    side = {}
                blk = side.get("physics")
                if blk and "error" not in blk:
                    continue
                proc = side.get("process")
                try:
                    bes = candidates(proc) if proc else []
                except Exception:
                    bes = []
                out.append({"sidecar": fp, "process": proc, "backend": bes[0] if bes else None, "backends": bes,
                            "reason": "physics error" if blk else "no physics block"})
    return n, out


def standalone_kind(d):
    """which MadGraph process output `d` is: "madloop" (an output keeping Cards/proc_card_mg5.dat: the MadLoop
    *_virt_standalone ones and some trees), "standalone_cpp" (output standalone_cpp keeps no proc card: Cards/param_card.dat
    with SubProcesses/), or None (not a process output, or a symlink: never pruned)"""
    if os.path.islink(d) or not os.path.isdir(d):
        return None
    if os.path.isfile(os.path.join(d, "Cards", "proc_card_mg5.dat")):
        return "madloop"
    if os.path.isfile(os.path.join(d, "Cards", "param_card.dat")) and os.path.isdir(os.path.join(d, "SubProcesses")):
        return "standalone_cpp"
    return None


def standalone_dirs(work_dir):
    """The compiled process outputs directly under the MadGraph install `work_dir`: <name>_standalone directories that
    standalone_kind recognises (the install's own directories never end so) [D208]"""
    return sorted(os.path.join(work_dir, x) for x in os.listdir(work_dir)
                  if x.endswith("_standalone") and standalone_kind(os.path.join(work_dir, x)))


def standalones_census(work_dir):
    """every <name>_standalone entry directly under `work_dir` sorted by why it is or is not a prune candidate
    (standalone_dirs): symlink, no Cards/ at all, Cards/ without proc_card_mg5.dat (with or without a param_card), or
    a candidate; per class the count, the bytes (files only, symlinks not followed) and the name prefixes"""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from lhe_parts import _du as du
    out = {}
    for x in sorted(os.listdir(work_dir)):
        if not x.endswith("_standalone"):
            continue
        d = os.path.join(work_dir, x)
        cards = os.path.join(d, "Cards")
        if os.path.islink(d):
            cls = "symlink"
        elif not os.path.isdir(d):
            cls = "not a directory"
        elif standalone_kind(d):
            cls = "prune candidate (" + standalone_kind(d) + ")"
        elif not os.path.isdir(cards):
            cls = "no Cards/"
        else:
            has_pc = os.path.isfile(os.path.join(cards, "param_card.dat"))
            cls = "Cards/ without proc_card_mg5.dat" + (", param_card present" if has_pc else ", no param_card")
        c = out.setdefault(cls, {"n": 0, "bytes": 0, "prefixes": {}, "first": []})
        c["n"] += 1
        c["bytes"] += 0 if os.path.islink(d) else du(d)
        pre = x.split("_", 1)[0]
        c["prefixes"][pre] = c["prefixes"].get(pre, 0) + 1
        if len(c["first"]) < 5:
            c["first"].append([x, sorted(os.listdir(d))[:8] if os.path.isdir(d) and not os.path.islink(d) else None])
    for c in out.values():
        c["gb"] = round(c.pop("bytes") / 1e9, 2)
    return out


def prune_standalones(work_dir, keep, apply=False, manifest_dir=None):
    """The standalones of standalone_dirs(work_dir) not in `keep` (the backends of pools without a sidecar physics
    block), each deleted with --apply after a manifest of every path and size [D208]. The install itself is never
    touched: only <name>_standalone outputs directly under it."""
    import shutil, time
    keep = {os.path.realpath(k) for k in keep if k}
    cand = [d for d in standalone_dirs(work_dir) if os.path.realpath(d) not in keep]
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from lhe_parts import _du as du
    ent = [{"path": d, "bytes": du(d)} for d in cand]
    mdir = manifest_dir or os.path.join(work_dir, ".prune_manifests")
    os.makedirs(mdir, exist_ok=True)
    man = os.path.join(mdir, f"standalones_{time.strftime('%Y%m%d_%H%M%S')}{'' if apply else '_dry'}.json")
    with open(man, "w") as f:
        json.dump({"work_dir": work_dir, "apply": apply, "kept": sorted(keep), "entries": ent}, f)
    freed = 0
    if apply:
        root = os.path.realpath(work_dir)
        for e in ent:
            r = os.path.realpath(e["path"])
            if os.path.dirname(r) != root or not r.endswith("_standalone"):
                raise RuntimeError(f"{e['path']}: not a standalone directly under {root}; nothing more deleted")
            shutil.rmtree(r)
            freed += e["bytes"]
    return {"manifest": man, "apply": apply, "standalones": len(standalone_dirs(work_dir)) + (len(ent) if apply else 0),
            "deletable": len(ent), "kept": len(keep), "gb": round(sum(e["bytes"] for e in ent) / 1e9, 2),
            "freed_gb": round(freed / 1e9, 2)}


def main(argv=None):
    """python tools/pool_physics.py check --train <training recipe> <probe recipe> [...]: every probe's check, without a
    model (what tools/zero_shot_parent.py would refuse); one PHYSICS_CHECK line each, the counts as the last line.
    python tools/pool_physics.py block <process> [<pool .npy>]: the block of a process (and the pool's sidecar's)."""
    import argparse, yaml
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("check")
    c.add_argument("--train", required=True)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--quiet", action="store_true", help="only the counts and the mismatching probes")
    c.add_argument("--role", default="val", choices=("val", "test"), help="the probe pool read (EE4: test)")
    c.add_argument("--sidecar-only", action="store_true",
                   help="blocks from the pools' sidecars only, no backend card: the check without the standalones")
    c.add_argument("probes", nargs="+")
    b = sub.add_parser("block")
    b.add_argument("process")
    b.add_argument("pool", nargs="?")
    sc = sub.add_parser("sidecars", help="pools whose sidecar has no physics block, with the standalones they need")
    sc.add_argument("--out")
    sub.add_parser("standalones-census", help="every *_standalone under the install by why it is (not) a prune candidate")
    pr = sub.add_parser("prune-standalones", help="delete the standalones no pool needs (manifest first) [D208]")
    pr.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    mg = _mg()
    if a.mode == "standalones-census":
        print(json.dumps({"work_dir": mg.WORK_DIR, "classes": standalones_census(mg.WORK_DIR)}))
        return
    if a.mode in ("sidecars", "prune-standalones"):
        import datagen
        dirs = sorted({os.path.realpath(datagen.frozen_dir()), os.path.realpath(datagen.train_cache_dir())})
        n, bad = sidecar_scan(dirs)
        keep = sorted({b for e in bad for b in e["backends"]})
        unmapped = sorted({str(e["process"]) for e in bad if not e["backend"]})
        if a.mode == "sidecars":
            out = {"dirs": dirs, "sidecars": n, "without_block": len(bad), "standalones_needed": keep,
                   "unmapped_processes": unmapped, "pools": bad}
            if a.out:
                json.dump(out, open(a.out, "w"), indent=1)
            print(json.dumps({k: v for k, v in out.items() if k != "pools"}))
            return
        if unmapped and a.apply:
            raise SystemExit(f"{len(unmapped)} processes of pools without a physics block are not in the pipeline's "
                             f"table (their standalone is unknown), e.g. {unmapped[:5]}: nothing deleted")
        print(json.dumps(dict(prune_standalones(mg.WORK_DIR, keep, a.apply), sidecars=n, without_block=len(bad),
                              unmapped_processes=len(unmapped))))
        return
    if a.mode == "block":
        out = {"entry": physics_block(a.process)}
        if a.pool and os.path.exists(a.pool + ".recipe.json"):
            out["sidecar"] = json.load(open(a.pool + ".recipe.json")).get("physics")
        print(json.dumps(out, indent=1, default=str))
        return
    ref = TrainingReference(a.train, seed=a.seed, backend=not a.sidecar_only)
    counts, fields, refs, excluded = {}, {}, {}, {}
    for f in a.probes:
        doc = yaml.safe_load(open(f))
        procs = doc.get("processes", doc) if isinstance(doc, dict) else doc
        mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics")}
                                      for p in procs],
                                     default_sampling=recipe_sampling(doc))
        for p in procs:
            name = p["name"]
            if name in DROPPED:
                counts["excluded"] = counts.get("excluded", 0) + 1
                excluded[name] = DROPPED[name]
                continue
            try:
                pool = val_pool_path(name, p, doc, a.seed, a.role)
                chk = check_probe(name, p.get("physics"), pool, ref, backend=not a.sidecar_only)
            except Exception as e:
                chk = {"probe": name, "status": "error", "error": repr(e)[:200]}
            counts[chk["status"]] = counts.get(chk["status"], 0) + 1
            kind = (chk.get("reference") or {}).get("kind")
            refs[kind] = refs.get(kind, 0) + 1
            for m in chk.get("mismatches", []):
                fields[m["field"]] = fields.get(m["field"], 0) + 1
            if not a.quiet or chk["status"] != "ok":
                print("PHYSICS_CHECK " + json.dumps(chk, default=str), flush=True)
    print(json.dumps({"train": a.train, "probes": a.probes, "role": a.role, "sidecar_only": a.sidecar_only,
                      "reference_sources": ref.sources, "reference_unreadable": len(ref.unreadable),
                      "reference_unreadable_first": ref.unreadable[:10],
                      "status": counts, "excluded": excluded, "reference": refs,
                      "mismatching_fields": dict(sorted(fields.items(), key=lambda x: -x[1]))}))


if __name__ == "__main__":
    main()
