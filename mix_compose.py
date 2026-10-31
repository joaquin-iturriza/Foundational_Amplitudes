"""Load-time default-share mixtures (catalog_coverage step 14; the user, 2026-10-31: "just build the mixture from the
data that already exists when you load the data"). A recipe entry with sampling.f_default > 0 (catalog_v4_default50 /
default25) is composed when it is loaded, from pools that exist anyway, instead of a stored _smixd<NN> pool:
  * the entry's catalog_v4 pool (the same entry without f_default: _smix_, 0.35 flat in log|M|^2, 0.15 forward, the
    rest RAMBO bulk), whose rows are told apart by replaying each chunk's bulk and forward draws (they come first from
    the chunk's seeded generator and depend on no label; mg.build_mixture_dataset), matched to the pool's rows by their
    momenta, exactly;
  * the default-sampled events of `default_source` (its MadEvent lhe pools, mg.default_share_pools), the same rows
    [a, b) a stored pool's chunk reads (mg.default_share_rows), in the partonic frame and inside the pool's window and
    cuts (mg.default_share_momenta), with the |M|^2 the lhe pool carries.
Per chunk the shares are the stored pool's (the same roundings): every flat row, the forward rows, the first bulk rows
and the default rows; then shuffled with the chunk's seed. The composed pool is written under the job's node-local
$TMPDIR (removed with the job), with its per-event alpha_s (mg.compute_alphas at sqrt(s_hat)) and a recipe sidecar
carrying the stored pool's recipe id, so the loader reads it as it read the stored pool.
Nothing is generated and no matrix element is evaluated."""
import copy, hashlib, json, os, tempfile

import numpy as np

import mg5_pipeline_final as mg

KIND = {"bulk": 0, "fwd": 1, "flat": 2, "default": 3}


def _cfg(process):
    return mg.PROCESSES[process]


def control_config(process):
    """The entry's process config without the default share (its catalog_v4 pool's)."""
    c = copy.deepcopy(_cfg(process))
    s = dict(c.get("sampling") or {})
    s.pop("f_default", None); s.pop("default_source", None)
    c["sampling"] = s
    return c


class _Swap:
    """mg.PROCESSES[process] replaced by `cfg` inside the block."""
    def __init__(self, process, cfg):
        self.p, self.c = process, cfg

    def __enter__(self):
        self.old = mg.PROCESSES[self.p]; mg.PROCESSES[self.p] = self.c

    def __exit__(self, *a):
        mg.PROCESSES[self.p] = self.old


def _sampler_args(cfg, role):
    nfinal = int(cfg["nfinal"])
    m_list = list(cfg["m_finals"]) if "m_finals" in cfg else \
        (list(cfg["m_final"]) if isinstance(cfg["m_final"], (list, tuple)) else [float(cfg["m_final"])] * nfinal)
    cuts = dict(cfg.get("fiducial_cuts") or mg.FIDUCIAL_CUTS) if mg.FIDUCIAL_CUTS_ENABLED else None
    return m_list, cuts, mg.sampling_policy(cfg, role), cfg.get("sqrts_measure", "uniform"), cfg.get("beam_masses")


def _key(row_mom):
    return hashlib.blake2b(np.ascontiguousarray(row_mom, dtype=np.float64).tobytes(), digest_size=16).digest()


REL_TOL = 1e-9      # a replayed event and its pool row agree to rounding (the build node's floating point: ~1 ulp seen)


def _shat(P):
    q = P[:, 0] + P[:, 1]
    return np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0.0, None))


def match_rows(C, P, window=8):
    """For each event of C (m, k, 4) the index of the row of P (n, k, 4) equal to it within REL_TOL of its energy
    scale, -1 when none: candidates are the rows nearest in the first outgoing particle's p_x (continuous; sqrt(s_hat)
    is not: an e+e- pool sits at a few beam energies)."""
    sp, sc = P[:, 2, 1], C[:, 2, 1]
    o = np.argsort(sp, kind="stable"); ss = sp[o]
    at = np.searchsorted(ss, sc)
    scale = np.maximum(np.abs(C).reshape(len(C), -1).max(1), 1.0)
    out = np.full(len(C), -1, np.int64)
    for off in sorted(range(-window, window), key=abs):
        todo = out < 0
        if not todo.any():
            break
        i = o[np.clip(at[todo] + off, 0, len(o) - 1)]
        ok = np.abs(P[i] - C[todo]).reshape(len(i), -1).max(1) <= REL_TOL * scale[todo]
        out[np.where(todo)[0][ok]] = i[ok]
    return out


def chunk_plan(process, sqrts_min, sqrts_max, n_events, role, seed):
    import datagen
    return datagen.chunk_tasks(process, sqrts_min, sqrts_max, n_events, role, seed, "/nonexistent")


def provenance(process, sqrts_min, sqrts_max, n_events, role, seed, pool):
    """int8 per row of the entry's catalog_v4 pool `pool` (array): bulk 0, forward 1, flat 2. Raises unless every
    replayed bulk and forward event is found in its chunk's rows and the counts are the sampler's."""
    cfg = control_config(process)
    k = int(cfg["nfinal"]) + 2
    out = np.full(len(pool), KIND["flat"], np.int8)
    with _Swap(process, cfg):
        tasks = chunk_plan(process, sqrts_min, sqrts_max, n_events, role, seed)
        m_list, cuts, pol, measure, beam_mass = _sampler_args(cfg, role)
    start = 0
    for t in tasks:
        n = int(t["count"])
        rng = np.random.default_rng(t["seed"])
        f_fwd = float(pol.get("f_fwd", 0.0))
        n_flat = int(round(pol["f_flat"] * n)); n_fwd = int(round(f_fwd * n)); n_bulk = n - n_flat - n_fwd
        uni = mg._sqrts_sampler(float(sqrts_min), float(sqrts_max), measure, rng)
        drawn = {}
        if n_bulk:
            drawn["bulk"] = mg._candidates(n_bulk, uni, m_list, cfg["pdg_ids"], rng, cuts, 0.0, pol["y_lo"], 0.0,
                                           beam_mass)[0]
        if n_fwd:
            drawn["fwd"] = mg._candidates(n_fwd, uni, m_list, cfg["pdg_ids"], rng, cuts, 0.0, pol["y_lo"], 1.0,
                                          beam_mass)[0]
        Pc = np.asarray(pool[start:start + n, :4 * k], np.float64).reshape(n, k, 4)
        for kind, evs in drawn.items():
            m = match_rows(np.stack([np.asarray(e[0], np.float64) for e in evs]), Pc)
            if (m < 0).any():
                raise RuntimeError(f"{process} {role} chunk {t['idx']}: {int((m < 0).sum())} of {len(m)} replayed {kind} "
                                   f"events not in the pool")
            if (out[start + m] != KIND["flat"]).any() or len(np.unique(m)) != len(m):
                raise RuntimeError(f"{process} {role} chunk {t['idx']}: a pool row matched twice")
            out[start + m] = KIND[kind]
        c = np.bincount(out[start:start + n], minlength=3)
        if (c[0], c[1], c[2]) != (n_bulk, n_fwd, n_flat):
            raise RuntimeError(f"{process} {role} chunk {t['idx']}: provenance {c.tolist()} against bulk {n_bulk}, "
                               f"forward {n_fwd}, flat {n_flat}")
        start += n
    assert start == len(pool), (start, len(pool))
    return out


def default_rows_labelled(source, role, seed, rows, accept):
    """mg.default_share_momenta's rows [a, b), with the |M|^2 the lhe pools carry for them."""
    a, b = int(rows[0]), int(rows[1])
    parts, labels = [], []
    for name, rec, path in mg.default_share_pools(source, role, seed):
        with open(path + ".recipe.json") as f:
            if json.load(f).get("recipe_id") != mg.recipe_id(rec):
                raise RuntimeError(f"default share of {source} ({role}): pool {path} stale")
        k = int(mg.PROCESSES[name]["nfinal"]) + 2
        x = np.load(path, mmap_mode="r")
        parts.append(np.asarray(x[:, :4 * k], dtype=np.float64).reshape(len(x), k, 4))
        labels.append(np.asarray(x[:, -1], dtype=np.float64))
    P = mg.to_partonic_frame(np.concatenate(parts)); me = np.concatenate(labels)
    Q = P[:, 0] + P[:, 1]
    sq = np.sqrt(np.clip(Q[:, 0] ** 2 - (Q[:, 1:] ** 2).sum(1), 0.0, None))
    lo, hi, m_finals, pdg_ids, cuts = accept
    ok = (sq >= float(lo)) & (sq <= float(hi))
    if cuts:
        ok &= mg.fiducial_pass_mask(P, np.asarray(m_finals, float), np.asarray(pdg_ids, int), cuts)
    P, sq, me = P[ok], sq[ok], me[ok]
    if len(P) < b:
        raise RuntimeError(f"default share of {source} ({role}): rows [{a}, {b}) asked, {len(P)} accepted")
    return P[a:b], sq[a:b], me[a:b]


def compose(process, sqrts_min, sqrts_max, n_events, role, seed, control_pool_path):
    """(rows (n_events, 5k+1) as the stored pool's, provenance int8) of the entry's default-share mixture."""
    cfg = _cfg(process)
    k = int(cfg["nfinal"]) + 2
    m_list, cuts, pol, _, _ = _sampler_args(cfg, role)
    pool = np.load(control_pool_path, mmap_mode="r")
    prov = provenance(process, sqrts_min, sqrts_max, n_events, role, seed, pool)
    tasks = chunk_plan(process, sqrts_min, sqrts_max, n_events, role, seed)
    f_fwd = float(pol.get("f_fwd", 0.0)); f_def = float(pol["f_default"])
    out, src, start = [], [], 0
    pdg = np.asarray(cfg["pdg_ids"], np.float64)
    for t in tasks:
        n = int(t["count"]); blk = slice(start, start + n)
        a, b = t["default_rows"]
        n_flat = int(round(pol["f_flat"] * n)); n_fwd = int(round(f_fwd * n)); n_def = b - a
        n_bulk = n - n_flat - n_fwd - n_def
        if n_bulk < 0 or abs(float(pol["f_flat"]) + f_fwd + f_def - 1.0) < 1e-9:     # mg.build_mixture_dataset
            n_fwd += n_bulk; n_bulk = 0
        p = prov[blk]; idx = np.arange(start, start + n)
        fwd = idx[p == KIND["fwd"]]
        # the stored chunk's forward share can exceed the catalog_v4 chunk's by its rounding remainder (a share or two
        # of ~1e4): that remainder is read from the bulk rows, and counted as bulk in the shares
        short = max(n_fwd - len(fwd), 0)
        take = np.concatenate([idx[p == KIND["flat"]], fwd[:n_fwd], idx[p == KIND["bulk"]][:n_bulk + short]])
        assert len(idx[p == KIND["flat"]]) == n_flat and len(take) == n - n_def, (t["idx"], len(take), n, n_def)
        Pd, _, med = default_rows_labelled(pol["default_source"], role or "train", t.get("default_seed", seed), (a, b),
                                           (sqrts_min, sqrts_max, m_list, cfg["pdg_ids"], cuts))
        dr = np.concatenate([Pd.reshape(len(Pd), -1), np.tile(pdg, (len(Pd), 1)), med.reshape(-1, 1)], axis=1)
        rows = np.concatenate([np.asarray(pool[take], np.float64), dr])
        kinds = np.concatenate([p[take - start], np.full(len(dr), KIND["default"], np.int8)])
        order = np.random.default_rng([int(t["seed"]), 1]).permutation(len(rows))
        out.append(rows[order]); src.append(kinds[order])
        start += n
    rows = np.concatenate(out)
    assert rows.shape == (n_events, 5 * k + 1), rows.shape
    return rows, np.concatenate(src)


def alphas_of_rows(rows, k, cfg):
    P = rows[:, :4 * k].reshape(len(rows), k, 4)
    q = P[:, 0] + P[:, 1]
    sq = np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(axis=1), 0.0, None))
    return mg.compute_alphas(sq, alphas_mz=float(cfg.get("alphas_mz", 0.118)))


def compose_dir():
    t = os.environ.get("TMPDIR")
    d = os.path.join(t if t and os.path.isdir(t) else tempfile.gettempdir(), "amp_mix_compose")
    os.makedirs(d, exist_ok=True)
    return d


def ensure_composed(process, sqrts_min, sqrts_max, n_events, role, seed, recipe, stored_path, require_cache=True):
    """The path of the composed pool for the stored-pool identity `recipe` (whose path would be stored_path), written
    under compose_dir() with its alpha_s and recipe sidecar; reused within the job."""
    import datagen
    out = os.path.join(compose_dir(), os.path.basename(stored_path))
    wanted = mg.recipe_id(recipe)
    if datagen._is_cached(out, wanted):
        return out
    cfg_c = control_config(process)
    with _Swap(process, cfg_c):
        rec_c = mg.variable_energy_recipe(process, sqrts_min, sqrts_max, n_events, role=role, seed=seed)
        ctrl = mg.recipe_output_path(rec_c, datagen.dest_for_role(role))
    if not datagen._is_cached(ctrl, mg.recipe_id(rec_c)):
        raise RuntimeError(f"load-time mixture of {process} ({role}): its catalog_v4 pool {ctrl} is not built "
                           f"(recipe_id {mg.recipe_id(rec_c)})")
    rows, kinds = compose(process, sqrts_min, sqrts_max, n_events, role, seed, ctrl)
    tmp = out[:-len(".npy")] + f".tmp{os.getpid()}.npy"
    np.save(tmp, rows)
    k = int(_cfg(process)["nfinal"]) + 2
    np.save(out[:-len(".npy")] + ".alphas.npy", alphas_of_rows(rows, k, _cfg(process)))
    os.replace(tmp, out)
    # the sidecar as the stored pool's (its recipe id, the backend's physics block, read from the control pool's: the
    # same backend labelled both), the composition under derived_from (not identity); no manifest upsert
    side = json.load(open(ctrl + ".recipe.json"))
    full = dict(recipe)
    full["physics"] = side.get("physics")
    full["derived_from"] = {"composed_at_load": True, "control_pool": ctrl, "control_recipe_id": mg.recipe_id(rec_c),
                            "shares": {kk: int((kinds == v).sum()) for kk, v in KIND.items()}}
    full["output_file"] = out
    assert mg.recipe_id(full) == wanted
    full["recipe_id"] = wanted
    with open(out + ".recipe.json", "w") as f:
        json.dump(full, f, indent=2, sort_keys=True)
    return out
