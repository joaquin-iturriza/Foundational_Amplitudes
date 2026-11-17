#!/usr/bin/env python3
"""Predict |M|^2 with our "dimensionless + coupling orders" network (trained on the arXiv:2606.23791 catalog) on events
of any process of that catalog, every step done by the repository's own code: the particle features from the PDG ids,
the coupling orders of the process, the dimensionless momenta, the physics factor and standardization of the target,
the forward pass, and back to log|M|^2 (and the network's sigma). It runs tools/zero_shot_parent.py, the scorer every
zero-shot read of ours used, on a pool made from your arrays.

    python evaluate_daniels_catalog/evaluate.py --process "b b~ > h h" --events events.npz --out preds.npz [--n 2000]

--process: the process as in the catalog's process_catalog_sm.json ("b b~ > h h") or its folder name (bbx_hh).
--events: .npz with momenta (N, n, 4) in GeV as (E, px, py, pz), the two incoming partons first then the outgoing in
  the catalog's order; pdg (N, n) or (n,); alphas (N,), each event's alpha_s (MadEvent's AQCDUP; needed unless the
  process has no alpha_s); optionally amp (N,), the true |M|^2. Or one of our pools, a .npy of rows [momenta flat
  (n*4), PDG ids (n), |M|^2], with --alphas <file.npy>.
--out: .npz with logM2_pred (log|M|^2), M2_pred (|M|^2), sigma_logM2 (the network's sigma of log|M|^2), and when the
  truth was given logM2_true, M2_true; one row per event, in the input's order (its first --n events).
Model: evaluate_daniels_catalog/model (runs/catalog_norm_short/DlessInOrd/sigma_stage on CC-IN2P3: 5 epochs of 1e4
events per process on the catalog's 1352 training processes, the mean frozen at its best checkpoint, then a sigma fit).
CPU is fine for a few thousand events; a GPU is used when there is one.
"""
import argparse, json, os, runpy, sys, tarfile, tempfile

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def folder_of(process):
    """the catalog's folder name of a process given as its string or its folder"""
    procs = json.load(open(os.path.join(ROOT, "analysis", "catalog_daniel", "classes_default.json")))["processes"]
    by_str = {" ".join(p["process"].split()): p["folder"] for p in procs}
    folders = {p["folder"] for p in procs}
    f = process[3:] if process.startswith("dc_") else process
    if f in folders:
        return f
    s = " ".join(process.split())
    if s in by_str:
        return by_str[s]
    sys.exit(f"evaluate: {process!r} is not a process of the catalog (process_catalog_sm.json string or folder name)")


def rows_from(events, alphas):
    """(rows [momenta flat, pdg, amp], alphas or None, has_truth) from the input file"""
    if events.endswith(".npy"):
        rows = np.load(events).astype(np.float64)
        return rows, (np.load(alphas) if alphas else None), True
    z = np.load(events)
    mom = np.asarray(z["momenta"], np.float64)
    N, n, _ = mom.shape
    pdg = np.broadcast_to(np.asarray(z["pdg"], np.float64), (N, n))
    amp = np.asarray(z["amp"], np.float64) if "amp" in z.files else None
    a_s = np.asarray(z["alphas"], np.float64) if "alphas" in z.files else (np.load(alphas) if alphas else None)
    return np.concatenate([mom.reshape(N, 4 * n), pdg, (np.ones(N) if amp is None else amp)[:, None]], axis=1), a_s, \
        amp is not None


def event_keyed_chunks(com, seed, uid, rs=None, mom_div=None, chunk=1 << 16, dtype=np.float32):
    """keyed_aug.model_momenta_chunks with each event's Lorentz augmentation keyed on the event itself (the sha256 of
    its centre-of-mass momenta) instead of on its pool file and its place there: the same draw distribution as in
    training, so a prediction is a function of (event, seed) alone. `uid` is ignored."""
    import hashlib
    import keyed_aug
    com = np.asarray(com)
    for i in range(0, len(com), chunk):
        c = com[i:i + chunk]
        uids = np.array([int.from_bytes(hashlib.sha256(np.ascontiguousarray(r, np.float32).tobytes()).digest()[:8],
                                        "little") for r in c], dtype=np.uint64)
        x = keyed_aug.apply_components(keyed_aug.trafo_components(seed, uids, np.zeros(len(c), np.int64)), c).astype(dtype)
        if rs is not None:
            x = (x.astype(np.float64) / np.asarray(rs[i:i + chunk])[:, None, None]).astype(dtype)
        yield x if mom_div is None else x / mom_div


def score(name, pool, n_val, n_score, model, seed=42):
    """tools/zero_shot_parent.py's read of the first n_score events of the val pool `pool` (n_val rows) of process
    `name` by the network in `model`: the scorer's .npz (t, q, s, logF, ... in the scorer's shuffled order). The
    Lorentz augmentation the network was trained with is drawn per event from (seed, the event's momenta)
    (event_keyed_chunks), and the frames' degeneracy noise is a hash of each vector (rebuild_run.deterministic_frame_noise),
    so a prediction depends on the event and the seed alone, never on the file it came in or its place there."""
    tmp = tempfile.mkdtemp(prefix="evaluate_daniels_catalog_")
    # the network's own files, its config pointed at this checkout (training catalog, diagram sidecars)
    from omegaconf import OmegaConf, open_dict
    parent = os.path.join(tmp, "model")
    os.makedirs(parent)
    for f in ("data_stats.json", "particle_tokenizer.json", "models"):
        os.symlink(os.path.join(os.path.abspath(model), f), os.path.join(parent, f))
    diagrams = os.path.join(tmp, "diagrams")
    with tarfile.open(os.path.join(HERE, "diagrams.tar.gz")) as t:
        t.extractall(diagrams, filter="data")
    cfg = OmegaConf.load(os.path.join(model, "config.yaml"))
    with open_dict(cfg):
        cfg.data.processes_file = os.path.join(ROOT, "recipes", "catalog_daniel_train.yaml")
        cfg.model.diagrams_dir = diagrams
        cfg.run_dir = parent
        cfg.seed = int(seed)
        cfg.data.unaugmented_roles = ["train", "val"]    # uid None reaches model_momenta_chunks: keyed per event below
    OmegaConf.save(cfg, os.path.join(parent, "config.yaml"))
    recipe = os.path.join(tmp, name + ".yaml")
    yaml.safe_dump({"processes": [{"name": name, "sqrts": [13000, 13000], "n_train": 0, "n_val": n_val, "n_test": 0}]},
                   open(recipe, "w"))
    # every read of this pool comes from the file above, never from the generator or a site's data directory
    import datagen
    real = datagen.ensure_split_set

    def split(specs, role, seed, dest_dir=None, require_cache=False):
        mine = [s for s in specs if s["process"] == name]
        out = {name: pool} if mine else {}
        rest = [s for s in specs if s["process"] != name]
        if rest:
            out.update(real(rest, role=role, seed=seed, dest_dir=dest_dir, require_cache=require_cache))
        return out
    datagen.ensure_split_set = split
    import keyed_aug
    from rebuild_run import deterministic_frame_noise
    keyed_aug.model_momenta_chunks = event_keyed_chunks
    deterministic_frame_noise()
    os.environ.setdefault("SCRATCH", tmp)
    save = os.path.join(tmp, "preds")
    os.makedirs(save)
    sys.argv = ["zero_shot_parent.py", "--parent", parent, "--save-dir", save, "--split", "val", "--signed-magnitude",
                "--eval-subsample", str(n_score), "--train-subsample", str(n_score),
                "--allow-physics-mismatch", "evaluate_daniels_catalog: events given by the user, no pool sidecar", recipe]
    try:
        runpy.run_path(os.path.join(ROOT, "tools", "zero_shot_parent.py"), run_name="__main__")
    finally:
        datagen.ensure_split_set = real
    return np.load(os.path.join(save, name + ".npz"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--process", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--alphas", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None, help="the first n events only (default all)")
    ap.add_argument("--model", default=os.path.join(HERE, "model"))
    ap.add_argument("--seed", type=int, default=42, help="the seed of the per-event Lorentz augmentation draws")
    a = ap.parse_args()

    name = "dc_" + folder_of(a.process)
    rows, a_s, has_truth = rows_from(a.events, a.alphas)
    n = len(rows) if a.n is None else min(a.n, len(rows))
    rows = rows[:n].copy()
    truth = np.log(rows[:, -1]) if has_truth else None
    # the loader reads a role in a shuffled order and the scorer keeps it; each event gets a distinct stand-in |M|^2,
    # exp(-1 + 2 i / n), by which its prediction is put back in the input's order. The target only sets the probe's own
    # standardization, a representation the scorer inverts exactly, so the prediction does not depend on it
    tag = -1.0 + 2.0 * np.arange(n) / n
    rows[:, -1] = np.exp(tag)
    tmp = tempfile.mkdtemp(prefix="evaluate_daniels_catalog_")
    # the pool as the loader reads it, its per-event alpha_s beside it (data.target_physics_orders)
    pool = os.path.join(tmp, f"{name}_13000-13000GeV_val_lhe_amplitudes.npy")
    np.save(pool, rows)
    if a_s is not None:
        np.save(pool[:-len(".npy")] + ".alphas.npy", np.asarray(a_s, np.float64)[:n])
    z = score(name, pool, n, n, a.model, seed=a.seed)
    # q and t are log(|M|^2 F), F the per-event physics factor of the target (logF): log|M|^2 = q - logF
    logF = z["logF"].astype(np.float64) if z["logF"].size else 0.0
    t = z["t"].astype(np.float64) - logF
    pos = np.rint((t + 1.0) * n / 2.0).astype(int)              # the stand-in tag of each scored event
    assert len(t) == n and np.array_equal(np.sort(pos), np.arange(n)) and np.abs(t - tag[pos]).max() < 0.25 / n, \
        "the scored events do not map back one to one onto the input's"
    q = np.empty(n); s = np.empty(n)
    q[pos] = z["q"].astype(np.float64) - logF
    s[pos] = z["s"].astype(np.float64)
    out = {"logM2_pred": q, "M2_pred": np.exp(q), "sigma_logM2": s}
    if has_truth:
        out.update(logM2_true=truth, M2_true=np.exp(truth))
    np.savez(a.out, **out)
    msg = f"evaluate: {name}, {n} events -> {a.out}"
    if has_truth:
        msg += f"; MSE of log|M|^2 {float(np.mean((q - out['logM2_true']) ** 2)):.4g}"
    print(msg)


if __name__ == "__main__":
    main()
