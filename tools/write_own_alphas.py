"""Write <pool>.alphas.npy, each event's alpha_s, for our own (non-MadEvent) pools, so a run with
data.target_physics_orders (D93, the coupling-order factor) can read them: the order axis of the dimensionless round
scores the catalog runs on our one-loop pools and their trees (recipes/catalog_loop_probes.yaml). Nothing is rerun:
our generators set alpha_s from the event alone (mg5_pipeline_final.compute_alphas, one-loop running from alpha_s(M_Z)
at mu = sqrt(s_hat)), so it is recomputed here exactly:
  * a tree pool (mg5_pipeline_final, per_event_alphas true): alpha_s(sqrt(s_hat); alpha_s(M_Z));
  * a stripped one-loop QCD virtual (tools/nlo_virtual_pipeline.py, no alphas_prefactor): its Born is evaluated at the
    fixed alpha_s(M_Z) (nlo_virtual_pipeline: asrun = alphas_mz), so the constant;
  * a loop-induced entry (|M_1|^2, no Born): the per-event running alpha_s, like a tree (nlo_virtual_pipeline's
    loopind branch);
  * a virtual with alphas_prefactor or an electroweak one-loop entry: refused (each would need its own rule; an EW
    entry has alpha_s power 0, so a run with the factor needs none).
alpha_s(M_Z) is the recipe's alphas_mz, else 0.118 (the generators' config["alphas_mz"] and its default). CPU only.
    python tools/write_own_alphas.py recipes/catalog_loop_probes.yaml [--roles train val test]
(run after the pools exist: the missing tree train pools are built by prebuild_recipes.py first)
"""
import argparse, glob, json, os, sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def alphas_of(path, rec, cfg):
    import mg5_pipeline_final as mg
    # the generators' reference: mg5_pipeline_final reads config["alphas_mz"] (default 0.118; not the param card's aS),
    # recorded in the recipe only when it differs from 0.118; a stripped virtual is generated at the default 0.118
    # (datagen passes --alphas-mz only with alphas_prefactor, refused below)
    amz = float(rec.get("alphas_mz", cfg.get("alphas_mz", 0.118)))
    n = np.load(path, mmap_mode="r").shape[0]
    if cfg.get("kind") == "virt":
        from tools.nlo_virtual_pipeline import VIRT_PROCESSES
        vb = VIRT_PROCESSES.get(cfg.get("virt_base"), {})
        if cfg.get("alphas_prefactor") or rec.get("alphas_prefactor") or cfg.get("ew") or vb.get("ew"):
            raise ValueError(f"{path}: a virtual with its own alpha_s rule (prefactor or EW), refused")
        if not (cfg.get("loopind") or vb.get("loopind")):
            if cfg.get("order") is not None:
                raise ValueError(f"{path}: a virtual with an explicit order and no loop-induced flag, refused")
            return np.full(n, amz)
        # loop-induced (|M_1|^2, no Born): evaluated at the per-event running alpha_s(sqrt(s_hat); alpha_s(M_Z)) like a
        # tree (nlo_virtual_pipeline.generate_virt_dataset, the loopind branch), so it takes the tree rule below
    if rec.get("per_event_alphas") is not True:
        raise ValueError(f"{path}: recipe does not say per_event_alphas true ({rec.get('per_event_alphas')!r})")
    npart = len(rec["pdg_ids"])
    mom = np.asarray(np.load(path, mmap_mode="r")[:, :4 * npart], dtype=np.float64).reshape(n, npart, 4)
    q = mom[:, 0] + mom[:, 1]
    sq = np.sqrt(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(axis=1))
    lo, hi = float(rec["sqrts_min"]), float(rec["sqrts_max"])
    # the generator's sqrt(s_hat) lies in the pool's window: a check that the incoming legs are the first two rows
    assert np.all(sq > lo * (1 - 1e-6)) and np.all(sq < hi * (1 + 1e-6)), (path, sq.min(), sq.max(), lo, hi)
    return mg.compute_alphas(sq, alphas_mz=amz)


def main():
    import mg5_pipeline_final as mg
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--roles", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    # val and test frozen in $DATA_DIR, train in the $SCRATCH cache (as tools/recover_alphas.py)
    dirs = {"val": os.environ["DATA_DIR"], "test": os.environ["DATA_DIR"],
            "train": os.path.join(os.environ["SCRATCH"], "amp_data_cache")}
    done = 0
    for p in yaml.safe_load(open(a.spec))["processes"]:
        lo, hi = p["sqrts"]
        for role in a.roles:
            # a shaped pool carries the sampling tag in its name (catalog_v3: _smix_)
            fs = glob.glob(os.path.join(dirs[role], f"{p['name']}_{lo}-{hi}GeV_{role}_*amplitudes.npy"))
            assert len(fs) == 1, (p["name"], role, fs)
            out = fs[0][:-len(".npy")] + ".alphas.npy"
            if os.path.exists(out):
                continue
            rec = json.load(open(fs[0] + ".recipe.json"))
            try:
                a_s = alphas_of(fs[0], rec, mg.PROCESSES[p["name"]])
            except ValueError as e:                # no alpha_s rule: listed, no file (a run with the factor refuses it)
                print(f"[ALPHAS] refused: {e}")
                continue
            np.save(out, a_s)
            done += 1
            print(f"[ALPHAS] {os.path.basename(out)}: {a_s.size} events, {a_s.min():.4f} to {a_s.max():.4f}")
    print(f"WRITE_OWN_ALPHAS_DONE {done} written")


if __name__ == "__main__":
    main()
