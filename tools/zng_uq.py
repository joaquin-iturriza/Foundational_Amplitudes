#!/usr/bin/env python3
"""zng_uq_benchmark: the Zenodo 16793011 q qbar -> Z + n g datasets (lloca-experiments-uq, branch uq) in the KIT
benchmark's HDF5 layout (tools/kit_hardps.py), so a run reads them as kit_uq_z<g..>__me2 through scripts/job_kit.sh.

    python tools/zng_uq.py convert <src_dir> [--out $DATA_DIR/zng_uq]   (src_dir/<proc>/<proc>_<i>.npy)
    python tools/zng_uq.py crosscheck [--n 100]                          (our MadGraph |M|^2 on their events)

The source: rows of flat four-momenta (d, d~, Z, gluons; lab frame, beams along z) and the tree |M|^2 of a MadGraph
standalone at fixed alpha_s = 0.13 (their config data.alpha_s; the flavour d from their target_norm.zng_process).
Their split (experiments/amplitudes/experiment_uq.py, init_data, data.mode=single): the files <proc>_0, _1, _2
concatenated, torch.randperm(N, generator=torch.Generator().manual_seed(data_seed + 1)), data_seed 0, cut into
train 5e5, val 1e5, calib 1e6, test 1e6 in that order (torch 2.1.2 and their 2.9.1 give the same permutation).
Written per multiplicity to <out>/uq_<proc>.h5:
  bulk_train/bulk  their train then their val events (kit_hardps: train = the first n_train, val = the last n_val)
  bulk_test/bulk   their test events
  split_idx/<role> the indices into the concatenated files, to compare with their runs' split_indices_<i>.npz
Momenta are stored as given (their loader resets each energy to its mass shell with m = 1e-5 GeV for massless legs,
a relative change of order 1e-10 at their pT > 20 GeV). The metadata's alpha_s 0.13 is what the pool's per-event
alpha_s sidecar carries, so data.target_physics_orders strips the file's own coupling.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROCS = {"zg": 1, "zgg": 2, "zggg": 3, "zgggg": 4}
COUNTS = (("train", 500000), ("val", 100000), ("calib", 1000000), ("test", 1000000))
FILES = (0, 1, 2)
DATA_SEED = 0
ALPHA_S = 0.13
MZ = 91.188


def metadata(n):
    parts = [{"pdg": 1, "mass": 0.0}, {"pdg": -1, "mass": 0.0}, {"pdg": 23, "mass": MZ}] + [{"pdg": 21, "mass": 0.0}] * n
    return {"particles": parts, "alpha_s": ALPHA_S,
            "targets": {"me2": {"n_loops": 0, "alpha_s_power": n, "alpha_power": 1}},
            "source": "Zenodo 16793011 (lloca-experiments-uq branch uq), files " +
                      ", ".join(f"z{'g' * n}_{i}" for i in FILES),
            "split": {"data_seed": DATA_SEED, "counts": dict(COUNTS), "torch_randperm_seed": DATA_SEED + 1}}


def split_idx(N):
    import torch
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(DATA_SEED + 1)).numpy()
    out, i = {}, 0
    for role, c in COUNTS:
        out[role] = perm[i:i + c]
        i += c
    assert i <= N, f"split {COUNTS} needs {i} events, the files hold {N}"
    return out


def find(src, proc, i):
    hits = glob.glob(os.path.join(src, "**", f"{proc}_{i}.npy"), recursive=True)
    assert len(hits) == 1, f"{proc}_{i}.npy: {hits or 'not found'} under {src}"
    return hits[0]


def convert(src, out):
    import h5py
    os.makedirs(out, exist_ok=True)
    for proc, n in PROCS.items():
        dst = os.path.join(out, f"uq_{proc}.h5")
        if os.path.exists(dst):
            print(f"{dst}: exists, kept")
            continue
        rows = np.concatenate([np.load(find(src, proc, i)) for i in FILES], axis=0).astype(np.float64)
        P = 3 + n
        assert rows.shape[1] == 4 * P + 1, f"{proc}: {rows.shape[1]} columns, expected {4 * P + 1}"
        assert np.all(rows[:, -1] > 0), f"{proc}: a non-positive |M|^2"
        idx = split_idx(len(rows))
        mom, y = rows[:, :-1].reshape(len(rows), P, 4), rows[:, -1]
        tmp = dst + ".tmp"
        with h5py.File(tmp, "w") as f:
            f.attrs["metadata"] = json.dumps(metadata(n))
            for grp, sel in (("bulk_train/bulk", np.concatenate([idx["train"], idx["val"]])),
                             ("bulk_test/bulk", idx["test"])):
                g = f.create_group(grp)
                g["momenta"] = mom[sel]
                g["me2"] = y[sel]
                g["log10_me2"] = np.log10(y[sel])
            for role, v in idx.items():
                f[f"split_idx/{role}"] = v
        os.replace(tmp, dst)
        print(f"{dst}: {len(rows)} events, train {len(idx['train'])} val {len(idx['val'])} test {len(idx['test'])}")


def crosscheck(n_events, out_json):
    """Our MadGraph standalone (mg5_pipeline_final PROCESSES ddbar_Z<g..>, sm card) at each of the first n_events
    test events of every file, at alpha_s 0.13: the ratio ours / theirs per event (1 when the conventions agree; a
    constant other than 1 is a convention, e.g. the electroweak scheme; a spread is a mismatch), and the same for u u~
    at their momenta."""
    import datagen
    import mg5_pipeline_final as mg
    from tools import kit_hardps as kh
    res = {}
    for proc, n in [(p, n) for p, n in PROCS.items()] + [(f"{p}_as_uubar", n) for p, n in PROCS.items()]:
        # uubar: our u u~ > Z + n g at the same momenta; |M|^2 of q q~ > Z + gluons is (g_L^2 + g_R^2) times a function
        # of the momenta alone, so ours(u) / theirs(d) must be one constant per process (what a u-flavour model's
        # prediction is shifted by on their events)
        name = f"{'uubar' if proc.endswith('_as_uubar') else 'ddbar'}_Z{'g' * n}"
        r = kh.stratum_rows(f"uq_{proc.split('_as_')[0]}", "me2", "bulk_test", "bulk")[:n_events]
        mom = r[:, :4 * (3 + n)].reshape(len(r), 3 + n, 4)
        sa = datagen.ensure_backend(name)
        backend, _, drv, _ = mg.detect_compiled_backend(sa)
        assert backend == "cpp" and drv, f"{name}: no C++ driver under {sa}"
        perm = mg.row_to_slot_perm(mg.PROCESSES[name]["pdg_ids"], mg.PROCESSES[name]["mg5_generate"])
        with mg.CppDriverPipe(drv, sa) as pipe:
            ours = np.asarray(pipe.compute([(m, None) for m in mom], perm=perm, alphas=np.full(len(mom), ALPHA_S)))
        ratio = ours / r[:, -1]
        res[proc] = {"n": len(r), "median": float(np.median(ratio)), "min": float(ratio.min()),
                     "max": float(ratio.max()), "rel_spread": float(ratio.max() / ratio.min() - 1)}
        print(f"CROSSCHECK {proc}: ours/theirs median {res[proc]['median']:.6g} "
              f"[{res[proc]['min']:.6g}, {res[proc]['max']:.6g}] over {len(r)} events")
    json.dump(res, open(out_json, "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("src")
    c.add_argument("--out", default=os.path.join(os.environ.get("DATA_DIR", "."), "zng_uq"))
    x = sub.add_parser("crosscheck")
    x.add_argument("--n", type=int, default=100)
    x.add_argument("--out", default=os.path.join(os.environ.get("DATA_DIR", "."), "zng_uq", "crosscheck.json"))
    a = ap.parse_args()
    if a.cmd == "convert":
        convert(a.src, a.out)
    else:
        crosscheck(a.n, a.out)
