#!/usr/bin/env python3
"""The Laplacian positional encodings of every diagram of the catalog (diagrams.tar.gz), computed on this machine, as
evaluate_daniels_catalog/model/lap_pe.json {lap_pe_key(adjacency): (n, k) list}. Run where the model was trained (a
CC-IN2P3 or lxplus GPU node: both reproduce the stored read, MSE 0.2791): a degenerate eigenspace's basis depends on
the machine's LAPACK kernel, and evaluate.py loads this table so a read anywhere uses the trained basis.
    python evaluate_daniels_catalog/lap_pe_table.py [--out <path>] [--k 8]
"""
import argparse, json, os, sys, tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from models.diagram_encoder_llm4lhc import (graph_from_sidecar_diagram, lap_pe_key, laplacian_pe,  # noqa: E402
                                            particle_label_vocab)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "model", "lap_pe.json"))
ap.add_argument("--k", type=int, default=8, help="model.diagram_encoder_llm4lhc.laplacian_pe_dim")
a = ap.parse_args()
vocab, table, n_diag, n_degen = particle_label_vocab(), {}, 0, 0
with tarfile.open(os.path.join(HERE, "diagrams.tar.gz")) as t:
    for m in t:
        if not m.name.endswith(".diagrams.json"):
            continue
        for sub in json.load(t.extractfile(m))["subprocesses"]:
            for d in sub["diagrams"]:
                nt, edges = graph_from_sidecar_diagram(d, sub.get("external"), vocab)
                A = np.zeros((len(nt), len(nt)), dtype=bool)
                for i, j, _ in edges:                        # as load_llm4lhc_diagram_set builds adj
                    A[i, j] = A[j, i] = True
                n_diag += 1
                key = lap_pe_key(A)
                if key not in table:
                    table[key] = laplacian_pe(A, a.k).tolist()
                    deg = A.sum(1)
                    dinv = np.where(deg > 0, 1 / np.sqrt(np.maximum(deg, 1)), 0)
                    ev = np.linalg.eigvalsh(np.eye(len(A)) - dinv[:, None] * A * dinv[None, :])
                    n_degen += bool(np.any(np.diff(ev[1:a.k + 1]) < 1e-6))
json.dump(table, open(a.out, "w"))
print(f"LAP_PE_TABLE {n_diag} diagrams, {len(table)} distinct graphs, {n_degen} with a degenerate eigenspace among "
      f"their PE columns -> {a.out}")
