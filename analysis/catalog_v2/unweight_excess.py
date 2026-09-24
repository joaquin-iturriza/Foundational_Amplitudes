"""Undo the reference weight in the per-process validation of the reference-weighted arms.

Until commit 5a6a7b7 the multi-process validation loop scored each process through the
training aggregator, so under loss_aggregation=excess the recorded per-process value was
m_p * w_p with w_p = (1/L_ref(n_p)) / mean_q(1/L_ref(n_q)) (the reference weight of the
training loss), and the combined value was GM_p(m_p w_p). This rebuilds w_p from the run's
config (training.excess_reference, particle counts from n_particles.json; a process takes the
reference of its own multiplicity, and a multiplicity without one is an error), divides every
per-process value of every validation in per_process_metrics.json by w_p, recomputes the
combined value as their geometric mean (the validation aggregate, training.val_aggregation),
and writes the corrected record to runs/_unweighted/<run>/per_process_metrics.json, which
census.py, loss_vs_range.py and excess_ratio.py read like any run, at its best checkpoint under
the corrected aggregate.
    python analysis/catalog_v2/unweight_excess.py runs/<run> [runs/<run> ...]
"""
import glob, json, os, sys
import numpy as np
import yaml
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import census as C
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
for path in sys.argv[1:]:
    d = C.metrics(path)
    cfg_path = sorted(glob.glob(os.path.join(path, "**", "config.yaml"), recursive=True))[-1]
    ref = {int(k): float(v) for k, v in yaml.safe_load(open(cfg_path))["training"]["excess_reference"].items()}
    names = [n for n in d["proc_val_losses_no_reg"] if n in NP]
    missing = sorted({NP[n] for n in names} - set(ref))
    if missing:
        raise SystemExit(f"{path}: no excess_reference for multiplicities {missing}")
    inv = np.array([1.0 / ref[NP[n]] for n in names])
    w = dict(zip(names, inv / inv.mean()))
    proc = {n: [x / w[n] for x in d["proc_val_losses_no_reg"][n]] for n in names}
    nval = len(d["val_loss_no_reg"])
    d["proc_val_losses_no_reg"] = proc
    d["val_loss_no_reg"] = [float(np.exp(np.mean(np.log([proc[n][i] for n in names])))) for i in range(nval)]
    out_dir = os.path.join(ROOT, "runs", "_unweighted", os.path.basename(path.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(d, open(os.path.join(out_dir, "per_process_metrics.json"), "w"))
    i, comb, _ = C.at_best(d)
    print(f"{path}: refs {ref}; weights " + ", ".join(f"2->{k-2}: {w[next(n for n in names if NP[n] == k)]:.3g}"
          for k in sorted(ref) if any(NP[n] == k for n in names))
          + f"; best checkpoint #{i + 1} of {nval}, combined {comb:.4g} -> {os.path.relpath(out_dir, ROOT)}")
