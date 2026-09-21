"""Undo the reference weight in the per-process validation of the reference-weighted arms.

Until commit 5a6a7b7 the multi-process validation loop scored each process through the
training aggregator, so under loss_aggregation=excess the recorded per-process value was
m_p * w_p with w_p = (1/L_ref(n_p)) / mean_q(1/L_ref(n_q)) (the reference weight of the
training loss), and the combined value was GM_p(m_p w_p). This rebuilds w_p from the run's
config (training.excess_reference, particle counts from n_particles.json), divides every
per-process value of every "Val loss (combined)" line by w_p, recomputes the combined value
as the geometric mean, and writes the corrected log to runs/_unweighted/<run>/out_0.log so
census.py, loss_vs_range.py and excess_ratio.py read it like any run.
    python analysis/catalog_v2/unweight_excess.py runs/<run> [runs/<run> ...]
"""
import glob, json, os, re, sys
import numpy as np
import yaml
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
PAT = re.compile(r"^(.*Val loss \(combined\): )([0-9.eE+-]+)( \| )(.*)$")
for path in sys.argv[1:]:
    log = sorted(glob.glob(os.path.join(path, "**", "out_0.log"), recursive=True))[-1]
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(log), "config.yaml")))
    ref = {int(k): float(v) for k, v in cfg["training"]["excess_reference"].items()}
    keys = np.array(sorted(ref))
    out_dir = os.path.join(ROOT, "runs", "_unweighted", os.path.basename(path.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)
    w = None; n_lines = 0
    with open(os.path.join(out_dir, "out_0.log"), "w") as fo:
        for line in open(log, errors="replace"):
            m = PAT.match(line.rstrip("\n"))
            if not m:
                fo.write(line); continue
            toks = [t.partition("=") for t in m.group(4).split(", ")]
            names = [k.strip() for k, _, _ in toks]
            if w is None:
                inv = np.array([1.0 / ref[int(keys[np.argmin(np.abs(keys - NP[n]))])] for n in names])
                w = dict(zip(names, inv / inv.mean()))
                print(f"{path}: refs {ref}; weights by multiplicity "
                      + ", ".join(f"2->{k-2}: {w[next(n for n in names if NP[n]==k)]:.3g}" for k in keys))
            vals = {k.strip(): float(v) / w[k.strip()] for k, _, v in toks}
            comb = float(np.exp(np.mean(np.log(np.clip(list(vals.values()), 1e-10, None)))))
            fo.write(f"{m.group(1)}{comb:.4f}{m.group(3)}" + ", ".join(f"{k}={v:.4f}" for k, v in vals.items()) + "\n")
            n_lines += 1
    last = vals
    print(f"  {n_lines} validation lines rewritten -> {os.path.relpath(out_dir, ROOT)}; final combined {comb:.4f}")
    for k in keys:
        sel = [n for n in names if NP[n] == k]
        raw = [float(v) for kk, _, v in toks if kk.strip() in sel]
        cor = [last[n] for n in sel]
        print(f"  2->{k-2}: n={len(sel)}  above 0.05 as recorded {sum(v > 0.05 for v in raw)}  corrected {sum(v > 0.05 for v in cor)}"
              f"  median recorded {np.median(raw):.3g} corrected {np.median(cor):.3g}")
    print(f"  all: above 0.05 as recorded {sum(float(v) > 0.05 for _, _, v in toks)}  corrected {sum(v > 0.05 for v in last.values())}")
