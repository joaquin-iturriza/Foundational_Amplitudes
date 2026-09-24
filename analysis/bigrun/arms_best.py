"""The 448-process big run and its arms (docs/results.tex, sec:bigrun) at the best checkpoint, without a
pass/fail threshold (CLAUDE.md, Reported values). Per arm: the geometric mean, median and 90th percentile
of the per-process val_loss_no_reg over all processes, and the median / 90th percentile per catalog
category (alpha_s-scan variants, M_Z-scan variants of the 2->2 processes, the m_t / m_H / M_Z^{4l} scan
families, the plain processes).
    python analysis/bigrun/arms_best.py "label=<run dir>" ["label=<run dir>" ...]   (on the site holding the runs)
A run dir's value is its result.json "proc_val_losses_no_reg" (experiment._result_extra: every process at
the best checkpoint) when present, else its per_process_metrics.json read at the best checkpoint
(analysis/catalog_v2/census.at_best)."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_v2"))
import census as C

def category(n):
    if "__" not in n: return "plain"
    tag = n.split("__", 1)[1]
    if re.fullmatch(r"s\d+", tag): return "alpha_s scan"
    if re.fullmatch(r"mz\d+", tag): return "M_Z scan (2->2)"
    return "m_t / m_H / M_Z^4l families"

def load(path):
    rj = os.path.join(path, "result.json")
    if os.path.exists(rj):
        d = json.load(open(rj))
        if d.get("proc_val_losses_no_reg"):
            return d["proc_val_losses_no_reg"], "result.json"
    return C.at_best(C.metrics(path))[2], "per_process_metrics.json"

CATS = ["alpha_s scan", "M_Z scan (2->2)", "m_t / m_H / M_Z^4l families", "plain"]
arms = [a.split("=", 1) for a in sys.argv[1:]]
print(f"{'arm':28s} {'n':>4s} {'GM all':>9s} {'median':>9s} {'90%':>8s} | " + " | ".join(f"{c}: median / 90%" for c in CATS))
for label, path in arms:
    v, src = load(os.path.join(ROOT, path))
    x = np.array(list(v.values()))
    cells = []
    for c in CATS:
        y = np.array([s for n, s in v.items() if category(n) == c])
        cells.append(f"{np.median(y):.3g} / {np.percentile(y, 90):.3g} ({len(y)})" if len(y) else "-")
    print(f"{label:28s} {len(x):4d} {np.exp(np.mean(np.log(x))):9.3g} {np.median(x):9.3g} {np.percentile(x, 90):8.3g} | "
          + " | ".join(cells) + f"   [{src}]")
