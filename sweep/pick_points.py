"""The candidate of a sweep's DyHPO pool nearest each chosen HP point, to run with generate_sweep --extend --fixed-hp.

A sweep's candidates are drawn once from its seed, so a chosen point is run as the pool candidate closest to it in
(log10 lr, log10 lambda), lambda weighted by --w-lambda per decade; the knobs that move nothing (warm-up, eta_min, EMA;
docs/results.tex fig:finetune_hp) stay as the candidate has them. Candidates this sweep has observed, started, or
that --exclude names (trials of the same sweep run on another site) are skipped, and no candidate is picked twice.
    python sweep/pick_points.py <sweep_dir> --target 1.2e-3,3e-10 --target 1.8e-3,3e-10 [--exclude 186 110 100]
prints the chosen indices, one per target, on stdout, and what each is on stderr.
"""
import argparse, math, os, pickle, sys

ap = argparse.ArgumentParser()
ap.add_argument("sweep_dir")
ap.add_argument("--target", action="append", required=True, help="lr,lambda")
ap.add_argument("--exclude", type=int, nargs="*", default=[])
ap.add_argument("--w-lambda", type=float, default=0.25, help="weight of a decade of lambda against a decade of lr")
a = ap.parse_args()
st = pickle.load(open(os.path.join(a.sweep_dir, "dyhpo_state.pkl"), "rb"))
cand = st["candidates_raw"]
taken = {int(h) for h, _ in st.get("eval_order", [])} | {int(i) for i in st.get("init_conf_indices", [])}
taken |= {int(x[0]) if isinstance(x, tuple) else int(x) for x in st.get("in_flight", [])} | set(a.exclude)
out = []
for t in a.target:
    lr, lam = (float(x) for x in t.split(","))
    dist = lambda c: (abs(math.log10(c["training.lr"] / lr))
                      + a.w_lambda * abs(math.log10(c["training.regularization_lambda"] / lam)))
    i = min((i for i in range(len(cand)) if i not in taken and i not in out), key=lambda i: dist(cand[i]))
    c = cand[i]
    print(f"{os.path.basename(a.sweep_dir.rstrip('/'))}: target lr {lr:.1e} lambda {lam:.0e} -> hp{i}: "
          f"lr {c['training.lr']:.2e} lambda {c['training.regularization_lambda']:.1e} ema {c.get('ema')}", file=sys.stderr)
    out.append(i)
print(" ".join(map(str, out)))
