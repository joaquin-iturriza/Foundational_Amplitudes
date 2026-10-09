"""Fine-tune cells at the transfer study's two fixed points instead of a search (site decision D25; the paper's fine-tune
protocol: the landscape is flat to D = 10^2 and from 10^3 on only lr > 3e-3 and lambda near 1e-6 cost anything, so the
later arms ran lr 1.3e-3 and 2.2e-3 at lambda ~ 3e-10, the W-pole and electroweak-Sudakov arms first).

Per config: the sweep is generated empty if it does not exist yet, the pool candidates nearest the targets are found
(sweep/pick_points.py's rule: distance in log10 lr plus a quarter decade per decade of lambda; candidates already run,
in flight or initial are skipped), and one fixed-HP trial per target is added (generate_sweep --extend --fixed-hp),
not submitted: hand the printed sweep dirs to sweep_manager submit, or leave them to sweep/feed_capped.py.
Runs on the site (site run), where the sweeps live.
    python sweep/fixed_points.py sweep/sweep_config_A.yaml [...] [--target 1.3e-3,3e-10 --target 2.2e-3,3e-10]
prints one JSON line {sweep_name: [hp, hp]} (for expect_tp3.json) and the sweep dirs.
"""
import argparse, json, math, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import siteconf  # noqa: E402
from generate_sweep import load_config, run_generate  # noqa: E402

TARGETS = ["1.3e-3,3e-10", "2.2e-3,3e-10"]


def pick(sweep_dir, targets, w_lambda=0.25):
    st = pickle.load(open(os.path.join(sweep_dir, "dyhpo_state.pkl"), "rb"))
    cand = st["candidates_raw"]
    taken = {int(h) for h, _ in st.get("eval_order", [])} | {int(i) for i in st.get("init_conf_indices", [])}
    taken |= {int(x[0]) if isinstance(x, tuple) else int(x) for x in st.get("in_flight", [])}
    out = []
    for t in targets:
        lr, lam = (float(x) for x in t.split(","))
        dist = lambda c: (abs(math.log10(c["training.lr"] / lr))
                          + w_lambda * abs(math.log10(c["training.regularization_lambda"] / lam)))
        i = min((i for i in range(len(cand)) if i not in taken and i not in out), key=lambda i: dist(cand[i]))
        c = cand[i]
        print(f"{os.path.basename(sweep_dir)}: target lr {lr:.1e} lambda {lam:.0e} -> hp{i}: lr {c['training.lr']:.2e} "
              f"lambda {c['training.regularization_lambda']:.1e}", file=sys.stderr)
        out.append(i)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("configs", nargs="+")
ap.add_argument("--target", action="append", default=None, help="lr,lambda (default: the two fixed points)")
a = ap.parse_args()
chosen, dirs = {}, []
for c in a.configs:
    name = load_config(os.path.abspath(c))["sweep_name"]
    d = os.path.join(siteconf.SWEEP_DIR, name)
    if not os.path.exists(os.path.join(d, "dyhpo_state.pkl")):
        run_generate(c, n_trials=0, submit=False)
    hp = pick(d, a.target or TARGETS)
    run_generate(c, extend=True, submit=False, fixed_hp=hp)
    chosen[name] = hp
    dirs.append(d)
print(json.dumps(chosen))
print("DIRS " + " ".join(dirs))
