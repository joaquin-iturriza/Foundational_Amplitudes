"""Does the DyHPO sampler learn, the way sweeps really run it (a save and a load between every trial, one process per
trial)? On a synthetic objective at our scale (loss 1e-6..1e-3, a log-quadratic bowl in lr and lambda over the
transfer study's search space), the median loss of the guided picks is compared with random picks from the same pool,
and a replay of a real sweep's observations shows where its next picks go. Before the 2026-10-10 fix the surrogate
was never kept between processes and the objective sat below the GP noise floor, so guided picks were no better than
random. CPU, a few minutes, on a login node:
    python sweep/test_sampler_learns.py [--trials 30] [--startup 5] [--replay <sweep_dir>]
"""
import argparse, math, os, pickle, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep.dyhpo_sampler import DyHPOSampler  # noqa: E402

SPACE = [{"name": "training.lr", "type": "float_log", "low": 1e-3, "high": 1e-2},
         {"name": "training.regularization_lambda", "type": "float_log", "low": 1e-10, "high": 1e-6},
         {"name": "training.cosanneal_warmup_frac", "type": "float_uniform", "low": 0.05, "high": 0.2},
         {"name": "training.cosanneal_eta_min", "type": "float_log", "low": 1e-10, "high": 1e-7},
         {"name": "ema", "type": "categorical", "choices": ["false", "true"]},
         {"name": "training.ema_decay", "type": "float_uniform", "low": 0.9, "high": 0.999}]


def loss(hp):
    """A bowl in log lr (optimum 2.5e-3) and log lambda (optimum 1e-9), floor 2e-6, three decades deep."""
    x = (math.log10(hp["training.lr"]) - math.log10(2.5e-3)) / 0.5
    y = (math.log10(hp["training.regularization_lambda"]) + 9) / 2
    return 2e-6 * 10 ** (1.5 * (x * x + y * y))


def run(trials, startup, seed):
    d = tempfile.mkdtemp(prefix="zz_sampler_test_")
    path = os.path.join(d, "state.pkl")
    s = DyHPOSampler(SPACE, {"t_steps": [8000]}, n_candidates=200, seed=seed, output_path=d, n_startup=startup)
    s.save(path)
    got = []
    for _ in range(trials):
        s = DyHPOSampler.load(path, d, force_cpu=True)          # a new process per trial, as in a sweep
        hp_idx, hp, t = s.suggest()
        s.save(path)
        s = DyHPOSampler.load(path, d, force_cpu=True)
        v = loss(hp)
        s.observe(hp_idx, t, v)
        s.save(path)
        got.append(v)
    pool = [loss(c) for c in s.candidates_raw]
    return got, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--startup", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--replay", help="a sweep dir: load its state and print the next 3 guided picks (state not saved)")
    a = ap.parse_args()
    for seed in range(a.seeds):
        got, pool = run(a.trials, a.startup, 1000 + seed)
        guided = got[a.startup:]
        rng = np.random.default_rng(seed)
        rand = [np.median(rng.choice(pool, len(guided), replace=False)) for _ in range(200)]
        print(f"seed {seed}: guided picks median {np.median(guided):.3g}, best {min(got):.3g} | random picks median "
              f"{np.median(rand):.3g} (5-95%: {np.percentile(rand, 5):.3g}-{np.percentile(rand, 95):.3g}) | "
              f"pool best {min(pool):.3g}")
    if a.replay:
        st = os.path.join(a.replay, "dyhpo_state.pkl")
        s = DyHPOSampler.load(st, tempfile.mkdtemp(prefix="zz_sampler_replay_"), force_cpu=True)
        obs = sorted(((v, h) for h, d in s._val_loss_history.items() for v in d.values()))
        print("observed (val_loss, hp):", [(f"{v:.3g}", h) for v, h in obs])
        best = {k: f"{v:.3g}" if isinstance(v, float) else v for k, v in s.candidates_raw[obs[0][1]].items()}
        print("best observed:", best)
        for _ in range(3):
            h, hp, _ = s.suggest()
            print("next pick", h, {k: f"{v:.3g}" if isinstance(v, float) else v for k, v in hp.items()})


if __name__ == "__main__":
    main()
