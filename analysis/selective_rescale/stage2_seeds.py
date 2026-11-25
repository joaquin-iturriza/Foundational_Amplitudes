"""Plan selective_rescale (D552) stage 2: the chosen trial of each driven search (sweeps_local/<sweep>, lowest val_loss of
its 16 trials; one search shares its data, so its prepd_std, so this is the lowest MSE(log|M|^2)) and the chunk files that
rerun it at seeds 43 and 44 (run_fixed_hp.py through job_seed_batch.sh, the stage-1 format).

    python3 analysis/selective_rescale/stage2_seeds.py   -> seed_chunks/stage2/<run>.txt, data/stage2_chosen.json
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SEARCHES = ["zng_1e4_softbelow_x0p3", "zng_1e4_eventgate_x0p1", "zng_1e5_softbelow_x0p3", "zng_1e5_eventgate_x0p1",
            "zng_1e5_off", "zng_1e5_eps1e-8", "uug_1e4_softbelow_x0p1", "uug_1e5_softbelow_x0p1"]
SEEDS = (43, 44)


def best_params(summary):
    """The 'Best params:' block of a drive.py summary.txt as ordered key=value strings."""
    lines = open(summary).read().split("Best params:\n", 1)[1].split("\n\n", 1)[0].splitlines()
    return [l.strip().replace(" = ", "=") for l in lines if " = " in l]


def main():
    out, chosen = os.path.join(HERE, "seed_chunks", "stage2"), {}
    os.makedirs(out, exist_ok=True)
    for s in SEARCHES:
        sweep = f"selres_{s}_b2048"
        d = os.path.join(ROOT, "sweeps_local", sweep)
        res = [json.load(open(f)) for f in glob.glob(os.path.join(d, "results", "*.json"))]
        assert len(res) == 16 and all(r.get("val_loss") is not None for r in res), (sweep, len(res))
        b = min(res, key=lambda r: r["val_loss"])
        hp = best_params(os.path.join(d, "summary.txt"))
        top = open(os.path.join(d, "summary.txt")).read().split("Top-10 results", 1)[1].split("hp_", 1)[1][:4]
        assert int(top) == b["hp_idx"], (sweep, top, b["hp_idx"])   # summary's best params are this trial's
        chosen[s] = {"sweep": sweep, "hp_idx": b["hp_idx"], "val_loss": b["val_loss"], "run_dir": b["run_dir"], "hp": hp}
        for seed in SEEDS:
            run = f"selres_s2_{s}_s{seed}"
            with open(os.path.join(out, run + ".txt"), "w") as f:
                f.write(" ".join([f"sweep/sweep_config_{sweep}.yaml", run, "1953", *hp, f"seed={seed}"]) + "\n")
    json.dump(chosen, open(os.path.join(HERE, "data", "stage2_chosen.json"), "w"), indent=1)
    print(json.dumps({s: {"hp_idx": c["hp_idx"], "val_loss": c["val_loss"]} for s, c in chosen.items()}))


if __name__ == "__main__":
    main()
