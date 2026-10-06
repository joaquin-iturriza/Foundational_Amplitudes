"""Seed repeats of the transfer study's fine-tune cells (docs/results.tex sec:ladder-open, plan step 5): each cell's
best trial re-run at its full HPs with extra seeds, so the final curves are a seed mean with a band, not a search
minimum. A trial's record holds its candidate index only; the HPs are rebuilt from the sweep config's own candidate
pool (dyhpo_sampler._sample_candidates with the config's n_candidates and seed: the same draw every site made). A
cell's best is read as the figures read it (analysis/transfer/cells.py: best over the sweep and its _002
continuation; a 32k cell over its chosen points). Writes chunk files of run lines for scripts/job_seed_batch.sh, one
run per line: <config> <run name> <steps> key=value ...
    python3 sweep/seed_list.py --families tp3_r1fte ... --ks 2 3 4 5 6 [--horizon32] --seeds 1 2 --chunk 40 --out DIR
"""
import argparse, os, sys
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "sweep"), os.path.join(ROOT, "analysis", "transfer")]
from dyhpo_sampler import _sample_candidates  # noqa: E402
import cells  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--families", nargs="+", required=True)
ap.add_argument("--probes", nargs="+", default=list(cells.ARM))
ap.add_argument("--ks", nargs="+", type=int, required=True)
ap.add_argument("--skip", nargs="*", default=[], help="cells probe:k to leave out (final at another horizon)")
ap.add_argument("--horizon32", action="store_true", help="the 32k cells (<fam>32k_<probe>_d<k>, chosen points)")
ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
ap.add_argument("--chunk", type=int, default=40)
ap.add_argument("--out", required=True)
ap.add_argument("--prefix", default="chunk")
a = ap.parse_args()

pools, lines, missing = {}, [], []
skip = {(c.split(":")[0], int(c.split(":")[1])) for c in a.skip}
for fam in a.families:
    for p in a.probes:
        for k in a.ks:
            name = f"{fam}{'32k' if a.horizon32 else ''}_{p}_d{k}"
            if (p, k) in skip or not os.path.exists(os.path.join(ROOT, "sweep", f"sweep_config_{name}.yaml")):
                continue
            tr = [(n, t) for n in (name, name + "_002") for t in cells.S.get(n, []) if t.get("val_loss") is not None
                  and t.get("prepd_std") and (not a.horizon32 or name in cells.ALL_TRIALS32
                                              or t["hp"] in cells.CHOSEN32.get(name, []))]
            if not tr:
                missing.append(name)
                continue
            src, b = min(tr, key=lambda nt: nt[1]["val_loss"] * nt[1]["prepd_std"] ** 2)
            cfg_path = os.path.join("sweep", f"sweep_config_{src}.yaml")
            if cfg_path not in pools:
                c = yaml.safe_load(open(os.path.join(ROOT, cfg_path)))
                d = c.get("dyhpo", {})
                pools[cfg_path] = (c, _sample_candidates(c["search_space"], d.get("n_candidates", 300), d.get("seed", 42)))
            c, cand = pools[cfg_path]
            assert b["hp"] < len(cand), f"{src}: hp{b['hp']} is past the pre-sampled pool (an --extend draw)"
            hp = cand[b["hp"]]
            if "training.lr" in hp and b.get("lr"):
                assert abs(hp["training.lr"] / b["lr"] - 1) < 1e-4, f"{src}: hp{b['hp']} lr {hp['training.lr']} != {b['lr']}"
            T = c["fidelity_schedule"]["t_steps"][-1]
            for s in a.seeds:
                lines.append(f"{cfg_path} {name}_seed{s} {T} " + " ".join(f"{kk}={v}" for kk, v in hp.items()) + f" seed={s}")
os.makedirs(a.out, exist_ok=True)
for i in range(0, len(lines), a.chunk):
    open(os.path.join(a.out, f"{a.prefix}_{i // a.chunk:03d}.txt"), "w").write("\n".join(lines[i:i + a.chunk]) + "\n")
print(f"{len(lines)} runs in {(len(lines) + a.chunk - 1) // a.chunk} chunks; cells without a result: {len(missing)}")
for m in missing:
    print("  no result:", m)
