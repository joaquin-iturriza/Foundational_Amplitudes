"""Submit the null-parent 2x2 (design 2026-11-12, after the suspects batch): through the fine-tune configs at each
cell's stage-2 scratch best HPs (lr_scale = scratch lr / the configs' fixed training.lr, layer_decay 1), arm null
(fine_tune.shrink_perturb=[0,1]: every parameter a fresh init) and arm pre (the parent's weights), 16 runs on CC.
--smoke: one tree and one V/B null run at D=316, 50 steps. --dry: print only."""
import json, os, re, subprocess, sys
import yaml

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
K = (5, 6, 7, 8)
SCR_HP = {"epem_ttx": dict(zip(K, (1605, 844, 640, 657))), "ee_ttbar_nlo": dict(zip(K, (1663, 713, 358, 816)))}
ARMS = [("null", 9107, {"fine_tune.shrink_perturb": "[0.0,1.0]"}), ("pre", 9108, {})]


def hp_of(sweep, idx):
    tr = json.load(open(f"{R}/sweeps_local/{sweep}/driver_state.json"))["trials"]
    tr = tr.values() if isinstance(tr, dict) else tr
    return dict(next(v for v in tr if v.get("hp_idx") == idx)["hp"])


jobs = []
for t in ("epem_ttx", "ee_ttbar_nlo"):
    for k in K:
        cfg = f"dsave_diag_{'hmap' if t == 'epem_ttx' else 'parscale'}_ft_{t}_d{k}_t8500"
        lr0 = yaml.safe_load(open(f"{R}/sweep/sweep_config_{cfg}.yaml"))["fixed_params"]["training.lr"]
        hp = hp_of(f"dsave_s2_scr_{t}_d{k}_t8500", SCR_HP[t][k])
        hp["fine_tune.lr_scale"] = hp.pop("training.lr") / lr0
        hp["fine_tune.layer_decay"] = 1.0
        for name, idx, ch in ARMS:
            jobs.append((cfg, idx, {**hp, **ch}, f"nullparent {name} {t} d{k} (scratch hp{SCR_HP[t][k]})",
                         "00:25:00" if k <= 6 else "00:32:00", 8500))
if "--smoke" in sys.argv:
    jobs = [(c, 9197, hp, f"nullparent smoke {c} 50 steps", "00:15:00", 50)
            for c, i, hp, n, _, _ in jobs if i == 9107 and "_d5_" in c]
out = []
for cfg, idx, hp, note, lim, T in jobs:
    cmd = ["site", "submit", "ccin2p3", "FA", "sweep/trial_job.sh", "--item",
           "pretrain_data_saving", "--note", note, f"--hdr=--time={lim}", "--hdr=--mem=16G", "--hdr=--cpus-per-task=4",
           "--", "--sweep-config", f"sweep/sweep_config_{cfg}.yaml", "--detached", "--trial-idx", str(idx),
           "--hp-idx", str(idx), "--t-steps", str(T)]
    for kk, v in hp.items():
        cmd += ["--hp", f"{kk}={v}"]
    if "--dry" in sys.argv:
        print(" ".join(cmd)); continue
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"\brun\s+(FA-\S+)", r.stdout + r.stderr)
    out.append({"note": note, "cfg": f"sweep/sweep_config_{cfg}.yaml", "run_dir": f"runs/{cfg}/trial_{idx:04d}",
                "hp_idx": idx, "hp": hp, "t_steps": T, "rid": m.group(1) if m else None,
                "err": None if m else (r.stdout + r.stderr)[-400:]})
    print(note, out[-1]["rid"] or out[-1]["err"])
if out:
    json.dump(out, open(f"{R}/analysis/data_saving/data/nullparent_{'smoke' if '--smoke' in sys.argv else 'runs'}.json", "w"), indent=1)
