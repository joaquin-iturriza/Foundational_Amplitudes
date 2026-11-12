"""Submit the D408 suspects batch (design PASS required): 40 fine-tune + 16 scratch runs at D=316..1e4 on CC.
--smoke: the one 50-step smoke (tree ft D=316, MSE+WSD+AdamW). --dry: print only."""
import json, re, subprocess, sys
R = "/home/joaquin/work/FA"
K = (5, 6, 7, 8)
FT_HP = {"epem_ttx": dict(zip(K, (1647, 507, 1817, 1387))), "ee_ttbar_nlo": dict(zip(K, (340, 1544, 120, 406)))}
SCR_HP = {"epem_ttx": dict(zip(K, (1605, 844, 640, 657))), "ee_ttbar_nlo": dict(zip(K, (1663, 713, 358, 816)))}
WSD = {"training.scheduler": "WSD", "training.wsd_cooldown_frac": 0.125}
FT_ARMS = [("lr3", 9101, 1 / 3), ("lr10", 9102, 1 / 10), ("wsd", 9104, WSD), ("adamw", 9105, {"training.optimizer": "AdamW"}),
           ("mse", 9106, {"training.heterosc_mu_only": True})]
SCR_ARMS = [("wsd", 9204, WSD), ("mse", 9206, {"training.heterosc_mu_only": True})]


def hp_of(sweep, idx):
    tr = json.load(open(f"{R}/sweeps_local/{sweep}/driver_state.json"))["trials"]
    tr = tr.values() if isinstance(tr, dict) else tr
    return dict(next(v for v in tr if v.get("hp_idx") == idx)["hp"])


jobs = []
for t in ("epem_ttx", "ee_ttbar_nlo"):
    for k in K:
        tt = "t8500"
        base = hp_of(f"dsave_s2_ft_{t}_d{k}_{tt}", FT_HP[t][k])
        cfg = f"dsave_diag_{'hmap' if t == 'epem_ttx' else 'parscale'}_ft_{t}_d{k}_{tt}"
        for name, idx, ch in FT_ARMS:
            hp = dict(base)
            if isinstance(ch, float):
                hp["fine_tune.lr_scale"] = base["fine_tune.lr_scale"] * ch
            else:
                hp.update(ch)
            jobs.append((cfg, idx, hp, f"suspects ft {name} {t} d{k} (control {cfg} hp{FT_HP[t][k]})", "00:25:00" if k <= 6 else "00:32:00", 8500))
        sbase = hp_of(f"dsave_s2_scr_{t}_d{k}_{tt}", SCR_HP[t][k])
        for name, idx, ch in SCR_ARMS:
            hp = dict(sbase); hp.update(ch)
            jobs.append((f"dsave_s2_scr_{t}_d{k}_{tt}", idx, hp, f"suspects scr {name} {t} d{k} (control hp{SCR_HP[t][k]})", "00:25:00" if k <= 6 else "00:32:00", 8500))
if "--smoke" in sys.argv:
    sm = []
    for j in (jobs[0], next(j for j in jobs if "ee_ttbar_nlo" in j[0] and j[0].startswith("dsave_diag"))):
        cfg, _, hp, _, _, _ = j
        hp = dict(hp); hp.update(WSD); hp.update({"training.optimizer": "AdamW", "training.heterosc_mu_only": True})
        sm.append((cfg, 9199, hp, f"suspects smoke ft mse+wsd+adamw {cfg} 50 steps", "00:15:00", 50))
    jobs = sm
out = []
for cfg, idx, hp, note, lim, T in jobs:
    cmd = ["site", "submit", "ccin2p3", "FA", "sweep/trial_job.sh", "--item", "pretrain_data_saving", "--note", note,
           f"--hdr=--time={lim}", "--hdr=--mem=16G", "--hdr=--cpus-per-task=4"]
    cmd[5:5] = ["--no-sync", "--allow-dirty"]
    cmd += ["--", "--sweep-config", f"sweep/sweep_config_{cfg}.yaml", "--detached", "--trial-idx", str(idx), "--hp-idx", str(idx), "--t-steps", str(T)]
    for kk, v in hp.items():
        cmd += ["--hp", f"{kk}={v}"]
    if "--dry" in sys.argv:
        print(" ".join(cmd)); continue
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"\brun\s+(FA-\S+)", r.stdout + r.stderr)
    out.append({"note": note, "cfg": f"sweep/sweep_config_{cfg}.yaml", "run_dir": f"runs/{cfg}/trial_{idx:04d}", "hp_idx": idx,
                "hp": hp, "t_steps": T, "rid": m.group(1) if m else None, "err": None if m else (r.stdout + r.stderr)[-400:]})
    print(note, out[-1]["rid"] or out[-1]["err"])
if out:
    f = f"{R}/analysis/data_saving/data/suspects_{'smoke' if '--smoke' in sys.argv else 'runs'}.json"
    json.dump(out, open(f, "w"), indent=1)
