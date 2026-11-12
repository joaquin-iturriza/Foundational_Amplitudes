"""Submit the validation-with-training test (pretrain_data_saving, design 2026-11-12 after D420): does a change of
schedule, horizon or regularisation make validation go down together with training (curve_quality.py) without
losing validation? Each cell's stage-2 search best re-run at its own HPs through its own sweep config, one change per
arm: control (unchanged), wsd (warmup-stable-decay, cooldown the last 12.5% of steps as the suspects batch's WSD arm (submit_suspects.py), in place of the cosine's long
decay), half (T/2). Regularisation above the searched ranges goes through a widened DyHPO sweep, not
single values (D241). Cells: t t~ LO, u u~ LO, b b~ > Z Z V/B at D=316 and 3162, scratch and fine-tune.
--dry: print only."""
import json, os, re, subprocess, sys

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CELLS = json.load(open(f"{R}/analysis/data_saving/data/stage2_readout.json"))["cells"]
TARGETS, KS = ("epem_ttx", "ee_uu", "bbx_zz_nlo"), (5, 7)
ARMS = [("control", 9301, {}, 1.0), ("wsd", 9302, {"training.scheduler": "WSD", "training.wsd_cooldown_frac": 0.125}, 1.0),
        ("half", 9303, {}, 0.5)]   # 2T dropped for D412's 30 GPU-h beside the lr-up searches (results.tex ~l.7650: runs already go past improvement)
WORK_SEC_PER_STEP = 0.085  # training steps at k<=7, V100 (cal_vb runs: 0.074-0.091 s/step), the work estimate
SEC_PER_STEP = 0.12        # nullparent runs, 8500 steps at k<=7: ~1020 s submit to end (overhead), for wall limits


def hp_of(sweep, idx):
    tr = json.load(open(f"{R}/sweeps_local/{sweep}/driver_state.json"))["trials"]
    tr = tr.values() if isinstance(tr, dict) else tr
    return dict(next(v for v in tr if v.get("hp_idx") == idx)["hp"])


def jobs():
    out = []
    for c in CELLS:
        if c["target"] in TARGETS and c["k"] in KS:
            hp = hp_of(c["sweep"], c["best_hp"])
            for name, idx, ch, f in ARMS:
                T = int(round(c["T"] * f))
                lim = max(15, int(T * SEC_PER_STEP / 60 * 1.5) + 10)
                out.append((c["sweep"], idx, {**hp, **ch}, f"together {name} {c['arm']} {c['target']} d{c['k']} "
                            f"(best hp{c['best_hp']}, T {T})", lim, T))
    return out


if __name__ == "__main__":
    J = jobs()
    work = sum(T * WORK_SEC_PER_STEP for *_, T in J) / 3600
    print(f"{len(J)} runs, work {work:.1f} GPU-h, wall limits {sum(l for *_, l, _ in J) / 60:.1f} GPU-h")
    out = []
    for cfg, idx, hp, note, lim, T in J:
        cmd = ["site", "submit", "ccin2p3", "FA", "sweep/trial_job.sh", "--item", "pretrain_data_saving", "--note", note,
               f"--hdr=--time=00:{lim:02d}:00" if lim < 60 else f"--hdr=--time=01:{lim - 60:02d}:00",
               "--hdr=--mem=16G", "--hdr=--cpus-per-task=4", "--est-gpu-hours", f"{T * WORK_SEC_PER_STEP / 3600:.2f}",
               "--", "--sweep-config", f"sweep/sweep_config_{cfg}.yaml", "--detached", "--trial-idx", str(idx),
               "--hp-idx", str(idx), "--t-steps", str(T)]
        for k, v in hp.items():
            cmd += ["--hp", f"{k}={v}"]
        if "--dry" in sys.argv:
            print(" ".join(cmd))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        m = re.search(r"\brun\s+(FA-\S+)", r.stdout + r.stderr)
        out.append({"note": note, "cfg": f"sweep/sweep_config_{cfg}.yaml", "run_dir": f"runs/{cfg}/trial_{idx:04d}",
                    "hp_idx": idx, "hp": hp, "t_steps": T, "rid": m.group(1) if m else None,
                    "err": None if m else (r.stdout + r.stderr)[-400:]})
        print(note, out[-1]["rid"] or out[-1]["err"])
    if out:
        f = f"{R}/analysis/data_saving/data/together_runs.json"
        old = json.load(open(f)) if os.path.exists(f) else []
        json.dump(old + out, open(f, "w"), indent=1)
