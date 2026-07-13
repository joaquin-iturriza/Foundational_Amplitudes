#!/usr/bin/env python3
"""Self-test for the .claude guards. Lives in a .py so the test payloads (which by design
contain the very patterns the guards match) do not themselves trip the guards when the
harness inspects the Bash command that launches this.

Run:  python3 .claude/hooks/test_guards.py
"""
import json
import os
import subprocess
import tempfile

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
HOOKS = os.path.join(REPO, ".claude", "hooks")

# built from pieces so this file contains no literal match either
OFF = "plot" + "ting.plot_mse_het=" + "false"
BAD_SH = f"#!/bin/bash\n#SBATCH --array=0-1\npython run.py exp_name=x {OFF} plot=true\n"
GOOD_SH = "#!/bin/bash\npython run.py exp_name=x plot=true training.lr=0.004\n"
HP_GRID = ("#!/bin/bash\n#SBATCH --array=0-5\n"
           "LRS=(1e-4 1e-3 1e-2)\nLR=${LRS[$SLURM_ARRAY_TASK_ID]}\n"
           "python run.py training.lr=${LR}\n")
ABLATION = ("#!/bin/bash\n#SBATCH --array=0-1\n"
            "LOSSES=(MSE HETEROSC)\nLOSS=${LOSSES[$SLURM_ARRAY_TASK_ID]}\n"
            "python run.py training.loss=${LOSS} training.lr=0.004\n")


def run(hook, payload):
    p = subprocess.run([os.path.join(HOOKS, hook)], input=json.dumps(payload),
                       capture_output=True, text=True)
    return p.returncode == 2, p.stderr


def check(desc, hook, payload, want_block):
    blocked, _ = run(hook, payload)
    ok = blocked == want_block
    print(f"  [{'PASS' if ok else 'FAIL'}] {'BLOCK' if blocked else 'allow':5s} "
          f"(want {'BLOCK' if want_block else 'allow'})  {desc}")
    return ok


def bash(cmd):
    return {"tool_name": "Bash", "tool_input": {"command": cmd}}


def write(fp, content):
    return {"tool_name": "Write", "tool_input": {"file_path": fp, "content": content}}


def main():
    d = tempfile.mkdtemp()
    bad, good = os.path.join(d, "bad.sh"), os.path.join(d, "good.sh")
    grid, abl = os.path.join(d, "grid.sh"), os.path.join(d, "abl.sh")
    for f, c in ((bad, BAD_SH), (good, GOOD_SH), (grid, HP_GRID), (abl, ABLATION)):
        open(f, "w").write(c)

    ok = []
    print("plot_guard — must BLOCK runs configured with plotting off:")
    ok.append(check("sbatch script carrying the flag", "plot_guard.sh", bash(f"sbatch {bad}"), True))
    ok.append(check("inline run.py with the flag", "plot_guard.sh",
                    bash(f"python run.py exp_name=x {OFF}"), True))
    ok.append(check("Write a .sh carrying the flag", "plot_guard.sh",
                    write(os.path.join(d, "w.sh"), f"python run.py {OFF}\n"), True))
    print("plot_guard — must ALLOW (cleanup / clean runs must stay possible):")
    ok.append(check("sbatch a clean script", "plot_guard.sh", bash(f"sbatch {good}"), False))
    ok.append(check("grep FOR the flag (searching)", "plot_guard.sh",
                    bash(f"grep -rn {OFF} analysis/"), False))
    ok.append(check("sed REMOVING the flag (cleanup)", "plot_guard.sh",
                    bash(f"sed -i s/{OFF}//g analysis/x.sh"), False))
    ok.append(check("editing the guard's own source", "plot_guard.sh",
                    write("/x/.claude/hooks/plot_guard.sh", "plot=false"), False))

    print("\nhpo_guard — must BLOCK hand-rolled HP grids:")
    ok.append(check("array over training.lr", "hpo_guard.sh", bash(f"sbatch {grid}"), True))
    print("hpo_guard — must ALLOW non-HP ablations:")
    ok.append(check("array over training.loss (ablation)", "hpo_guard.sh", bash(f"sbatch {abl}"), False))
    ok.append(check("sweep_manager submit", "hpo_guard.sh",
                    bash("python sweep/sweep_manager.py submit sweeps/x"), False))

    print(f"\n{sum(ok)}/{len(ok)} passed")
    raise SystemExit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
