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

REPO = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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


def agents_on_opus(d):
    """one result per agent definition: its front matter says `model: inherit`, the main session's Opus with its 1M
    window (agent redesign B8; Joaquin 2026-10-31: 300k for the agents, beyond plain opus's window), except
    mechanical-worker, which says `model: sonnet` (D213: mechanical steps on Sonnet)."""
    want_of = {"mechanical-worker.md": "sonnet"}
    out = []
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not f.endswith(".md"):
            continue
        head = open(os.path.join(d, f)).read().split("---")
        front = head[1] if len(head) > 2 else ""
        want = want_of.get(f, "inherit")
        good = any(l.strip() == "model: " + want for l in front.splitlines())
        print(f"  [{'PASS' if good else 'FAIL'}] {f}: model {want if good else 'NOT ' + want}")
        out.append(good)
    return out


# a clean line of each form the project's jobs print (2026-10-30: the gridpack probes' two-word name was sorted broke)
CLEAN_SUMMARIES = ("RUN_SUMMARY lhe_probe gridpack rc=0", "RUN_SUMMARY lhe_parts rc=0")


def end_markers():
    """`site supervise` sorts a COMPLETED run clean only on an end marker of projects.toml; each clean RUN_SUMMARY
    form must match one (none may be broken by a k=v line), and a failed one (rc=1) must match none."""
    import re, tomllib
    try:
        pr = tomllib.load(open(os.path.expanduser("~/.config/ccorch/projects.toml"), "rb"))["projects"]["FA"]
    except (OSError, KeyError) as e:
        print(f"  [FAIL] projects.toml unreadable: {e}")
        return [False]
    marks = (pr.get("run_ends") or {}).get("end_markers") or []
    out = []
    for line in CLEAN_SUMMARIES:
        good = any(re.search(m, line) for m in marks)
        print(f"  [{'PASS' if good else 'FAIL'}] clean: {line}")
        out.append(good)
    bad = not any(re.search(m, "RUN_SUMMARY lhe_probe gridpack rc=1") for m in marks if "RUN_SUMMARY" in m)
    print(f"  [{'PASS' if bad else 'FAIL'}] rc=1 is no end marker")
    return out + [bad]


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

    print("\nmd_guard — must BLOCK a memory file (no persistent memory, CLAUDE.md rule 3):")
    ok.append(check("Write ~/.claude/projects/x/memory/MEMORY.md", "md_guard.sh",
                    write(os.path.expanduser("~/.claude/projects/x/memory/MEMORY.md"), "remember this\n"), True))

    print("\nagents — every .claude/agents/*.md inherits the main session's opus (mechanical-worker on sonnet, D213):")
    ok += agents_on_opus(os.path.join(REPO, ".claude", "agents"))

    print("\nrun ends — FA's end markers take the RUN_SUMMARY forms its jobs print:")
    ok += end_markers()

    print(f"\n{sum(ok)}/{len(ok)} passed")
    raise SystemExit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
