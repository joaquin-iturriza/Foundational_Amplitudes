#!/usr/bin/env python3
"""Several read-only lookups in one Bash call (agent_redesign step 29): FA's own stand-in for Code Mode, which cannot
run over Remote Control. Anything that writes, submits or commits stays a separate guarded call, so every hook still
sees it; this tool runs only the read operations below, each as an argv (no shell), from the FA checkout.

    python3 tools/batch.py OP [OP ...]        each OP one quoted string; results printed in order, each under '## OP'

    read FILE[:A-B]           lines A..B of FILE (numbered; whole file capped at --max lines)
    grep PATTERN [PATH ...]   grep -rnE over PATH (default .), skipping worktrees/, .git, .venv*
    def NAME [PATH ...]       where NAME is defined (def/class), with the whole body
    calls NAME [PATH ...]     each call site of NAME with its enclosing def
    ls GLOB                   files matching GLOB
    plan ENTRY                tools/plan.py show ENTRY
    site VERB ARGS            site runs|logs|status|decisions|items|where (read-only verbs only)

--max N caps the lines each operation prints (default 120); a cut output says how many lines it dropped.
"""
import glob, os, re, shlex, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = ["--exclude-dir=worktrees", "--exclude-dir=.git", "--exclude-dir=.venv*", "--exclude-dir=__pycache__"]
SITE_READ = {"runs", "logs", "status", "decisions", "items", "where"}


def sh(argv):
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    return p.stdout + (p.stderr if p.returncode not in (0, 1) else "")


def enclosing(path, line):
    """The innermost def/class at or above LINE whose indentation is below the line's."""
    try:
        lines = open(os.path.join(ROOT, path), errors="replace").read().splitlines()
    except OSError:
        return ""
    ind = len(lines[line - 1]) - len(lines[line - 1].lstrip())
    for k in range(line - 1, -1, -1):
        m = re.match(r"(\s*)(?:async\s+)?(def|class)\s+(\w+)", lines[k])
        if m and len(m.group(1)) < ind:
            return m.group(3)
    return "<module>"


def body(path, line):
    lines = open(os.path.join(ROOT, path), errors="replace").read().splitlines()
    ind = len(lines[line - 1]) - len(lines[line - 1].lstrip())
    end = line
    while end < len(lines) and (not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) > ind):
        end += 1
    while end > line and not lines[end - 1].strip():
        end -= 1
    return "\n".join("%d: %s" % (i + 1, lines[i]) for i in range(line - 1, end))


def op(text):
    a = shlex.split(text)
    v, args = a[0], a[1:]
    if v == "read":
        f, _, rng = args[0].partition(":")
        lines = open(os.path.join(ROOT, f), errors="replace").read().splitlines()
        lo, hi = (int(x) for x in rng.split("-")) if rng else (1, len(lines))
        return "\n".join("%d: %s" % (i, lines[i - 1]) for i in range(lo, min(hi, len(lines)) + 1))
    if v == "grep":
        return sh(["grep", "-rnE", *SKIP, "--", args[0], *(args[1:] or ["."])])
    if v in ("def", "calls"):
        name = args[0]
        pat = r"^\s*(async\s+)?(def|class)\s+%s\b" % name if v == "def" else r"\b%s\s*\(" % name
        out = []
        for hit in sh(["grep", "-rnHE", "--include=*.py", *SKIP, "--", pat, *(args[1:] or ["."])]).splitlines():
            path, ln, src = hit.split(":", 2)
            path = os.path.relpath(path, ".") if not path.startswith("./") else path[2:]
            if v == "def":
                out.append("%s:%s\n%s" % (path, ln, body(path, int(ln))))
            elif not re.match(r"\s*(async\s+)?def\s", src):
                out.append("%s:%s:%s  %s" % (path, ln, enclosing(path, int(ln)), src.strip()))
        return "\n".join(out)
    if v == "ls":
        return "\n".join(sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, args[0]),
                                                                            recursive=True)))
    if v == "plan":
        return sh([sys.executable, "tools/plan.py", "show", args[0]])
    if v == "site":
        if args[0] not in SITE_READ:
            return "refused: site %s is not a read-only verb (%s); run it as its own call" % (
                args[0], ", ".join(sorted(SITE_READ)))
        return sh(["site", *args])
    return "refused: unknown operation %r (read, grep, def, calls, ls, plan, site)" % v


def main(argv):
    mx = 120
    if "--max" in argv:
        i = argv.index("--max")
        mx = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    for t in argv:
        try:
            out = op(t).rstrip("\n").splitlines()
        except Exception as e:  # one bad op must not lose the others' results
            out = ["error: %s: %s" % (type(e).__name__, e)]
        cut = len(out) - mx
        print("## " + t)
        print("\n".join(out[:mx]) + ("\n... %d more lines (narrow it or raise --max)" % cut if cut > 0 else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
