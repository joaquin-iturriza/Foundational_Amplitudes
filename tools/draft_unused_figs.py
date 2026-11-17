"""List files under <draft>/figs not added by the received-draft commit and not included by paper.tex.

Usage: draft_unused_figs.py <draft dir> [received_commit=d742281]; exit 1 if any are listed.
"""
import re
import subprocess
import sys
from pathlib import Path


def unused(draft, received="d742281"):
    draft = Path(draft)
    git = lambda *a: subprocess.run(["git", "-C", str(draft), *a], capture_output=True, text=True, check=True).stdout
    theirs = set(git("show", "--name-only", "--diff-filter=A", "--format=", received).split("\n"))
    tex = (draft / "paper.tex").read_text()
    # an include names a path; match by basename with or without extension
    inc = {Path(p.strip()).stem for p in re.findall(r"\{([^{}]*figs/[^{}]*)\}", tex)}
    inc |= {Path(p).stem for p in re.findall(r"figs/[\w./-]+", tex)}
    return sorted(str(f.relative_to(draft)) for f in (draft / "figs").rglob("*")
                  if f.is_file() and str(f.relative_to(draft)) not in theirs and f.stem not in inc)


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        run = lambda *a: subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)
        run("init", "-q")
        (Path(d) / "figs").mkdir()
        (Path(d) / "figs/theirs.pdf").write_text("x")
        run("add", "."); run("commit", "-qm", "recv")
        rec = run("rev-parse", "--short", "HEAD") and subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        for n in ("used.pdf", "stale.pdf"):
            (Path(d) / "figs" / n).write_text("x")
        (Path(d) / "paper.tex").write_text(r"\includegraphics{figs/used.pdf}")
        assert unused(d, rec) == ["figs/stale.pdf"]
        (Path(d) / "figs/stale.pdf").unlink()
        assert unused(d, rec) == []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _selftest(); print("selftest ok"); sys.exit(0)
    out = unused(*sys.argv[1:3])
    print("\n".join(out))
    sys.exit(1 if out else 0)
