#!/usr/bin/env python3
"""Split .claude/context/sections_v2.txt into one file per section where its `file:` line says (agent redesign D3):
FA's in ~/work/FA/.claude/context/, the cross-project ones in ~/.claude/context/. Each file keeps the section's trigger
lines (paths, cmds, tools, session), then `---`, then its text, which is what .claude/hooks/context_push.py reads.
    python3 tools/context_split.py [--check]      (--check: exit 1 when a split file differs from the source)
FA's files are written into the checkout this tool runs from (a worktree writes its own copies)."""
import argparse, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, ".claude", "context", "sections_v2.txt")
FA = "~/work/FA/"


def parse(text):
    out = []
    for blk in re.split(r"(?m)^=== SECTION ", text)[1:]:
        name, rest = blk.split("\n", 1)
        head, body = rest.split("\n---\n", 1)
        f = re.search(r"(?m)^file:\s*(\S+)", head).group(1)
        trig = "\n".join(l for l in head.splitlines() if re.match(r"(paths|cmds|tools|session):", l))
        out.append((name.strip(), f, trig + "\n---\n" + body.rstrip() + "\n"))
    return out


def target(f):
    return os.path.join(ROOT, f[len(FA):]) if f.startswith(FA) else os.path.expanduser(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    bad = 0
    for name, f, text in parse(open(SRC).read()):
        p = target(f)
        cur = open(p).read() if os.path.exists(p) else None
        if a.check:
            if cur != text:
                print("differs: %s (%s)" % (name, p))
                bad = 1
            continue
        if cur != text:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(text)
            print("wrote %s" % p)
    return bad


if __name__ == "__main__":
    sys.exit(main())
