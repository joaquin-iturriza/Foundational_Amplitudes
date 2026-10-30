#!/usr/bin/env python3
"""No attribution to an assistant in a commit (CLAUDE.md rule 4; agent redesign B7). One pattern, used by
.githooks/commit-msg (the gate, on the message file) and git_verify_guard.py (at the command, before it runs).

    python3 tools/attribution.py check-file MSGFILE        exit 1 naming the line, else 0
    python3 tools/attribution.py scan [--since 2026-10-01]  commits of the current branch whose message has one"""
import argparse, re, subprocess, sys

PATTERN = re.compile(
    r"^\s*co-authored-by:.*(?:\bclaude\b|anthropic\.com)"     # a trailer naming Claude or an anthropic.com address
    r"|generated with \[?claude code"                         # the PR / commit footer
    r"|^\s*claude-session:",                                  # the session link trailer
    re.I | re.M)


def offending(msg):
    """the lines of a message that attribute the work to an assistant (git comment lines left out)."""
    lines = [l for l in msg.splitlines() if not l.startswith("#")]
    return [l.strip() for l in lines if PATTERN.search(l)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check-file", "scan"))
    ap.add_argument("path", nargs="?")
    ap.add_argument("--since", default="2026-10-01")
    a = ap.parse_args(argv)
    if a.mode == "check-file":
        bad = offending(open(a.path, errors="replace").read())
        if bad:
            sys.stderr.write("commit-msg: refused, the message attributes the work to an assistant (CLAUDE.md rule 4): "
                             "%s\nRemove the line(s) and commit again.\n" % "; ".join(bad))
            return 1
        return 0
    out = subprocess.run(["git", "log", "--since", a.since, "--format=%H%x00%B%x01"], capture_output=True,
                         text=True).stdout
    n = 0
    for rec in out.split("\x01"):
        if "\x00" not in rec:
            continue
        sha, msg = rec.strip("\n").split("\x00", 1)
        bad = offending(msg)
        if bad:
            n += 1
            print("%s %s" % (sha[:10], "; ".join(bad)))
    print("%d commit(s) since %s carry an attribution" % (n, a.since))
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
