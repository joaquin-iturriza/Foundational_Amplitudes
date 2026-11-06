#!/usr/bin/env python3
"""Has the pillar review passed a change? (agent redesign, step B6). Read by git_verify_guard.py (a merge into trunk)
and tools/plan.py (`plan done`), so neither merges nor closes work no reviewer has read.

A pillar (claudemd-keeper, notes-editor, repo-reviewer; the table PILLARS of .claude/hooks/review_backlog.sh, read from
that file so there is one table) has passed a commit C when its watermark W (.claude/.review_state/<pillar>, in the
main checkout) contains C (C is an ancestor of W), or when the pillar's files are the same at C as at W (the reviewer
passed that content, committed later or elsewhere).

    A pass is the watermark or any tip listed in .claude/.review_state/<pillar>.passed (every `advance` appends its tip), so
    one branch's pass is not lost when another branch's pass moves the watermark.

    python3 tools/review_state.py unreviewed <base>..<tip> [--root R]   pillars whose files the range changes and
                                                                        whose review has not passed its tip
    python3 tools/review_state.py commit <sha> [--root R]               the same for one commit"""
import argparse, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(root, *a):
    return subprocess.run(["git", "-C", root] + list(a), capture_output=True, text=True)


def pillars(root=ROOT):
    """{name: [pathspecs]} from review_backlog.sh's PILLARS table."""
    try:
        text = open(os.path.join(root, ".claude", "hooks", "review_backlog.sh")).read()
    except OSError:
        return {}
    m = re.search(r"^PILLARS='(.*?)'", text, re.S | re.M)
    out = {}
    for line in (m.group(1).splitlines() if m else []):
        f = line.split("|")
        if len(f) >= 4:
            out[f[0].strip()] = f[3].split()
    return out


def state_root(root):
    common = git(root, "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()
    return os.path.dirname(common) if common.endswith("/.git") else root


def watermark(root, name):
    try:
        return open(os.path.join(state_root(root), ".claude", ".review_state", name)).read().split()[0]
    except (OSError, IndexError):
        return None


def changed(root, rng, paths):
    return bool(git(root, "-c", "core.quotePath=false", "diff", "--name-only", rng, "--", *paths).stdout.strip())


def passes(root, name):
    """the watermark and every tip the pillar's reviewer passed (review_backlog.sh appends them to <pillar>.passed)"""
    out = []
    w = watermark(root, name)
    if w:
        out.append(w)
    try:
        out += open(os.path.join(state_root(root), ".claude", ".review_state", name + ".passed")).read().split()
    except OSError:
        pass
    return list(dict.fromkeys(out))


def passed(root, name, paths, tip):
    for w in passes(root, name):
        if git(root, "cat-file", "-e", w + "^{commit}").returncode != 0:
            continue
        if git(root, "merge-base", "--is-ancestor", tip, w).returncode == 0:
            return True
        if git(root, "diff", "--quiet", w, tip, "--", *paths).returncode == 0:
            return True
    return False


def unreviewed(rng, root=ROOT):
    """[(pillar, watermark)] whose files the range changes and whose review has not passed its tip."""
    tip = rng.split("..")[-1] if ".." in rng else rng
    tip = git(root, "rev-parse", tip).stdout.strip() or tip
    rng = rng if ".." in rng else "%s^..%s" % (rng, rng)
    return [(n, watermark(root, n)) for n, paths in pillars(root).items()
            if changed(root, rng, paths) and not passed(root, n, paths, tip)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("unreviewed", "commit"))
    ap.add_argument("ref")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args(argv)
    bad = unreviewed(a.ref, a.root)
    for n, w in bad:
        print("%s: not reviewed past its watermark %s" % (n, (w or "none")[:10]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
