"""Keep the repository at trunk, main and the worktrees of live work (D334, Joaquin 2026-11-09: "make sure things are
set up so this can't happen again"). Run by the supervisor's maintain pass; a non-zero exit there is a MAINTAIN event.

    python3 tools/repo_hygiene.py [--fold-agents] [--no-fetch] [--json]

--fold-agents  every Agent-isolation worktree (.claude/worktrees/agent-<id>) whose agent has ended (its transcript's last
               assistant message ended the turn or called SubagentHandback, and nothing was written for 15 min; or no
               transcript at all) and whose HEAD is in trunk is folded and removed with scripts/remove_worktree.sh
               (results folded into the trunk, the worktree removed, its branch deleted here and on origin).
The check (always): what live work does not explain, each listed with why:
  worktree_merged     a worktree (not the main checkout) whose HEAD is in trunk and that is not an agent's running one
  agent_unmerged      an ended agent's worktree with commits trunk lacks (folding it would lose them: a person decides)
  branch_merged       a local branch other than trunk/main, checked out in no worktree, already in trunk
  branch_orphan       a local branch other than trunk/main, checked out in no worktree, with commits trunk lacks
  remote_merged       a branch on origin other than trunk/main (and the retired ccin2p3, jeanzay) whose tip is in trunk
  remote_unmerged     a branch on origin other than trunk/main with commits trunk lacks
Exit 0 when nothing is listed, 1 otherwise (one summary line last, so the MAINTAIN event names the counts).
"""
import argparse, glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = {"trunk", "main"}
KEEP_REMOTE = KEEP | {"ccin2p3", "jeanzay"}   # retired, kept on GitHub (fa-git-and-release)
TRANSCRIPTS = os.path.expanduser(os.environ.get("HYGIENE_TRANSCRIPTS", "~/.claude/projects/-home-joaquin-work-FA"))
QUIET_S = 900


def git(*a, cwd=ROOT, check=False):
    r = subprocess.run(["git", "-C", cwd] + list(a), capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError("git %s: %s" % (" ".join(a), r.stderr.strip()))
    return r.stdout.strip(), r.returncode


def in_trunk(rev, cwd=ROOT):
    return git("merge-base", "--is-ancestor", rev, "trunk", cwd=cwd)[1] == 0


def worktrees():
    out, _ = git("worktree", "list", "--porcelain")
    wts, cur = [], {}
    for line in out.splitlines() + [""]:
        if not line:
            if cur:
                wts.append(cur)
            cur = {}
        elif line.startswith("worktree "):
            cur["path"] = line[9:]
        elif line.startswith("HEAD "):
            cur["head"] = line[5:]
        elif line.startswith("branch "):
            cur["branch"] = line[7:].replace("refs/heads/", "", 1)
    return wts


def agent_state(aid):
    """'running', 'ended' or 'unknown' (no transcript) for an agent id."""
    fs = glob.glob(os.path.join(TRANSCRIPTS, "*", "subagents", "agent-%s.jsonl" % aid))
    if not fs:
        return "unknown"
    f = max(fs, key=os.path.getmtime)
    if time.time() - os.path.getmtime(f) < QUIET_S:
        return "running"
    last = None
    with open(f, errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") in ("assistant", "user"):
                last = d
    if not last or last.get("type") != "assistant":
        return "running"          # a tool result or a message waiting for the model: not ended
    m = last.get("message") or {}
    ended = m.get("stop_reason") == "end_turn" or any(
        isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "SubagentHandback"
        for b in m.get("content") or [])
    return "ended" if ended else "running"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-agents", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    found, done = [], []
    if not a.no_fetch:
        git("fetch", "--prune", "--quiet", "origin")
    agent_dir = os.path.join(ROOT, ".claude", "worktrees")
    live_branches = set()
    for w in worktrees():
        p, br = w["path"], w.get("branch")
        if os.path.realpath(p) == os.path.realpath(ROOT):
            continue
        rel = os.path.relpath(p, ROOT)
        merged = in_trunk(w.get("head") or "HEAD", cwd=p) if os.path.isdir(p) else False
        if os.path.realpath(p).startswith(os.path.realpath(agent_dir) + os.sep):
            aid = os.path.basename(p)[len("agent-"):]
            st = agent_state(aid)
            if st == "running":
                if br:
                    live_branches.add(br)
                continue
            if not merged:
                found.append(("agent_unmerged", rel, "agent %s %s; branch %s has commits trunk lacks" % (aid, st, br)))
                if br:
                    live_branches.add(br)
                continue
            if a.fold_agents:
                r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "remove_worktree.sh"), rel], cwd=ROOT,
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    done.append("folded and removed %s (agent %s %s, branch %s)" % (rel, aid, st, br))
                    continue
                found.append(("worktree_merged", rel, "agent %s %s; remove_worktree.sh refused: %s" % (
                    aid, st, " ".join((r.stderr or r.stdout).split())[-200:])))
            else:
                found.append(("worktree_merged", rel, "agent %s %s, HEAD in trunk" % (aid, st)))
            if br:
                live_branches.add(br)
            continue
        if br:
            live_branches.add(br)
        if merged:
            found.append(("worktree_merged", rel, "branch %s is in trunk: finished work? remove it with "
                          "scripts/remove_worktree.sh" % br))
    out, _ = git("for-each-ref", "--format=%(refname:short)", "refs/heads")
    for br in out.split():
        if br in KEEP or br in live_branches:
            continue
        kind = "branch_merged" if in_trunk(br) else "branch_orphan"
        found.append((kind, br, "local branch in no worktree"))
    out, _ = git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin")
    for rb in out.split():
        br = rb.split("/", 1)[1] if "/" in rb else rb
        if br in KEEP_REMOTE or br == "HEAD" or rb == "origin":
            continue
        found.append(("remote_merged" if in_trunk(rb) else "remote_unmerged", rb, "branch on origin"))
    for line in done:
        print("  " + line)
    for kind, what, why in found:
        print("  %-16s %s: %s" % (kind, what, why))
    counts = {}
    for kind, _, _ in found:
        counts[kind] = counts.get(kind, 0) + 1
    if a.json:
        print(json.dumps({"found": [dict(kind=k, what=w, why=y) for k, w, y in found], "done": done}))
    print("repo_hygiene: %s" % (", ".join("%d %s" % (n, k) for k, n in sorted(counts.items())) or "clean") +
          ("; %d agent worktree(s) folded" % len(done) if done else ""))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
