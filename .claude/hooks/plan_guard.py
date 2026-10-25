#!/usr/bin/env python3
"""Stop hook: the turn cannot end while an entry of the plan (docs/results.tex, sec:ladder-open, its top-level items)
is neither moving nor waiting on something named.

Why (2026-10-25): every earlier guard checked a proxy the agent satisfies while the plan sits still: an armed waiter,
an item that once had a run, a commit on some path, an item with one running job. The plan's entries with no work
item of their own were invisible to all of them, and after each job event the turn ended with those entries untouched
(the user: "44 minutes without nothing else being done, why?").

.claude/plan_map.json maps each entry (by the start of its \\textbf title) to its status:
  "item:<id>"            a site work item; it must have a run unfinished or a launch queued (site's item_moving), or
                         a wait recorded with `site item set --waiting`
  "user:<YYYY-MM-DD> <the question as sent>"   a call only the user can make, put to them in a message on that date
  "held:D<n>"            held or stopped by the user's decision D<n>
  "reference:<why>"      a state or design record, not work (listed on every stop, so it stays visible)
An entry missing from the map, an item that is idle, or a "done:" status (a finished entry belongs in the results,
not the hand-off: delete it) blocks the turn.
"""
import json, os, re, sqlite3, sys, time

REPO = os.environ.get("CLAUDE_PROJECT_DIR",
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MAP = os.path.join(REPO, ".claude", "plan_map.json")
DB = os.path.expanduser("~/.local/share/ccorch/runs.db")
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "REMOVED", "BOOT_FAIL")


def entries():
    s = open(os.path.join(REPO, "docs", "results.tex")).read()
    body = s[s.index("\\label{sec:ladder-open}"):]
    depth, out = 0, []
    for line in body.splitlines()[1:]:
        st = line.strip()
        if st.startswith("\\begin{itemize}"):
            depth += 1
            continue
        if st.startswith("\\end{itemize}"):
            depth -= 1
            if depth == 0:
                break
        if depth == 1 and st.startswith("\\item"):
            m = re.match(r"\\item \\textbf\{(.*)", st)
            out.append((m.group(1) if m else st[5:]).strip())
    return out


def item_state(c, iid):
    q = ",".join("?" * len(TERMINAL))
    if c.execute("SELECT 1 FROM runs WHERE project='FA' AND item=? AND COALESCE(state,'') NOT IN (%s) LIMIT 1" % q,
                 (iid,) + TERMINAL).fetchone():
        return "running"
    if c.execute("SELECT 1 FROM launches WHERE project='FA' AND item=? AND (status='queued' OR (status='launching' "
                 "AND COALESCE(checked, created) > ?)) LIMIT 1", (iid, int(time.time()) - 3600)).fetchone():
        return "queued"
    r = c.execute("SELECT status, waiting FROM items WHERE project='FA' AND id=?", (iid,)).fetchone()
    if not r:
        return "no such item"
    if r[0] != "active":
        return r[0]
    return "waiting: " + r[1] if r[1] else "idle"


def main():
    try:
        json.load(sys.stdin)
    except ValueError:
        pass
    try:
        ents = entries()
        pm = json.load(open(MAP)) if os.path.exists(MAP) else {}
        c = sqlite3.connect(DB)
    except Exception as e:                                      # fail closed: a broken guard is a guard that is off
        sys.stderr.write("plan guard failed (%r); fix it before ending the turn\n" % e)
        return 2
    bad, ref = [], []
    for e in ents:
        key = next((k for k in pm if e.startswith(k)), None)
        if key is None:
            bad.append("  %s: not in .claude/plan_map.json" % e[:90])
            continue
        st = pm[key]
        kind, _, val = st.partition(":")
        if kind == "item":
            s = item_state(c, val.strip())
            if s in ("idle", "no such item", "done"):
                bad.append("  %s: item %s is %s" % (key, val.strip(), s))
        elif kind == "done":
            bad.append("  %s: marked done, so its result belongs in the results section: delete the entry" % key)
        elif kind == "user":
            if not re.match(r"\s*\d{4}-\d{2}-\d{2} \S", val):
                bad.append("  %s: a user call needs the date it was sent and the question" % key)
        elif kind == "held":
            if not re.match(r"\s*D\d+", val):
                bad.append("  %s: a hold names the user's decision (D<n>)" % key)
        elif kind == "reference":
            ref.append(key)
        else:
            bad.append("  %s: unknown status %r" % (key, st))
    if bad:
        sys.stderr.write("PLAN GUARD: the turn cannot end while an entry of the plan (sec:ladder-open) is neither "
                         "moving nor waiting on something named:\n" + "\n".join(bad) +
                         "\nTake each one's next step now (submit, queue a launch, or do the analysis), or record in "
                         ".claude/plan_map.json what it waits on (user:<date> <question sent>, held:D<n>).\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
