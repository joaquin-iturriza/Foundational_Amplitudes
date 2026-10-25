#!/usr/bin/env python3
"""Build the plan from today's five places: the hand-off entries of docs/results.tex (sec:ladder-open), their map
(.claude/plan_map.json), the site work items (active and held) with their decisions' quotes, and the runs filed under
them. Each hand-off entry keeps its full prose as its question and its stated next step; every quote is carried over.
Whatever cannot be mapped, and every field that still needs a human (a machine done_when, a site item), is printed.
    python3 tools/plan_migrate.py --out <plan.yaml>     (never the live plan by default)"""
import argparse, json, os, re, sqlite3, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan  # noqa: E402

ROOT = [plan.ROOT]



def handoff():
    """[(title, prose)] of the top-level hand-off items."""
    s = open(os.path.join(ROOT[0], "docs", "results.tex")).read()
    body, depth, out = s[s.index("\\label{sec:ladder-open}"):], 0, []
    for line in body.splitlines()[1:]:
        st = line.strip()
        if st.startswith("\\begin{itemize}"):
            depth += 1
            if depth == 1:
                continue
        if st.startswith("\\end{itemize}"):
            depth -= 1
            if depth == 0:
                break
        if depth == 1 and st.startswith("\\item"):
            out.append([st, []])
        elif out:
            out[-1][1].append(st)
    res = []
    for head, rest in out:
        m = re.match(r"\\item \\textbf\{((?:[^{}]|\{[^{}]*\})*)\}", head)
        title = m.group(1) if m else head[5:80]
        res.append((plain(title), plain(" ".join([head[m.end():] if m else head] + rest))))
    return res


def plain(t):
    t = re.sub(r"\\(code|emph|textbf)\{((?:[^{}]|\{[^{}]*\})*)\}", r"\2", t)
    t = re.sub(r"\\(ref|label)\{[^}]*\}|~", " ", t)
    t = t.replace("\\_", "_").replace("$", "").replace("\\,", " ")
    return re.sub(r"\s+", " ", t).strip()


def next_of(prose):
    m = re.search(r"(?:Proposed next|Next, [^:.]{0,60}:|Next:|Next is|Next,)\s*(.+?\.)(?=\s|$)", prose)
    return m.group(1).strip() if m else None


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=plan.ROOT, help="the checkout whose hand-off is read (default: the main one)")
    a = ap.parse_args()
    ROOT[0] = a.root
    c = sqlite3.connect(plan.RUNS_DB)
    items = {r[0]: dict(zip(("id", "title", "decision", "scope", "status", "match"), r)) for r in c.execute(
        "SELECT id, title, decision, scope, status, match FROM items WHERE project='FA' AND status IN ('active','held')")}
    decs = [dict(zip(("id", "kind", "quote", "said_at", "covers"), r)) for r in c.execute(
        "SELECT id, kind, quote, said_at, covers FROM decisions WHERE project='FA' AND in_force=1")]
    pm = json.load(open(os.path.join(ROOT[0], ".claude", "plan_map.json")))

    def quotes(iid, dec):
        names = lambda x: [y.strip() for y in (x["covers"] or "").split(",")]
        return [{"said_at": x["said_at"], "text": x["quote"], "decision": x["id"], "covers": names(x)} for x in decs
                if x["id"] == dec or iid in names(x)]     # covers as they stand now: a later hold/release edits them

    def blank(**k):
        if re.search(r"\bsetup\b", "%s %s" % (k.get("id", ""), k.get("title", "")), re.I):
            k.setdefault("kind", "setup")          # the setup's own entries: their record may be CLAUDE.md or a tool
        e = {"status": "running", "quotes": [], "decisions": [], "next": None, "done_when": None, "match": None,
             "notes_ref": None, "evidence": {"runs": [], "commit": None, "paths": []}, "dead_ends": [], "asks": []}
        e.update(k)
        return e

    entries, notes, by_id = [], [], {}
    for title, prose in handoff():
        key = next((k for k in pm if title.startswith(plain(k))), None)
        kind, _, val = pm.get(key, "").partition(":")
        val = val.strip()
        if kind == "item" and val in by_id:                           # a second hand-off entry on the same item
            e = by_id[val]
            e["question"] += "\n\n" + title + " " + prose
            e["next"] = e["next"] or next_of(prose)
        elif kind == "item" and val in items:
            it = items[val]
            e = blank(id=val, title=it["title"], question=title + " " + prose, next=next_of(prose) or it["scope"],
                      status="running" if it["status"] == "active" else "held",
                      quotes=quotes(val, it["decision"]), decisions=[it["decision"]], match=it["match"])
            entries.append(e); by_id[val] = e
        elif kind == "user":
            m = re.match(r"(\d{4}-\d{2}-\d{2}) (.*)", val)
            e = blank(id=slug(title), title=title, question=prose, status="waiting_user", ask=m.group(2),
                      waiting_since=m.group(1) + "T00:00:00+00:00", next="act on the user's answer",
                      asks=[{"ask": m.group(2), "at": m.group(1) + "T00:00:00+00:00"}])
            entries.append(e); notes.append("%s: no site item yet (created when the user approves)" % e["id"])
        elif kind == "held":
            did = re.match(r"(D\d+)", val).group(1)
            e = blank(id=slug(title), title=title, question=prose, status="held", hold_decision=did, decisions=[did],
                      next=next_of(prose) or "resumes when the user releases it",
                      quotes=[{"said_at": x["said_at"], "text": x["quote"], "decision": x["id"]} for x in decs
                              if x["id"] == did])
            it = items.get(e["id"])
            if it:                        # paused, and since given a site item: its globs and the hold naming it
                e["match"] = it["match"]
                e["hold_decision"] = next((x["id"] for x in decs if x["kind"] in ("hold", "forbid") and e["id"] in
                                           [y.strip() for y in (x["covers"] or "").split(",")]), did)
                e["quotes"] += [q for q in quotes(e["id"], it["decision"]) if q["decision"] != did]
            entries.append(e); by_id[e["id"]] = e
        else:
            notes.append("hand-off entry %r (%s): a record, not work; stays as prose" % (title[:60], kind or "no map"))
    for iid, it in items.items():
        if iid in by_id:
            continue
        e = blank(id=iid, title=it["title"], question=it["scope"], next=it["scope"],
                  status="held" if it["status"] == "held" else "running", quotes=quotes(iid, it["decision"]),
                  decisions=[it["decision"]], match=it["match"])
        if it["status"] == "held":
            e["hold_decision"] = next((x["id"] for x in decs if x["kind"] in ("hold", "forbid")
                                       and iid in [y.strip() for y in (x["covers"] or "").split(",")]), None)
        entries.append(e); by_id[iid] = e
    for e in entries:
        if e["status"] == "held" and not e.get("hold_decision"):
            notes.append("%s is held but no hold or forbid in force names it: record the user's hold (plan answer "
                         "--kind hold) or release it" % e["id"])
        elif e["status"] == "held":
            dd = plan.decision(e["hold_decision"])
            if not dd or dd["kind"] not in ("hold", "forbid") or e["id"] not in dd["covers"]:
                notes.append("%s: its hold %s is a %s that does not name it: record the user's hold for it" % (
                    e["id"], e["hold_decision"], dd and dd["kind"]))
        if e["status"] in ("running", "waiting_user") and not e.get("next"):
            notes.append("%s: no next step found" % e["id"])
    open_ = [e for e in entries if e["status"] != "done"]
    notes.append("%d open entries have no machine check yet: set each with plan check before plan done" % sum(
        not e["done_when"] for e in open_))
    notes += ["%s: waits on the user with nothing recorded as searched first; check CLAUDE.md's setup and HPO rules "
              "and the decisions in force before the question is sent" % e["id"] for e in open_
              if e["status"] == "waiting_user" and not e.get("searched")]
    notes += ["%s: no site item (plan item once the user approves it)" % e["id"] for e in open_
              if e["status"] != "waiting_user" and e["id"] not in items]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    import yaml
    # the switch-over time, in plan.yaml (committed): only runs that end after it count as "ended since the last
    # step" (without it every run ever finished would list every entry); the state file starts empty
    t = plan.now().isoformat()
    open(a.out, "w").write("# The plan: the only source of truth for open work. Edit through tools/plan.py.\n" +
                           yaml.safe_dump({"migrated_at": t, "entries": entries}, sort_keys=False,
                                          allow_unicode=True, width=120))
    sp = os.path.join(os.path.dirname(os.path.abspath(a.out)), ".state.json")
    head = subprocess.run(["git", "-C", ROOT[0], "rev-parse", "-q", "--verify", "trunk^{commit}"],
                          capture_output=True, text=True).stdout.strip()
    open(sp, "w").write(json.dumps({"_claims_verified": head} if head else {}))   # the claims watermark starts here
    print("plan: %d entries written to %s (empty state %s)" % (len(entries), a.out, sp))
    for n in notes:
        print("NOTE: " + n)


if __name__ == "__main__":
    main()
