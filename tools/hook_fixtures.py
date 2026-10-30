#!/usr/bin/env python3
"""Fixtures of the night of 29 to 30 Oct 2026 (agent redesign, step A1): the real states the new hooks and the
supervisor are tested on. Everything is copied, nothing moved or changed at its source.

    python3 tools/hook_fixtures.py build [--out DIR] [--no-remote]
    python3 tools/hook_fixtures.py check [--fixture DIR]

build writes DIR (default tests/fixtures/hooks_20261029/):
  ledger.json       rows of ~/.local/share/ccorch/runs.db: the events of the window (FA and project-less), the runs
                    they name or that were submitted or ended in it, and every FA item and decision (the state the
                    events were judged against), read through a read-only connection
  plan_before/      plan/plan.yaml at 2720709b^ (the plan as it stood when "the acked-runs demand" refused the turn
                    ends) and plan/.state.json cut to the steps recorded before that commit (the state file is not in
                    git; its steps carry their time)
  plan_head/        plan/plan.yaml at the commit the fixture was built at and the live plan/.state.json
  queue_catalog_v4_default.json   the lhe_parts queue of the default-share generation on CC-IN2P3, read with one
                    `site run` (a python reader printing one JSON line, no write): per state the task names with their
                    mtime, the spec files with their sha1, and the first line of every .err file (the 908 IndexError
                    tasks of the replan under live workers, now in failed_requeued/)
  the night's transcripts are not copied (they carry each session's whole context): transcript_records() reads
                    the main-session and subagent transcripts in place, records of the window only, trimmed (text and
                    tool calls kept, tool results cut to 1500 characters, thinking dropped); the manifest lists them
  manifest.json     window, source commits, counts
check verifies the fixture reproduces the night's counts: 227 events (224 RUN: 159 TIMEOUT, 34 COMPLETED, 31 FAILED;
3 MAINTAIN), all acked, and 908 IndexError tasks in the queue snapshot."""
import argparse, datetime as dt, glob, hashlib, json, os, re, sqlite3, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "tests", "fixtures", "hooks_20261029")
DB = os.path.expanduser("~/.local/share/ccorch/runs.db")
TRANSCRIPTS = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
CET = dt.timezone(dt.timedelta(hours=1))
START = dt.datetime(2026, 10, 29, 20, 0, 0, tzinfo=CET)
END = dt.datetime(2026, 10, 30, 3, 56, 37, tzinfo=CET)        # the last event of the night (id 2416)
DEMAND_COMMIT = "2720709b"                                    # plan: an acked run end counts as handled
EXPECTED = {"events": 227, "RUN": 224, "TIMEOUT": 159, "COMPLETED": 34, "FAILED": 31, "MAINTAIN": 3, "acked": 227,
            "IndexError": 908}
QUEUE = "catalog_v4_default"
MARK = "HOOK_FIXTURE_QUEUE "

# the reader run on the site: stdlib only, reads, prints one line
QUEUE_READER = r'''
import hashlib, json, os, sys
q = os.path.join(os.environ["SCRATCH"], "lhe_queue", sys.argv[1])
out = {"queue": q, "states": {}, "specs": {}, "errors": {}}
for st in sorted(os.listdir(q)):
    p = os.path.join(q, st)
    if not os.path.isdir(p) or st in ("locks", "gc", "logs"):
        continue
    rows = []
    for dp, dn, fn in os.walk(p):
        for f in sorted(fn):
            fp = os.path.join(dp, f)
            rel = os.path.relpath(fp, p)
            rows.append([rel, int(os.path.getmtime(fp))])
            if f.endswith(".json") and st in ("specs",):
                out["specs"][rel] = hashlib.sha1(open(fp, "rb").read()).hexdigest()
            if ".err" in f:
                try:
                    out["errors"][st + "/" + rel] = open(fp, errors="replace").readline().strip()[:200]
                except OSError:
                    pass
    out["states"][st] = rows
for f in sorted(os.listdir(q)):
    fp = os.path.join(q, f)
    if os.path.isfile(fp):
        out["specs"][f] = hashlib.sha1(open(fp, "rb").read()).hexdigest()
print("HOOK_FIXTURE_QUEUE " + json.dumps(out, separators=(",", ":")))
'''


def _rows(c, sql, args=()):
    cur = c.execute(sql, args)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def build_ledger(db=DB):
    c = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    t0, t1 = int(START.timestamp()), int(END.timestamp())
    events = _rows(c, "SELECT * FROM events WHERE ts BETWEEN ? AND ? AND (project='FA' OR project IS NULL) ORDER BY id",
                   (t0, t1))
    named = set()
    for e in events:
        named.update(re.findall(r"\b[A-Za-z0-9]+-[a-z0-9]+-[0-9a-f]{8}\b", e.get("msg") or ""))
    runs = _rows(c, "SELECT * FROM runs WHERE project='FA' AND ((submitted BETWEEN ? AND ?) OR (finished BETWEEN ? AND ?))"
                    " ORDER BY submitted", (t0, t1, t0, t1))
    have = {r["id"] for r in runs}
    for rid in sorted(named - have):
        runs += _rows(c, "SELECT * FROM runs WHERE id=?", (rid,))
    items = _rows(c, "SELECT * FROM items WHERE project='FA' ORDER BY id")
    decisions = _rows(c, "SELECT * FROM decisions WHERE project='FA' ORDER BY recorded")
    return {"window": [t0, t1], "events": events, "runs": runs, "items": items, "decisions": decisions}


def _git(*a):
    return subprocess.run(["git", "-C", ROOT] + list(a), capture_output=True, text=True, check=True).stdout


def build_plan(out):
    before = _git("rev-parse", DEMAND_COMMIT + "^").strip()
    head = _git("rev-parse", "HEAD").strip()
    t_demand = int(_git("log", "-1", "--format=%ct", DEMAND_COMMIT).strip())
    common = _git("rev-parse", "--git-common-dir").strip()          # the state is gitignored: the main checkout's
    main = os.path.dirname(common if os.path.isabs(common) else os.path.join(ROOT, common))
    state = json.load(open(os.path.join(main, "plan", ".state.json")))
    for sub, rev in (("plan_before", before), ("plan_head", head)):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
        open(os.path.join(out, sub, "plan.yaml"), "w").write(_git("show", "%s:plan/plan.yaml" % rev))
    json.dump(state, open(os.path.join(out, "plan_head", ".state.json"), "w"), indent=1, sort_keys=True)
    cut = {}
    for k, v in state.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        steps = [s for s in v.get("steps") or [] if _epoch(s.get("at")) <= t_demand]
        if steps or not v.get("steps"):
            cut[k] = dict(v, steps=steps)
    json.dump(cut, open(os.path.join(out, "plan_before", ".state.json"), "w"), indent=1, sort_keys=True)
    return {"plan_before": before, "plan_head": head, "demand_commit_time": t_demand}


def _epoch(s):
    if not s:
        return 0
    try:
        return int(dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def build_queue(site="ccin2p3"):
    cmd = "python3 -c %s %s" % (_shq(QUEUE_READER), QUEUE)
    r = subprocess.run(["site", "run", site, "FA", "--", cmd], capture_output=True, text=True, timeout=600)
    for line in r.stdout.splitlines():
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    raise SystemExit("hook_fixtures: the queue reader printed no %s line (exit %d): %s"
                     % (MARK.strip(), r.returncode, (r.stderr or r.stdout)[-800:]))


def _shq(s):
    return "'" + s.replace("'", "'\\''") + "'"


def _trim(rec):
    """a transcript record kept for the tests: text, tool calls and hook records; tool results cut; thinking out."""
    keep = {k: rec[k] for k in ("type", "subtype", "timestamp", "uuid", "parentUuid", "isSidechain", "agentId",
                                "sessionId", "content", "hookInfos", "hookErrors", "preventedContinuation",
                                "stopReason", "compactMetadata", "level") if k in rec}
    if rec.get("type") == "attachment":
        a = dict(rec.get("attachment") or {})
        for k in ("content", "prompt"):
            if isinstance(a.get(k), str):
                a[k] = a[k][:4000]
        keep["attachment"] = a
    msg = rec.get("message")
    if isinstance(msg, dict):
        m = {k: msg[k] for k in ("role", "usage", "model", "stop_reason") if k in msg}
        content = msg.get("content")
        if isinstance(content, list):
            out = []
            for b in content:
                t = b.get("type")
                if t == "thinking" or t == "redacted_thinking":
                    continue
                if t == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                    b = dict(b, content=str(c)[:1500])
                out.append(b)
            m["content"] = out
        else:
            m["content"] = content
        keep["message"] = m
    return keep


def transcript_files(src=TRANSCRIPTS):
    """name -> path of the night's main-session and subagent transcripts (read in place, never copied: they carry the
    session's whole context and stay out of git)."""
    files = [p for p in glob.glob(os.path.join(src, "*.jsonl")) + glob.glob(os.path.join(src, "*", "subagents", "*.jsonl"))
             if os.path.getmtime(p) >= START.timestamp()]
    out = {}
    for p in sorted(files):
        name = os.path.basename(p)[:-len(".jsonl")]
        if "/subagents/" in p:
            name = os.path.basename(os.path.dirname(os.path.dirname(p)))[:8] + "__" + name
        out[name] = p
    return out


def load(fixture=DEFAULT_OUT):
    return json.load(open(os.path.join(fixture, "ledger.json")))


def transcript_records(name=None, src=TRANSCRIPTS):
    """(name, trimmed record) for every record of the window, main sessions first (their names have no __)."""
    lo = START.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    hi = END.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "~"
    for n, p in sorted(transcript_files(src).items(), key=lambda kv: ("__" in kv[0], kv[0])):
        if name and n != name:
            continue
        with open(p, errors="replace") as f:
            for line in f:
                if '"timestamp":"2026-10-' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if lo <= (r.get("timestamp") or "") <= hi and r.get("type") in ("user", "assistant", "attachment", "system"):
                    yield n, _trim(r)


def counts(fixture=DEFAULT_OUT):
    led = load(fixture)
    ev = [e for e in led["events"] if e["project"] == "FA"]
    c = {"events": len(ev), "acked": sum(1 for e in ev if e["acked"])}
    for e in ev:
        c[e["kind"]] = c.get(e["kind"], 0) + 1
        m = re.search(r" ended ([A-Z_]+)", e["msg"] or "") if e["kind"] == "RUN" else None
        if m:
            c[m.group(1)] = c.get(m.group(1), 0) + 1
    qp = os.path.join(fixture, "queue_%s.json" % QUEUE)
    if os.path.exists(qp):
        q = json.load(open(qp))
        c["IndexError"] = sum(1 for v in q["errors"].values() if v.startswith("IndexError"))
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=DEFAULT_OUT)
    b.add_argument("--no-remote", action="store_true", help="keep the queue snapshot already in --out")
    k = sub.add_parser("check")
    k.add_argument("--fixture", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        os.makedirs(a.out, exist_ok=True)
        man = {"window": [START.isoformat(), END.isoformat()], "built": dt.datetime.now(CET).isoformat(timespec="seconds")}
        json.dump(build_ledger(), open(os.path.join(a.out, "ledger.json"), "w"), indent=1, sort_keys=True)
        man["plan"] = build_plan(a.out)
        if not a.no_remote:
            json.dump(build_queue(), open(os.path.join(a.out, "queue_%s.json" % QUEUE), "w"), separators=(",", ":"))
        man["transcripts"] = sorted(transcript_files())
        man["counts"] = counts(a.out)
        json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1, sort_keys=True)
        print(json.dumps(man["counts"]))
        a.fixture = a.out
    got = counts(a.fixture)
    bad = {k: (got.get(k), v) for k, v in EXPECTED.items() if got.get(k) != v}
    for k, v in EXPECTED.items():
        print("  %-10s %6s  (expected %d)" % (k, got.get(k), v))
    if bad:
        print("hook_fixtures: MISMATCH %s" % bad)
        return 1
    print("hook_fixtures: the fixture reproduces the night's counts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
