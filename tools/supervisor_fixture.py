#!/usr/bin/env python3
"""Fixture of the night's run ends for the supervisor's run-end sorter (agent_redesign, switch-over C1).

    python3 tools/supervisor_fixture.py [--first 2188 --last 2416] [--raw DIR] [--out tests/fixtures/supervisor_20261029]

Copies (never moves) the RUN events of the window from the registry (~/.local/share/ccorch/runs.db: id, time, item,
state, the hand-written ack note) with their runs (job, note, scheduler id), and each run's log as it stands on the
site: its first 22 lines, its last 60, and every line (at most 20 per log) matching an error or end-marker pattern.
The logs are read through `site run <site> FA -- head/tail/grep <files>` (one command each, read only), or taken from
DIR (heads.raw, tails.raw, grep.raw of an earlier read of the same commands). Writes run_ends.json.
"""
import argparse, json, os, re, sqlite3, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DB = os.path.expanduser("~/.local/share/ccorch/runs.db")
# the job scripts' --output names (scripts/job_*.sh), %j = the scheduler id
LOG_PREFIX = {"job_lhe_parts.sh": "lhe_parts", "job_gen_daniel_catalog.sh": "gen_daniel", "job_lhe_gc.sh": "lhe_gc",
              "job_catalog_v3_short.sh": "catalog_v3_short"}
GREP = (r"Traceback|IndexError|OOM|[Oo]ut of memory|oom.kill|fail to reach target|No space left on device|\[FAIL\]|"
        r"StaleLayout|RUN_SUMMARY [^ ]+ rc=|DUE TO TIME LIMIT|LHE_PARTS_DONE|V3_SHORT_DONE|RESULT_JSON|RUN_DONE|"
        r"^Done: |exit [0-9]")
HEAD, TAIL = 22, 60


def rows(first, last):
    c = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    q = ("SELECT e.id, e.ts, e.item, e.kind, e.key, e.msg, e.acked, e.ack_note, r.id, r.job, r.note, r.sched_id, "
         "r.site, r.submitted, r.finished FROM events e LEFT JOIN runs r ON r.id = substr(e.key, 1, instr(e.key, ':') - 1) "
         "WHERE e.project = 'FA' AND e.id BETWEEN ? AND ? ORDER BY e.id")
    keys = ("event", "ts", "item", "kind", "key", "msg", "acked", "ack_note", "run", "job", "run_note", "sched_id",
            "site", "submitted", "finished")
    return [dict(zip(keys, r)) for r in c.execute(q, (first, last))]


def split_files(raw):
    """`head -v` / `tail -v` output -> {file: text}."""
    parts = re.split(r"^==> (\S+) <==$", raw, flags=re.M)
    return {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}


def read_site(site, files, what):
    cmd = {"head": "head -v -n %d %s" % (HEAD, " ".join(files)),
           "tail": "stat -c '%%n %%Y %%s' %s; tail -v -n %d %s" % (" ".join(files), TAIL, " ".join(files)),
           "grep": "grep -H -n -m 20 -E '%s' %s" % (GREP, " ".join(files))}[what]
    p = subprocess.run(["site", "--timeout", "570", "run", site, "FA", "--", cmd], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600)
    return p.stdout.decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2188)
    ap.add_argument("--last", type=int, default=2416)
    ap.add_argument("--raw", help="directory with heads.raw, tails.raw, grep.raw")
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "fixtures", "supervisor_20261029"))
    a = ap.parse_args()
    ev = rows(a.first, a.last)
    runs = [e for e in ev if e["kind"] == "RUN"]
    fname = {e["run"]: "%s_%s.out" % (LOG_PREFIX[os.path.basename(e["job"])], e["sched_id"]) for e in runs}
    files = sorted(set(fname.values()))
    site = runs[0]["site"]
    raw = {}
    for what in ("head", "tail", "grep"):
        if a.raw:
            raw[what] = open(os.path.join(a.raw, {"head": "heads", "tail": "tails", "grep": "grep"}[what] + ".raw")).read()
        else:
            raw[what] = read_site(site, files, what)
    heads, tails = split_files(raw["head"]), split_files(raw["tail"])
    stat = {}
    for ln in raw["tail"].splitlines():
        m = re.match(r"^(\S+\.out) (\d+) (\d+)$", ln)
        if m:
            stat[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    grep = {}
    for ln in raw["grep"].splitlines():
        m = re.match(r"^(\S+\.out):(\d+):(.*)$", ln)
        if m:
            grep.setdefault(m.group(1), []).append([int(m.group(2)), m.group(3)[:400]])
    missing = [f for f in files if f not in tails]
    if missing:
        sys.exit("logs not read: %s" % missing[:5])
    cut = lambda t: "\n".join(l[:400] for l in t.splitlines())
    out = []
    for e in ev:
        r = {k: e[k] for k in ("event", "ts", "item", "kind", "key", "acked", "ack_note")}
        if e["kind"] == "RUN":
            f = fname[e["run"]]
            r.update({"run": e["run"], "state": e["key"].split(":", 1)[1], "job": e["job"], "run_note": e["run_note"],
                      "sched_id": e["sched_id"], "submitted": e["submitted"], "finished": e["finished"],
                      "log": {"file": f, "head": cut(heads.get(f, "")), "tail": cut(tails[f]),
                              "matches": grep.get(f, []), "mtime": stat.get(f, (None, None))[0],
                              "size": stat.get(f, (None, None))[1]}})
        else:
            r["msg"] = e["msg"]
        out.append(r)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "run_ends.json"), "w") as f:
        json.dump(out, f, indent=0)
    kinds = {}
    for r in out:
        k = r["kind"] if r["kind"] != "RUN" else "RUN " + r["state"]
        kinds[k] = kinds.get(k, 0) + 1
    print(json.dumps({"events": len(out), "acked": sum(1 for r in out if r["acked"]), "by_kind": kinds}))


if __name__ == "__main__":
    main()
