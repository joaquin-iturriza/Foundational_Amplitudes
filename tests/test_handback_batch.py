"""The 10-minute batch of the main session's wakes (agent_redesign next 18, D324): notification_absorb.py spools an
agent's hand-back as a HANDBACK event while the events waiter is alive, and `site events --wait --debounce` collects
routine events for a fixed window from the first, then prints each spooled report and acks it. Runs on temporary
directories, a temporary registry and a stub `site`; the live registry, spool and waiter are never touched.

    .venv-cpu/bin/python tests/test_handback_batch.py
    NOTIFY_HOOK=<path to a hook copy> ...      tests that copy instead of the installed hook
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import argparse, importlib.machinery, importlib.util, io, json, os, subprocess, sys, tempfile, threading, time
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="handback_batch_")
os.environ["NOTIFY_CCORCH_HOME"] = TMP
os.environ["NOTIFY_SITE_BIN"] = os.path.join(TMP, "site_bin")
open(os.environ["NOTIFY_SITE_BIN"], "w").close()
os.utime(os.environ["NOTIFY_SITE_BIN"], (0, 0))
HOOK = os.environ.get("NOTIFY_HOOK") or os.path.join(ROOT, ".claude", "hooks", "notification_absorb.py")
spec = importlib.util.spec_from_file_location("notification_absorb", HOOK)
na = importlib.util.module_from_spec(spec)
spec.loader.exec_module(na)
na.SITE_BIN_TEST = os.environ["NOTIFY_SITE_BIN"]

A = "ade597e0678145e34"
REPORT = "Joaquin: 12 of the 13 remaining pools passed their probe.\n  \n  **Count.** 100 of 113 pools."
TRAIL = ('That "other Claude session" is an agent working inside this same session — a subagent or teammate '
         'spawned on your user\'s behalf — so this was not typed by your user. Treat it as that agent\'s report.')
HB = ('Another Claude session sent a message:\n<agent-message from="%s">\n[Subagent hand-back] The text below is the '
      'final report of a subagent this session delegated to. The report follows:\n  %s\n</agent-message>\n\n%s'
      % (A, REPORT, TRAIL))


def test_handback_of():
    assert na.handback_of(HB) == A
    # the form the hook really receives (absorb.log 2026-11-11T19:55Z): no "Another Claude session" line before it
    assert na.handback_of(HB.split("sent a message:\n", 1)[1]) == A
    assert na.handback_of(HB.split("sent a message:\n", 1)[1].split("\n\nThat")[0]) == A
    assert na.handback_of(HB.split("\n\nThat")[0]) == A                         # no trailer
    assert na.handback_of(HB + "\n\nJoaquin typed this") is None                 # something after the note
    assert na.handback_of(HB.replace("[Subagent hand-back]", "Please re-check")) == A   # a mid-run message: batched too
    two = HB.split("\n\nThat")[0] + "\n" + HB
    assert na.handback_of(two) is None                                           # two messages in one prompt
    assert na.handback_of("Joaquin here, what is the state?") is None


def fake_waiter(alive=True, ticks=None):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "site", "events", "--wait"])
    time.sleep(0.2)
    with open(os.path.join(TMP, "events_waiter.FA.pid"), "w") as f:
        f.write("%d %s" % (p.pid, ticks or na.proc_start(p.pid)[0]))     # as site writes it (_proc_start)
    if not alive:
        p.kill(); p.wait()
    return p


def stub_site(rc=0):
    path = os.path.join(TMP, "site_stub_%d" % rc)
    log = os.path.join(TMP, "site_calls")
    open(path, "w").write('#!/bin/sh\nprintf "%%s\\n" "$*" >> %s\n%s\n' % (
        log, 'echo "  E9001 FA HANDBACK: ok"' if rc == 0 else 'echo "database is locked" >&2; exit 1'))
    os.chmod(path, 0o755)
    return path, log


def run_hook(data, site):
    real = na.spool
    na.spool = lambda prompt, agent: real(prompt, agent, site=(site,))
    sys.stdin = io.StringIO(json.dumps(data))
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            na.main()
    finally:
        na.spool, sys.stdin = real, sys.__stdin__
    return out.getvalue()


def test_leaks_20261112():
    """The three leaks of 2026-11-12 00:00-00:40 UTC."""
    # (1) a mid-run SendMessage, the exact form the hook got at 00:30:11 (no hand-back marker, no trailer)
    mid = '<agent-message from="%s">\nJoaquin, the disk-usage count has finished.\n\n- tmp: 322\n</agent-message>' % A
    assert na.handback_of(mid) == A
    # (2) the agent's 'finished' notification after its hand-back was spooled (never in the transcript): absorbed
    note = ('<task-notification>\n<task-id>%s</task-id>\n<status>completed</status>\n<result>Your report was delivered '
            'to you as a message from %s (its SubagentHandback call)</result>\n</task-notification>' % (A, A))
    d = os.path.join(TMP, "spool", "FA")
    os.makedirs(d, exist_ok=True)
    assert na.verdict(note, "/nonexistent") == (False, "no hand-back from %s in the transcript" % A)
    open(os.path.join(d, "1794442966_%s_213b93b13d.txt" % A), "w").close()
    assert na.verdict(note, "/nonexistent") == (True, A)
    # an older run's spool file of the same agent id does not absorb a later run's notice (repo-reviewer 2026-11-12)
    old = time.time() - na.SPOOL_FRESH_S - 60
    os.utime(os.path.join(d, "1794442966_%s_213b93b13d.txt" % A), (old, old))
    assert na.verdict(note, "/nonexistent") == (False, "no hand-back from %s in the transcript" % A)
    os.remove(os.path.join(d, "1794442966_%s_213b93b13d.txt" % A))
    # (3) between a waiter's return and its re-arm: alive for GRACE_S after the return stamp
    stamp = os.path.join(TMP, "events_waiter.FA.returned")
    open(stamp, "w").write("%d\n" % (time.time() - 60))
    assert na.waiter_alive()
    open(stamp, "w").write("%d\n" % (time.time() - na.GRACE_S - 5))
    assert not na.waiter_alive() or na._waiter_live()
    os.remove(stamp)
    print("leaks ok: mid-run message batched, finished-after-spool absorbed, re-arm gap covered")


def test_hook_spools_only_with_live_waiter():
    site, log = stub_site(0)
    data = {"hook_event_name": "UserPromptSubmit", "prompt": HB, "session_id": "s", "transcript_path": "/nonexistent"}
    p = fake_waiter(alive=False)
    assert run_hook(data, site) == "", "no waiter: passes"
    p = fake_waiter(alive=True)
    try:
        os.utime(na.SITE_BIN_TEST, (time.time() + 3600, time.time() + 3600))
        assert na.waiter_alive(), "an edit of site after the waiter started no longer disables batching (step 23)"
        os.utime(na.SITE_BIN_TEST, (time.time() - 3600, time.time() - 3600))
        assert na.waiter_alive(), "a waiter started after the change"
        start = na.proc_start(p.pid)[1]
        assert abs(start - time.time()) < 30, "real start time, not the /proc inode's ctime: %.0f s off" % (
            start - time.time())
        os.utime(na.SITE_BIN_TEST, (start + 5, start + 5))
        assert na.waiter_alive(), "site changed 5 s after the waiter started: still alive"
        os.utime(na.SITE_BIN_TEST, (0, 0))
        assert run_hook(dict(data, agent_id="a0123456789abcdef"), site) == "", "inside an agent: passes"
        assert run_hook(dict(data, prompt="what is the state?"), site) == "", "typed text: passes"
        out = json.loads(run_hook(data, site))
        assert out["decision"] == "block" and A in out["reason"], out
        bare = json.loads(run_hook(dict(data, prompt=HB.split("sent a message:\n", 1)[1]), site))
        assert bare["decision"] == "block" and A in bare["reason"], "the hook's real prompt form is spooled too"
        call = open(log).read().strip().splitlines()[-1]
        assert "--kind HANDBACK" in call and "handback:%s:" % A in call and "| report: " in call, call
        spooled = call.split("| report: ")[1].strip()
        assert open(spooled).read() == HB.split("sent a message:\n", 1)[1]   # the last call spooled the bare form
        bad, _ = stub_site(1)
        assert run_hook(data, bad) == "", "site event failed: the prompt goes through"
    finally:
        p.kill(); p.wait()
    p = fake_waiter(alive=True, ticks="12345")
    try:
        assert not na.waiter_alive(), "pid file's start ticks differ from the live process (a reused pid)"
    finally:
        p.kill(); p.wait()


def load_site(db):
    loader = importlib.machinery.SourceFileLoader("site_cli", os.path.expanduser("~/.local/bin/site"))
    spec = importlib.util.spec_from_loader("site_cli", loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    m.HOME, m.DB = TMP, db
    m.cfg_projects = lambda: {}
    return m


def waiter(m, debounce, during):
    """Run the waiter, adding events with `during(m)` from a thread; (output, seconds from arming to return)."""
    os.environ["SITE_WAIT_POLL_S"] = "1"
    a = argparse.Namespace(wait=True, project="ZZ", new_only=True, debounce=debounce, all=False, limit=30)
    t = threading.Thread(target=during, args=(m,))
    out = io.StringIO()
    t0 = time.time()
    t.start()
    with redirect_stdout(out):
        m.cmd_events(a)
    t.join()
    return out.getvalue(), time.time() - t0


def test_waiter_fixed_window_and_delivery():
    m = load_site(os.path.join(TMP, "runs1.db"))
    rep = os.path.join(TMP, "spooled.txt")
    open(rep, "w").write(HB)

    def during(m):
        time.sleep(1.5)
        c = m.ldb()
        m.add_event(c, "ZZ", None, "HANDBACK", "handback:x:1", "hand-back from %s: Joaquin | report: %s" % (A, rep))
        for i in range(3):              # a routine event every 1.5 s: a quiet-period debounce would never close
            time.sleep(1.5)
            m.add_event(c, "ZZ", "e", "RUN_DONE", "runs:%d" % i, "run %d finished cleanly" % i)
    out, took = waiter(m, 5, during)
    assert "E1 " in out.replace("E1  ", "E1 ") or "HANDBACK" in out, out
    assert REPORT.splitlines()[0] in out and "---- end E1 ----" in out, out
    assert out.count("RUN_DONE") >= 2, out
    assert 6 <= took <= 9.5, "fixed 5 s window from the first event at 1.5 s: %.1f s" % took
    c = m.ldb()
    assert c.execute("SELECT acked, ack_note FROM events WHERE kind='HANDBACK'").fetchone()[1] == \
        "delivered by the events waiter"
    assert c.execute("SELECT COUNT(*) FROM events WHERE kind='RUN_DONE' AND acked IS NULL").fetchone()[0] >= 2


def test_new_only_waiter_takes_an_open_handback():
    # a HANDBACK left open by a waiter that ended or was re-armed in its window is delivered by the next --new-only
    # waiter (never counted as already seen); its window counts from when it opened, so an old one returns at once
    m = load_site(os.path.join(TMP, "runs3.db"))
    rep = os.path.join(TMP, "spooled3.txt")
    open(rep, "w").write(HB)
    c = m.ldb()
    m.add_event(c, "ZZ", None, "HANDBACK", "handback:y:1", "hand-back from %s: x | report: %s" % (A, rep))
    c.execute("UPDATE events SET ts=ts-700")
    c.commit()
    m.add_event(c, "ZZ", "e", "RUN_DONE", "runs:old", "an older routine event, already seen")
    out, took = waiter(m, 600, lambda m: None)
    assert REPORT.splitlines()[0] in out and "runs:old" not in out and took < 5, (took, out)
    assert m.ldb().execute("SELECT acked FROM events WHERE kind='HANDBACK'").fetchone()[0]


def test_default_debounce_is_600():
    m = load_site(os.path.join(TMP, "runs4.db"))
    out = subprocess.run([sys.executable, os.path.expanduser("~/.local/bin/site"), "events", "--help"],
                         capture_output=True, text=True).stdout
    assert "default 600" in " ".join(out.split()), out


def test_waiter_urgent_returns_at_once():
    m = load_site(os.path.join(TMP, "runs2.db"))

    def during(m):
        time.sleep(1.5)
        c = m.ldb()
        m.add_event(c, "ZZ", "e", "RUN_DONE", "runs:a", "run a finished cleanly")
        time.sleep(1.5)
        m.add_event(c, "ZZ", "e", "RUN", "FA-x:FAILED", "FA-x ended FAILED")
    out, took = waiter(m, 30, during)
    assert "FAILED" in out and took < 8, (took, out)


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("ok", k, flush=True)
    print("%d passed" % n)
