"""The 10-minute batch of the main session's wakes (agent_redesign next 18, D324): notification_absorb.py spools an
agent's hand-back as a HANDBACK event while the events waiter is alive, and `site events --wait --debounce` collects
routine events for a fixed window from the first, then prints each spooled report and acks it. Runs on temporary
directories, a temporary registry and a stub `site`; the live registry, spool and waiter are never touched.

    .venv-cpu/bin/python tests/test_handback_batch.py
    NOTIFY_HOOK=<path to a hook copy> ...      tests that copy instead of the installed hook
"""
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
    assert na.handback_of(HB.split("\n\nThat")[0]) == A                         # no trailer
    assert na.handback_of(HB + "\n\nJoaquin typed this") is None                 # something after the note
    assert na.handback_of(HB.replace("[Subagent hand-back]", "Please re-check")) is None   # a peer message, not a report
    two = HB.split("\n\nThat")[0] + "\n" + HB
    assert na.handback_of(two) is None                                           # two messages in one prompt
    assert na.handback_of("Joaquin here, what is the state?") is None


def fake_waiter(alive=True):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "site", "events", "--wait"])
    with open(os.path.join(TMP, "events_waiter.FA.pid"), "w") as f:
        f.write("%d 0" % p.pid)
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


def test_hook_spools_only_with_live_waiter():
    site, log = stub_site(0)
    data = {"hook_event_name": "UserPromptSubmit", "prompt": HB, "session_id": "s", "transcript_path": "/nonexistent"}
    p = fake_waiter(alive=False)
    assert run_hook(data, site) == "", "no waiter: passes"
    p = fake_waiter(alive=True)
    try:
        os.utime(na.SITE_BIN_TEST, (time.time() + 3600, time.time() + 3600))
        assert run_hook(data, site) == "", "a waiter older than its site code: passes"
        os.utime(na.SITE_BIN_TEST, (0, 0))
        assert run_hook(dict(data, agent_id="a0123456789abcdef"), site) == "", "inside an agent: passes"
        assert run_hook(dict(data, prompt="what is the state?"), site) == "", "typed text: passes"
        out = json.loads(run_hook(data, site))
        assert out["decision"] == "block" and A in out["reason"], out
        call = open(log).read().strip().splitlines()[-1]
        assert "--kind HANDBACK" in call and "handback:%s:" % A in call and "| report: " in call, call
        spooled = call.split("| report: ")[1].strip()
        assert open(spooled).read() == HB
        bad, _ = stub_site(1)
        assert run_hook(data, bad) == "", "site event failed: the prompt goes through"
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
