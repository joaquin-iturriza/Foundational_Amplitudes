"""A ROUTINE: hand-back opens its HANDBACK event already acked; any other hand-back opens it unacked."""
import _tmpdir  # noqa: F401
import os, sqlite3, stat, sys, types, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".claude", "hooks"))
import notification_absorb as na  # noqa: E402


def _site_module():
    m = types.ModuleType("site_tool")
    m.__file__ = os.path.expanduser("~/.local/bin/site")
    exec(compile(open(m.__file__).read().split("\nif __name__")[0], m.__file__, "exec"), m.__dict__)
    return m


def _spool_args(tmp, report):
    fake = os.path.join(tmp, "fakesite")
    open(fake, "w").write('#!/bin/sh\necho "$@" > %s/args\necho E1\n' % tmp)
    os.chmod(fake, 0o755)
    na.CCORCH = tmp
    prompt = '<agent-message from="a0123456789abcdef">\nThe report follows:\n%s\n</agent-message>' % report
    ok, _ = na.spool(prompt, "a0123456789abcdef", site=(fake,))
    assert ok
    return open(os.path.join(tmp, "args")).read()


def test_spool_passes_ack_only_for_routine():
    with tempfile.TemporaryDirectory() as t:
        assert "--acked-note routine hand-back, auto-acked" in _spool_args(t, "ROUTINE: did it")
    with tempfile.TemporaryDirectory() as t:
        assert "--acked-note" not in _spool_args(t, "Joaquin, a finding")


def test_add_event_acked_note():
    s = _site_module()
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts, project, item, kind, key, msg, acked, ack_note, mute_until)")
    assert s.add_event(c, "FA", None, "HANDBACK", "k1", "m", "routine hand-back, auto-acked")
    assert s.add_event(c, "FA", None, "HANDBACK", "k2", "m")
    assert c.execute("SELECT key, acked IS NULL FROM events ORDER BY id").fetchall() == [("k1", 0), ("k2", 1)]
