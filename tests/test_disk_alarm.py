#!/usr/bin/env python3
"""The disk alarm and the pre-submission disk check (plan entry agent_redesign step 24, D384), on a scratch event
table; the code under test is ~/.local/bin/site.

  D1  the replay of 2026-11-11: an ssh failure of the disk watch is acked and muted for 2 h (E4668), then the watch
      reports DISK: an ALERT opens at once (it was swallowed by the mute)
  D2  the same DISK line on the next passes opens nothing while that ALERT is open (one per key), and a new one
      after its ack's mute ends (it was a "repeat" for four days, 2026-11-07..11)
  S1  site submit's disk check: 52 GB free, batch needs 20 GB -> refused (32 GB left, under the 60 GB floor)
  S2  250 GB free, needs 20 GB -> goes;  S3  52 GB free, no --disk-gb -> goes with a warning;  S4  df unreadable -> goes

Run:  python3 tests/test_disk_alarm.py
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os
import tempfile
import time
import types

from test_run_done_waiter import load_site

def one_pass(m, tmp, line, rc=1):
    with open(os.path.join(tmp, "out.txt"), "w") as f:
        f.write(line + "\n")
    with open(os.path.join(tmp, "rc"), "w") as f:
        f.write(str(rc))
    cmd = "cat out.txt; exit $(test -s rc && cat rc || echo 1)"     # one command, as the watch is one entry
    m.cfg_projects = lambda: {"T": {"local": tmp, "maintain": [cmd]}}
    m.supervise_pass("T", log=lambda *a: None, beat=False)
    return cmd


def alerts(m):
    c = m.ldb()
    return c.execute("SELECT id, kind, key, acked FROM events WHERE project='T' ORDER BY id").fetchall()


def test_disk_alarm():
    with tempfile.TemporaryDirectory() as tmp:
        m = load_site(tmp)
        m.selftest = lambda p: []
        m.launch_pass = lambda *a, **k: 0
        m.ledger_on = lambda n: False
        cmd = one_pass(m, tmp, "kex_exchange_identification: Connection reset by peer", rc=255)
        ev = alerts(m)
        assert len(ev) == 1 and ev[0][1] == "MAINTAIN", ev
        c = m.ldb()   # ack with the 2 h mute E4668 had
        c.execute("UPDATE events SET acked=?, mute_until=? WHERE id=?", (int(time.time()), int(time.time()) + 7200,
                                                                          ev[0][0]))
        c.commit()
        one_pass(m, tmp, "DISK /sps/lpnhe at 99% (6 GB free)")
        ev = alerts(m)
        assert len(ev) == 2 and ev[1][1] == "ALERT" and ev[1][2].startswith("DISK "), ev
        print("D1 ok: DISK opens an ALERT under its own key despite the muted ssh-failure ack")
        one_pass(m, tmp, "DISK /sps/lpnhe at 93% (41 GB free)")
        assert len(alerts(m)) == 2, alerts(m)
        c.execute("UPDATE events SET acked=?, mute_until=? WHERE id=?", (int(time.time()) - 7300,
                                                                          int(time.time()) - 100, ev[1][0]))
        c.commit()
        one_pass(m, tmp, "DISK /sps/lpnhe at 93% (41 GB free)")
        ev = alerts(m)
        assert len(ev) == 3 and ev[2][1] == "ALERT", ev
        print("D2 ok: one open per key; a new ALERT once the ack's mute ended")


def test_submit_disk_check():
    with tempfile.TemporaryDirectory() as tmp:
        m = load_site(tmp)
        sc = {}
        for free_gb, need, ok, case in ((52, 20, False, "S1"), (250, 20, True, "S2"), (52, None, True, "S3"),
                                        (None, 20, True, "S4")):
            out = "%d\n" % int(free_gb * 1e9) if free_gb else "df: cannot read"
            m.run_ssh = lambda sc_, cmd, timeout=60, out=out: (0 if free_gb else 1, out)
            a = types.SimpleNamespace(site="ccin2p3", disk_gb=need)
            try:
                m.disk_check(a, sc, "/sps/x")
                went = True
            except SystemExit:
                went = False
            assert went == ok, (case, free_gb, need)
            print("%s ok: %s GB free, need %s -> %s" % (case, free_gb, need, "goes" if went else "refused"))


if __name__ == "__main__":
    test_disk_alarm()
    test_submit_disk_check()
