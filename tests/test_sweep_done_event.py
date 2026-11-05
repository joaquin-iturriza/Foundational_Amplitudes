"""Supervisor RUN_DONE for an ended sweep recorded on an entry (site_runends.sweep_done_events), mocked and read only."""
import importlib.util, os, sqlite3, sys, time, types

BIN = os.path.expanduser("~/.local/bin")


def load():
    spec = importlib.util.spec_from_file_location("site_runends", os.path.join(BIN, "site_runends.py"))
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)
    return rx


def mod():
    m = types.SimpleNamespace(os=os)

    def add_event(c, project, item, kind, key, msg):
        if c.execute("SELECT 1 FROM events WHERE project IS ? AND item IS ? AND kind=? AND key=? AND acked IS NULL",
                     (project, item, kind, key)).fetchone():
            return False
        c.execute("INSERT INTO events (ts,project,item,kind,key,msg) VALUES (?,?,?,?,?,?)",
                  (int(time.time()), project, item, kind, key, msg))
        return True
    m.add_event = add_event
    return m


def db():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts, project, item, kind, key, msg, acked)")
    return c


ROW = {"entry": "warm_start", "owner": "ag", "owner_state": "stopped", "site": "ccin2p3", "name": "ws_c1_k533_hpo8",
       "label": "x", "steps": ["3"]}
SUM = lambda r: {"evaluations": 8, "best_val_loss": "-2.75764"}


def run(rows, c=None, **kw):
    c = c or db()
    n = load().sweep_done_events(mod(), c, "FA", {"local": "~/work/FA"}, None, lambda *a: None, rows=rows, summary=SUM, **kw)
    return n, c


def test_complete_once_then_not_repeated_even_after_ack():
    n, c = run([ROW])
    assert n == 1
    msg = c.execute("SELECT msg FROM events").fetchone()[0]
    assert "ws_c1_k533_hpo8" in msg and "8 evaluations" in msg and "-2.75764" in msg
    c.execute("UPDATE events SET acked=1")
    assert run([ROW], c)[0] == 0 and c.execute("SELECT count(*) FROM events").fetchone()[0] == 1


def test_running_sweep_gives_no_event():
    assert run([])[0] == 0          # plan sweepends lists ended sweeps only


def test_sweep_no_wait_names_gives_no_event():
    assert run([dict(ROW, steps=[])])[0] == 0      # recorded on an entry, no step waits on it
    assert run([dict(ROW, entry="x", steps=[])])[0] == 0


def test_owner_working_no_event():
    assert run([dict(ROW, owner_state="working")])[0] == 0
