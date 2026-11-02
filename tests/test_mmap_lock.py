"""dirlock.dir_lock: the atomic-mkdir lock of the memmap cache and the sweep state (flock on CC-IN2P3's NFS /sps
answers ENOLCK on some nodes, E3550): exclusion across processes, a dead holder's lock broken after STALE_S, a live
holder's lock kept by its heartbeat past STALE_S, release on an exception."""
import multiprocessing as mp
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dirlock  # noqa: E402


def _worker(path, log, n):
    for _ in range(n):
        with dirlock.dir_lock(path, poll_s=0.001):
            with open(log, "a") as f:
                f.write("in\n")
            time.sleep(0.002)
            with open(log, "a") as f:
                f.write("out\n")


def test_excludes_across_processes():
    with tempfile.TemporaryDirectory() as d:
        path, log = os.path.join(d, "k"), os.path.join(d, "log")
        ps = [mp.Process(target=_worker, args=(path, log, 20)) for _ in range(4)]
        [p.start() for p in ps]
        [p.join() for p in ps]
        lines = open(log).read().split()
        assert len(lines) == 160 and lines == ["in", "out"] * 80   # never two holders at once
        assert not os.path.exists(path + ".lockdir")


def test_dead_holder_broken_after_stale():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k")
        os.mkdir(path + ".lockdir")                       # a holder killed while holding: no heartbeat
        old = time.time() - 700
        os.utime(path + ".lockdir", (old, old))
        with dirlock.dir_lock(path, stale_s=600):
            assert os.path.exists(os.path.join(path + ".lockdir", "owner"))


def test_live_holder_kept_by_heartbeat():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k")
        waited = []
        with dirlock.dir_lock(path, heartbeat_s=0.05):
            old = time.time() - 10
            os.utime(path + ".lockdir", (old, old))          # looks stale ...
            time.sleep(0.2)                                   # ... until the heartbeat touches it
            age = time.time() - os.stat(path + ".lockdir").st_mtime
            assert age < 1.0
            q = mp.Queue()
            p = mp.Process(target=_try, args=(path, q))
            p.start(); p.join()
            waited.append(q.get())
        assert waited == ["held"]


def _try(path, q):
    def sleep(s):
        raise TimeoutError
    try:
        with dirlock.dir_lock(path, stale_s=5, sleep=sleep):
            q.put("took it")
    except TimeoutError:
        q.put("held")


def test_released_on_exception():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k")
        try:
            with dirlock.dir_lock(path):
                raise ValueError
        except ValueError:
            pass
        assert not os.path.exists(path + ".lockdir")


def test_heartbeat_survives_transient_error(monkeypatch):
    calls = []
    real = os.utime

    def flaky(p, *a):
        calls.append(p)
        if len(calls) == 1:
            raise OSError(116, "Stale file handle")
        return real(p, *a)

    monkeypatch.setattr(dirlock.os, "utime", flaky)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k")
        with dirlock.dir_lock(path, heartbeat_s=0.02):
            old = time.time() - 10
            real(path + ".lockdir", (old, old))
            time.sleep(0.2)
            assert time.time() - os.stat(path + ".lockdir").st_mtime < 1.0   # still beating after the ESTALE
        assert len(calls) >= 3


def test_late_breaker_puts_a_fresh_lock_back():
    """Two waiters judged one lock stale; the first broke it and holds a fresh lock. The second's rename takes the
    fresh lock: it must see a different owner, put it back, and not remove it."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k")
        lock = path + ".lockdir"
        os.mkdir(lock)
        with open(os.path.join(lock, "owner"), "w") as f:
            f.write("dead 1 x 0\n")
        old = time.time() - 700
        os.utime(lock, (old, old))
        real_owner = dirlock._owner
        state = {"n": 0}

        def owner(l):
            # the late breaker read the dead owner before the first breaker replaced the lock
            state["n"] += 1
            if state["n"] == 1:
                out = real_owner(l)
                # first breaker acts now: breaks the dead lock and takes a fresh one (owner "fresh", new mtime)
                dirlock._rmtree(lock)
                os.mkdir(lock)
                with open(os.path.join(lock, "owner"), "w") as f:
                    f.write("fresh 2 y 0\n")
                return out
            return real_owner(l)

        dirlock._owner = owner
        try:
            def sleep(s):
                raise TimeoutError
            try:
                with dirlock.dir_lock(path, stale_s=600, sleep=sleep):
                    raise AssertionError("took a live lock")
            except TimeoutError:
                pass
        finally:
            dirlock._owner = real_owner
        assert real_owner(lock) == "fresh 2 y 0"       # the fresh lock is back in place and intact


def test_release_keeps_a_lock_not_ours():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k")
        lock = path + ".lockdir"
        with dirlock.dir_lock(path):
            with open(os.path.join(lock, "owner"), "w") as f:     # someone else's lock now sits at the path
                f.write("other 3 z 0\n")
        assert os.path.exists(lock) and dirlock._owner(lock) == "other 3 z 0"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
