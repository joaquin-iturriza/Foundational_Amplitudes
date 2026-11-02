"""data_mmap._lock_polling: the cache lock survives ENOLCK (CC-IN2P3 NFS, 2026-11-02) and still excludes a second
holder."""
import errno
import fcntl
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_mmap  # noqa: E402


def test_retries_enolck_then_holds(monkeypatch):
    calls, sleeps = [], []
    real = fcntl.flock

    def fake(f, op):
        calls.append(op)
        if len(calls) <= 2:
            raise OSError(errno.ENOLCK, "No locks available")
        return real(f, op)

    monkeypatch.setattr(data_mmap.fcntl, "flock", fake)
    with tempfile.NamedTemporaryFile("w") as f:
        data_mmap._lock_polling(f, poll=0.0, sleep=sleeps.append)
    assert len(calls) == 3 and len(sleeps) == 2
    assert all(op & fcntl.LOCK_NB for op in calls)


def test_other_errors_raise(monkeypatch):
    def fake(f, op):
        raise OSError(errno.EBADF, "bad fd")

    monkeypatch.setattr(data_mmap.fcntl, "flock", fake)
    with tempfile.NamedTemporaryFile("w") as f:
        try:
            data_mmap._lock_polling(f, poll=0.0, sleep=lambda s: None)
        except OSError as e:
            assert e.errno == errno.EBADF
        else:
            raise AssertionError("EBADF was swallowed")


def test_waits_for_holder():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.lock")
        with open(p, "w") as a, open(p, "w") as b:
            data_mmap._lock_polling(a, poll=0.0)
            waited = []

            def release(s):          # the first poll finds it held; release it, the second takes it
                waited.append(s)
                fcntl.flock(a, fcntl.LOCK_UN)

            data_mmap._lock_polling(b, poll=0.0, sleep=release)
            assert len(waited) == 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
