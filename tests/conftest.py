"""Every test's temp files live under one directory that is removed when the session ends (they piled up in /tmp)."""
import os, shutil, tempfile

_dir = []


def pytest_sessionstart(session):
    d = tempfile.mkdtemp(prefix="fa_tests_")
    _dir.append(d)
    tempfile.tempdir = os.environ["TMPDIR"] = d


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_dir.pop(), ignore_errors=True)
