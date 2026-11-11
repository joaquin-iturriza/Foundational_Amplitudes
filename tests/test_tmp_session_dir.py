"""tests/conftest.py puts every test's temp dirs under one fa_tests_* dir that is gone after the pytest run.

    .venv-cpu/bin/python -m pytest tests/test_tmp_session_dir.py
"""
import glob, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_session_dir_removed():
    with tempfile.TemporaryDirectory() as outer:
        t = os.path.join(outer, "tests")
        os.makedirs(t)
        for f in ("conftest.py", "_tmpdir.py"):
            open(os.path.join(t, f), "w").write(open(os.path.join(ROOT, "tests", f)).read())
        open(os.path.join(t, "test_tiny.py"), "w").write(
            "import tempfile, os\ndef test_x():\n    d = tempfile.mkdtemp(prefix='tiny_')\n"
            "    open(os.path.join(os.environ['TMPDIR'], 'seen'), 'w').write(d)\n    assert os.path.basename(os.path.dirname(d)).startswith('fa_tests_')\n")
        env = dict(os.environ, TMPDIR=outer)
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", t], capture_output=True,
                           text=True, env=env, cwd=outer)
        assert r.returncode == 0, r.stdout + r.stderr
        assert not glob.glob(os.path.join(outer, "fa_tests_*")), os.listdir(outer)


def test_plain_script_leaves_nothing():
    with tempfile.TemporaryDirectory() as outer:
        open(os.path.join(outer, "_tmpdir.py"), "w").write(open(os.path.join(ROOT, "tests", "_tmpdir.py")).read())
        open(os.path.join(outer, "x.py"), "w").write(
            "import _tmpdir\nimport tempfile\ntempfile.mkdtemp(prefix='tiny_')\n")
        r = subprocess.run([sys.executable, "x.py"], capture_output=True, text=True, cwd=outer,
                           env=dict(os.environ, TMPDIR=outer))
        assert r.returncode == 0, r.stderr
        assert not glob.glob(os.path.join(outer, "fa_tests_*")), os.listdir(outer)
