"""Import first: every temp file of this process goes under one fa_tests_* dir, removed at exit (pytest or plain script)."""
import atexit, os, shutil, tempfile

DIR = tempfile.mkdtemp(prefix="fa_tests_")
tempfile.tempdir = os.environ["TMPDIR"] = DIR
atexit.register(shutil.rmtree, DIR, True)
