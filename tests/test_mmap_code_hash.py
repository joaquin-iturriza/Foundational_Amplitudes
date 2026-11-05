"""data_mmap._code_hash (plan catalog_coverage step 36): the memmap cache key follows only the code the cached arrays
come from. An edit to the lock, the logging or the open in data_mmap.py keeps the key (no new ~23 GB cache); an edit to
the build, a streamed reduction or a constant of the format changes it.

CPU only:  python tests/test_mmap_code_hash.py   (or pytest)"""
import importlib.util
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = open(os.path.join(ROOT, "data_mmap.py")).read()


def _hash_of(text, tag):
    """_code_hash of a copy of data_mmap.py with `text` as its source, loaded as its own module"""
    d = tempfile.mkdtemp(prefix="zz_codehash_")
    p = os.path.join(d, f"data_mmap_{tag}.py")
    open(p, "w").write(text)
    spec = importlib.util.spec_from_file_location(f"data_mmap_{tag}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m._code_hash()


def _edit(old, new):
    assert SRC.count(old) == 1, old
    return SRC.replace(old, new)


def test_lock_and_logging_edits_keep_the_key():
    base = _hash_of(SRC, "base")
    lock = _edit('LOGGER.info(f"data.memmap: waiting for the lock of {key} on {socket.gethostname()}")',
                 'LOGGER.info(f"data.memmap: waiting on the lock of {key} ({socket.gethostname()})")')
    assert _hash_of(lock, "lock") == base
    opened = _edit('LOGGER.info(f"data.memmap: opening the cache {d}")', 'LOGGER.info(f"data.memmap: cache hit {d}")')
    assert _hash_of(opened, "open") == base


def test_build_reduction_and_format_edits_change_the_key():
    base = _hash_of(SRC, "base2")
    build = _edit("    # ---- pass 1: shapes", "    _ = None\n    # ---- pass 1: shapes")
    assert _hash_of(build, "build") != base
    red = _edit("def _rowsum(it):\n", "def _rowsum(it):\n    _ = None\n")
    assert _hash_of(red, "red") != base
    chunk = _edit("CHUNK = 1 << 22", "CHUNK = 1 << 21")
    assert _hash_of(chunk, "chunk") != base


if __name__ == "__main__":
    test_lock_and_logging_edits_keep_the_key()
    test_build_reduction_and_format_edits_change_the_key()
    print("test_mmap_code_hash: 2 tests passed")
