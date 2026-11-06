"""submit_guard's epoch check: a training job under one epoch is refused unless SUBEPOCH_OK=1."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".claude", "hooks"))
import cmdparse  # noqa: E402
import submit_guard as g  # noqa: E402

G = "site submit ccin2p3 FA scripts/job_catalog_v3_short.sh --item x --est-gpu-hours 4 %s -- %s"
LONG = "--env LONG_HEADS=8 --env LONG_FACTOR=1 --env LONG_MEMMAP=true"
ARM = "v4_het_default25"


def run(sub):
    old = os.getcwd()
    os.chdir(ROOT)
    try:
        return g.epoch(cmdparse.commands(sub))
    finally:
        os.chdir(old)


def test_whole_set_w8_refused_and_counted():
    r = g.epoch_count(open(os.path.join(ROOT, "scripts/job_catalog_v3_short.sh")).read(), ARM,
                      {"LONG_HEADS": "8", "LONG_FACTOR": "1"}, ROOT)
    assert r[1] == 4264 and r[2] == 16384 and r[0] < 1, r
    out = run(G % (LONG, ARM))
    assert len(out) == 1 and "epochs, under one" in out[0], out


def test_1e4_per_process_passes():
    r = g.epoch_count(open(os.path.join(ROOT, "scripts/job_catalog_v3_short.sh")).read(), ARM, {}, ROOT)
    assert r[0] >= 1 and r[3] == 11530000, r
    assert run(G % ("", ARM)) == []


def test_subepoch_ok_passes():
    assert run(G % (LONG + " --env SUBEPOCH_OK=1", ARM)) == []


if __name__ == "__main__":
    test_whole_set_w8_refused_and_counted(); test_1e4_per_process_passes(); test_subepoch_ok_passes(); print("ok")
