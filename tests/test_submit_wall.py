"""submit_guard's wall check: --hdr=--time below tools/short_wall.py's need for a job_catalog_v3_short.sh job is refused."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".claude", "hooks"))
import cmdparse  # noqa: E402
import submit_guard as g  # noqa: E402

G = "site submit ccin2p3 FA scripts/job_catalog_v3_short.sh --item x --est-gpu-hours 3 --hdr=--time=%s %s -- %s"
GROW = "--env LONG_HEADS=16 --env LONG_CKPT=true --env LONG_MEMMAP=true --env GROW_K=533 --env GROW_HP=1333"


def run(t, env, args):
    old = os.getcwd()
    os.chdir(ROOT)
    try:
        return g.wall(cmdparse.commands(G % (t, env, args)))
    finally:
        os.chdir(old)


def test_growth_pair_2h20_refused_with_value():          # FA-ccin2p3-92d91c6f / 760744cb
    out = run("02:20:00", GROW, "v4_het_default25_seed2")
    assert len(out) == 1 and "2.38 h" in out[0] and "--hdr=--time=02:30:00" in out[0], out


def test_growth_at_2h30_passes():
    assert run("02:30:00", GROW, "v4_het_default25_seed2") == []


def test_signhead_step61_3h_refused_3h30_passes():
    out = run("03:00:00", "--env SHORT_TAG=keyed", "v4_het_control_signhead")
    assert len(out) == 1 and "--hdr=--time=03:30:00" in out[0], out
    assert run("03:30:00", "--env SHORT_TAG=keyed", "v4_het_control_signhead") == []


def test_smoke_and_other_scripts_not_checked():
    assert run("00:10:00", "", "v4_het_control_signhead smoke") == []
    assert g.wall(cmdparse.commands("site submit ccin2p3 FA scripts/job_probe.sh --hdr=--time=00:01:00 -- x")) == []
