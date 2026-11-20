"""submit_guard's rundir check: a job_datasaving_parent.sh submission whose tag's run dir exists on the site is
refused before it is queued (FA-lxplus-a74362e3, 2026-11-20). The site read itself (`site run ... test -e`) is stubbed."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".claude", "hooks"))
import cmdparse  # noqa: E402
import submit_guard as g  # noqa: E402

SUB = ("site submit lxplus FA scripts/job_datasaving_parent.sh --item x --est-gpu-hours 1.6 --hdr=--time=02:30:00 -- "
       "heldoutbbzz ++model.input_pair_logs=6 ++training.scheduler=WSD %s")


def test_rundir():
    os.environ.pop("SUBMIT_TEST", None)
    seen = []
    g._site_path_exists = lambda site, p: (seen.append((site, p)), p.endswith("_wsdrerun2"))[1]
    assert g.rundir(cmdparse.commands(SUB % "tag=wsdrerun2"))                       # exists: refused
    assert seen[-1] == ("lxplus", "runs/data_saving/dsave_parent_heldoutbbzz_wsdrerun2")
    assert g.rundir(cmdparse.commands(SUB % "tag=wsdrerun3")) == []                 # new tag: passes
    g._site_path_exists = lambda site, p: None
    assert g.rundir(cmdparse.commands(SUB % "tag=x"))                               # unread: refused, retry
    n = len(seen)
    assert g.rundir(cmdparse.commands(SUB.replace("lxplus", "auto") % "tag=x"))     # auto: refused, no read
    assert g.rundir(cmdparse.commands("site submit lxplus FA scripts/job_config_smoke.sh -- sweep/a.yaml 50")) == []
    assert g.rundir(cmdparse.commands((SUB % "tag=wsdrerun2").replace("--item x", "--check --item x"))) == []
    assert len(seen) == n


if __name__ == "__main__":
    test_rundir(); print("ok")
