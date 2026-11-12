"""check_site_data --paths: a default-share mixture lists its component pools, and one recipe's sampling never
carries into the next recipe of the same call."""
import os, subprocess, sys, tempfile
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = {"name": "v4_bb_abbh", "sqrts": [155, 6000], "n_train": 100, "n_val": 10, "n_test": 10}
MIX = {"mode": "mixture", "f_fwd": 0.15, "fwd_cand": 1 / 3, "f_default": 0.25, "default_source": "dsh_bb_abbh"}


def run(t, *args):
    env = dict(os.environ, SCRATCH=t, DATA_DIR=t)
    return subprocess.run([sys.executable, "tools/check_site_data.py", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True).stdout


def paths(t, *specs):
    return {l.split(" ", 1)[1] for l in run(t, "--paths", *specs).splitlines() if l.startswith("POOL ")}


def line_of(out, spec):
    return next(l for l in out.splitlines() if l.startswith(os.path.basename(spec) + ":"))


def test_mixture_components_and_no_carry_over():
    with tempfile.TemporaryDirectory() as t:
        a, b = os.path.join(t, "a.yaml"), os.path.join(t, "b.yaml")
        yaml.safe_dump({"processes": [dict(P, sampling=MIX)]}, open(a, "w"))
        yaml.safe_dump({"processes": [P]}, open(b, "w"))
        pa, pb, pab = paths(t, a), paths(t, b), paths(t, a, b)
        assert any("dsh_bb_abbh" in f for f in pa)          # the mixture's default-share pools are listed
        assert not any("dsh_bb_abbh" in f for f in pb)
        assert pab == pa | pb
        assert line_of(run(t, a, b), b) == line_of(run(t, b), b)   # b after a resolves as b alone (no carried sampling)


if __name__ == "__main__":
    test_mixture_components_and_no_carry_over(); print("ok")
