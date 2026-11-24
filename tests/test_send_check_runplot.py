import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".claude", "hooks"))
import send_check as sc
from test_send_check_axes import run

RUN = os.path.expanduser("~/.local/share/ccorch/artifacts/FA/FA-x-b92c2413/sps/runs/r/trial_1/plots_0/loss.png")


def test_run_plot_passes():
    r = run(RUN, "loss against step")
    assert r.returncode == 0, r.stderr


def test_adhoc_png_refused(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "a.pdf").write_bytes(b"x")
    r = run(str(tmp_path / "a.png"), "loss against step")
    assert r.returncode == 2 and "legend.txt" in r.stderr
    assert not sc.is_run_plot(str(tmp_path / "plots_0" / "a.png"))
