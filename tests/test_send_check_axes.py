import json, os, subprocess, sys

HOOK = os.path.join(os.path.dirname(__file__), "..", ".claude", "hooks", "send_check.py")


def run(path, caption):
    ev = {"tool_name": "SendUserFile", "cwd": "/", "tool_input": {"file_path": path, "caption": caption}}
    return subprocess.run([sys.executable, HOOK], input=json.dumps(ev), capture_output=True, text=True)


def test_x_y_axis_passes(tmp_path):
    p = tmp_path / "f.svg"
    assert run(str(p), "x axis: energy, y axis: loss").returncode == 0


def test_against_passes(tmp_path):
    assert run(str(tmp_path / "f.svg"), "kept/quota against log10|M|^2").returncode == 0


def test_neither_refused(tmp_path):
    r = run(str(tmp_path / "f.svg"), "loss curves of the arms")
    assert r.returncode == 2 and "x and y axes" in r.stderr and "f.svg" in r.stderr


def test_panels_each_checked(tmp_path):
    assert run(str(tmp_path / "f.svg"), "(a) loss against step (b) the arms").returncode == 2
    assert run(str(tmp_path / "f.svg"), "(a) loss against step (b) error against energy").returncode == 0


def test_non_figure_not_checked(tmp_path):
    assert run(str(tmp_path / "d.json"), "nothing").returncode == 0
