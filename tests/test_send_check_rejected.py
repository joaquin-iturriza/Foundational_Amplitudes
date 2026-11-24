import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".claude", "hooks"))
import send_check  # noqa: E402


def test_figure_by_a_rejected_script_is_refused(tmp_path):
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan" / "plan.yaml").write_text(
        "entries:\n- id: cmp\n  intent:\n  - text: I sent you a very particular plot\n    rejects: old.py\n")
    for name in ("old", "new"):
        (tmp_path / (name + ".legend.txt")).write_text("# producer: %s/%s.py abc\nours\n" % (tmp_path, name))
        (tmp_path / (name + ".png")).write_text("")
    rej = send_check.rejected_producers(str(tmp_path))
    assert rej == {os.path.realpath(str(tmp_path / "old.py")): ("cmp", "I sent you a very particular plot")}
    send_check.rejected_producers = lambda root=None: rej
    bad, _ = send_check.legend_findings({"file_path": str(tmp_path / "old.png")}, str(tmp_path))
    assert any("Joaquin rejected on cmp" in b for b in bad)
    bad, _ = send_check.legend_findings({"file_path": str(tmp_path / "new.png")}, str(tmp_path))
    assert not any("rejected" in b for b in bad)
