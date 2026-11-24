import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".claude", "hooks"))
import send_check  # noqa: E402


def test_figure_by_a_rejected_script_is_refused_from_any_checkout(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan" / "plan.yaml").write_text(
        "entries:\n- id: cmp\n  intent:\n  - text: I sent you a very particular plot\n    rejects: a/old.py\n")
    wt = tmp_path / "worktrees" / "wt"
    (wt / "a").mkdir(parents=True)
    (wt / ".git").write_text("gitdir: elsewhere\n")       # a worktree's .git is a file
    (tmp_path / "a").mkdir()
    figs = {"main_old": tmp_path / "a" / "old.py", "wt_old": wt / "a" / "old.py", "new": tmp_path / "a" / "new.py"}
    for name, script in figs.items():
        (tmp_path / (name + ".legend.txt")).write_text("# producer: %s abc\nours\n" % script)
        (tmp_path / (name + ".png")).write_text("")
    rej = send_check.rejected_producers(str(tmp_path))
    assert rej == {"a/old.py": ("cmp", "I sent you a very particular plot")}
    monkeypatch.setattr(send_check, "rejected_producers", lambda root=None: rej)
    hit = lambda n: any("Joaquin rejected on cmp" in b for b in
                        send_check.legend_findings({"file_path": str(tmp_path / (n + ".png"))}, str(tmp_path))[0])
    assert hit("main_old") and hit("wt_old") and not hit("new")
