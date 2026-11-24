"""plan.lab_route (D541): a RUN event's resume line starts with the mechanical-worker's DeepSeek triage step."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import plan


def test_run_event_gets_the_lab_route():
    r = plan.lab_route({"id": "x"}, [("E1", None, "E1  11-24 01:15 FA RUN x FA-ccin2p3-b77878e1:FAILED ended"),
                                     ("E2", None, "E2  11-24 01:15 FA DONE x batch ended")])
    assert "labllm.py triage FA-ccin2p3-b77878e1 --event E1" in r and "E2" not in r


def test_no_run_event_no_route():
    assert plan.lab_route({"id": "x"}, [("E2", None, "E2  11-24 01:15 FA DONE x batch ended")]) == ""


def test_lookup_items_checked_against_the_copy(tmp_path):
    import labllm
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "a.py").write_text("class Foo:\n    def bar(self):\n        pass\n")
    (tmp_path / ".claude" / "hooks").mkdir(parents=True)
    (tmp_path / ".claude" / "hooks" / "x.py").write_text("def name():\n    pass\n")
    got = labllm.cited(str(tmp_path), "`tools/a.py:bar`\n%s/tools/a.py:Foo\ntools/a.py:baz\ntools/b.py:bar\n"
                       ".claude/hooks/x.py:name\n./tools/a.py:bar\n" % tmp_path)
    assert got == [("tools/a.py:bar", True), ("tools/a.py:Foo", True), ("tools/a.py:baz", False), ("tools/b.py:bar", False),
                   (".claude/hooks/x.py:name", True), ("tools/a.py:bar", True)]


def test_cheap_check_material():
    import lab_llm_trial as lt
    assert lt.excerpt("a\nb\nboom at 12\nc", "boom at 12", 1) == "2: b\n3: boom at 12\n4: c" and lt.excerpt("a", "zz") == ""
    c = lt.compact(" f | 2 +-", "+x", "ok t_a\nPASS  M4: a failed item\nFAILED t_b")
    assert "== test: 1 failure lines, 2 pass lines" in c and c.index("FAILED t_b") < c.index("ok t_a")
    c = lt.compact("", "", "ERROR collecting tests/x.py\n1 error in 0.1s\nok t_a", rc=2)
    assert "== test: 2 failure lines, 1 pass lines" in c and c.rstrip().endswith("test rc 2")
