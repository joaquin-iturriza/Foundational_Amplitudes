"""plan_check refuses a Stop whose last message has a numeric markdown table and no SendUserFile png/pdf in the turn.

    .venv-cpu/bin/python tests/test_plan_check_table_figure.py
"""
import json, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, ".claude", "hooks")]
import plan_check

TABLE = "Joaquin, results:\n\n| arm | loss |\n|---|---|\n| a | 1.5e-3 |\n| b | 0.25 |\n"


def check(text, tools=()):
    recs = [{"type": "user", "uuid": "u1", "message": {"content": "go"}}]
    for name, inp in tools:
        recs.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t", "name": name,
                                                                   "input": inp}]}})
    recs.append({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})
    p = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    open(p, "w").write("\n".join(json.dumps(r) for r in recs) + "\n")
    return plan_check.result_table_without_figure({"transcript_path": p})


def test_table_without_figure_refused():
    assert check(TABLE) == [plan_check.TABLE_MSG]
    assert check(TABLE, [("SendUserFile", {"file_path": "/x/notes.txt"})]) == [plan_check.TABLE_MSG]


def test_table_with_png_passes():
    assert check(TABLE, [("SendUserFile", {"file_path": "/x/fig.png"})]) == []
    assert check(TABLE, [("SendUserFile", {"file_path": "/x/fig.pdf"})]) == []


def test_plain_numbers_pass():
    assert check("Joaquin, loss is 0.25 and 1.5e-3.") == []
    assert check("| a | b |\n|---|---|\n| x | y |\n") == []


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
