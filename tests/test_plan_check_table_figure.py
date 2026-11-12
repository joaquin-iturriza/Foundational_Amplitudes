"""plan_check refuses a Stop whose turn has a markdown table of measured results, with or without a figure
(Joaquin 2026-11-12: "I don't want tables I need plots"); bookkeeping tables pass.

    .venv-cpu/bin/python tests/test_plan_check_table_figure.py
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
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
    return plan_check.result_table({"transcript_path": p})


def test_table_without_figure_refused():
    assert check(TABLE) == [plan_check.TABLE_MSG]
    assert check(TABLE, [("SendUserFile", {"file_path": "/x/notes.txt"})]) == [plan_check.TABLE_MSG]


def test_table_with_png_refused():
    assert check(TABLE, [("SendUserFile", {"file_path": "/x/fig.png"})]) == [plan_check.TABLE_MSG]
    assert check(TABLE, [("SendUserFile", {"file_path": "/x/fig.pdf"})]) == [plan_check.TABLE_MSG]


def test_table_earlier_in_turn_refused():
    """A table in an earlier text block of the turn reached Joaquin too, even when the last block is plain."""
    recs = [{"type": "user", "uuid": "u1", "message": {"content": "go"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": TABLE}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Joaquin, the plot is above."}]}}]
    p = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    open(p, "w").write("\n".join(json.dumps(r) for r in recs) + "\n")
    assert plan_check.result_table({"transcript_path": p}) == [plan_check.TABLE_MSG]


# the cell shapes of the main session's last six result tables up to 2026-11-12 00:25 UTC (21:41, 22:07, 22:49, 23:45,
# 00:01, 00:25), each sent with figures: interval, raised-dot power of ten, signed decimal, a decimal behind a count
REAL = ("| target | median [10%, 90%] | best |\n|---|---|---|\n| uū LO | 3.8 [2.7, 4.8] | 3.5 |\n",
        "| setup | MSE |\n|---|---|\n| tt̄ virtual from scratch, 10⁵ events | 1.6·10⁻⁴ |\n",
        "| bin | median against control |\n|---|---|\n| high-energy | +0.995 |\n",
        "| D | r5/r4 val | r5/r4 test |\n|---|---|---|\n| 316 | 1.89 | 1.70 |\n",
        "| arm | slope | MSE |\n|---|---|---|\n| Hard points added | −0.90 | 0.012 (within ±4% from there) |\n")


def test_replay_20261112():
    """Each real table shape is refused with a figure sent in the same turn. With TABLE_REPLAY=<json list of
    {ts, text}> (the six messages extracted from the main session's transcript, kept out of the repo) they are
    replayed whole."""
    for t in REAL:
        assert check("Joaquin, result:\n\n" + t, [("SendUserFile", {"file_path": "/x/fig.png"})]) == \
            [plan_check.TABLE_MSG], t
    if os.environ.get("TABLE_REPLAY"):
        msgs = json.load(open(os.environ["TABLE_REPLAY"]))
        for m in msgs:
            assert check(m["text"], [("SendUserFile", {"file_path": "/x/fig.png"})]) == [plan_check.TABLE_MSG], m["ts"]
        print("replayed %d real messages, all refused" % len(msgs))


# the four tables the gate refused on 2026-11-11 (main session 14:43, 15:09, 15:08 and the 19:03 audit): bookkeeping
AUDIT = ("| Cause | Failed jobs | When it failed |\n|---|---|---|\n| σ-fit stage looked for the wrong checkpoint name | 18 | "
         "after all 8000 training steps |\n| Training diverged at high learning rate | 5 | mid-training |\n")
TIMELINE = ("| When | What should have happened | What happened |\n|---|---|---|\n| 07:24 | Parents go out | Pools "
            "missing |\n| 10:50–11:30 | | Parents ran, then diverged around step 1000 at learning rate 1.88e-2 |\n")
COUNTS = "| Fine-tune target | Diverged / finished |\n|---|---|\n| e+e- → u ū LO | 0 / 34 |\n| e+e- → t t̄ LO | 2 / 22 |\n"
TRIGGERS = ("| Trigger | Kinds | Examples |\n|---|---|---|\n| I guessed the setup | 5 | \"the ledger can't delete\" |\n"
            "| Two rules disagree | 4 | who acknowledges events (about 40 acks) |\n")


def test_bookkeeping_tables_pass():
    for t in (AUDIT, TIMELINE, COUNTS, TRIGGERS):
        assert check("Joaquin, here:\n\n" + t) == [], t


def test_measured_cells_refused():
    for cell in ("72 s", "about 120×", "3×10⁻⁷", "19%", "~0.8 ± 0.1", "254M"):
        assert check("| arm | value |\n|---|---|\n| a | %s |\n" % cell) == [plan_check.TABLE_MSG], cell


def test_plain_numbers_pass():
    assert check("Joaquin, loss is 0.25 and 1.5e-3.") == []
    assert check("| a | b |\n|---|---|\n| x | y |\n") == []


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
