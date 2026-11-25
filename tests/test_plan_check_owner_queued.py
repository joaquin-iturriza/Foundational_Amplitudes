"""A SendMessage whose result says 'queued for delivery' resumes an owner that ends first, so the interim stop
notification written after it is not the owner's last word (2026-11-25, llm_compare owner afd7ef53b90b9afa4).

    .venv-cpu/bin/python tests/test_plan_check_owner_queued.py
"""
import _tmpdir  # noqa: F401
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, ".claude", "hooks")]
import plan_check
import plan

A = "afd7ef53b90b9afa4"


def _lines(queued):
    send = {"type": "assistant", "timestamp": "2026-11-25T12:05:39Z", "message": {"content": [
        {"type": "tool_use", "id": "tu1", "name": "SendMessage", "input": {"to": A, "message": "E9098"}}]}}
    res = {"type": "user", "timestamp": "2026-11-25T12:06:36Z", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "tu1", "content": [{"type": "text", "text":
         "Message queued for delivery to %s at its next tool round." % A if queued else "Resuming agent"}]}]}}
    note = {"type": "queue-operation", "operation": "enqueue", "timestamp": "2026-11-25T12:06:31Z", "content":
            "<task-notification><task-id>%s</task-id><status>completed</status></task-notification>" % A}
    return [json.dumps(r) for r in (send, note, res)]


def kinds(queued):
    since = plan.parse("2026-11-25T12:00:00Z")
    return [k for _, k, _, _ in plan_check.owner_events_of(_lines(queued), since)]


def test_queued_message_cancels_later_stop():
    assert kinds(True) == ["resumed"]       # the stop arrives before its result record, still not the last word


def test_unqueued_stop_still_counts():
    assert kinds(False) == ["resumed", "stopped"]


def test_repeated_refusal_prefixed_single_not():
    line = "llm_compare: resume its owner x (SendMessage): open events E1"
    assert plan_check.mark_repeats([line], []) == [line]
    assert plan_check.mark_repeats([line], [line]) == [plan_check.REPEAT + line]
    assert plan_check.mark_repeats([line, "other"], [line]) == [plan_check.REPEAT + line, "other"]


if __name__ == "__main__":
    test_queued_message_cancels_later_stop(); test_unqueued_stop_still_counts()
    test_repeated_refusal_prefixed_single_not(); print("ok")
