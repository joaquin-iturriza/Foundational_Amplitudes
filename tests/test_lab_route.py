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
