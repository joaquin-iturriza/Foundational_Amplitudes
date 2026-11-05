"""The site gate's 60-minute window counts an ended job at its elapsed time, a pending or running job at its limit
(site_gate.py, 2026-11-05). Mocked sacct records, no scheduler."""
import importlib.util
import json
import os
import tempfile
from unittest import mock

GATE = os.path.join(os.path.expanduser("~"), ".local", "bin", "site_gate.py")


def _gate():
    spec = importlib.util.spec_from_file_location("site_gate", GATE)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def _window(sacct_out, recs):
    g = _gate()
    path = os.path.join(tempfile.mkdtemp(), "w.jsonl")
    now = 10000.0
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(dict(r, t=now - 60, project="P", items=["i"])) + "\n")
    run = mock.Mock(return_value=mock.Mock(stdout=sacct_out))
    with mock.patch.object(g.subprocess, "run", run), mock.patch.object(g.shutil, "which", return_value="/x/sacct"):
        return g.window(path, "P", ["i"], now)


# JobID|State|ElapsedRaw(s)|TimelimitRaw(min)|ReqTRES|AllocTRES
def test_failed_after_3_min_counted_at_3_min():
    out = "11|FAILED|180|600|billing=1,gres/gpu=2|gres/gpu=2\n"
    assert abs(_window(out, [{"jobid": "11", "gpu_h": 20.0}]) - 2 * 3 / 60.0) < 1e-6


def test_running_job_counted_at_limit():
    out = "12|RUNNING|900|600|billing=1,gres/gpu=2|gres/gpu=2\n"
    assert abs(_window(out, [{"jobid": "12", "gpu_h": 20.0}]) - 20.0) < 1e-6


def test_pending_job_counted_at_limit():
    out = "13|PENDING|0|600|gres/gpu=2|\n"
    assert abs(_window(out, [{"jobid": "13", "gpu_h": 20.0}]) - 20.0) < 1e-6


def test_completed_job_counted_at_elapsed():
    out = "14|COMPLETED|7200|600|gres/gpu=1|gres/gpu=1\n"
    assert abs(_window(out, [{"jobid": "14", "gpu_h": 10.0}]) - 2.0) < 1e-6


def test_cancelled_unstarted_counts_zero_and_array_mixes():
    out = ("15|CANCELLED by 1|0|600|gres/gpu=1|\n"
           "16_0|FAILED|360|60|gres/gpu=1|gres/gpu=1\n16_1|RUNNING|100|60|gres/gpu=1|gres/gpu=1\n")
    assert _window(out, [{"jobid": "15", "gpu_h": 10.0}]) == 0.0
    assert abs(_window(out, [{"jobid": "16", "gpu_h": 2.0}]) - (1.0 + 0.1)) < 1e-6
