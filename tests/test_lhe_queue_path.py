"""A bare queue name resolves to $SCRATCH/lhe_queue/<name> (the workers' directory); a path is kept."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))


def test_queue_path(monkeypatch=None):
    os.environ["SCRATCH"] = "/scr"
    import lhe_parts
    assert lhe_parts.queue_path("hd_test_h5") == "/scr/lhe_queue/hd_test_h5"
    assert lhe_parts.queue_path("/scr/lhe_queue/q") == "/scr/lhe_queue/q"
    assert lhe_parts.queue_path("rel/q") == "rel/q"


if __name__ == "__main__":
    test_queue_path()
    print("ok")
