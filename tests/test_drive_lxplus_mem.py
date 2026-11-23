"""drive.lxplus_mem_ok: lxplus gives 3000 MB per CPU, so a request above 3000*cpus MB is refused there."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep.drive import lxplus_mem_ok  # noqa: E402


def test_lxplus_mem_ok():
    assert not lxplus_mem_ok("32G", 4)
    assert lxplus_mem_ok("12G", 8)
    assert lxplus_mem_ok("6000M", 2)
    assert not lxplus_mem_ok("6001", 2)
    assert not lxplus_mem_ok("lots", 4)


if __name__ == "__main__":
    test_lxplus_mem_ok()
    print("ok")
