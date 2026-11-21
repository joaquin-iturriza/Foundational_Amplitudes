"""drive.condor_floor: a config's HTCondor GPU requirements map to the 24G floor only when they name the devices (E8056)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep.drive import condor_floor  # noqa: E402


def test_condor_floor():
    assert condor_floor('regexp("A100|H100", TARGET.GPUs_DeviceName) && !regexp("MIG", TARGET.GPUs_DeviceName)') == "24G"
    assert condor_floor('!regexp("MIG", TARGET.GPUs_DeviceName)') is None
    assert condor_floor(None) is None
    assert condor_floor('regexp("A100|T4", TARGET.GPUs_DeviceName)') is None


if __name__ == "__main__":
    test_condor_floor()
    print("ok")
