"""BaseExperiment.__call__: a deliberate clean exit (SystemExit 0, e.g. data.memmap_prebuild after building the
cache) leaves the process with rc 0; any other exit or exception still fails it with rc 1."""
import pytest

from base_experiment import BaseExperiment


class _Exp(BaseExperiment):
    def __init__(self, exc):
        self.exc = exc

    def run_mlflow(self):
        raise self.exc


@pytest.mark.parametrize("exc,code", [(SystemExit(0), 0), (SystemExit(None), None), (SystemExit(2), 1),
                                      (RuntimeError("boom"), 1)])
def test_exit_code(exc, code):
    with pytest.raises(SystemExit) as e:
        _Exp(exc)()
    assert e.value.code == code
