"""generate_sweep.next_available_name: only its own three-digit counter is read as a counter (plan catalog_coverage
step 39a: cov_v4_long_hpo8_cc_17056 became cov_v4_long_hpo8_17057)."""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sweep.generate_sweep import next_available_name


def _state(d, name):
    os.makedirs(os.path.join(d, name)); open(os.path.join(d, name, "dyhpo_state.pkl"), "w").close()


def test_step_suffix_kept():
    with tempfile.TemporaryDirectory() as d:
        _state(d, "cov_x_17056")
        assert next_available_name("cov_x_17056", d) == "cov_x_17056_002"
        assert next_available_name("cov_new_17056", d) == "cov_new_17056"


def test_counter_advances():
    with tempfile.TemporaryDirectory() as d:
        _state(d, "cov_y"); _state(d, "cov_y_002")
        assert next_available_name("cov_y", d) == "cov_y_003"
        assert next_available_name("cov_y_002", d) == "cov_y_003"


if __name__ == "__main__":
    test_step_suffix_kept(); test_counter_advances(); print("ok")
