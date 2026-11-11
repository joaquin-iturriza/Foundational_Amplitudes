"""CPU check that base_plots' run figures (plot_mse, plot_metric) are written as a .pdf and .png pair from the .pdf
path the caller passes (port of l2-gen 29edaf2d; the project's figure-pair rule).
    .venv-cpu/bin/python -m pytest tests/test_base_plots_pair.py
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import base_plots  # noqa: E402


def test_plot_mse_and_metric_write_pair():
    with tempfile.TemporaryDirectory() as d:
        base_plots.plot_mse(os.path.join(d, "mse.pdf"), [[1.0, 0.5, 0.2], [1.1, 0.6, 0.3]], labels=["train", "val"])
        base_plots.plot_metric(os.path.join(d, "metric.pdf"), [[1.0, 2.0, 3.0]], "m", labels=["a"])
        assert sorted(os.listdir(d)) == ["metric.pdf", "metric.png", "mse.pdf", "mse.png"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
