"""plot_style.save refuses a legend wider than its plot box (LegendOverflow, nothing written) and saves one that fits."""
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style as ps  # noqa: E402


def _fig(labels, ncol):
    fig, ax = ps.figure()
    for i, lab in enumerate(labels):
        ax.plot([0, 1], [i, i + 1], label=lab)
    # a raw ax.legend: ps.legend would shrink it to fit, which is what the check exists to enforce
    ax.legend(loc="upper left", ncol=ncol)
    return fig


def test_wide_legend_refused():
    fig = _fig([f"a long legend label number {i}" for i in range(4)], ncol=4)
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "wide")
        try:
            ps.save(fig, base)
        except ps.LegendOverflow:
            assert not os.path.exists(base + ".png") and not os.path.exists(base + ".pdf")
            return
    raise AssertionError("a legend wider than its plot box was saved")


def test_narrow_legend_saved():
    fig = _fig(["a", "b"], ncol=1)
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "narrow")
        ps.save(fig, base)
        assert os.path.exists(base + ".png") and os.path.exists(base + ".pdf")


def _strip(n_chars, ncol):
    fig, ax = ps.figure()
    for i in range(4):
        ax.plot([0, 1], [i, i + 1], label="x" * n_chars + str(i))
    ps.shared_legend(fig, ax, ncol=ncol)
    return fig


def test_fig_strip_inside_canvas_saved():
    fig = _strip(3, 2)
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "strip")
        ps.save(fig, base)
        assert os.path.exists(base + ".png") and os.path.exists(base + ".pdf")


def test_fig_strip_off_canvas_refused():
    fig = _strip(60, 4)
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "strip")
        try:
            ps.save(fig, base)
        except ps.LegendOverflow:
            assert not os.path.exists(base + ".png") and not os.path.exists(base + ".pdf")
            return
    raise AssertionError("a figure legend running off the canvas was saved")


if __name__ == "__main__":
    test_fig_strip_inside_canvas_saved()
    test_fig_strip_off_canvas_refused()
    test_wide_legend_refused()
    test_narrow_legend_saved()
    print("ok")
