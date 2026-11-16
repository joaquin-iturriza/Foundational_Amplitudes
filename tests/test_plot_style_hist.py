"""plot_style.true_pred_hist: x range is the truth's filled range, a model with under 1% of its events there is not
drawn, the ratio panel shares x and its lines cover every bin with true events, clipped at the ratio limits."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style as ps  # noqa: E402


def test_true_pred_hist():
    e = np.arange(11.0)                                   # bins 0..9; truth fills bins 2..6
    t = np.array([0, 0, 5, 10, 20, 10, 5, 0, 0, 0])
    good = dict(label="good", edges=e, truth=t, pred=np.array([0, 0, 5, 10, 40, 10, 1, 0, 0, 0]))
    off = dict(label="off", edges=e, truth=t, pred=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1000]))
    fig, ax = ps.figure()
    r, dropped = ps.true_pred_hist(ax, [good, off], ratio=(0.5, 1.5))
    assert ax.get_xlim() == (2.0, 7.0)
    assert dropped == ["off"]
    assert r.get_shared_x_axes().joined(ax, r)
    labels = [h.get_label() for h in ax.patches]
    assert labels == ["truth", "good"]
    rat = r.patches[0].get_data().values                  # the good model's ratio
    assert np.isnan(rat[:2]).all() and np.isnan(rat[7:]).all() and not np.isnan(rat[2:7]).any()
    assert np.allclose(rat[2:7], [1, 1, 1.5, 1, 0.5])     # 2.0 and 0.2 drawn at the limits
    lo, hi = r.get_ylim()
    assert lo < 0.5 and hi > 1.5


if __name__ == "__main__":
    test_true_pred_hist(); print("ok")
