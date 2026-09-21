"""Scaled signed log: sgn(x) log(1+|x|/s) for sign-changing pools.

Run as a script (pytest is shadowed by the stray root-level py.py):
    python tests/test_signedlog_scale.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import (  # noqa: E402
    SIGNEDLOG_QUANTILE,
    get_fn,
    get_inv_fn,
    preprocess_amplitude,
    resolve_amp_trafos,
    signedlog_scale,
    undo_preprocess_amplitude,
)


def _signed_pool(n=200000, seed=0):
    """A stand-in for a sign-changing one-loop finite part: log-uniform |x| over
    twenty decades around 1e-5, random sign."""
    rng = np.random.default_rng(seed)
    mag = 10.0 ** rng.normal(-5.0, 4.0, size=n)
    return (rng.choice([-1.0, 1.0], size=n) * mag).reshape(-1, 1)


def test_positive_pool_keeps_plain_log():
    x = np.exp(np.random.default_rng(1).normal(size=(1000, 1)))
    assert resolve_amp_trafos(["log", "standardization"], x) == ["log", "standardization"]


def test_signed_pool_gets_a_scale_at_the_quantile():
    x = _signed_pool()
    tr = resolve_amp_trafos(["log", "standardization"], x)
    assert tr[0].startswith("signedlog:") and tr[1] == "standardization"
    s = signedlog_scale(tr[0])
    q = np.quantile(np.abs(x[x != 0]), SIGNEDLOG_QUANTILE)
    assert np.isclose(s, q, rtol=1e-5), (s, q)


def test_custom_quantile():
    x = _signed_pool()
    tr = resolve_amp_trafos(["log"], x, scale_quantile=0.5)
    assert np.isclose(signedlog_scale(tr[0]), np.median(np.abs(x)), rtol=1e-5)


def test_forward_inverse_roundtrip():
    x = _signed_pool()
    tr = resolve_amp_trafos(["log", "standardization"], x)
    y, m, s = preprocess_amplitude(x, trafos=tr)
    back = undo_preprocess_amplitude(y, m, s, trafos=tr)
    assert np.allclose(back, x, rtol=1e-6, atol=0.0)


def test_scale_tames_the_tail():
    """Without the scale the standardized target has a kurtosis of thousands (the
    transform is linear below |x|=1 and most events sit far below it); with it
    the tail is a few sigma."""
    x = _signed_pool()
    y0, _, _ = preprocess_amplitude(x, trafos=["signedlog", "standardization"])
    tr = resolve_amp_trafos(["log", "standardization"], x)
    y1, _, _ = preprocess_amplitude(x, trafos=tr)

    def kurt(v):
        v = v.ravel()
        return float(np.mean(v ** 4) / np.mean(v ** 2) ** 2)

    assert kurt(y0) > 5 * kurt(y1), (kurt(y0), kurt(y1))
    assert kurt(y1) < 10, kurt(y1)
    assert np.abs(y1).max() < 8, np.abs(y1).max()


def test_legacy_string_is_scale_one():
    x = np.array([[-3.0], [0.5], [2.0]])
    assert np.allclose(get_fn("signedlog")(x, None), np.sign(x) * np.log1p(np.abs(x)))
    assert np.allclose(get_inv_fn("signedlog")(get_fn("signedlog")(x, None), None), x)


def test_inverse_is_finite_for_wild_predictions():
    tr = resolve_amp_trafos(["log", "standardization"], _signed_pool())
    y = np.array([[-5000.0], [5000.0], [0.0]])
    out = undo_preprocess_amplitude(y, 0.0, 1.0, trafos=tr)
    assert np.isfinite(out).all()
    out = undo_preprocess_amplitude(y, 0.0, 1.0, trafos=["log", "standardization"])
    assert np.isfinite(out).all()


def test_abslog_for_the_sign_head():
    x = _signed_pool()
    tr = resolve_amp_trafos(["log", "standardization"], x, sign_head=True)
    assert tr[0].startswith("abslog:") and tr[1] == "standardization"
    assert np.isclose(signedlog_scale(tr[0]), np.quantile(np.abs(x[x != 0]), SIGNEDLOG_QUANTILE), rtol=1e-5)
    y, m, s = preprocess_amplitude(x, trafos=tr)
    mag = undo_preprocess_amplitude(y, m, s, trafos=tr)
    assert (mag > 0).all()
    big = np.abs(x) >= signedlog_scale(tr[0])
    assert np.allclose(mag[big], np.abs(x)[big], rtol=1e-6)
    assert np.allclose(mag * np.sign(x), x, rtol=1e-6, atol=signedlog_scale(tr[0]) * 1.01)
    # positive pools are untouched by the flag
    pos = np.exp(np.random.default_rng(2).normal(size=(500, 1)))
    assert resolve_amp_trafos(["log", "standardization"], pos, sign_head=True) == ["log", "standardization"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
