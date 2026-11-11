#!/usr/bin/env python
"""tools/kit_threshold_scan.py: the scan points are on shell and conserve momentum, and the quadrature's sigma has the
two-body normalisation (e+e- -> mu+mu- via a photon, massless: sigma = 4 pi alpha^2 / (3 s) from <|M|^2> = e^4 (1 + c^2)).

Run:  .venv-cpu/bin/python tests/test_kit_threshold_scan.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import kit_threshold_scan as ts  # noqa: E402


def test_grid_kinematics():
    sq, beta, c, w, mom = ts.grid()
    assert mom.shape == (len(sq) * len(c), 4, 4)
    tot = mom[:, 0] + mom[:, 1] - mom[:, 2] - mom[:, 3]
    assert np.abs(tot).max() < 1e-9
    m2 = mom[:, 2, 0] ** 2 - (mom[:, 2, 1:] ** 2).sum(-1)
    assert np.allclose(np.sqrt(m2), ts.MT, rtol=1e-9)
    assert np.allclose(np.sqrt(1 - 4 * ts.MT ** 2 / sq ** 2), beta, rtol=1e-9)


def test_sigma_normalisation():
    alpha, sq = 1 / 137.0, np.array([100.0, 500.0])
    c, w = np.polynomial.legendre.leggauss(ts.N_COS)
    me2 = np.concatenate([(4 * np.pi * alpha) ** 2 * (1 + c ** 2) for _ in sq])
    got = ts.sigma_pb(sq, np.ones(2), w, me2)
    assert np.allclose(got, ts.GEV2_PB * 4 * np.pi * alpha ** 2 / (3 * sq ** 2), rtol=1e-12)


def test_scale_band():
    # MG5_aMC's summary.txt layout (runs/kit/threshold_scan/amc_dry/Events/run_04 on CC)
    txt = ("  Scale variation (computed from histogram information):\n"
           "      Dynamical_scale_choice -2 (envelope of 9 values): \n"
           "          5.102e-01 pb  +9.3% -7.5%\n")
    assert ts.scale_band(txt) == (9.3, 7.5)
    assert ts.scale_band("Total cross section: 1.0e-01 +- 1e-4 pb") is None


if __name__ == "__main__":
    test_scale_band()
    test_grid_kinematics()
    test_sigma_normalisation()
    print("all ok")
