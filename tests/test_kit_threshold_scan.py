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



def test_nlo_analytic_against_mg5_amc():
    """The analytic NLO: the vector formula's massless limit (3/4) and threshold expansion (hep-ph/9705254 eq. 14),
    and sigma_LO, sigma_NLO and the scale band against MG5_aMC's fixed-order runs on CC (smoke FA-ccin2p3-84d3488e
    at beta 1.2e-2, dry run amc_dry at beta 0.062)."""
    import numpy as np
    from tools import kit_threshold_scan as ts
    assert abs(ts.r1_vector(1 - 1e-7) - 0.75) < 1e-4
    v = 1e-3
    assert abs(ts.r1_vector(v) - (np.pi ** 2 * (1 + v * v) / (2 * v) - 4)) < 1e-3
    for beta, lo, nlo, up, dn in ((0.0119, 0.02047, 0.4423, 11.2, 9.0), (0.062, 0.1063, 0.5102, 9.3, 7.5)):
        sq = 2 * ts.MT / np.sqrt(1 - beta ** 2)
        _, sv, sa = ts.lo_vector_axial(sq)
        k, ku, kd = ts.nlo_analytic(sq, [0.1174, -0.0946])      # alpha_s variation read off MG5_aMC's band
        assert abs((sv + sa) / lo - 1) < 2e-3 and abs(k * (sv + sa) / nlo - 1) < 1e-3
        assert abs(ku - up) < 0.1 and abs(kd - dn) < 0.1


if __name__ == "__main__":
    test_scale_band()
    test_grid_kinematics()
    test_sigma_normalisation()
    test_nlo_analytic_against_mg5_amc()
    print("all ok")
