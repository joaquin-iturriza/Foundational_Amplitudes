#!/usr/bin/env python3
"""
nlo_madloop.py — thin loader/evaluator for a MadLoop [virt=QCD] standalone.

Exposes the f2py `get_me_full` built from f2py_wrapper.f (see
tools/compare_madloop_vs_dat.py for how the wrapper is added), returning the
full Laurent array per phase-space point:
    born = ANS(0),  finite = ANS(1),  1/eps = ANS(2),  1/eps^2 = ANS(3)
with MadLoop's standard alpha_s/2pi normalization already applied.

MadLoop resolves MadLoop5_resources / ident_card relative to CWD, so we chdir
into the P0 dir on load. Load ONE standalone per Python process (the f2py module
and ML5 globals are singletons).
"""

import glob
import os
import sys
import numpy as np


def load(so_dir):
    """Import + initialise the standalone in <so_dir> (a P0_* dir). Returns the
    get_me_full callable."""
    so_dir = os.path.abspath(so_dir)
    if not glob.glob(os.path.join(so_dir, "matrix2py*.so")):
        raise FileNotFoundError(
            f"matrix2py*.so not in {so_dir} — run `make matrix2py.so` there "
            f"(needs GET_ME_FULL in f2py_wrapper.f).")
    sys.path.insert(0, so_dir)
    os.chdir(so_dir)
    import matrix2py
    param_card = os.path.abspath(os.path.join(so_dir, os.pardir, os.pardir,
                                              "Cards", "param_card.dat"))
    init = getattr(matrix2py, "ml5_0_initialise", None) or getattr(matrix2py, "initialise")
    init(param_card)
    full = getattr(matrix2py, "ml5_0_get_me_full", None) or getattr(matrix2py, "get_me_full")
    return full


def evaluate(get_me_full, momenta, alphas=0.118):
    """momenta: (nexternal,4) array [E,px,py,pz] in the PROCESS particle order.
    mu_R^2 = s is taken from the incoming pair. Returns a dict of the Laurent
    coefficients normalized to born/(alpha_s/2pi):  c0 (finite), c1, c2."""
    P = np.asfortranarray(np.asarray(momenta, dtype=np.float64).T)  # (0:3, nexternal)
    pin = momenta[0] + momenta[1]
    s = pin[0]**2 - pin[1]**2 - pin[2]**2 - pin[3]**2
    ans, rc = get_me_full(P, alphas, s, -1)
    born, fin, e1, e2 = (float(ans[i]) for i in range(4))
    ao2pi = alphas / (2.0 * np.pi)
    out = {"s": s, "born": born, "rc": int(rc)}
    if born != 0.0:
        out["c0"] = fin / born / ao2pi   # finite coefficient (= virt_fin/born)
        out["c1"] = e1 / born / ao2pi    # single pole 1/eps
        out["c2"] = e2 / born / ao2pi    # double pole 1/eps^2
    return out
