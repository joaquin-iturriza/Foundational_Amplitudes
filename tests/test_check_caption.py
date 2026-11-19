import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import check_caption as cc

FIG = os.path.join(os.path.dirname(__file__), "..", "analysis", "zng_uq", "figs", "zng_parent_learning.png")
GOOD = ("(a) NLL against step. (b) validation loss against step. (c) error against step. Legend: "
        "$u\\bar d\\to e^+\\nu_e g$; $u\\bar u\\to t\\bar t,\\ V$; $u\\bar u\\to t\\bar t$; "
        "training, all 28 processes; the other 25 processes; validation = training.")


def test_refuses_undefined_entries():
    assert cc.refusals(FIG, "Legend: - training events. (a) NLL against step.")


def test_accepts_defined_entries():
    assert cc.refusals(FIG, GOOD) == []
