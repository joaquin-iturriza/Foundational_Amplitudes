import os, sys
import matplotlib
matplotlib.use("Agg")
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style as ps  # noqa: E402


def _fig(labels, xlabel="step", ylabel="loss", title=""):
    fig, ax = ps.figure()
    for i, l in enumerate(labels):
        ax.plot([0, 1], [i, i + 1], label=l)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    return fig


def test_current_script_labels_refused(tmp_path):
    fig = _fig(["parent A", "scratch B", "w8 parent A", "w16 scratch B"])
    with pytest.raises(ps.CodedLabel, match="parent A"):
        ps.save(fig, str(tmp_path / "f"))
    assert not list(tmp_path.iterdir())


def test_words_pass(tmp_path):
    fig = _fig(["parent, share 0", "from scratch", "width-8 parent", "width-16 from scratch"])
    ps.save(fig, str(tmp_path / "f"))
    assert (tmp_path / "f.png").exists() and (tmp_path / "f.pdf").exists()


def test_physics_math_passes(tmp_path):
    fig = _fig([r"$e^+e^-\to W^+W^-$"], xlabel=r"$\log_{10}|\mathcal{M}|^2$", ylabel="MSE (GeV$^{-2}$)")
    ps.save(fig, str(tmp_path / "f"))


@pytest.mark.parametrize("t", ["C1t", "C10w", "H2p5", "PP4", "EE4", "HD5", "D213", "E45", "k=533", "w16",
                               "FA-ccin2p3-d521160b", "b688a859", "arm B", "W"])
def test_codes_refused(t):
    assert ps.code_in_label(t) is not None


@pytest.mark.parametrize("t", ["GeV", "val loss", "PP4"[:0] or "pp", "MSE of log", "step", r"$k=533$", "10 TeV"])
def test_plain_allowed(t):
    assert ps.code_in_label(t) is None


@pytest.mark.parametrize("t", ["AdamW L2 weight $\\lambda$", "training wall time on one V100 [h]", "A100 / H100", "NLO QCD MSE"])
def test_real_names_allowed(t):
    assert ps.code_in_label(t) is None


@pytest.mark.parametrize("t", ["C1t", "parent A"])
def test_still_refused(t):
    assert ps.code_in_label(t) is not None
