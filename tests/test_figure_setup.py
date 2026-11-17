"""The figure setup fix of 2026-11-16 (Joaquin: 'why do you not do something about it instead of fixing each individual
occurrence of the mistake?'), on the recorded cases: kitvm_dist and kit_paper_figs / muon_long (a ratio drawn as a
panel of its own), the INPUT_ARMS legends (sentences over several lines), the V-beta caption (the threshold scan sent
with a caption that did not define its legend), an ad hoc montage, a draft figure sent loose."""
import os, subprocess, sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, ".claude", "hooks"))
import plot_style as ps  # noqa: E402
import send_check as sc  # noqa: E402

E = np.linspace(0, 1, 11)
T = np.arange(1, 11, dtype=float)


def test_ratio_as_own_cell_warned():
    """kitvm_dist (a87aeec4), kit_paper_figs and muon_long: the ratio as a cell of its own grid row"""
    fig, axs = ps.figure(ncols=2, nrows=2)
    axs[0][0].stairs(T, E, label="truth"); axs[1][0].stairs(T / T, E); axs[1][0].set_ylabel("pred / truth")
    w = ps.style_warnings(fig)
    assert any("ratio panel of its own" in x for x in w)


def test_ratio_label_forms():
    """a stand-alone ratio figure ("MSE ratio", repo-review 2026-11-16) is allowed; \\frac{pred}{true} is caught"""
    for lab, warned in (("MSE ratio", False), ("density ratio", False), (r"$\frac{\rm pred}{\rm true}$", True),
                        ("pred/true", True)):
        fig, ax = ps.figure()
        ax.set_ylabel(lab)
        assert any("ratio panel" in x for x in ps.style_warnings(fig)) == warned, lab


def test_greek_mapping():
    """the TeX-greek table (repo-review 2026-11-16: it was off by one from sigma on)"""
    assert sc.norm(r"$\sigma$") == sc.norm("σ") and sc.norm(r"$\phi$") == sc.norm("φ")
    assert sc.norm(r"$\tau$") not in sc.norm("only σ here") and sc.norm(r"$\omega$") == "ω"


def test_true_pred_hist_not_warned():
    fig, ax = ps.figure()
    ps.true_pred_hist(ax, [{"label": "model", "edges": E, "pred": T, "truth": T}], xlabel="x")
    assert ps.style_warnings(fig) == []


def test_input_arms_sentences_warned_and_new_labels_short():
    old = ["scratch,\n+ soft rescale + $\\log(s_{ij}/s)$\n+ $\\log\\beta^2$ of massive pairs",   # INPUT_ARMS @2f2cd689
           "fine-tuned, $10^5$ events, parent r5 + threshold tops"]
    for lab in old:
        fig, ax = ps.figure()
        ax.plot([0, 1], [0, 1], label=lab)
        ps.legend(ax)
        assert any("too long" in x for x in ps.style_warnings(fig)), lab
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import kit_figures as K
    for tag in list(K.INPUT_ARMS) + ["offoff", "plain", ""]:
        for mode in ("A", "A_ft", "B", "B_bulk"):
            lab = K.arm_label({"mode": mode, "tag": tag, "n": 900000})
            assert "\n" not in lab and len(ps.plain_text(lab)) <= ps.LEGEND_MAX_CHARS, lab


def test_save_writes_legend_file(tmp_path):
    fig, ax = ps.figure()
    ax.plot([1, 2], [1, 2], label=r"$V\beta$")
    ps.legend(ax)
    ps.save(fig, str(tmp_path / "f"))
    lines = open(tmp_path / "f.legend.txt").read().splitlines()
    assert lines[0].startswith("# producer: ") and lines[-1] == r"$V\beta$"


def _sidecar(base, labels, script=None, warns=()):
    script = script or os.path.join(ROOT, "tools", "kit_threshold_scan.py")
    blob = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD:tools/kit_threshold_scan.py"],
                          capture_output=True, text=True).stdout.strip()
    with open(base + ".legend.txt", "w") as f:
        f.write(f"# producer: {script} {blob}\n")
        f.writelines(f"# warn: {w}\n" for w in warns)
        f.writelines(l + "\n" for l in labels)
    open(base + ".png", "wb").close()


# the legend of kit_threshold_scan_d405_legend (label() @818fd569), as sent on 2026-11-13 03:28
VBETA_LEGEND = ["NLO scale band", "LO", "NLO", r"$\pm2\,\delta_{\rm MC}$ of the NLO (lower panel)",
                r"from scratch, $10^5$ events, log inputs", r"from scratch, $10^5$ events, log inputs + $\log\beta^2$",
                r"fine-tuned, $10^5$ events, parent r5", r"fine-tuned, $10^5$ events, parent r5 + threshold tops"]
VBETA_CAPTION = ("e⁺e⁻→tt̄ threshold scan from 11 November. KIT data, m_t = 173 GeV, α_s = 0.118. The plot is the "
                 "network's NLO cross section over the exact NLO, against β = √(1−4m_t²/s) from 10⁻³ to 0.6.\n- One "
                 "curve per tt̄ virtual model: from scratch with plain inputs, with the log β² input, fine-tuned from "
                 "the held-out parents, with hard threshold points added, and trained on V·β.\n- The legend is in the "
                 "second file.")


def test_vbeta_caption_refused(tmp_path):
    """since 2026-11-17 an entry in plain words names itself; a symbol entry still needs the caption"""
    base = str(tmp_path / "kit_threshold_scan_d405")
    _sidecar(base, VBETA_LEGEND + [r"FT+$V\beta$"])
    bad, shown = sc.legend_findings({"files": [base + ".png"], "caption": VBETA_CAPTION}, "/", drafts=str(tmp_path))
    assert any("does not define" in b and "FT+" in b for b in bad)
    assert not any("parent r5" in b.split("the legend holds")[0] for b in bad)   # plain words: no caption needed
    assert shown and "NLO scale band" in shown[0]          # the legend is shown at the send


def test_panel_files_and_strips_refused(tmp_path):
    """Joaquin 2026-11-17: one file per figure, legend inside; save_panels and legend_strip leave a warning"""
    figs = ps.panels(2)
    for fig, ax in figs:
        ax.plot([0, 1], [0, 1], label="AdamW optimizer")
    ps.save_panels(figs, str(tmp_path / "p"))
    ps.legend_strip(figs[0][1], str(tmp_path / "p_legend"))
    for b in ("p_a", "p_legend"):
        assert any("one file holding all its panels" in w for w in sc.read_legend(str(tmp_path / (b + ".png")))[2]), b


def test_legend_cell_fills_the_grid(tmp_path, capsys):
    fig, axs = ps.figure(ncols=2, nrows=2)
    for ax in (axs[0][0], axs[0][1], axs[1][0]):
        ax.plot([0, 1], [0, 1], label="Muon optimizer")
    ps.legend_cell(axs[1][1], axs[0][0])
    ps.save(fig, str(tmp_path / "g"))
    assert "empty" not in capsys.readouterr().out
    assert sc.read_legend(str(tmp_path / "g.png"))[3] == ["Muon optimizer"]


def test_caption_defining_every_entry_passes(tmp_path):
    base = str(tmp_path / "scan")
    _sidecar(base, ["truth", "scratch", r"FT+LO+$\beta$", r"$V\beta$"])
    cap = "truth: exact NLO; scratch: from scratch; FT+LO+β: fine-tuned with LO tops and the β input; Vβ: target V·β"
    bad, _ = sc.legend_findings({"files": [base + ".png"], "caption": cap}, "/", drafts=str(tmp_path))
    assert bad == []


def test_montage_and_uncommitted_and_warned_refused(tmp_path):
    open(tmp_path / "montage.png", "wb").close()           # no legend file: not drawn through ps.save
    bad, _ = sc.legend_findings({"files": [str(tmp_path / "montage.png")], "caption": ""}, "/", drafts=str(tmp_path))
    assert bad and "not drawn through ps.save" in bad[0]
    adhoc = tmp_path / "adhoc.py"; adhoc.write_text("print(1)\n")
    _sidecar(str(tmp_path / "a"), [], script=str(adhoc))
    bad, _ = sc.legend_findings({"files": [str(tmp_path / "a.png")], "caption": ""}, "/", drafts=str(tmp_path))
    assert any("not committed" in b for b in bad)
    _sidecar(str(tmp_path / "w"), [], warns=["axes 'pred / truth' is a ratio panel of its own"])
    bad, _ = sc.legend_findings({"files": [str(tmp_path / "w.png")], "caption": ""}, "/", drafts=str(tmp_path))
    assert any("ratio panel" in b for b in bad)


def test_draft_figure_goes_as_pages(tmp_path):
    (tmp_path / "paper.tex").write_text("\\includegraphics{figs/example_dist_a.pdf}\n")
    _sidecar(str(tmp_path / "example_dist_a"), [])
    bad, _ = sc.legend_findings({"files": [str(tmp_path / "example_dist_a.png")], "caption": ""}, "/",
                                drafts=str(tmp_path))
    assert bad and "tools/draft_figure.py" in bad[0]
    assert sc.legend_findings({"files": [str(tmp_path / "draft_v2.pdf")], "caption": ""}, "/",
                              drafts=str(tmp_path)) == ([], [])
    (tmp_path / "paper.tex").write_text("\\includegraphics{figs/muon_ft.pdf}\n")   # a one-file figure goes as itself
    _sidecar(str(tmp_path / "muon_ft"), ["AdamW optimizer"])
    bad, _ = sc.legend_findings({"files": [str(tmp_path / "muon_ft.png")], "caption": ""}, "/", drafts=str(tmp_path))
    assert not any("draft_figure" in b for b in bad)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))


def test_draft_figure_tool_output_accepted(tmp_path):
    """tools/draft_figure.py's file carries a legend file naming the tool; same stem as a draft figure, still sendable"""
    (tmp_path / "paper.tex").write_text("\\includegraphics{figs/draft_fig_x.pdf}\n")
    tool = os.path.join(ROOT, "tools", "draft_figure.py")
    blob = subprocess.run(["git", "-C", ROOT, "hash-object", tool], capture_output=True, text=True).stdout.strip()
    base = str(tmp_path / "draft_fig_x")
    open(base + ".png", "wb").close()
    open(base + ".legend.txt", "w").write(f"# producer: {tool} {blob}\nfrom scratch\n")
    bad, _ = sc.legend_findings({"files": [base + ".png"], "caption": "from scratch: the baseline"}, "/",
                                drafts=str(tmp_path))
    assert not any("paper draft" in b for b in bad)


def test_outside_legend_refused(tmp_path):
    """Joaquin 2026-11-17: ps.shared_legend and a raw fig.legend leave a warning, so send_check refuses the figure"""
    for i, draw in enumerate((lambda f, a: ps.shared_legend(f, a), lambda f, a: f.legend(loc="upper center"))):
        (fig, ax), = ps.panels(1)
        ax.plot([0, 1], [0, 1], label="AdamW optimizer")
        draw(fig, ax)
        ps.save(fig, str(tmp_path / f"o{i}"))
        assert any("outside legend" in w for w in sc.read_legend(str(tmp_path / f"o{i}.png"))[2]), i
