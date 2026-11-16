"""Compile one figure of the paper draft standalone (legend strip plus panels, no caption).

    python3 tools/draft_figure.py <label> [--out DIR] [--tex ~/drafts/fm_amplitudes_2026-11/paper.tex]

Writes DIR/draft_fig_<label>.{pdf,png} (a leading 'fig:' dropped, ':' becomes '_') and a .legend.txt naming this tool and its
git blob, with the legend strip's text as the entries; send_check accepts that file instead of 'compiled draft pages'.
"""
import argparse, os, re, subprocess, sys, tempfile

HDR = (r"\documentclass{standalone}\usepackage{graphicx}\usepackage{amsmath,amssymb}" "\n"
       r"\graphicspath{{%s/}}" "\n"
       r"\newcommand{\pnl}[1]{\raisebox{-\height}{\includegraphics{#1}}}" "\n"
       r"\newcommand{\pnld}[2]{\raisebox{\dimexpr-\height-#1\relax}{\includegraphics{#2}}}" "\n"
       r"\newcommand{\widerow}[2][1]{#2}" "\n"
       r"\begin{document}\begin{tabular}{c}" "\n")


def block(tex, label):
    """Body of the figure environment holding \\label{label}, from after \\centering to before \\caption."""
    for m in re.finditer(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", tex, re.S):
        if "\\label{%s}" % label in m.group(0):
            b = m.group(0)
            b = b[b.index("\\centering") + len("\\centering"):b.index("\\caption")] if "\\centering" in b else b[:b.index("\\caption")]
            return b.strip().replace("\\hfill", r"\hspace{1em}")
    raise SystemExit("no figure with label %s" % label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--out", default=os.getcwd())
    ap.add_argument("--tex", default=os.path.expanduser("~/drafts/fm_amplitudes_2026-11/paper.tex"))
    a = ap.parse_args()
    d = os.path.dirname(os.path.abspath(a.tex))
    body = block(open(a.tex, encoding="utf-8").read(), a.label)
    base = os.path.join(a.out, "draft_fig_" + re.sub(r"^fig:", "", a.label).replace(":", "_"))
    with tempfile.TemporaryDirectory() as t:
        open(os.path.join(t, "f.tex"), "w").write(HDR % d + body + "\n\\end{tabular}\\end{document}\n")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"], cwd=t,
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stdout[-2000:])
        subprocess.run(["cp", os.path.join(t, "f.pdf"), base + ".pdf"], check=True)
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "150", base + ".pdf", base], check=True)
    me = os.path.abspath(__file__)
    blob = subprocess.run(["git", "-C", os.path.dirname(me), "hash-object", me], capture_output=True, text=True).stdout.strip()
    leg = re.search(r"includegraphics\{([^}]*legend[^}]*)\}", body)
    ents = []
    if leg:
        p = os.path.join(d, leg.group(1))
        txt = subprocess.run(["pdftotext", "-layout", p, "-"], capture_output=True, text=True).stdout
        ents = [e for e in re.split(r"\s{2,}|\n", txt) if e.strip()]
    with open(base + ".legend.txt", "w", encoding="utf-8") as f:
        f.write("# producer: %s %s\n" % (me, blob))
        f.writelines(e + "\n" for e in ents)
    print(base + ".png")


if __name__ == "__main__":
    main()
