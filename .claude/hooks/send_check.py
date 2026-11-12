#!/usr/bin/env python3
"""PreToolUse hook for SendUserFile: a figure or deck that reaches the user reads on its own (2026-10-27, the user:
"the rules say that your figures should be understandable and self contained ... And no cryptic unexplained names
too. So it's a general problem"; CLAUDE.md 0e).

For each file the tool call names: a .png is read through its sibling .pdf (every figure ships as both; a .png
without one is refused), a .pdf (figure or deck) through pdftotext, page by page. The send is refused (exit 2) when
the text holds, on any page:
  - an identifier with an underscore (snake_case, a config key, a file stem); a math subscript (one letter, then up
    to three characters: s_0, m_k, c_v) is not one, so formulas keep their symbols;
  - a code-like token that lost its underscore in the PDF (TeX draws \\_ as a rule, so "catalog\\_v3" extracts as
    "catalog v3"): a short lowercase stem glued to digits, such as v3, tp3, pp13 (a math word glued to digits, min2pi from "min 2p_i p_j", is mathtext residue, not one);
  - a run id (FA-<site>-<hash>, trial_NNNN) or a bare hex hash;
  - a recipe or arm prefix (tp3_, tp2_, v3_, v4_, pp13_, dc_, zz_);
  - a held-out axis code (H1..H9 with any suffix, H2p5, H7E, H7c, XV3, XV4) or a decision or event code (D12, E345);
  - a label with "ratio" or "gain" that is not written as a formula a/b, or a "/" with one side empty (a quantity is math, never spelled out in words)
    on one side.

Axes rule (2026-11-07, the user: "what is the x axis in the top plots?"): a figure (.png, .pdf, .svg) sent with a
`caption` is also refused unless the caption names both axes: an x description ("x axis", "x-axis", "x:",
"horizontal", or an x followed later by "against") and a y description ("y axis", "y-axis", "y:", "vertical"); the
pattern "<y quantity> against <x quantity>" gives both. When the caption names panels "(a)", "(b)", ..., each
panel's text must do so (a text before the first letter counts for all panels, so a shared "x axis ..." sentence
there is enough).

A term listed in docs/glossary.txt (term<TAB>plain definition) is allowed, in its underscore or its spaced form.

Read-only use: python3 .claude/hooks/send_check.py --scan FILE... prints the findings (a .txt file is read as one
page, for tests) and exits 1 when there are any.
"""
import json, os, re, subprocess, sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
PREFIXES = ("tp3", "tp2", "v3", "v4", "pp13", "dc", "zz")
RULES = [  # (what it is, pattern); order = priority when several match the same string
    ("run id", re.compile(r"\bFA-[A-Za-z0-9]+-[0-9a-f]{4,}\b|\btrial_\d+\b")),
    ("recipe or arm prefix", re.compile(r"\b(?:%s)_\w*" % "|".join(PREFIXES))),
    ("held-out axis code", re.compile(r"\bH[1-9](?:p\d|[A-Za-z])?\b|\bXV[0-9]\b")),
    ("decision or event code", re.compile(r"\b[DE]\d+\b")),
    ("hex hash", re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b")),
    ("identifier with an underscore", re.compile(r"\b(?![A-Za-z]_[A-Za-z0-9]{1,3}\b)\w*[A-Za-z0-9]_\w*|\b_\w+")),
    ("code-like token (a lost underscore?)", re.compile(r"(?<![\w.])(?!(?:log|ln|min|max|exp|sin|cos|tan)\d)(?:[a-z]{2,4}\d+[a-z0-9]*|v\d+)\b")),
]
RATIO = re.compile(r"\b(ratios?|gains?)\b", re.I)
BOTH_SIDES = re.compile(r"[A-Za-z]\S*\s*/\s*\S*[A-Za-z0-9]")


def glossary(root=ROOT):
    terms = []
    p = os.path.join(root, "docs", "glossary.txt")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            if "PROPOSED" in line and line.lstrip().startswith("#"):
                break                     # entries below the marker wait for the user's acceptance: not allowed yet
            t = line.split("\t", 1)[0].strip()
            if t and not t.startswith("#"):
                terms.append(t)
    return terms


def mask(text, terms):
    """Blank out every glossary term (case-insensitive, whole words; a term's underscores also match a space, since
    TeX's \\_ extracts as a gap) so nothing inside one is reported."""
    for t in sorted(terms, key=len, reverse=True):
        pat = r"(?<!\w)" + r"[_ ]".join(re.escape(w) for w in re.split(r"[_ ]", t)) + r"(?!\w)"
        text = re.sub(pat, lambda m: " " * len(m.group(0)), text, flags=re.I)
    return text


def pages(path):
    """[(page number, text)] of a file, or raise ValueError with why it cannot be read."""
    if path.lower().endswith(".txt"):
        return [(1, open(path, encoding="utf-8").read())]
    if path.lower().endswith(".png"):
        pdf = path[:-4] + ".pdf"
        if not os.path.exists(pdf):
            raise ValueError("no sibling .pdf (%s): every figure ships as both, and the check reads the PDF's text"
                             % pdf)
        path = pdf
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", path, "-"], capture_output=True, text=True)
    if r.returncode:
        raise ValueError("pdftotext failed: %s" % r.stderr.strip()[:200])
    return [(i + 1, t) for i, t in enumerate(r.stdout.split("\f")) if t.strip()]


def findings(text, terms):
    """[(what, string)] in the order found, each string once."""
    out, seen = [], set()
    clean = mask(text, terms)
    for line, raw in zip(clean.splitlines(), text.splitlines()):   # masking keeps lengths and newlines
        taken = []                         # spans already reported by a higher-priority rule
        for what, rx in RULES:
            for m in rx.finditer(line):
                if any(a < m.end() and m.start() < b for a, b in taken):
                    continue
                taken.append(m.span())
                if m.group(0) not in seen:
                    seen.add(m.group(0))
                    out.append((what, m.group(0)))
        s = raw.strip()                  # unmasked: a glossary term names a side of a ratio
        if RATIO.search(s) and not BOTH_SIDES.search(s):
            key = ("ratio or gain not written as a formula a/b", s)
        elif "/" in s and re.search(r"(^|\s)/|/(\s|$)", s) and not BOTH_SIDES.search(s):
            key = ("'/' with a side unnamed", s)
        else:
            continue
        if key[1] not in seen:
            seen.add(key[1])
            out.append(key)
    return out


def check(path, terms):
    """Lines of findings for one file (empty if it passes)."""
    if "run_plots" in os.path.normpath(path).split(os.sep):   # a run's own plot, copied unchanged: labels are the training code's
        return []
    try:
        pp = pages(path)
    except (OSError, ValueError) as e:
        return ["%s: cannot be read: %s" % (path, e)]
    lines = []
    for n, t in pp:
        for what, s in findings(t, terms):
            lines.append("%s, page %d: %s: %r" % (path, n, what, s))
    return lines


FIG_EXT = (".png", ".pdf", ".svg")
X_AXIS = re.compile(r"\bx[\s-]*axis|\bx\s*:|horizontal|\bx\b.*\bagainst\b", re.I | re.S)
Y_AXIS = re.compile(r"\by[\s-]*axis|\by\s*:|vertical", re.I)
AGAINST = re.compile(r"\S.*\bagainst\b\s*\S", re.I | re.S)
PANEL = re.compile(r"\(([a-h])\)")


def has_axes(text):
    return bool(AGAINST.search(text)) or (bool(X_AXIS.search(text)) and bool(Y_AXIS.search(text)))


def axes_missing(caption):
    """True when a caption does not name both axes (per panel when it names panels)."""
    caption = caption or ""
    parts = PANEL.split(caption)          # [shared, letter, text, letter, text, ...]
    if len(parts) < 5:
        return not has_axes(caption)
    shared = parts[0]
    return any(not has_axes(shared + " " + parts[i]) for i in range(2, len(parts), 2))


def is_deck(path):
    """A slide-deck PDF (under a talks/ directory) has no single pair of axes: exempt from the axes rule."""
    return path.lower().endswith(".pdf") and "talks" in os.path.normpath(path).split(os.sep)


def axes_findings(tool_input, cwd):
    cap = tool_input.get("caption") if isinstance(tool_input, dict) else None
    cap = cap if isinstance(cap, str) else ""
    out = []
    for p in sent_files(tool_input, FIG_EXT):
        if is_deck(p):
            continue
        if axes_missing(cap):
            out.append("%s: its caption must say what the x and y axes are (e.g. \"x axis: ..., y axis: ...\" or "
                       "\"<y quantity> against <x quantity>\", for each panel)" % p)
    return out


def sent_files(tool_input, exts=(".png", ".pdf")):
    """Every .png/.pdf path (or exts) in the tool input, whatever its field is called."""
    out = []
    def walk(x):
        if isinstance(x, str):
            if x.lower().endswith(exts):
                out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(tool_input)
    return out


def main(argv):
    terms = glossary()
    if argv[1:2] == ["--scan"]:
        bad = [l for p in argv[2:] for l in check(p, terms)]
        print("\n".join(bad) if bad else "clean")
        return 1 if bad else 0
    try:
        ev = json.load(sys.stdin)
    except ValueError:
        return 0
    if ev.get("tool_name") != "SendUserFile":
        return 0
    cwd = ev.get("cwd") or ROOT
    bad = []
    for p in sent_files(ev.get("tool_input") or {}):
        bad += check(p if os.path.isabs(p) else os.path.join(cwd, p), terms)
    ax = axes_findings(ev.get("tool_input") or {}, cwd)
    if ax:
        sys.stderr.write("send_check: a figure needs its axes named in the caption:\n  " + "\n  ".join(ax) + "\n")
    if not bad:
        return 2 if ax else 0
    sys.stderr.write(
        "send_check: this file does not read on its own (CLAUDE.md 0e: plain names, no internal codes). Redraw it with "
        "plain words for each of these, or, for a term the user has accepted, add it with its plain definition to "
        "docs/glossary.txt:\n  " + "\n  ".join(bad) + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
