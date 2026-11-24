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

Legend rule (2026-11-16, Joaquin: "why do you not do something about it instead of fixing each individual occurrence
of the mistake?"): every figure (.png, .pdf) other than a deck, a draft or a run's own plot is read through
<base>.legend.txt, which plot_style.save and legend_strip write beside it. The send is refused when that file is
missing (the figure was not drawn through ps.save: an ad hoc montage or a hand-saved image), when the script it names
is not committed at the content that drew it (git blob), when the save recorded a style warning (a legend entry over
two lines of 40 characters, a ratio panel that is not a hist_ratio inset, a panel file or a legend strip of its own:
Joaquin 2026-11-17, one file per figure with its legend inside), or when the caption leaves a symbol entry undefined:
an entry in plain words (two words of three letters or more outside math) names its curve itself (Joaquin 2026-11-17,
figures read without the text); any other entry, reduced to its letters and digits, TeX symbols as their Unicode
letters, must appear in the caption reduced the same way. The legend entries are shown at every send. A panel file or
legend strip the paper draft includes (an includegraphics stem ending in _a.._j or _legend, of a .tex under ~/drafts)
is never sent loose: it goes as made by tools/draft_figure.py <label>; a one-file draft figure goes as itself.

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


def is_draft(path):
    """Compiled pages of the paper draft (draft_*.pdf, or under ~/drafts/) are a document: its prose is never linted."""
    n = os.path.normpath(os.path.expanduser(path))
    return n.lower().endswith(".pdf") and (os.path.basename(n).startswith("draft_")
                                           or n.startswith(os.path.expanduser("~/drafts") + os.sep))


def check(path, terms):
    """Lines of findings for one file (empty if it passes)."""
    if is_draft(path):
        return []
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
PANEL = re.compile(r"(?:^|(?<=\. ))\(([a-h])\)", re.M)   # only a letter opening a line or sentence


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
        if is_deck(p) or is_draft(p):
            continue
        if axes_missing(cap):
            out.append("%s: its caption must say what the x and y axes are (e.g. \"x axis: ..., y axis: ...\" or "
                       "\"<y quantity> against <x quantity>\", for each panel)" % p)
    return out


GREEK = dict(zip("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma tau upsilon phi "
                 "chi psi omega".split(), "αβγδεζηθικλμνξπρστυφχψω"))
GREEK.update({"varepsilon": "ε", "varphi": "φ", "vartheta": "θ", "ell": "l"})
WORDS = ("log", "ln", "exp", "sin", "cos", "tan", "min", "max", "lim")


def norm(text):
    """Letters and digits only, lowercase, TeX greek as its letter, Unicode super/subscripts as plain digits (NFKC):
    a legend entry and a caption reduced this way compare as substrings."""
    import unicodedata
    t = re.sub(r"\\([A-Za-z]+)", lambda m: GREEK.get(m.group(1).lower(), "") if m.group(1).lower() in GREEK
               else (m.group(1) if m.group(1) in WORDS else ""), text)
    t = unicodedata.normalize("NFKC", t)
    return "".join(c for c in t.lower() if c.isalnum())


def plain_words(entry):
    """True when a legend entry names its curve in words: two words of three letters or more outside $...$ math."""
    return len(re.findall(r"[A-Za-z]{3,}", re.sub(r"\$[^$]*\$", " ", entry))) >= 2


def legend_file(path):
    return re.sub(r"\.(png|pdf|svg)$", "", path, flags=re.I) + ".legend.txt"


def read_legend(path):
    """(producer script, blob, [warnings], [entries]) of a figure's legend file, None when it has none."""
    p = legend_file(path)
    if not os.path.exists(p):
        return None
    script, blob, warns, labels = "", "", [], []
    for line in open(p, encoding="utf-8").read().splitlines():
        if line.startswith("# producer: "):
            script, _, blob = line[len("# producer: "):].rpartition(" ")
        elif line.startswith("# warn: "):
            warns.append(line[len("# warn: "):])
        elif line.strip():
            labels.append(line)
    return script, blob, warns, labels


def committed(script, blob):
    """True when `script` is in git and some commit holds it at exactly `blob` (the content that drew the figure).
    shortcut: only the top-level script is checked, not the repo modules it imports (plot_style, helpers); hash
    those too if an uncommitted helper ever draws a sent figure."""
    if not blob or not os.path.isfile(script):
        return False
    d = os.path.dirname(script)
    r = subprocess.run(["git", "-C", d, "log", "--all", "-1", "--format=%h", "--find-object=" + blob, "--",
                        os.path.basename(script)], capture_output=True, text=True, timeout=30)
    return r.returncode == 0 and bool(r.stdout.strip())


def draft_stems(drafts=None):
    """Stems of every figure an \\includegraphics of a .tex under ~/drafts names."""
    import glob
    drafts = drafts or os.path.expanduser("~/drafts")
    out = set()
    for tex in glob.glob(os.path.join(drafts, "**", "*.tex"), recursive=True):
        try:
            body = open(tex, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
            out.add(re.sub(r"\.(pdf|png)$", "", os.path.basename(m.group(1).strip())))
    return out


def repo_rel(path):
    """`path` relative to the top of its own checkout (the main one or a worktree: the nearest parent holding .git),
    so the same script compares equal from any checkout; the absolute path when none holds it."""
    p = os.path.realpath(path)
    d = os.path.dirname(p)
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, ".git")):
            return os.path.relpath(p, d)
        d = os.path.dirname(d)
    return p


def rejected_producers(root=ROOT):
    """{repo-relative path: (entry, Joaquin's words)} of every file an entry's intent record says he rejected
    (`plan intent --rejects`, D534): a figure drawn by one, from any checkout, never reaches him again."""
    try:
        import yaml
        with open(os.path.join(root, "plan", "plan.yaml"), encoding="utf-8") as f:
            d = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) or {}   # 4 MB: C loader 0.2 s vs 3.6 s
    except Exception:
        return {}
    es = d.get("entries", d) if isinstance(d, dict) else d
    es = es.values() if isinstance(es, dict) else es
    out = {}
    for e in es or []:
        for x in (e.get("intent") or []) if isinstance(e, dict) else []:
            if x.get("rejects"):
                out[x["rejects"]] = (e.get("id"), x.get("text"))
    return out


def legend_findings(tool_input, cwd, drafts=None):
    """([refusal lines], [legend lines shown]) for the figures of a SendUserFile call (see the legend rule)."""
    cap = tool_input.get("caption") if isinstance(tool_input, dict) else None
    ncap = norm(cap if isinstance(cap, str) else "")
    bad, shown, stems = [], [], None
    rejects = None
    for p in sent_files(tool_input, (".png", ".pdf")):
        p = p if os.path.isabs(p) else os.path.join(cwd, p)
        if is_deck(p) or is_draft(p) or "run_plots" in os.path.normpath(p).split(os.sep):
            continue
        stems = draft_stems(drafts) if stems is None else stems
        lg0 = read_legend(p)
        made_by_tool = lg0 is not None and os.path.basename(lg0[0]) == "draft_figure.py"   # the figure itself
        stem = re.sub(r"\.(png|pdf)$", "", os.path.basename(p), flags=re.I)
        if not made_by_tool and stem in stems and re.search(r"_(?:[a-j]|legend)$", stem):
            bad.append("%s: a figure of the paper draft: make it standalone with tools/draft_figure.py <label> "
                       "(legend strip plus panels, no caption) and send that file, never the loose panel" % p)
            continue
        lg = read_legend(p)
        if lg is None:
            bad.append("%s: no %s: not drawn through ps.save by a committed tool (an ad hoc montage or a hand-saved "
                       "image); draw it with a committed script. If this is compiled pages of the paper draft (a document, "
                       "not a figure), name it draft_<something>.pdf and send it again; draft_*.pdf is never linted "
                       "as a figure." % (p, os.path.basename(legend_file(p))))
            continue
        script, blob, warns, labels = lg
        if script and rejects is None:
            rejects = rejected_producers()       # read only when a sent figure names a producer
        rej = rejects.get(repo_rel(script)) if script else None
        if rej:
            bad.append('%s: drawn by %s, which Joaquin rejected on %s ("%s"): draw it with the tool his reference '
                       'calls for (plan brief %s, first lines)' % (p, script, rej[0], rej[1], rej[0]))
            continue
        if not committed(script, blob):
            bad.append("%s: drawn by %s, not committed at the content that drew it: commit the script and redraw"
                       % (p, script or "no script"))
        bad += ["%s: %s" % (p, w) for w in warns]
        shown.append("%s: %s" % (os.path.basename(p), " | ".join(labels) if labels else "(no legend)"))
        miss = [l for l in labels if norm(l) and not plain_words(l) and norm(l) not in ncap]
        if miss:
            bad.append("%s: the caption does not define these legend entries: %s (the legend holds: %s)"
                       % (p, " | ".join(miss), " | ".join(labels)))
    return bad, shown


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
    lg, shown = legend_findings(ev.get("tool_input") or {}, cwd)
    if lg:
        sys.stderr.write("send_check: a figure must be one file drawn by a committed tool through ps.save, its legend "
                         "entries in plain words or each defined in the caption:\n  " + "\n  ".join(lg) + "\n")
        return 2
    if ax:
        sys.stderr.write("send_check: a figure needs its axes named in the caption:\n  " + "\n  ".join(ax) + "\n")
    if not bad:
        if not ax and shown:
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext":
                                                     "send_check: legend entries sent:\n" + "\n".join(shown)}}))
        return 2 if ax else 0
    sys.stderr.write(
        "send_check: this file does not read on its own (CLAUDE.md, Talking to me: name a thing by what it is, no code alone). Redraw it with "
        "plain words for each of these, or, for a term the user has accepted, add it with its plain definition to "
        "docs/glossary.txt:\n  " + "\n  ".join(bad) + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
