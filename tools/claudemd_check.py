#!/usr/bin/env python3
"""Composition of the CLAUDE.md rules files: how much of the text is rule, reference, provenance or duplicate.

    python3 tools/claudemd_check.py [FILE ...] [--list] [--json] [--save-baseline P] [--baseline P [--tol 1.0]]

Default files: this repo's CLAUDE.md and ~/work/CLAUDE.md (the two every agent call loads). Size is counted in
characters of whitespace-normalised text; tokens are estimated as characters / 4.

The classification is by the explicit rules below, applied in order, so the same text always gets the same numbers.

1. Units. The text is cut into blocks: a blank line, a heading (`#`), a list item (`-`, `*`, `1.`, `0b.`) or a table
   row starts a new block; a fenced code block (```) is one block. A block is cut into sentences at `.`, `!` or `?`
   followed by a space and a capital, digit, backtick, quote, bracket or `*` (common abbreviations excepted); a piece
   of four words or fewer (a list number, a run-in title such as "Routing.") joins the next one. A table row and a
   code block are one unit each.
2. HEAD (not counted in the shares): headings, `---` rules, a table's header and separator rows.
3. PROVENANCE, by characters: a bracket or parenthesis group (under 450 characters) that contains a date
   (2026-10-26), a commit (`commit abc1234`, `fixed abc1234`), "my call|decision|approval|preference|coverage
   decision" or "open (unsourced)"; and the whole cell of a table column headed "Source". A bare decision pointer
   ([D145]) is not provenance: the rules file may keep it.
4. DUPLICATE, whole unit: (a) hook mechanics: the unit names a hook script (`*_guard.sh|py`, `*_check.py`,
   `review_backlog.sh gate|check|bashpre|bashpost`, `auto_push.sh`, `stop_once`, `reviewer_resume.py`,
   `context_guard.py`) or a hook event (PreToolUse, PostToolUse, SessionStart, PreCompact, "Stop hook",
   "`Stop` hook"), or `autoMemoryEnabled`; (b) a repeat: a sentence with at least 8 content words whose word
   3-shingles are at least 70% contained in one earlier sentence (earlier in the same file, or in an earlier file
   on the command line). Semantic restatements in other words are not caught: that is the reviewer's job.
5. REFERENCE, whole unit: a table row; a code block; a list item that opens with a backticked name followed by
   " — ", " - " or ":" (a definition) unless it carries a strong marker (never, always, must, do not, don't); a
   sentence that ends in ":" (an introduction); any other sentence without a normative marker (6).
6. RULE: a sentence with a normative marker (NORMATIVE below): a modal or prohibition word (never, always, must,
   only, should, do not, don't, no, not, avoid, unless, instead); a prescriptive verb (keep, use, go(es), come(s),
   stay, wins, needs, requires, means, waits, asks, confirm, refuses, names, says, passes, carries, belongs, reads,
   records, "is shown|read|reported|fine|free", exempt, "is my call", "is a bug"); a party a rule binds or serves (me,
   my, I, you, your, the user, agent(s), main session); or a sentence that opens with an imperative verb (IMPERATIVE).

7. Topic-bound: a RULE sentence that names, in backticks, a code area (`sweep/`, `models/`, `config/`, `recipes/`,
   `analysis/`, `data/`, `optim/`, `sites/`, `mg5_pipeline`, `experiment.py`, `dataset.py`, `wrappers.py`, `run.py`,
   `particle_ids`), a config key (`training.`, `data.`, `evaluation.`, `model.`, `net.`, `fine_tune.`, `cluster.`,
   `dyhpo.`) or an `LLOCA_` toggle. It stays in the rule share and is reported apart ("topic-bound"): such a rule
   has a file trigger and belongs in that topic's pushed section unless it is one of the rules kept on purpose.

The shares reported are of the counted characters (everything but HEAD). `--list` prints every non-rule unit (and every topic-bound rule) with
its file:line, class and first words, so each one can be found and moved. `--save-baseline P` writes the shares as
JSON; `--baseline P` exits 1 when the rule share falls more than `--tol` points below the saved one.
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILES = [os.path.join(REPO, "CLAUDE.md"), os.path.join(os.path.dirname(REPO), "CLAUDE.md")]

NEW_BLOCK = re.compile(r"^(#|\s{0,3}[-*] |\s{0,3}\d+[a-z]?\. |\|)")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"“(*\[])")
ABBREV = re.compile(r"\b(e\.g|i\.e|vs|etc|cf|approx|resp|Fig|Eq|Sec|al)\.$")
PROV_IN_GROUP = re.compile(r"20\d\d-\d\d-\d\d|\bcommit [0-9a-f]{7}|\bfixed [0-9a-f]{7}|"
                           r"\bmy (?:call|decision|approval|preference|coverage decision)\b|open \(unsourced\)|"
                           r"\*\*open \(unsourced\)\*\*")
HOOK = re.compile(r"\b\w+_(?:guard|check)\.(?:sh|py)\b|review_backlog\.sh`?\s*(?:gate|check|bashpre|bashpost)\b|"
                  r"\bauto_push\.sh\b|\bstop_once\b|\breviewer_resume\.py\b|\bcontext_guard\.py\b|"
                  r"\b(?:PreToolUse|PostToolUse|SessionStart|PreCompact)\b|\bStop`? hook\b|autoMemoryEnabled")
STRONG = re.compile(r"\b(?:never|always|must|do not|don't)\b", re.I)
NORMATIVE = re.compile(
    r"\b(?:never|always|must|only|should|do not|don't|avoid|unless|instead|no|not|"
    r"keeps?|uses?|go(?:es)?|comes?|stays?|wins|needs?|requires?|is my call|is a bug|means|waits?|asks?|confirm|"
    r"refuses?|names?|says?|passe?s?|carr(?:y|ies)|belongs?|reads?|records?|is (?:shown|read|reported|fine|free)|"
    r"exempt|me|my|I|you|your|the user|agents?|main session)\b", re.I)
IMPERATIVE = {
    "add", "answer", "ask", "check", "commit", "compare", "confirm", "copy", "delegate", "edit", "estimate", "flag",
    "ignore", "keep", "launch", "make", "name", "pass", "plan", "put", "read", "record", "remove", "reuse", "rewrite",
    "run", "save", "say", "search", "send", "source", "spawn", "start", "submit", "tell", "test", "use", "verify",
    "work", "write", "state", "report", "treat", "set", "follow", "sweep", "tune", "seed", "center", "centre",
}
DEFINITION = re.compile(r"^\s*(?:[-*]|\d+[a-z]?\.)?\s*(?:\*\*)?`[^`]+`(?:\*\*)?[^—:]{0,60}?(?: — | - |:)")
TOPIC = re.compile(r"`[^`]*(?:\b(?:sweep|models|config|recipes|analysis|data|optim|sites)/|mg5_pipeline|experiment\.py|"
                   r"dataset\.py|wrappers\.py|\brun\.py|particle_ids|\b(?:training|data|evaluation|model|net|fine_tune|"
                   r"cluster|dyhpo)\.\w|LLOCA_)[^`]*`")
STOP = set("the a an of to in on at for and or is are be by it its as with from that this than then into".split())


def blocks(text):
    """(first line number, block text) in file order (rule 1)."""
    out, cur, start, fence = [], [], 0, False
    for n, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        if fence:
            cur.append(line)
            if s.startswith("```"):
                out.append((start, "\n".join(cur)))
                cur, fence = [], False
            continue
        if s.startswith("```"):
            if cur:
                out.append((start, "\n".join(cur)))
            cur, start, fence = [line], n, True
            if s.count("```") > 1:
                out.append((start, line))
                cur, fence = [], False
            continue
        if (s == "" or NEW_BLOCK.match(line)) and cur:
            out.append((start, "\n".join(cur)))
            cur = []
        if s:
            if not cur:
                start = n
            cur.append(line)
    if cur:
        out.append((start, "\n".join(cur)))
    return out


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def sentences(text):
    parts, buf = [], ""
    for piece in SENT_SPLIT.split(text):
        buf = (buf + " " + piece).strip() if buf else piece
        if ABBREV.search(buf) or len(buf.split()) <= 4:
            continue
        parts.append(buf)
        buf = ""
    if buf:
        parts.append(buf)
    return parts


def prov_mask(text):
    """Character mask of provenance groups (rule 3): outermost bracket/paren groups that match PROV_IN_GROUP."""
    mask = [False] * len(text)
    stack, spans = [], []
    for i, c in enumerate(text):
        if c in "([":
            stack.append(i)
        elif c in ")]" and stack:
            j = stack.pop()
            seg = text[j:i + 1]
            if len(seg) < 450 and PROV_IN_GROUP.search(seg):
                spans.append((j, i + 1))
    for a, z in spans:
        for k in range(a, z):
            mask[k] = True
    return mask


def words(s):
    return [w for w in re.findall(r"[a-z0-9_μα∞]+", s.lower()) if w not in STOP]


def shingles(ws):
    return {tuple(ws[i:i + 3]) for i in range(max(0, len(ws) - 2))}


class Doc:
    def __init__(self):
        self.units = []      # (file, line, cls, chars, text)
        self.seen = []       # (shingles, file, line) of earlier sentences

    def add(self, path, cls, chars, line, text):
        if chars > 0:
            self.units.append((path, line, cls, chars, text))

    def repeat_of(self, sent):
        ws = words(sent)
        if len(ws) < 8:
            return None
        sh = shingles(ws)
        for other, f, ln in self.seen:
            if sh and len(sh & other) / len(sh) >= 0.7:
                return "%s:%d" % (label(f), ln)
        return None

    def remember(self, sent, path, line):
        ws = words(sent)
        if len(ws) >= 8:
            self.seen.append((shingles(ws), path, line))


def classify_sentence(s):
    if s.rstrip().endswith(":"):
        return "reference"
    if NORMATIVE.search(s):
        return "rule"
    first = re.sub(r"^\s*(?:[-*]|\d+[a-z]?\.)?\s*(?:\*\*)?", "", s).split(" ", 1)[0].strip("*`\"(,").lower()
    if first in IMPERATIVE:
        return "rule"
    return "reference"


def analyse(paths):
    doc = Doc()
    for path in paths:
        text = open(os.path.expanduser(path), encoding="utf-8").read()
        table_source_col = None
        for line, raw in blocks(text):
            b = norm(raw)
            s = raw.strip()
            if s.startswith("#") and not s.startswith("```") or re.fullmatch(r"-{3,}", s):
                doc.add(path, "head", len(b), line, b)
                continue
            if s.startswith("|"):
                cells = [c.strip() for c in s.strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    doc.add(path, "head", len(b), line, b)
                    continue
                if any(c.lower() == "source" for c in cells) and not any("`" in c for c in cells):
                    table_source_col = [c.lower() for c in cells].index("source")
                    doc.add(path, "head", len(b), line, b)
                    continue
                if any(c.lower() in ("knob", "what") for c in cells[:1]):
                    table_source_col = None
                    doc.add(path, "head", len(b), line, b)
                    continue
                src = 0
                if table_source_col is not None and table_source_col < len(cells):
                    src = len(cells[table_source_col])
                rest = " | ".join(c for k, c in enumerate(cells) if k != table_source_col or table_source_col is None)
                p = sum(prov_mask(rest))
                doc.add(path, "provenance", src + p, line, b)
                doc.add(path, "reference", len(b) - src - p, line, b)
                continue
            table_source_col = None if not s.startswith("|") else table_source_col
            if s.startswith("```"):
                doc.add(path, "duplicate" if HOOK.search(b) else "reference", len(b), line, b)
                continue
            mask = prov_mask(b)
            pos = 0
            definition = bool(DEFINITION.match(b)) and not STRONG.search(b)
            for sent in sentences(b):
                k = b.find(sent, pos)
                k = pos if k < 0 else k
                pos = k + len(sent)
                pchars = sum(mask[k:k + len(sent)])
                rest = len(sent) - pchars
                doc.add(path, "provenance", pchars, line, sent)
                rep = doc.repeat_of(sent)
                if HOOK.search(sent):
                    cls = "duplicate"
                elif rep:
                    cls = "duplicate"
                    sent = "[repeats %s] %s" % (rep, sent)
                elif definition:
                    cls = "reference"
                else:
                    cls = classify_sentence(sent)
                    if cls == "rule" and TOPIC.search(sent):
                        cls = "rule_topic"
                doc.add(path, cls, rest, line, sent)
                doc.remember(sent, path, line)
    return doc


CLASSES = ("rule", "reference", "provenance", "duplicate")


def cls_of(c):
    return "rule" if c == "rule_topic" else c


def summary(doc, paths):
    res = {"files": {}, "total": {}}
    for path in paths + [None]:
        units = [u for u in doc.units if path is None or u[0] == path]
        counted = sum(u[3] for u in units if u[2] != "head")
        d = {c: sum(u[3] for u in units if cls_of(u[2]) == c) for c in CLASSES}
        d["rule_topic"] = sum(u[3] for u in units if u[2] == "rule_topic")
        d["head"] = sum(u[3] for u in units if u[2] == "head")
        d["counted_chars"] = counted
        d["tokens_est"] = round((counted + d["head"]) / 4)
        d["shares"] = {c: round(100.0 * d[c] / counted, 1) if counted else 0.0 for c in CLASSES + ("rule_topic",)}
        if path is None:
            res["total"] = d
        else:
            res["files"][path] = d
    return res


def label(path):
    return path if path == "TOTAL" else "/".join(os.path.abspath(os.path.expanduser(path)).split("/")[-2:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--list", action="store_true", help="print every non-rule unit with file:line")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--save-baseline")
    ap.add_argument("--baseline")
    ap.add_argument("--tol", type=float, default=1.0)
    a = ap.parse_args()
    paths = a.files or [p for p in DEFAULT_FILES if os.path.exists(p)]
    doc = analyse(paths)
    res = summary(doc, paths)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for path, d in list(res["files"].items()) + [("TOTAL", res["total"])]:
            sh = d["shares"]
            print("%-34s ~%5d tok  rule %5.1f%% (topic-bound %4.1f%%)  reference %5.1f%%  provenance %4.1f%%  "
                  "duplicate %4.1f%%" % (label(path), d["tokens_est"], sh["rule"], sh["rule_topic"],
                                         sh["reference"], sh["provenance"], sh["duplicate"]))
    if a.list:
        for f, line, cls, chars, text in doc.units:
            if cls in ("rule", "head"):
                continue
            print("%s:%d\t%s\t%d\t%s" % (label(f), line, cls, chars, text[:90]))
    if a.save_baseline:
        with open(a.save_baseline, "w") as fh:
            json.dump(res["total"]["shares"], fh)
    if a.baseline:
        base = json.load(open(a.baseline))
        now = res["total"]["shares"]["rule"]
        if now < base["rule"] - a.tol:
            print("rule share %.1f%% fell below the baseline %.1f%% (tolerance %.1f points)"
                  % (now, base["rule"], a.tol), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
