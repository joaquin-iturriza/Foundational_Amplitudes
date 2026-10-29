#!/usr/bin/env python3
"""Weighted size of the code reviewer's backlog, in average hand-written lines (the user's call of 2026-10-29: the
trigger fires at about two blocking errors waiting, "Okay, two errors waiting seems reasonable then" [D171]).

Each changed line counts by how often the code review found a blocking error in its area and kind of change
(docs/results.tex sec:review-rates, analysis/review_rates/review_rates.json): area rate times kind rate over the
square of the hand-written rate, so an average line weighs 1. Generated sweep configs and recipes (sweep/*.yaml,
recipes/*.yaml) count once per family: a family is the files whose names agree up to their last `_` part (a lone
file is a family of one); only its largest file counts, at weight 1 (one generator run writes the same mistake into every
file of it). Untracked files count at their full length as new files.

  python3 tools/review_weight.py <since-commit> <pathspec...>     prints the weighted line count (an integer)
"""
import json, os, re, subprocess, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATES = os.path.join(HERE, "analysis", "review_rates", "review_rates.json")
CONFIGS = re.compile(r"^(sweep|recipes)/.*\.ya?ml$")


def area(p):
    """The review-rate analysis's areas (analysis/review_rates/review_rates.py)."""
    sys.path.insert(0, os.path.join(HERE, "analysis", "review_rates"))
    import review_rates
    return review_rates.area(p)


def weights():
    try:
        d = json.load(open(RATES))
        base = d["hand_written"]["per_kline"]
        # a rate measured on no finding is no measurement: that area or kind weighs 1 (an area with 0 findings in
        # 511 lines, the analysis's "other", weighed 0 and never fired a review)
        wa = {k: v["per_kline"] / base for k, v in d["by_area"].items() if v.get("lines") and v.get("n")}
        wk = {k: v["per_kline"] / base for k, v in d["by_kind"].items() if v.get("lines") and v.get("n")}
        return wa, wk
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        return {}, {}                                  # no rates: every line weighs 1


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout


def changes(since, paths):
    """[(path, changed lines, kind)] since the commit, untracked files included; kinds as review_rates.py measures
    them (no rename detection; A = new file, D = deletion, anything else = edit)."""
    kinds = {}
    for line in git("diff", "--no-renames", "--name-status", since, "--", *paths).splitlines():
        st, _, p = line.partition("\t")
        kinds[p] = {"A": "new file", "D": "deletion"}.get(st[:1], "edit")
    out = []
    for line in git("diff", "--no-renames", "--numstat", since, "--", *paths).splitlines():
        a, d, p = (line.split("\t") + ["", "", ""])[:3]
        if a == "-" or not p:
            continue                                   # binary
        out.append((p, int(a) + int(d), kinds.get(p, "edit")))
    for p in git("ls-files", "--others", "--exclude-standard", "--", *paths).splitlines():
        try:
            out.append((p, sum(1 for _ in open(os.path.join(HERE, p), errors="replace")), "new file"))
        except OSError:
            pass
    return out


def weighted(since, paths):
    wa, wk = weights()
    total, fam = 0.0, defaultdict(list)
    for p, n, kind in changes(since, paths):
        if CONFIGS.match(p):
            fam[(os.path.dirname(p), re.sub(r"_[^_]*$", "", os.path.basename(p)))].append((p, n))
            continue
        total += n * wa.get(area(p), 1.0) * wk.get(kind, 1.0)
    for members in fam.values():
        total += max(n for _, n in members)          # every family once, a lone file a family of one (D171)
    return int(round(total))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    print(weighted(sys.argv[1], sys.argv[2:]))
