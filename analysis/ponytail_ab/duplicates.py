#!/usr/bin/env python3
"""How much existing code the analysis scripts reimplement (the user, 2026-10-29: "each figure doesn't need one
individual script to be reproduced, when making a lot of similar figures, you can have one script, and each figure gets
reproduced by passing different arguments to it"). Two counts, written to analysis/ponytail_ab/duplicates.json and
printed as one JSON object:
  reviews   from the blind reviews of the ponytail A/B's second round (reviews2/<task>_<label>.txt, sec:ponytail-ab):
            every "REIMPL: <new code> | <existing code> | <why>" line, the existing code parsed into path:function
            targets (a bare file name is resolved against the line's other paths, then the repository's files; a
            target with no function counts as its file). Per target: in how many REIMPL lines and in how many of the
            reviewed diffs it is named. Four reviews relayed only their count (REIMPLEMENTED: n), so the itemised lines
            are fewer than the reviewers' total, both given.
  scan      the near-duplicate functions of analysis/ and tools/ at a commit (default HEAD), by tools/dupcheck.py's
            normalisation (names and constants stripped, functions over 6 statements, token-sequence similarity >= 0.8,
            a function never against one nested in it; the 0.8 and the 6 are open, unsourced): every pair, and the
            clusters they form (connected components), largest first.
    python3 analysis/ponytail_ab/duplicates.py [--rev REV]"""
import argparse, glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import dupcheck  # noqa: E402

# a path (with or without directories) and optionally :function or :f, g
TARGET = re.compile(r"((?:[\w.-]+/)*[\w-]+\.py)(?::(\w+(?:, \w+(?=$|[\s,)]))*))?")


def targets(field, files):
    """[path:function] named in a REIMPL line's existing-code field."""
    out, dirs = [], []
    for m in TARGET.finditer(field):
        path, fns = m.group(1), m.group(2)
        if "/" in path:
            dirs.append(os.path.dirname(path))
        else:
            cand = [d + "/" + path for d in dirs if d + "/" + path in files] or \
                   sorted(f for f in files if f.endswith("/" + path))
            path = cand[0] if cand else path
        for fn in (fns.split(", ") if fns else [None]):
            t = path + (":" + fn if fn else "")
            if t not in out:
                out.append(t)
    return out


def reviews(files):
    lines, per_target, totals, itemised = [], {}, {}, {}
    for f in sorted(glob.glob(os.path.join(HERE, "reviews2", "*.txt"))):
        diff = os.path.basename(f)[:-4]
        text = open(f).read()
        m = re.search(r"^REIMPLEMENTED: (\d+)", text, re.M)
        totals[diff] = int(m.group(1)) if m else None
        rows = [ln for ln in text.splitlines() if ln.startswith("REIMPL:")]
        itemised[diff] = len(rows)
        for ln in rows:
            parts = [p.strip() for p in ln[len("REIMPL:"):].split("|")]
            ts = targets(parts[1], files)
            lines.append({"diff": diff, "new": parts[0], "existing": ts, "why": parts[2] if len(parts) > 2 else ""})
            for t in ts:
                for key in (t, t.split(":")[0] + " (file)"):
                    e = per_target.setdefault(key, {"lines": 0, "diffs": set()})
                    e["lines"] += 1
                    e["diffs"].add(diff)
    rank = lambda file_level: [
        {"target": k.replace(" (file)", ""), "lines": v["lines"], "diffs": len(v["diffs"])}
        for k, v in sorted(per_target.items(), key=lambda kv: (-kv[1]["lines"], -len(kv[1]["diffs"]), kv[0]))
        if k.endswith(" (file)") == file_level]
    return {"n_reviews": len(totals), "reviewer_total": sum(v or 0 for v in totals.values()),
            "itemised_lines": len(lines), "reviews_with_lines": sum(1 for v in itemised.values() if v),
            "count_only_reviews": sorted(k for k, v in itemised.items() if not v),
            "per_diff": {k: {"reviewer_count": totals[k], "itemised": itemised[k]} for k in sorted(totals)},
            "targets": rank(False), "files": rank(True), "lines": lines}


def scan(rev):
    corpus = dupcheck.corpus(ROOT, rev)
    fs = [(p, f) for p in sorted(corpus) for f in corpus[p] if f[2] > dupcheck.MIN_STMTS]
    pairs = []
    for i, (p, f) in enumerate(fs):
        for p2, g in fs[i + 1:]:
            if dupcheck.nested(p, f[0], p2, g[0]):
                continue
            s = dupcheck.similarity(f[3], g[3])
            if s >= dupcheck.SIM:
                pairs.append((f"{p}:{f[0]}", f"{p2}:{g[0]}", round(s, 3)))
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x
    for a, b, _ in pairs:
        parent[find(a)] = find(b)
    groups = {}
    for a, b, _ in pairs:
        for x in (a, b):
            groups.setdefault(find(x), set()).add(x)
    clusters = []
    for mem in groups.values():
        sims = [s for a, b, s in pairs if a in mem]
        clusters.append({"size": len(mem), "names": sorted({m.split(":")[1].split(".")[-1] for m in mem}),
                         "files": len({m.split(":")[0] for m in mem}), "min_similarity": min(sims),
                         "members": sorted(mem)})
    clusters.sort(key=lambda c: (-c["size"], c["members"][0]))
    return {"rev": rev, "functions_compared": len(fs), "pairs": len(pairs),
            "functions_in_a_pair": len({x for a, b, _ in pairs for x in (a, b)}), "clusters": clusters,
            "pair_list": [{"a": a, "b": b, "similarity": s} for a, b, s in pairs]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rev", default="HEAD", help="the commit whose analysis/ and tools/ are scanned")
    a = ap.parse_args()
    rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", a.rev], capture_output=True, text=True,
                         check=True).stdout.strip()
    files = set(subprocess.run(["git", "-C", ROOT, "ls-tree", "-r", "--name-only", rev], capture_output=True,
                               text=True).stdout.split("\n"))
    out = {"reviews": reviews(files), "scan": scan(rev)}
    with open(os.path.join(HERE, "duplicates.json"), "w") as f:
        json.dump(out, f, indent=1)
    r, s = out["reviews"], out["scan"]
    summary = {"reviews": {k: r[k] for k in ("n_reviews", "reviewer_total", "itemised_lines", "reviews_with_lines")},
               "top_targets": {t["target"]: t["lines"] for t in r["targets"][:8]},
               "top_files": {t["target"]: t["lines"] for t in r["files"][:6]},
               "scan": {"rev": rev, "functions_compared": s["functions_compared"], "pairs": s["pairs"],
                        "functions_in_a_pair": s["functions_in_a_pair"], "clusters": len(s["clusters"]),
                        "largest": {"%s (%d files)" % ("+".join(c["names"]), c["files"]): c["size"]
                                    for c in s["clusters"][:8]}}}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
