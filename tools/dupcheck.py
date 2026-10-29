#!/usr/bin/env python3
"""Near-duplicate functions in analysis/ and tools/ (the user, 2026-10-29: agents keep writing one-use scripts that
reimplement existing code). One normalisation, used by the repo scan (analysis/ponytail_ab/duplicates.py) and by the
commit check (.githooks/pre-commit):
  - a function's body becomes a token sequence: the AST node types in pre-order, with its arguments and the names it
    assigns replaced by one placeholder (renaming a variable changes nothing), every constant replaced by its type
    (another path or threshold changes nothing), its docstring dropped; global names, called functions and attribute
    names are kept (np.load and json.load stay different);
  - two functions are near duplicates when difflib's ratio of their sequences is >= SIM (0.8);
  - only functions of more than MIN_STMTS (6) statements, counted recursively without the docstring, are compared,
    except against analysis/common (the shared readers, all short): any function of at least COMMON_MIN (2)
    statements is compared with them.
SIM (0.8), MIN_STMTS (6) and COMMON_MIN (2) are open (unsourced): set by the setup request, not by a call of the user's.
Commit check (`--staged`): every function of a staged .py under analysis/ or tools/ that is new, or whose body
changed, is compared with every function of the index's analysis/ and tools/ .py files (itself excluded); the commit is
refused when one is >= SIM similar, naming it as path:function. A changed function is refused only for a match its
committed version did not already have (editing one of an old pair is not refused for the old pair). Override: a
comment `# dup-ok: <reason>` on the def line.
    python3 tools/dupcheck.py --staged [--root R]        exit 1 with the report on a duplicate
    python3 tools/dupcheck.py FILE.py [FILE.py ...]      print each function's best match in the repo's working tree"""
import argparse, ast, difflib, os, re, subprocess, sys, warnings

SIM = 0.8
MIN_STMTS = 6
COMMON_MIN = 2
COMMON = "analysis/common/"
SCOPE = ("analysis/", "tools/")
DUP_OK = re.compile(r"#\s*dup-ok:\s*\S")


def _stmts(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body, sum(isinstance(x, ast.stmt) for b in body for x in ast.walk(b))


def _locals(node):
    a = node.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    names |= {x.arg for x in (a.vararg, a.kwarg) if x}
    for x in ast.walk(node):
        if isinstance(x, ast.Name) and isinstance(x.ctx, (ast.Store, ast.Del)):
            names.add(x.id)
        elif isinstance(x, ast.arg):
            names.add(x.arg)
        elif isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and x is not node:
            names.add(x.name)
        elif isinstance(x, ast.ExceptHandler) and x.name:
            names.add(x.name)
        elif isinstance(x, (ast.Import, ast.ImportFrom)):
            names |= {(n.asname or n.name).split(".")[0] for n in x.names}
    return names


def tokens(node):
    """The normalised token sequence of a function's body (see the module docstring)."""
    body, _ = _stmts(node)
    loc = _locals(node)
    out = []

    def visit(n):
        t = type(n).__name__
        if isinstance(n, ast.Name):
            out.append("V" if n.id in loc else "N:" + n.id)
            return
        if isinstance(n, ast.Constant):
            out.append("C:" + type(n.value).__name__)
            return
        if isinstance(n, (ast.expr_context,)):
            return
        out.append(t)
        if isinstance(n, ast.Attribute):
            out.append("A:" + n.attr)
        elif isinstance(n, ast.keyword):
            out.append("K:" + str(n.arg))
        for c in ast.iter_child_nodes(n):
            visit(c)
    for b in body:
        visit(b)
    return out


def functions(src, path="?"):
    """[(qualname, lineno, n_statements, tokens, dup_ok)] of every function (nested ones too) in `src`."""
    try:
        with warnings.catch_warnings():           # a scanned file's invalid escape is not this check's business
            warnings.simplefilter("ignore")
            tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()
    out = []

    def walk(n, prefix):
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + c.name
                _, ns = _stmts(c)
                line = lines[c.lineno - 1] if c.lineno - 1 < len(lines) else ""
                out.append((q, c.lineno, ns, tokens(c), bool(DUP_OK.search(line))))
                walk(c, q + ".")
            elif isinstance(c, ast.ClassDef):
                walk(c, prefix + c.name + ".")
            else:
                walk(c, prefix)
    walk(tree, "")
    return out


def similarity(a, b, floor=SIM):
    """difflib's ratio of two token sequences, or 0.0 when cheap upper bounds already put it under `floor`."""
    la, lb = len(a), len(b)
    if not la or not lb or 2.0 * min(la, lb) / (la + lb) < floor:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < floor or sm.quick_ratio() < floor:
        return 0.0
    return sm.ratio()


def git(root, *a):
    return subprocess.run(["git", "-C", root] + list(a), capture_output=True, text=True)


def _blobs(root, spec_paths):
    """{path: text} of `<rev>:<path>` specs read in one git cat-file --batch (a path may hold spaces)."""
    if not spec_paths:
        return {}
    req = ("\n".join(s for s, _ in spec_paths) + "\n").encode()
    r = subprocess.run(["git", "-C", root, "cat-file", "--batch=%(objecttype) %(objectsize)"], input=req,
                       capture_output=True)
    out, buf, i = {}, r.stdout, 0
    for spec, path in spec_paths:
        nl = buf.index(b"\n", i)
        head = buf[i:nl].decode("utf-8", "replace")
        i = nl + 1
        if head.endswith(" missing") or head.endswith(" ambiguous"):
            continue
        n = int(head.rsplit(" ", 1)[1])
        out[path] = buf[i:i + n].decode("utf-8", "replace")
        i += n + 1
    return out


def in_scope(p):
    return p.endswith(".py") and p.startswith(SCOPE) and "/__pycache__/" not in p


def corpus(root, rev=""):
    """{path: [functions]} of the analysis/ and tools/ .py files of the index (rev "") or of a commit."""
    ls = git(root, "ls-files") if not rev else git(root, "ls-tree", "-r", "--name-only", rev)
    paths = [p for p in ls.stdout.split("\n") if in_scope(p)]
    texts = _blobs(root, [("%s:%s" % (rev, p), p) for p in paths])
    return {p: [f for f in functions(t, p) if f[2] >= COMMON_MIN] for p, t in texts.items()}


def eligible(p, ns, p2, ns2):
    """Whether two functions are compared: both over MIN_STMTS statements, or one of analysis/common and both of at
    least COMMON_MIN."""
    if p.startswith(COMMON) or p2.startswith(COMMON):
        return ns >= COMMON_MIN and ns2 >= COMMON_MIN
    return ns > MIN_STMTS and ns2 > MIN_STMTS


def nested(p, q, p2, q2):
    """True when one function is the other, or is defined inside it (an outer function holds its inner one's body)."""
    return p == p2 and (q == q2 or q.startswith(q2 + ".") or q2.startswith(q + "."))


def best_matches(fn, corpus_, skip, floor=SIM):
    """[(similarity, path:qualname, line)] of the corpus functions >= floor similar to `fn`, leaving out the function
    skip = (path, qualname) itself and the functions nested in it or holding it."""
    out = []
    for p, fs in corpus_.items():
        for q, ln, ns, tk, _ in fs:
            if nested(p, q, *skip) or not eligible(p, ns, skip[0], fn[2]):
                continue
            s = similarity(fn[3], tk, floor)
            if s >= floor:
                out.append((s, "%s:%s" % (p, q), ln))
    return sorted(out, reverse=True)


def staged(root):
    """Problems of the staged change: one line per new or changed function that duplicates another."""
    r = git(root, "diff", "--cached", "--name-status", "-M", "-z", "--diff-filter=AMR")
    f, was, i = r.stdout.split("\0"), {}, 0
    while i < len(f) - 1:                  # -z: status, path (a rename: status, old path, new path)
        st = f[i]
        if st.startswith("R"):
            was[f[i + 2]] = f[i + 1]
            i += 3
        else:
            was[f[i + 1]] = f[i + 1]
            i += 2
    changed = [p for p in was if in_scope(p)]
    if not changed:
        return []
    idx = corpus(root)
    has_head = git(root, "rev-parse", "--verify", "-q", "HEAD").returncode == 0
    # a renamed file is compared with its old path's committed version: moving code is not writing it
    head = _blobs(root, [("HEAD:%s" % was[p], p) for p in changed]) if has_head else {}
    head_corpus = {}                  # the committed corpus, read only when a changed function has a match
    bad = []
    for p in changed:
        old = {q: (tk, ln, ns) for q, ln, ns, tk, _ in functions(head.get(p, ""), p)}
        for fn in idx.get(p, []):
            q, ln, ns, tk, ok = fn
            if ok or (q in old and old[q][0] == tk):
                continue
            hits = best_matches(fn, idx, (p, q))
            if q in old and hits:     # a changed function: only matches its committed version did not have
                prev = (q, old[q][1], old[q][2], old[q][0], False)
                head_corpus = head_corpus or corpus(root, "HEAD")
                had = {m for _, m, _ in best_matches(prev, head_corpus, (was[p], q))}
                had |= {p + m[len(was[p]):] for m in had if m.startswith(was[p] + ":")}
                hits = [h for h in hits if h[1] not in had]
            if hits:
                s, m, mln = hits[0]
                bad.append("%s:%s (line %d) is %.2f similar to %s (line %d)%s" % (
                    p, q, ln, s, m, mln, "" if len(hits) == 1 else " and %d more" % (len(hits) - 1)))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    root = a.root or git(os.getcwd(), "rev-parse", "--show-toplevel").stdout.strip() or os.getcwd()
    if a.staged:
        bad = staged(root)
        if bad:
            print("dupcheck: a new or changed function duplicates an existing one (normalised body >= %.1f similar; "
                  "tools/dupcheck.py):" % SIM)
            for b in bad:
                print("  " + b)
            print("Import the existing function (analysis/common, `python3 tools/readers_index.py`) or extend it with "
                  "an argument; a family of similar figures is one script with arguments. A genuine exception carries "
                  "`# dup-ok: <reason>` on its def line.")
            return 1
        return 0
    idx = corpus(root)
    for f in a.files:
        p = os.path.relpath(os.path.abspath(f), root)
        for fn in functions(open(f).read(), p):
            if fn[2] >= COMMON_MIN:
                h = best_matches(fn, idx, (p, fn[0]))
                print("%s:%s  %s" % (p, fn[0], "%.2f %s" % h[0][:2] if h else "no match"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
