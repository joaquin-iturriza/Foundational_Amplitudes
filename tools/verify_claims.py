#!/usr/bin/env python3
"""Recompute every number a change adds to docs/results.tex (setup redesign, D121; run by .githooks/pre-commit and
pre-merge-commit). A paragraph (or itemize item) that states numbers carries their provenance as LaTeX comments:
    % src: python3 <a tracked script under analysis/ or tools/> [args] @<commit where that script exists>
    % claim: <key, dotted into the JSON object the script prints as its last {...} line>=<value as written, 0.80, 5.8e-8, 40>
    % allow: <tokens that are not results: 2\to4, H100> [<a plain number> because <why it is no result>]  (optional)
For each paragraph or item the change touches (only numbers the change ADDS to it count: those on its added lines
less those on the lines its own hunks removed, so a reflow or a typo fix around existing numbers needs no claim, while a
number moved in from another part of the file does):
  - every added number in its text (decimals, integers, a x 10^b, 5.8e-8, braced \\textbf{0.80} or \\num{..}, table cells,
    suffixed 40k or 2.35x; -- is a range) must be the written value of one of its claims or an allowed token;
    multiplicities (2\\to4), years, dates, times, subscripts, lengths and the insides of \\label, \\ref, \\cite,
    \\code, \\texttt and \\url are skipped;
  - every claim whose line is added, or whose value is an added number, must be written in the text;
  - every src must be `python3 <path>`, the path tracked, under analysis/ or tools/, and present at @commit; it is rerun
    on the local artifacts (range mode: where the script is the one at the range's end, see Where) and each claim must equal the recomputed value up to the rounding the text shows (half a
    unit of its last digit).
The generated plan view between % plan:begin and % plan:end is skipped. Exit 1 with the report on any failure.
    python3 tools/verify_claims.py [--root R] [--staged | --worktree | --rev A..B] [--file docs/results.tex]"""
import argparse, json, os, re, shlex, subprocess, sys

SKIP_CMDS = r"\\(?:label|ref|eqref|cite|code|texttt|url|includegraphics|pnl|input)\{[^}]*\}"
SKIP = [SKIP_CMDS, r"\d+\s*\\to\s*\d+", r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?", r"\b\d{1,2}:\d{2}\b",
        r"\b(?:19|20)\d{2}\b", r"_\{[^}]*\}", r"_\d", r"\\(?:multicolumn|multirow)\{\d+\}",
        r"\d+(?:\.\d+)?\s*\\(?:line|text|column)width", r"\b\d+(?:\.\d+)?\s*(?:pt|mm|cm|em|ex)\b",
        r"\\(?:hspace|vspace|setlength|rule)\*?(?:\{[^}]*\})+", r"\[\s*[\d.]+\s*\]"]
SCI = re.compile(r"(-?\d+(?:\.\d+)?)\s*\\times\s*10\^\{?(-?\d+(?:\.\d+)?)\}?")
POW = re.compile(r"(?<![\d.])10\^\{?(-?\d+(?:\.\d+)?)\}?")
ENOT = re.compile(r"(?<![A-Za-z_\d.])-?\d+(?:\.\d+)?[eE][+-]?\d+(?![\d.])")
# a number not inside an identifier (tp3, H100, catalog\_v3): not after a letter, digit, point or underscore; a minus
# is a sign only where no digit stands before it (0.3-0.5 is a range); a letter suffix is kept (40k, 2.35x, 8GeV)
PLAIN = re.compile(r"(?:(?<![\w.])-)?(?<![A-Za-z_\d.])(\d+(?:\.\d+)?)([A-Za-z]*)")
SCALE = {"k": 1e3, "K": 1e3, "M": 1e6}
PLAIN_TOKEN = re.compile(r"-?\d+(?:\.\d+)?[kKM]?")
def git(root, *a, check=True):
    r = subprocess.run(["git", "-C", root] + list(a), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(a), r.stderr.strip()))
    return r


def changed_lines(diff):
    out = set()
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.M):
        start, n = int(m.group(1)), int(m.group(2) or 1)
        out.update(range(start, start + n))
    return out


def blocks(text):
    """[(first line, last line, lines)]: paragraphs split at blank lines, and every itemize item on its own."""
    out, cur, start = [], [], None
    for i, line in enumerate(text.splitlines(), 1):
        st = line.strip()
        if not st or st.startswith("\\item") or st.startswith("\\begin{itemize") or st.startswith("\\end{itemize"):
            if cur:
                out.append((start, i - 1, cur))
            cur, start = ([line], i) if st.startswith("\\item") else ([], None)
            continue
        if not cur:
            start = i
        cur.append(line)
    if cur:
        out.append((start, start + len(cur) - 1, cur))
    return out


def numbers(body, allowed):
    """(token text, value) of every number in the text, skipped spans and allowed tokens removed. A suffix k or M
    scales (40k is 40000); any other letters after a number are its unit or a multiplier sign (2.35x is 2.35)."""
    for t in sorted(allowed, key=len, reverse=True):
        body = re.sub(r"(?<![\w.])" + re.escape(t) + r"(?![\w.]*\d)", " ", body)
    for p in SKIP:
        body = re.sub(p, " ", body)
    body = body.replace("---", " ").replace("--", " ")          # en and em dashes: ranges, not minus signs
    out = []
    for rx, val in ((SCI, lambda m: float(m.group(1)) * 10 ** float(m.group(2))),
                    (POW, lambda m: 10 ** float(m.group(1))), (ENOT, lambda m: float(m.group(0)))):
        out += [(m.group(0), val(m)) for m in rx.finditer(body)]
        body = rx.sub(" ", body)
    for m in PLAIN.finditer(body):
        sign = -1.0 if m.group(0).startswith("-") else 1.0
        suf = m.group(2)
        out.append((m.group(0), sign * float(m.group(1)) * SCALE.get(suf[:1], 1.0) if suf in ("k", "K", "M")
                    else sign * float(m.group(1))))
    return out


def allow_line(rest):
    """(tokens, problem) of a % allow: line. A token that is a plain number (40, 0.5, 8k) needs its reason on the
    line ("% allow: 40 because it is the number of probes, a design count"); other tokens (2\to4, H100) need none."""
    head, _, reason = rest.partition(" because ")
    toks = head.split()
    plain = [t for t in toks if PLAIN_TOKEN.fullmatch(t)]
    if plain and len(reason.split()) < 3:
        return toks, "allowed number(s) %s need a reason (%% allow: <number> because <why it is not a result>)" % (
            " ".join(plain))
    return toks, None


def forms(value):
    v = value.strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)", v)
    if m:
        ex = str(int(m.group(2)))
        return [v, "%s\\times10^{%s}" % (m.group(1), ex), "%s\\times 10^{%s}" % (m.group(1), ex),
                "%s\\times10^%s" % (m.group(1), ex)]
    return [v]


def claim_value(v):
    """The number a claim's written value stands for (40k is 40000)."""
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)([kKM]?)", v.strip())
    return float(m.group(1)) * SCALE.get(m.group(2), 1.0) if m else None


def matches(claim, got):
    m = re.fullmatch(r"(-?\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?([kKM]?)", claim.strip())
    if not m or isinstance(got, bool) or not isinstance(got, (int, float)):
        return False
    dec, ex, sc = len(m.group(2) or ""), int(m.group(3) or 0), SCALE.get(m.group(4), 1.0)
    return abs(float(got) - claim_value(claim)) <= 0.5 * 10 ** (ex - dec) * sc * (1 + 1e-9)


def flatten(x, pre=""):
    if isinstance(x, dict):
        return {k: v for kk, vv in x.items() for k, v in flatten(vv, pre + str(kk) + ".").items()}
    if isinstance(x, list):
        return {k: v for i, vv in enumerate(x) for k, v in flatten(vv, pre + str(i) + ".").items()}
    return {pre[:-1]: x}


def src_problem(root, cmd, commit, rev=None):
    """Why the source is not acceptable (None: it is). With `rev` (range mode) the script must be tracked at rev,
    not in the checkout the check runs in: a branch that adds a script and its paragraph is checked as it stands."""
    parts = shlex.split(cmd)
    if len(parts) < 2 or parts[0] != "python3":
        return "src must be `python3 <script> ...`, not %r" % cmd
    path = os.path.normpath(parts[1])
    if not (path.startswith("analysis/") or path.startswith("tools/")):
        return "src script %s is not under analysis/ or tools/" % path
    tracked = git(root, "cat-file", "-e", "%s:%s" % (rev, path), check=False) if rev else \
        git(root, "ls-files", "--error-unmatch", path, check=False)
    if tracked.returncode != 0:
        return "src script %s is not tracked%s" % (path, " at " + rev if rev else "")
    if not commit:
        return "src has no @<commit>"
    if git(root, "cat-file", "-e", "%s:%s" % (commit, path), check=False).returncode != 0:
        return "src script %s does not exist at %s" % (path, commit)
    return None


class Where:
    """Where a source runs. Staged or working-tree mode: the checkout. Range mode: the checkout when its copy of the
    script is the one at rev; else a worktree whose HEAD is rev (it has that branch's local artifacts); else a
    temporary detached worktree at rev, removed afterwards."""
    STALE_S = 3600           # with no pid file, a temporary worktree older than this is left over

    def __init__(self, root, rev=None):
        self.root, self.rev, self.tmp = root, rev, None
        self.sha = git(root, "rev-parse", "--verify", "-q", rev + "^{commit}", check=False).stdout.strip() if rev \
            else None
        if rev:
            self.cleanup()

    @staticmethod
    def owner_alive(wt):
        """The run that made this temporary worktree is still going: its pid file (beside the worktree) names a live
        process. With no pid file (an older run) it counts as gone once older than STALE_S."""
        import time
        try:
            pid = int(open(os.path.join(os.path.dirname(wt), "owner.pid")).read().strip())
        except (OSError, ValueError):
            return os.path.isdir(wt) and time.time() - os.path.getmtime(wt) <= Where.STALE_S
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def cleanup(self):
        """Remove temporary worktrees of earlier runs that were cut off (a timeout kills a run before close), never one
        a live run still uses."""
        for blk in git(self.root, "worktree", "list", "--porcelain", check=False).stdout.split("\n\n"):
            f = dict(l.split(" ", 1) for l in blk.splitlines() if " " in l)
            wt = f.get("worktree", "")
            if "/verify_claims_" in wt and not self.owner_alive(wt):
                git(self.root, "worktree", "remove", "--force", wt, check=False)
                try:
                    os.remove(os.path.join(os.path.dirname(wt), "owner.pid"))
                    os.rmdir(os.path.dirname(wt))
                except OSError:
                    pass
        git(self.root, "worktree", "prune", check=False)

    def dir_for(self, path):
        if not self.rev:
            return self.root
        at = git(self.root, "rev-parse", "-q", "--verify", "%s:%s" % (self.rev, path), check=False).stdout.strip()
        here = os.path.join(self.root, path)
        if os.path.exists(here) and git(self.root, "hash-object", here, check=False).stdout.strip() == at:
            return self.root
        wt = None
        for blk in git(self.root, "worktree", "list", "--porcelain", check=False).stdout.split("\n\n"):
            f = dict(l.split(" ", 1) for l in blk.splitlines() if " " in l)
            if f.get("HEAD") == self.sha and f.get("worktree") and os.path.isdir(f["worktree"]):
                wt = f["worktree"]
        if wt:
            return wt
        if not self.tmp:
            import tempfile
            base = tempfile.mkdtemp(prefix="verify_claims_")
            open(os.path.join(base, "owner.pid"), "w").write(str(os.getpid()))    # cleanup leaves it while we live
            self.tmp = os.path.join(base, "wt")
            git(self.root, "worktree", "add", "-q", "--detach", self.tmp, self.sha)
        return self.tmp

    def close(self):
        if self.tmp:
            git(self.root, "worktree", "remove", "--force", self.tmp, check=False)
            try:
                os.remove(os.path.join(os.path.dirname(self.tmp), "owner.pid"))
                os.rmdir(os.path.dirname(self.tmp))
            except OSError:
                pass


SCRIPT_TIMEOUT = [1800]       # seconds per source script (the git hooks); plan claims passes 0, no limit


def run_src(where, cmd, cache):
    if cmd not in cache:
        cwd = where.dir_for(os.path.normpath(shlex.split(cmd)[1]))
        try:
            r = subprocess.run(shlex.split(cmd), cwd=cwd, capture_output=True, text=True,
                               timeout=SCRIPT_TIMEOUT[0] or None)
        except subprocess.TimeoutExpired:
            cache[cmd] = RuntimeError("took over %d s (--script-timeout; plan claims runs with no limit)"
                                      % SCRIPT_TIMEOUT[0])
            return cache[cmd]
        js = [l for l in r.stdout.splitlines() if l.strip().startswith("{")]
        if r.returncode != 0 or not js:
            cache[cmd] = RuntimeError("exit %d, %s" % (r.returncode, (r.stderr or "no JSON line").strip()[-200:]))
        else:
            try:
                cache[cmd] = flatten(json.loads(js[-1]))
            except ValueError as e:
                cache[cmd] = RuntimeError("bad JSON: %s" % e)
    return cache[cmd]


def hunks(diff):
    """[(new start, new count, removed lines)] of a -U0 diff."""
    out, cur = [], None
    for l in diff.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", l)
        if m:
            cur = (int(m.group(1)), int(m.group(2) or 1), [])
            out.append(cur)
        elif cur is not None and l.startswith("-"):
            cur[2].append(l[1:])
    return out


def added_numbers(lines, first, last, added, hks):
    """The numbers a paragraph gains: those on its added lines, joined, less those on the lines its own hunks
    removed, joined. A reflow or a typo fix inside the paragraph adds none; a number moved in from elsewhere (the
    hand-off) counts as added. Comment lines are provenance, not numbers."""
    import collections
    text = lambda ls: " ".join(l for l in ls if not l.lstrip().startswith("%"))
    plus = text(l for i, l in enumerate(lines) if first + i in added)
    gone = []
    for c, n, rem in hks:
        lo, hi = (c, c + n - 1) if n else (c, c + 1)          # a pure deletion sits after line c
        if lo <= last and hi >= first:
            gone += rem
    return collections.Counter(numkey(v) for _, v in numbers(plus, [])) - collections.Counter(
        numkey(v) for _, v in numbers(text(gone), []))


def numkey(v):
    """A number by value, so 5.8\\times10^{-8} split over two lines by a reflow is the same number."""
    return "%.12g" % v


def verify(root, text, added, rev=None, diff=None, cache_file=None):
    where = Where(root, rev)
    try:
        return _verify(root, text, added, rev, where, diff, cache_file)
    finally:
        where.close()


def _verify(root, text, added, rev, runner, diff=None, cache_file=None):
    """Policy (the user's approval wording, "every number and figure added"): only numbers ADDED by the diff need a
    claim; a claim is rerun when its claim line or its src line is added, or its value is an added number."""
    import hashlib
    problems, cache = [], {}
    hks = hunks(diff) if diff is not None else None
    try:
        done = json.load(open(cache_file)) if cache_file else {}
    except (OSError, ValueError):
        done = {}
    tl = text.splitlines()
    gen = [i for i, l in enumerate(tl, 1) if l.lstrip().startswith("% plan:begin")]
    gen = [(b, next((j for j in range(b, len(tl) + 1) if tl[j - 1].lstrip().startswith("% plan:end")), len(tl)))
           for b in gen]
    for first, last, lines in blocks(text):
        if not any(first <= n <= last for n in added):
            continue
        if any(b <= first and last <= e for b, e in gen):      # the plan's generated view (tools/plan.py view)
            continue
        key = hashlib.sha256("\n".join(lines).encode()).hexdigest()
        if key in done:                                        # this paragraph, as it stands, passed before
            continue
        n_before = len(problems)
        where = "lines %d-%d" % (first, last)
        body = " ".join(l for l in lines if not l.lstrip().startswith("%"))
        src, commit, claims, allowed, src_added = None, None, [], [], False
        for i, l in enumerate(lines):
            m = re.match(r"\s*%\s*src:\s*(.+?)(?:\s+@(\S+))?\s*$", l)
            if m:
                src, commit, src_added = m.group(1), m.group(2), first + i in added
                why = src_problem(root, src, commit, rev)
                if why:                                        # reported if the line is new or a claim of it is rerun
                    src = "!" + why
                    if first + i in added:
                        problems.append("%s: %s" % (where, why))
                continue
            m = re.match(r"\s*%\s*claim:\s*([^=\s]+)\s*=\s*(\S+)\s*$", l)
            if m:
                claims.append((src, m.group(1), m.group(2), first + i in added or src_added))
                continue
            m = re.match(r"\s*%\s*allow:\s*(.+)$", l)
            if m:
                toks, why = allow_line(m.group(1))
                allowed += toks
                if why and first + i in added:
                    problems.append("%s: %s" % (where, why))
        values = [claim_value(v) for _, _, v, _ in claims]
        for _, key_, v, _ in claims:
            if claim_value(v) is None:
                problems.append("%s: claim %s=%s is not a number" % (where, key_, v))
        values = [v for v in values if v is not None]
        left = None if hks is None else added_numbers(lines, first, last, added, hks)
        new_vals = []
        for tok, val in numbers(body, allowed):
            if left is not None:
                if left[numkey(val)] <= 0:
                    continue                                   # already in the paragraph before this change
                left[numkey(val)] -= 1
            new_vals.append(val)
            if not any(abs(val - v) <= 1e-12 * max(1.0, abs(v)) for v in values):
                problems.append("%s: %s is no claim's value (add a %% claim, or %% allow it if it is no result)" % (
                    where, tok))
        for s_, key_, val, line_added in claims:
            cv = claim_value(val)
            if not line_added and not any(cv is not None and abs(cv - x) <= 1e-12 * max(1.0, abs(x))
                                          for x in new_vals):
                continue                                       # an existing claim whose number did not change
            if not any(f in body for f in forms(val)):
                problems.append("%s: claim %s=%s is not written in the text" % (where, key_, val))
            if s_ is None or s_.startswith("!"):
                problems.append("%s: claim %s has no valid src above it%s" % (where, key_, ": " + s_[1:] if s_ else ""))
                continue
            got = run_src(runner, s_, cache)
            if isinstance(got, Exception):
                problems.append("%s: %s failed: %s" % (where, s_, got))
            elif key_ not in got:
                problems.append("%s: %s prints no %s" % (where, s_, key_))
            elif not matches(val, got[key_]):
                problems.append("%s: %s=%s, recomputed %r (%s)" % (where, key_, val, got[key_], s_))
        if cache_file and len(problems) == n_before:           # saved paragraph by paragraph: a cut run keeps them
            done[key] = 1
            tmp = cache_file + ".tmp"
            open(tmp, "w").write(json.dumps(done))
            os.replace(tmp, cache_file)
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--file", default="docs/results.tex")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--staged", action="store_true")
    g.add_argument("--worktree", action="store_true", help="the working file against HEAD")
    g.add_argument("--rev")
    ap.add_argument("--cache", help="a JSON file of paragraphs (by content hash) that passed; they are skipped")
    ap.add_argument("--script-timeout", type=float, default=SCRIPT_TIMEOUT[0],
                    help="seconds per source script, 0 for no limit (default %(default)s)")
    a = ap.parse_args(argv)
    SCRIPT_TIMEOUT[0] = a.script_timeout
    new_rev = None
    if a.rev:
        new_rev = a.rev.split("..")[-1] or "HEAD"
        text = git(a.root, "show", "%s:%s" % (new_rev, a.file)).stdout
        diff = git(a.root, "diff", "-U0", a.rev, "--", a.file).stdout
    elif a.worktree:
        text, diff = open(os.path.join(a.root, a.file)).read(), git(a.root, "diff", "-U0", "HEAD", "--", a.file).stdout
    else:
        text, diff = git(a.root, "show", ":" + a.file).stdout, git(a.root, "diff", "--cached", "-U0", "--", a.file).stdout
    problems = verify(a.root, text, changed_lines(diff), new_rev if a.rev else None, diff, a.cache)
    for p in problems:
        print("CLAIM: " + p)
    print("verify_claims: %d problem(s)" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
