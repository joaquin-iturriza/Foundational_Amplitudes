#!/usr/bin/env python3
"""What the tool calls of a session's subagents were: re-reads of what the agent already had, repeats, refusals,
orientation, or productive work, and the context each class cost.

  python3 tools/agent_calls.py [SESSION_ID] [--after YYYY-MM-DDTHH:MM] [--before ...] [--show AGENT_ID[,ID] [--from 0] [--n 10]]

Agents are selected by launch time (UTC, the Agent call's timestamp; an agent with no launch record found, by its
first record) and grouped by launch parameters (subagent type, effort), read from the Agent calls in the main
transcript and in the subagent transcripts (nested launches), as tools/agent_costs.py does.

Classes, first match wins:
  refused      the result is a hook refusal, a permission denial or a tool error (is_error), except a grep that
               only found nothing (exit 1, no output)
  repeat       the same tool and input (a Bash call: the same command) as an earlier call of that agent; a Read
               is judged by the re-read rule instead, which knows whether the file was written in between
  re-read      every file the call reads was already read by that agent, with no write to it in between: a Read, or
               a Bash sed -n or head range, at least half of whose lines were read before; a Bash cat, grep, awk or
               tail of a file already read whole. grep, awk, tail, a sed program and a range piped into a filter
               (grep without -v, head, tail, awk) show only part of what they scan, so they count as read nothing. A call that also runs, writes or reads anything new
               is not a re-read. Logs and task outputs (*.log, *.output, *.out, *.err) grow, so they are never re-read.
  orientation  the first read of CLAUDE.md or of docs/results.tex by the agent (later new ranges of it are lookups,
               productive), plan.py show, site decisions, any --help
  productive   everything else: edits and writes, commits, site and plan commands, script runs, new reads and searches

A write that resets a file's read history: Edit/Write/NotebookEdit of it; in Bash a redirect, tee, sed -i, cp/mv
onto it, a script run whose text (heredoc included) names it, any script run for outputs (png, pdf,
json, npz, txt, csv, yaml, ...), plan.py (plan/plan.yaml); git checkout, merge, pull, rebase,
reset, stash, cherry-pick, worktree (every file).

Context per call: the input context of the API response that issued it (input + cache read + cache creation
tokens, as tools/agent_costs.py), split equally between the calls of one response, so a group's classes sum to the
context its agents re-read in all.

A content check that needs no rule about commands: the share of each class's result text whose lines were already
in the agent's context (launch prompt, earlier inputs and results; lines of 25+ characters, line numbers stripped).
A fork's inherited context is not in its transcript, so for forks it is a lower bound.

--show prints each call of the named agents (from call --from, --n of them) with its class, reason, context and the
share of its result already in context, for checking the classifier by hand.

--refusals [--windows A..B,C..D] [--sample 5] [--texts] breaks the refused calls down by source: the hook (named from
its refusal text, with its mode: review_backlog.sh gate), Claude Code's own refusals (permission: auto-mode
classifier, worktree isolation, ...; harness: sleep block) and tool errors by kind; per source and window the calls,
their context and the refusal text's size. For each hook, a sample of up to --sample of its refusals (evenly spaced
in time over all windows) is listed with the hand verdict recorded for it in analysis/agent_calls/refusal_verdicts.json
(tool_use id -> {"wrong": true when the call was legitimate, "why"}); --texts prints each sampled call and refusal."""
import glob, json, os, re, shlex, statistics as st, sys
from collections import Counter, defaultdict

PROJ = os.path.expanduser("~/.claude/projects/-home-joaquin-work-FA")
CLASSES = ("re-read", "repeat", "refused", "orientation", "productive")
VOLATILE = re.compile(r"\.(log|output|out|err)$")
ARTEFACT = re.compile(r"\.(png|pdf|json|jsonl|npz|npy|txt|csv|pkl|yaml)$")
READERS = {"cat", "sed", "head", "tail", "grep", "awk", "nl", "less", "more", "rg", "egrep", "fgrep"}
NEUTRAL = {"cd", "echo", "printf", "true", "false", "export", "set", "pwd", ":", "unset", "wait", "date", "for", "done",
           "fi", "}"}               # "for": the loop header; its body is classified as its own segments
GIT_RESET = {"checkout", "merge", "pull", "rebase", "reset", "stash", "cherry-pick", "worktree", "switch", "restore"}
REDIR = {">", ">>", ">|", "&>", "&>>", ">&", "<", "<<", "<<<", "<&"}
OPS = {";", "&&", "||", "|", "&", "\n", "(", ")", "|&", ";;"}
INF = float("inf")


def strip_heredocs(cmd):
    """Drop heredoc bodies (their text is a script's source, not the command line)."""
    out, lines, i = [], cmd.split("\n"), 0
    while i < len(lines):
        ln = lines[i]; out.append(ln)
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", ln)
        if m:
            end = m.group(1); i += 1
            while i < len(lines) and lines[i].strip() != end:
                i += 1
        i += 1
    return "\n".join(out)


def segments(cmd):
    """Split a shell command into (operator before it, tokens, redirect targets). None if it does not parse."""
    try:
        lx = shlex.shlex(strip_heredocs(cmd), posix=True, punctuation_chars=";&|()<>")
        lx.whitespace = " \t\r"; lx.whitespace_split = True
        toks = list(lx)
    except ValueError:
        return None
    segs, cur, red, op, i = [], [], [], ";", 0
    while i < len(toks):
        t = toks[i]
        if t in OPS:
            if cur or red:
                segs.append((op, cur, red))
            cur, red, op = [], [], t
        elif t in REDIR or re.fullmatch(r"\d?(>>?|>&|<)", t):
            if cur and re.fullmatch(r"\d", cur[-1]):
                cur.pop()
            if i + 1 < len(toks):
                tgt = toks[i + 1]
                if t.startswith(">") or t.startswith("&>"):
                    if not re.fullmatch(r"&?\d|/dev/null", tgt):
                        red.append(tgt)
                i += 1
        else:
            cur.append(t)
        i += 1
    if cur or red:
        segs.append((op, cur, red))
    return segs


def absp(p, cwd):
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(cwd, os.path.expanduser(p)))


def read_targets(argv, cwd):
    """[(path, lo, hi, whole, adds)] a reader command reads (whole: it needs the file read whole before to be a
    re-read; adds: what it prints is that range, so it counts as read; grep, awk, tail and sed programs print only a
    part); [] if it reads stdin; None if it is a search (grep -r) or names files by a variable or glob."""
    cmd, args = os.path.basename(argv[0]), argv[1:]
    if cmd in ("grep", "egrep", "fgrep", "rg"):
        files, pat_given, skip = [], False, False
        if cmd == "rg" or any(re.fullmatch(r"-\w*[rR]\w*|--recursive", a) for a in args):
            return None
        for k, a in enumerate(args):
            if skip:
                skip = False; continue
            if a in ("-e", "-f", "--regexp", "--file"):
                pat_given = True; skip = True; continue
            if a in ("-m", "-A", "-B", "-C", "--max-count", "--include", "--exclude"):
                skip = True; continue
            if a.startswith("-") and a != "-":
                if re.match(r"-\w*e$", a):
                    pat_given = True; skip = True
                continue
            if not pat_given:
                pat_given = True; continue
            files.append(a)
        return [(absp(f, cwd), 1, INF, True, False) for f in files]
    if cmd == "sed":
        if any(a == "-i" or a.startswith("-i") or a.startswith("--in-place") for a in args):
            return "write", [absp(a, cwd) for a in args if not a.startswith("-")][1:]
        rng, files, script = None, [], None
        for a in args:
            if a.startswith("-"):
                continue
            if script is None:
                script = a; continue
            files.append(a)
        m = re.fullmatch(r"\s*(\d+)\s*(?:,\s*(\d+|\$))?\s*p\s*;?\s*", script or "")
        if m:
            lo = int(m.group(1)); hi = INF if m.group(2) == "$" else int(m.group(2) or lo)
            return [(absp(f, cwd), lo, hi, hi == INF and lo == 1, True) for f in files]
        return [(absp(f, cwd), 1, INF, True, False) for f in files]    # a sed program: only part of it is seen
    if cmd in ("head", "tail"):
        n, files, skip = 10, [], False
        for k, a in enumerate(args):
            if skip:
                skip = False; continue
            if a in ("-n", "-c"):
                skip = True
                try:
                    n = int(args[k + 1].lstrip("+-"))
                except (IndexError, ValueError):
                    pass
                continue
            if re.fullmatch(r"-n?\d+", a):
                n = int(a.lstrip("-n")); continue
            if a.startswith("-"):
                continue
            files.append(a)
        if cmd == "head":
            return [(absp(f, cwd), 1, n, False, True) for f in files]
        return [(absp(f, cwd), 1, INF, True, False) for f in files]     # the end of a file of unknown length
    if cmd == "awk":
        files = [a for a in args if not a.startswith("-")][1:]
        return [(absp(f, cwd), 1, INF, True, False) for f in files if "=" not in f]
    files = [a for a in args if not a.startswith("-")]
    return [(absp(f, cwd), 1, INF, True, True) for f in files]


def is_orient_file(p):
    return p.endswith("CLAUDE.md") or p.endswith("docs/results.tex")


class Agent:
    def __init__(self):
        self.cov = defaultdict(list)     # path -> [(lo, hi)] read since the last write
        self.seen = set()
        self.oriented = set()            # orientation files already read once (a later new range is a lookup)

    def orient(self, p):
        if is_orient_file(p) and p not in self.oriented:
            self.oriented.add(p); return True
        return False

    def covered(self, p, lo, hi, whole):
        if VOLATILE.search(p):
            return False
        iv = self.cov.get(p)
        if not iv:
            return False
        if whole or hi == INF:
            return any(a <= lo and b == INF for a, b in iv)
        seen = sum(1 for k in range(lo, hi + 1) if any(a <= k <= b for a, b in iv))
        return seen >= 0.5 * (hi - lo + 1)             # at least half of the range was read before

    def add(self, p, lo, hi):
        self.cov[p].append((lo, hi))

    def wipe(self, p=None):
        if p is None:
            self.cov.clear()
        else:
            self.cov.pop(p, None)

    def name_wipe(self, toks):
        for p in list(self.cov):
            b = os.path.basename(p)
            if any(t == b or t.endswith("/" + b) for t in toks):
                self.cov.pop(p)


STATE = {"ls", "find", "wc", "du", "df", "stat", "which", "file", "diff", "cmp", "tree", "ps", "jq", "sqlite3"}
SITE_STATE = {"runs", "logs", "status", "poll", "where", "events", "items", "list", "pick", "digest"}
GIT_STATE = {"log", "diff", "show", "status", "blame", "grep", "rev-parse", "branch", "ls-files", "describe"}


def label(c, argv):
    """The kind of work a non-reader segment does, for the reason column."""
    sub = argv[1] if len(argv) > 1 else ""
    if c in STATE or (c == "git" and sub in GIT_STATE) or (c == "site" and sub in SITE_STATE):
        return "state read"
    if c == "git":
        return "git " + sub
    if c == "site":
        return "site " + sub
    if c in ("python", "python3", "bash", "sh", "pytest", "make") or c.endswith((".py", ".sh")):
        return "script run"
    return "other command"


def classify_bash(ag, cmd, cwd, shown=True):
    """(class, reason) for a Bash command; updates the agent's read history (not when the output was not shown:
    a result over the size limit is saved to a file and only a 2 KB preview reaches the agent)."""
    segs = segments(cmd)
    if segs is None:
        ag.wipe(); return "productive", "unparsed command"
    kinds, reads = [], []            # (kind, label); kind: orient / newread / reread / work
    for j, (op, argv, red) in enumerate(segs):
        for t in red:
            ag.wipe(absp(t, cwd))
        if red:
            kinds.append(("work", "write (redirect)"))
        while argv and re.fullmatch(r"\w+=.*", argv[0]):
            argv = argv[1:]
        if argv and argv[0] in ("timeout", "nice", "time", "env", "command"):
            argv = argv[1:]
            while argv and (argv[0].startswith("-") or re.fullmatch(r"[\d.]+[smh]?|\w+=.*", argv[0])):
                argv = argv[1:]
        while argv and argv[0] in ("do", "then", "else", "until", "while", "if", "!", "{"):
            argv = argv[1:]
        if not argv:
            continue
        c = os.path.basename(argv[0])
        if "--help" in argv or (len(argv) == 2 and argv[1] == "-h"):
            kinds.append(("orient", "--help")); continue
        if c == "cd":
            if len(argv) > 1:
                cwd = absp(argv[1], cwd)
            continue
        if c in NEUTRAL:
            continue
        if op in ("|", "|&") and c in READERS | {"sort", "uniq", "wc", "cut", "tr", "column", "xargs", "jq"}:
            continue                    # a filter of the previous segment's output
        if c in READERS or c == "cat":
            tg = read_targets(argv, cwd)
            if isinstance(tg, tuple):   # sed -i
                for p in tg[1]:
                    ag.wipe(p)
                kinds.append(("work", "write (sed -i)")); continue
            if tg is None or any("$" in p or "*" in p or "?" in p for p, *_ in tg):
                kinds.append(("newread", "search")); continue
            nxt = segs[j + 1] if j + 1 < len(segs) else None
            filtered = bool(nxt and nxt[0] in ("|", "|&") and nxt[1] and nxt[1][0] in ("grep", "head", "tail", "awk")
                            and "-v" not in nxt[1])     # piped into a filter: only part of the range is seen
            for p, lo, hi, whole, adds in tg:
                if ag.covered(p, lo, hi, whole):
                    kinds.append(("reread", os.path.basename(p)))
                else:
                    kinds.append(("orient" if ag.orient(p) else "newread", "read " + os.path.basename(p)))
                if adds and not filtered:
                    reads.append((p, lo, hi))
            continue
        if c == "git" and len(argv) > 1 and argv[1] in GIT_RESET:
            ag.wipe(); kinds.append(("work", "git " + argv[1])); continue
        if c in ("cp", "mv", "tee", "rm", "touch", "ln", "mkdir"):
            ag.name_wipe(argv[1:]); kinds.append(("work", "write (" + c + ")")); continue
        if argv[0].endswith("plan.py") or (c in ("python", "python3") and len(argv) > 1 and argv[1].endswith("plan.py")):
            ag.wipe(absp("plan/plan.yaml", cwd))
            sub = next((a for a in argv[1:] if not a.startswith("-") and not a.endswith(".py")), "")
            kinds.append(("orient", "plan show") if sub == "show" else ("work", "plan " + sub)); continue
        if c == "site" and len(argv) > 1 and argv[1] == "decisions":
            kinds.append(("orient", "site decisions")); continue
        lab = label(c, argv)
        if lab == "script run":         # it may rewrite any file its text names (heredoc included) and its outputs
            ag.name_wipe(re.findall(r"[\w./-]+", cmd))
            for p in [p for p in ag.cov if ARTEFACT.search(p)]:
                ag.wipe(p)
        kinds.append(("work", lab))
    for p, lo, hi in reads if shown else []:
        ag.add(p, lo, hi)
    ks = [k for k, _ in kinds]
    for k, cls in (("work", "productive"), ("newread", "productive"), ("orient", "orientation"),
                   ("reread", "re-read")):
        if k in ks:
            labs = sorted({lab for kk, lab in kinds if kk == k})
            return cls, ("already read: " if k == "reread" else "") + ", ".join(labs)
    return "productive", "neutral"


HOOK = re.compile(r"hook error: \[([^\]]*)\]")
PERMISSIONS = (("auto mode classifier", "permission: auto-mode classifier"),
               ("isolated in the worktree", "permission: worktree isolation"),
               ("Only the auto-mode classifier can allow", "permission: not in auto mode"),
               ("built-in Claude Code safety check", "permission: built-in safety check"),
               ("doesn't want to proceed", "permission: user rejected"),
               ("Permission for this tool use was denied", "permission: denied"),
               ("Blocked: sleep", "harness: sleep block"))
TOOL_ERRORS = (("No such tool available", "unknown tool"), ("InputValidationError", "bad tool input"),
               ("Missing required parameter", "bad tool input"), ("has not been read yet", "edit before read"),
               ("String to replace not found", "edit string not found"), ("exceeds maximum allowed tokens", "read too large"),
               ("File does not exist", "missing file"), ("API Error", "API error"))


def source(res, call):
    """Where an is_error result came from: 'hook <script> [mode]' named from the refusal text, 'permission: ...' or
    'harness: ...' (Claude Code itself), or 'tool error: <kind>'."""
    txt = text_of(res.get("content")) if not isinstance(res.get("content"), str) else res["content"]
    m = HOOK.search(txt)
    if m:
        toks = m.group(1).replace('"', "").split()
        k = next((i for i, t in enumerate(toks) if t.endswith((".sh", ".py"))), len(toks) - 1)
        return "hook " + " ".join([os.path.basename(toks[k])] + toks[k + 1:k + 2])
    for key, name in PERMISSIONS:
        if key in txt:
            return name
    for key, name in TOOL_ERRORS:
        if key in txt:
            return "tool error: " + name
    if re.match(r"Exit code \d+", txt):
        body = txt.split("\n", 1)[1] if "\n" in txt else ""
        for key, name in (("Traceback", "python exception"), ("command not found", "command not found"),
                          ("No such file", "missing file"), ("cannot access", "missing file"),
                          ("cannot open", "missing file"), ("Terminated", "killed or timed out"),
                          ("usage:", "usage error")):
            if key in body:
                return "tool error: exit, " + name
        return "tool error: exit, nonzero status" if body.strip() else "tool error: exit, no output"
    return "tool error: other"


def refused(res, call):
    if not res.get("is_error"):
        return None
    txt = json.dumps(res.get("content"))
    if call["name"] == "Bash" and re.match(r'"Exit code 1(\\n)?"$', txt) and "grep" in call["input"].get("command", ""):
        return None                     # grep with no match
    return "tool error" if source(res, call).startswith("tool error") else "hook/permission refusal"


def launch_records(sid):
    """tool_use id -> {type, effort, model, ts, prompt, description, file} of every Agent call, in the main and
    subagent transcripts (file: the transcript that made the call)."""
    out = {}
    files = [os.path.join(PROJ, sid + ".jsonl")] + glob.glob(os.path.join(PROJ, sid, "subagents", "agent-*.jsonl"))
    for f in files:
        for line in open(f):
            if '"Agent"' not in line:
                continue
            r = json.loads(line)
            if r.get("type") != "assistant":
                continue
            for x in (r.get("message") or {}).get("content") or []:
                if isinstance(x, dict) and x.get("type") == "tool_use" and x.get("name") == "Agent":
                    i = x.get("input") or {}
                    out[x["id"]] = {"type": i.get("subagent_type") or "general-purpose",
                                    "effort": i.get("effort") or "default", "model": i.get("model") or "inherit",
                                    "ts": r.get("timestamp") or "", "prompt": i.get("prompt") or "",
                                    "description": i.get("description") or "", "file": f}
    return out


def launches(sid):
    """tool_use id -> (type, effort, timestamp) of every Agent call, in the main and subagent transcripts."""
    return {k: (v["type"], v["effort"], v["ts"]) for k, v in launch_records(sid).items()}


NUMPFX = re.compile(r"^\s*\d+[\t:-]")


def text_of(content):
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text")


def lines_of(text):
    """The lines of a text that can tell a repeat, without the line-number prefix Read, grep -n and nl add."""
    out = []
    for ln in text.split("\n"):
        ln = NUMPFX.sub("", ln, count=1).strip()
        if len(ln) >= 25:
            out.append(ln)
    return out


def analyse(f):
    """[(call, class, reason, context share, timestamp, result chars, result chars already in context)] of one agent
    transcript, and its first timestamp. A result line (25 characters or more, line-number prefix stripped) is already
    in context when the launch prompt, an earlier call's input or an earlier result had it."""
    ag, recs, first = Agent(), [], None
    for line in open(f):
        r = json.loads(line)
        if r.get("timestamp") and first is None:
            first = r["timestamp"]
        recs.append(r)
    calls, results, ctx_of, n_of = [], {}, {}, Counter()
    known = set()
    for r in recs:
        if r.get("type") == "user" and isinstance((r.get("message") or {}).get("content"), str):
            known.update(lines_of(r["message"]["content"])); break
    for r in recs:
        m = r.get("message") or {}
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for x in c:
            if not isinstance(x, dict):
                continue
            if r.get("type") == "assistant" and x.get("type") == "tool_use":
                mid = m.get("id"); u = m.get("usage") or {}
                ctx_of[mid] = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) \
                    + u.get("cache_creation_input_tokens", 0)
                n_of[mid] += 1
                calls.append((x, mid, r.get("cwd") or "/home/joaquin/work/FA", r.get("timestamp")))
            elif x.get("type") == "tool_result":
                results[x.get("tool_use_id")] = (x, r.get("toolUseResult"))
    out = []
    for x, mid, cwd, ts in calls:
        name, inp = x["name"], x.get("input") or {}
        res, tur = results.get(x["id"], ({}, None))
        key = (name, inp.get("command")) if name == "Bash" else \
            (name, json.dumps({k: v for k, v in inp.items() if k != "description"}, sort_keys=True))
        why = refused(res, x)
        shown = "<persisted-output>" not in text_of(res.get("content"))[:200]
        ctxs = ctx_of[mid] / n_of[mid]
        if why:
            cls = "refused"
        elif key in ag.seen and name != "Read":
            cls, why = "repeat", "same input as an earlier call"
            if name == "Bash":
                classify_bash(ag, inp.get("command", ""), cwd, shown)      # keep the read history in step
        elif name == "Read":
            p = absp(inp.get("file_path", ""), cwd)
            fl = (tur or {}).get("file") if isinstance(tur, dict) else None
            if fl and fl.get("startLine"):
                lo = fl["startLine"]; hi = lo + max(fl.get("numLines", 1), 1) - 1
                if lo == 1 and fl.get("totalLines") and hi >= fl["totalLines"]:
                    hi = INF
            else:
                lo = inp.get("offset") or 1; hi = lo + (inp.get("limit") or 2000) - 1
                if not inp.get("limit"):
                    hi = INF
            if ag.covered(p, lo, hi, False):
                cls, why = "re-read", "overlaps an earlier read of " + os.path.basename(p)
            elif ag.orient(p):
                cls, why = "orientation", "first read of " + os.path.basename(p)
            else:
                cls, why = "productive", "new read"
            if shown:
                ag.add(p, lo, hi)
        elif name in ("Edit", "Write", "NotebookEdit"):
            ag.wipe(absp(inp.get("file_path") or inp.get("notebook_path") or "", cwd))
            cls, why = "productive", "edit/write"
        elif name == "Bash":
            cls, why = classify_bash(ag, inp.get("command", ""), cwd, shown)
        else:
            cls, why = "productive", name
        ag.seen.add(key)
        rl = lines_of(text_of(res.get("content")))
        nch, dup = sum(len(l) for l in rl), sum(len(l) for l in rl if l in known)
        known.update(rl)
        known.update(lines_of("\n".join(str(v) for v in inp.values())))
        out.append((x, cls, why, ctxs, ts, nch, dup, res))
    return out, first


def kind_of(why):
    """One kind for a productive call's reason, the first that applies: write, plan, site, git, script run, state
    read, new read, other (Agent, SendMessage, SubagentHandback, ToolSearch, ...)."""
    for k, keys in (("write", ("edit/write", "write (")), ("plan", ("plan ",)), ("site", ("site ",)),
                    ("git", ("git ",)), ("script run", ("script run",)), ("state read", ("state read",)),
                    ("new read", ("read ", "new read", "search"))):
        if any(x in why for x in keys):
            return k
    return "other"


VERDICTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "agent_calls",
                        "refusal_verdicts.json")


def refusal_rows(sid, after, before):
    """[(timestamp, agent id, call, source, context share, result)] of the refused calls of the agents launched in
    the window (the same selection as the class tables)."""
    launch, out = launches(sid), []
    for meta in glob.glob(os.path.join(PROJ, sid, "subagents", "agent-*.meta.json")):
        m = json.load(open(meta))
        f = meta[:-10] + ".jsonl"
        if not os.path.exists(f):
            continue
        ts = launch.get(m.get("toolUseId"), ("", "", ""))[2]
        if not ts:
            with open(f) as fh:
                ts = next((t for t in (json.loads(l).get("timestamp") for l in fh) if t), "")
        if not (after <= ts < before):
            continue
        rows, _ = analyse(f)
        aid = os.path.basename(f)[6:-6]
        out += [(t or "", aid, x, source(res, x), ctx, res) for x, cls, why, ctx, t, nch, dup, res in rows
                if cls == "refused"]
    return sorted(out, key=lambda r: r[0])


def refusals(sid, windows, sample, show):
    """Refusals and errors per source and window; for each hook, the hand verdicts (VERDICTS: tool_use id ->
    {wrong: bool, why}) on a sample of up to `sample` of its refusals, evenly spaced in time over all windows."""
    verdicts = json.load(open(VERDICTS)) if os.path.exists(VERDICTS) else {}
    per = {w: refusal_rows(sid, *w) for w in windows}
    srcs = Counter()
    for rows in per.values():
        for r in rows:
            srcs[r[3]] += 1
    hooks = defaultdict(list)
    for rows in per.values():
        for r in rows:
            if r[3].startswith("hook "):
                hooks[r[3]].append(r)
    picked = {}
    for h, rs in hooks.items():
        rs = sorted({r[2]["id"]: r for r in rs}.values(), key=lambda r: r[0])
        k = min(sample, len(rs))
        picked[h] = [rs[round(i * (len(rs) - 1) / max(k - 1, 1))] for i in range(k)] if k > 1 else rs[:k]
    print("session %s; agents launched in each window (UTC); context = the input context of the response that issued"
          " the call, M tokens; result = the refusal text, k characters" % sid)
    size = lambda r: len(text_of(r[5].get("content")) if not isinstance(r[5].get("content"), str) else r[5]["content"])
    for w, rows in per.items():
        c, ctx, ch = Counter(), Counter(), Counter()
        for r in rows:
            c[r[3]] += 1; ctx[r[3]] += r[4]; ch[r[3]] += size(r)
        print("\nwindow %s .. %s: %d refused calls, %.1fM context" % (w[0], w[1], len(rows), sum(ctx.values()) / 1e6))
        print("%-46s %6s %8s %8s" % ("source", "calls", "ctx M", "result k"))
        for src in sorted(c, key=lambda k: (-ctx[k], k)):
            print("%-46s %6d %8.2f %8.1f" % (src, c[src], ctx[src] / 1e6, ch[src] / 1e3))
    print("\nhooks over all windows: calls, context, and the hand sample (wrong = the call was legitimate)")
    print("%-36s %6s %8s %8s %6s" % ("hook", "calls", "ctx M", "sampled", "wrong"))
    for h in sorted(hooks, key=lambda h: -sum(r[4] for r in hooks[h])):
        ids = [r[2]["id"] for r in picked[h]]
        done = [i for i in ids if i in verdicts]
        print("%-36s %6d %8.2f %8d %6d%s" % (h, len(hooks[h]), sum(r[4] for r in hooks[h]) / 1e6, len(done),
                                             sum(verdicts[i]["wrong"] for i in done),
                                             "  (%d without verdict)" % (len(ids) - len(done)) if len(done) < len(ids)
                                             else ""))
    print("\nhooks' stated reasons over all windows (the refusal's first line, numbers as N)")
    for h in sorted(hooks, key=lambda h: -len(hooks[h])):
        why = Counter()
        for r in hooks[h]:
            t = r[5].get("content"); t = t if isinstance(t, str) else text_of(t)
            t = t.split("]: ", 1)[-1].split("\n")[0]
            why[re.sub(r"\d+", "N", re.sub(r"/tmp/\S+|'[^']*'", "X", t))[:110]] += 1
        for k, v in why.most_common():
            print("%-36s %4d  %s" % (h, v, k))
    print("\nhand verdicts (sample over all windows)")
    for h in sorted(picked, key=lambda h: -len(hooks[h])):
        for r in picked[h]:
            i = r[2]["id"]; v = verdicts.get(i)
            print("%-36s %s %s %s" % (h, i, "WRONG" if v and v["wrong"] else ("ok" if v else "?"),
                                      (v or {}).get("why", "")))
            if show:
                inp = r[2].get("input") or {}
                c = (inp.get("command") or json.dumps(inp)).replace("\n", " | ")
                print("    call:", c if len(c) <= 700 else c[:250] + " [...] " + c[-450:])
                res = r[5].get("content")
                print("    refusal:", (res if isinstance(res, str) else text_of(res))[:500].replace("\n", " | "))


def main(argv):
    opt = {}
    for flag in ("--after", "--before", "--show", "--n", "--from", "--windows", "--sample"):
        if flag in argv:
            i = argv.index(flag); opt[flag] = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    modes = {a for a in argv if a in ("--refusals", "--texts")}
    argv = [a for a in argv if a not in modes]
    sid = argv[1] if len(argv) > 1 else os.path.basename(max(glob.glob(PROJ + "/*.jsonl"), key=os.path.getmtime))[:-6]
    if "--refusals" in modes:
        wins = [tuple(w.split("..")) for w in opt["--windows"].split(",")] if "--windows" in opt else \
            [(opt.get("--after", ""), opt.get("--before", "9999"))]
        refusals(sid, wins, int(opt.get("--sample", 5)), "--texts" in modes)
        return
    if "--show" in opt:
        n, lo = int(opt.get("--n", 10)), int(opt.get("--from", 0))
        for aid in opt["--show"].split(","):
            rows, _ = analyse(os.path.join(PROJ, sid, "subagents", "agent-%s.jsonl" % aid))
            print("== agent %s: %d calls" % (aid, len(rows)))
            for k, (x, cls, why, ctx, ts, nch, dup, _) in enumerate(rows[lo:lo + n], lo):
                inp = x.get("input") or {}
                s = inp.get("command") or inp.get("file_path") or json.dumps(inp)
                if x["name"] == "Read" and (inp.get("offset") or inp.get("limit")):
                    s += " [%s+%s]" % (inp.get("offset"), inp.get("limit"))
                print("%3d %-11s %-34s %6.0fk %3.0f%% of %6d %s: %s" % (
                    k, cls, why[:34], ctx / 1e3, 100 * dup / max(nch, 1), nch, x["name"], s.replace("\n", " ")[:150]))
        return
    after, before = opt.get("--after", ""), opt.get("--before", "9999")
    launch = launches(sid)
    new = lambda: {"n": 0, "calls": Counter(), "ctx": Counter(), "why": Counter(), "ch": Counter(), "dup": Counter(),
                   "kind": Counter()}
    groups = defaultdict(new)
    for meta in glob.glob(os.path.join(PROJ, sid, "subagents", "agent-*.meta.json")):
        m = json.load(open(meta))
        f = meta[:-10] + ".jsonl"
        if not os.path.exists(f):
            continue
        typ, eff, ts = launch.get(m.get("toolUseId"), (m.get("agentType") or "?", "?", ""))
        if m.get("agentType") == "fork":
            typ = "fork"
        if not ts:
            with open(f) as fh:
                for line in fh:
                    ts = json.loads(line).get("timestamp") or ""
                    if ts:
                        break
        if not (after <= ts < before):
            continue
        rows, _ = analyse(f)
        g = groups[(typ, eff)]
        g["n"] += 1
        for x, cls, why, ctx, _, nch, dup, _ in rows:
            g["calls"][cls] += 1; g["ctx"][cls] += ctx; g["ch"][cls] += nch; g["dup"][cls] += dup
            if cls == "refused":
                g["why"][why] += 1
            elif cls == "productive":
                g["kind"][kind_of(why)] += 1
    allg = new(); allg["n"] = sum(g["n"] for g in groups.values())
    for g in groups.values():
        for k in ("calls", "ctx", "why", "ch", "dup", "kind"):
            allg[k].update(g[k])
    print("window %s .. %s, session %s" % (after or "start", before if before != "9999" else "end", sid))
    hdr = "%-34s %4s %6s" % ("type, effort", "n", "calls") + "".join(" %15s" % c for c in CLASSES)
    print("calls per class (share of the group's calls)\n" + hdr)
    order = sorted(groups.items(), key=lambda kv: -sum(kv[1]["ctx"].values())) + [(("all", ""), allg)]
    for key, g in order:
        tot = sum(g["calls"].values()) or 1
        print("%-34s %4d %6d" % (", ".join(k for k in key if k), g["n"], tot)
              + "".join(" %8d (%3.0f%%)" % (g["calls"][c], 100 * g["calls"][c] / tot) for c in CLASSES))
    print("context tokens per class, M (share of the group's context)\n" + hdr.replace("calls", "  ctxM"))
    for key, g in order:
        tot = sum(g["ctx"].values()) or 1
        print("%-34s %4d %6.1f" % (", ".join(k for k in key if k), g["n"], tot / 1e6)
              + "".join(" %7.1fM (%3.0f%%)" % (g["ctx"][c] / 1e6, 100 * g["ctx"][c] / tot) for c in CLASSES))
    print("result text already in the agent's context, share of the class's result characters (all of the group)")
    print(hdr.replace("calls", "  dup%"))
    for key, g in order:
        tot, dtot = sum(g["ch"].values()) or 1, sum(g["dup"].values())
        print("%-34s %4d %5.0f%%" % (", ".join(k for k in key if k), g["n"], 100 * dtot / tot)
              + "".join(" %14.0f%%" % (100 * g["dup"][c] / max(g["ch"][c], 1)) for c in CLASSES))
    tk = sum(allg["kind"].values()) or 1
    print("productive calls by what they did: " + ", ".join("%s %d (%.0f%%)" % (k, v, 100 * v / tk)
                                                         for k, v in allg["kind"].most_common()))
    print("refusals: " + ", ".join("%s %d" % kv for kv in allg["why"].most_common()))


if __name__ == "__main__":
    main(sys.argv)
