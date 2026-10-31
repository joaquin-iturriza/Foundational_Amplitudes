#!/usr/bin/env python3
"""One submission guard (agent redesign, step B2), PreToolUse(Bash). The checks of placement_guard.sh, design_guard.py,
hpo_guard.sh and the Bash half of plot_guard.sh, on one reading of the command (cmdparse.py), plus the GPU budget.
A command is refused (exit 2, every reason found) when
  placement  it runs raw sbatch / condor_submit; it streams bulk data through this laptop (ssh | ssh, scp or rsync
             between two remote ends, sshfs mounts); or it is a named submission (`site submit <site> <project>`, or
             `site run <site> <project> -- ...` that submits) with no unit left for that site in the last `site pick`
             (~/.local/share/ccorch/placement/<project>.json, 60 min; a unit is spent per submission). `auto` passes.
  design     a submission for a plan entry (`site submit|launch add ... --item X`, `site run --item X -- <submits>`,
             scripts/launch_*.sh X) whose entry has no design recorded in the last DESIGN_H hours passed by the design
             reviewer (a REREAD=<run> of the entry under its passed design passes); `plan design-pass|design-fix` from
             anyone but the design-reviewer subagent.
  hpo        an sbatch or site submit of a script that is a job array feeding a hyperparameter a per-task variable
             (allowlist .claude/hpo_grid_allowlist.txt).
  plot       a run.py invocation or a submitted script that turns plotting off (allowlist
             .claude/plot_disable_allowlist.txt); plot_guard.sh keeps the Write half.
  budget     a GPU submission (`site submit` of a script that asks for a GPU) without --est-gpu-hours; or one whose
             estimate plus the item's GPU submissions of the last 60 min (registry runs.est_gpu_hours) passes 30 GPU-h,
             or 10 on Jean Zay (a submission there, or `auto --allow-jeanzay`), unless the item's recorded cost
             (items.est_gpu_hours, set from the decision that approved it) covers that total (CLAUDE.md rule 2).
             The early warning only: the budget is enforced at the site's submission gate (site_gate.py beside the
             site tool), which every sbatch / condor_submit passes, launch scripts and sweep_manager included.
  data       a FA training job whose recipe's pools are not prebuilt on the named site (tools/check_site_data.py run
             there), unless it is chained afterok on that site's prebuild run of the recipe (see data()).
Each check's verdict is the old guard's on the same command (tests/test_hooks_real.py replays the night's submissions
through both). It fails closed on a submission like design_guard did; a command that submits nothing is never refused.
Env for tests: the PLAN_* variables of tools/plan.py; SUBMIT_RUNS_DB (default the site registry); HOME (placement)."""
import json, os, re, shlex, sqlite3, sys, time

HOOKS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HOOKS)
import cmdparse  # noqa: E402

HERE = os.path.dirname(os.path.dirname(HOOKS))
REPO = os.environ.get("CLAUDE_PROJECT_DIR") or HERE
RUNS_DB = os.environ.get("SUBMIT_RUNS_DB") or os.path.expanduser("~/.local/share/ccorch/runs.db")
PLACEMENT_MAX_AGE = 3600
BUDGET_GPU_H, BUDGET_JZ_GPU_H, BUDGET_WINDOW_S = 30.0, 10.0, 3600
RAW_SUBMIT = re.compile(r"(^|[;&|(]|\$\()\s*(sbatch|condor_submit|condor_submit_dag)([^A-Za-z0-9_]|$)", re.M)
HP_RE = (r"(training\.(lr|clip_grad_norm|clip_grad_value|regularization_lambda|cosanneal_warmup_frac|cosanneal_eta_min|"
         r"heterosc_beta|ema_decay|sampler_alpha_ema)|fine_tune\.(lr_scale|layer_decay))")
K1 = "plot"           # assembled so this file never contains a literal match
OFF_RE = re.compile(r"(^|\s)%s=(false|False)|(^|\s)%s:\s*'?(false|False)|%sting\.[a-zA-Z_]+=(false|False)|"
                    r"%sting\.[a-zA-Z_]+:\s*'?(false|False)" % (K1, K1, K1, K1))
GPU_ASK = re.compile(r"^\s*#SBATCH\s+(?:--gres=gpu|--gpus(?:-per-\w+)?[= ]|-G\s*\d)|^\s*request_gpus\s*=\s*[1-9]", re.M)


def _path(s):
    return s if os.path.isabs(s) else os.path.join(REPO, s)


def _allowlisted(name, text, mode):
    """hpo: a line equal to the script's path or repo-relative path; plot: a line that is a substring of text."""
    p = os.path.join(REPO, ".claude", name)
    if not os.path.exists(p):
        return False
    for line in open(p):
        line = line.split("#")[0].strip()
        if line and ((mode == "eq" and line in text) or (mode == "sub" and line in text[0])):
            return True
    return False


# ------------------------------------------------------------------ placement (placement_guard.sh)
def bulk(cmd):
    try:
        lx = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lx.whitespace_split = True
        toks = list(lx)
    except ValueError:
        return None
    starts = {"|", ";", "&&", "||", "(", "&", "$("}
    heads = [t for i, t in enumerate(toks) if i == 0 or toks[i - 1] in starts]
    mounts = {m for t in toks for m in re.findall(r"/mnt/(eos|afs|ccin2p3)\b", t)}
    remote = [t for t in toks if re.match(r"^[A-Za-z0-9_.@-]+:(/|~|[^/]*$)", t) and not t.startswith("http")]
    if heads.count("ssh") >= 2 and "|" in toks:
        return "an `ssh ... | ssh ...` pipe streams the data through this laptop"
    if "ssh" in heads and mounts:
        return "mixing an sshfs mount with ssh streams the data through this laptop"
    if any(h in ("scp", "rsync") for h in heads) and len(remote) >= 2:
        return "scp/rsync between two remote ends streams the data through this laptop"
    if any(h in ("cp", "rsync", "tar", "mv") for h in heads) and len(mounts) >= 2:
        return "copying between two sshfs mounts streams the data through this laptop"
    return None


def named_submissions(scan):
    subs = [m.group(1, 2) for m in re.finditer(r"(?:^|[;&|(]|\$\()\s*site\s+submit\s+(\S+)\s+(\S+)", scan, re.M)]
    if re.search(r"sweep_manager\.py\s+submit|generate_sweep\.py[^;|&]*--submit|condor_submit|sbatch", scan):
        subs += [m.group(1, 2) for m in re.finditer(r"(?:^|[;&|(]|\$\()\s*site\s+run\s+(\S+)\s+(\S+)", scan, re.M)]
    return sorted(set(subs))


def placement(cmd, scan):
    if RAW_SUBMIT.search(scan):
        return ["placement: raw sbatch/condor_submit. Submit through the site tool so the run is registered and "
                "placed:   site pick <project> --jobs N [--groups G --mem 8G]   then   site submit auto <project> "
                "<script> [-- args]"]
    b = bulk(cmd)
    if b:
        return ["placement: %s. Move data site to site:   site copy <project> <src-site> <dst-site> <path...> "
                "[--to DIR] [--overwrite]   (results back here: site fetch <run>; code to a site: site sync)" % b]
    for site, project in named_submissions(scan):
        if site == "auto" or site.startswith("-"):
            continue
        verdict = spend_unit(project, site, dry=True)     # spent in problems() once no check refuses the command
        if verdict != "ok":
            return ["placement: submission to '%s' for '%s' — %s. Placement is not the agent's call, and a batch is "
                    "spread over sites by headroom. Run:   site pick %s --jobs <trials> --groups <sweeps> --mem <G>   "
                    "(or: site submit auto %s <script> [-- args]) and submit only what the plan allots to each site "
                    "(--allow-jeanzay ONLY if the user asked for Jean Zay)." % (site, project, verdict, project, project)]
    return []


def spend_unit(project, site, dry=False):
    f = os.path.join(os.path.expanduser("~/.local/share/ccorch/placement"), "%s.json" % project)
    if not os.path.exists(f):
        return "no placement recorded for this project"
    try:
        d = json.load(open(f))
    except Exception as e:
        return "unreadable placement file: %s" % e
    age = int(time.time()) - int(d.get("ts", 0))
    quota = d.get("quota") or ({d["site"]: 1} if d.get("site") else {})
    if age > PLACEMENT_MAX_AGE:
        return "last `site pick` is %d min old; pick again" % (age // 60)
    left = int(quota.get(site, 0))
    if left <= 0:
        have = ", ".join("%s %d" % kv for kv in quota.items() if kv[1] > 0) or "nothing"
        return "no quota left for %r in the last plan (left: %s)" % (site, have)
    if dry:
        return "ok"
    quota[site] = left - 1
    d["quota"] = quota
    json.dump(d, open(f, "w"))
    return "ok"


# ------------------------------------------------------------------ hpo (hpo_guard.sh)
def hpo(cmd):
    if not re.search(r"(^|[;&|(]|\$\()\s*(sbatch|site +submit)([^A-Za-z0-9_]|$)", cmd, re.M):
        return []
    for s in [t for t in re.split(r"[ \t\n]", cmd) if t.endswith(".sh")]:
        path = s if os.path.isfile(s) else os.path.join(REPO, s)
        if not os.path.isfile(path):
            continue
        rel = path[len(REPO) + 1:] if path.startswith(REPO + "/") else path
        if _allowlisted("hpo_grid_allowlist.txt", {path, rel}, "eq"):
            continue
        text = open(path, errors="replace").read()
        if not re.search(r"^#SBATCH\s+--array", text, re.M):
            continue
        hit = [l for l in text.splitlines() if re.search(HP_RE + r"=[\"']?\$", l)][:3]
        if hit:
            return ["hpo: '%s' is a hand-rolled HYPERPARAMETER GRID (an array varying an HP: %s). HP search goes "
                    "through DyHPO, single-fidelity (sweep/generate_sweep.py, then sweep_manager.py submit); arrays "
                    "stay fine for non-HP ablations. If the user explicitly approved this grid: add '%s' to "
                    ".claude/hpo_grid_allowlist.txt." % (rel, " | ".join(h.strip() for h in hit), rel)]
    return []


# ------------------------------------------------------------------ plot (plot_guard.sh, Bash half)
def plot(cmd, cmds):
    if _allowlisted("plot_disable_allowlist.txt", (cmd,), "sub"):
        return []
    if cmds is None:
        runpy = "\n".join(l for l in cmd.splitlines() if "run.py" in l)
        scripts = [t for t in re.split(r"[ \t\n]", cmd) if t.endswith(".sh")] if re.search(
            r"(^|[;&|(]|\$\()\s*(sbatch|site +submit)([^A-Za-z0-9_]|$)", cmd, re.M) else []
    else:
        runpy = "\n".join(" ".join(ws) for ws in cmdparse.runs_runpy(cmds))
        scripts = cmdparse.submitted_scripts(cmds)
    if runpy and OFF_RE.search(runpy):
        return ["plot: this configures a RUN with PLOTTING OFF (run.py invocation). CLAUDE.md: always keep plot: "
                "true; ask what the flag gates before disabling it (allowlist .claude/plot_disable_allowlist.txt "
                "if the user explicitly approved it)."]
    for s in scripts:
        path = s if os.path.isfile(s) else os.path.join(REPO, s)
        if not os.path.isfile(path) or _allowlisted("plot_disable_allowlist.txt", (path,), "sub"):
            continue
        hit = [l for l in open(path, errors="replace").read().splitlines() if OFF_RE.search(l)][:3]
        if hit:
            return ["plot: this configures a RUN with PLOTTING OFF (sbatch %s: %s). CLAUDE.md: always keep plot: "
                    "true (allowlist .claude/plot_disable_allowlist.txt if the user explicitly approved it)."
                    % (s, " | ".join(h.strip() for h in hit))]
    return []


# ------------------------------------------------------------------ design (design_guard.py)
SUBMITS = re.compile(r"sweep_manager\.py\s+submit\b|\bsbatch\b|\bcondor_submit(?:_dag)?\b")
GENERATOR = re.compile(r"generate_\w*sweeps?\w*\.py")
VERDICT = re.compile(r"tools/plan\.py\b[^;&|\n]*\bdesign-(?:pass|fix)\b")
REREAD = re.compile(r"(?:^|[\s;&|(])REREAD=['\"]?([A-Za-z0-9_.:-]+)")


def items_of(toks):
    out = []
    for i, t in enumerate(toks):
        if t == "--item" and i + 1 < len(toks):
            out.append(toks[i + 1])
        elif t.startswith("--item="):
            out.append(t.split("=", 1)[1])
    return [x for part in out for x in part.split(",") if x]


def remote_submits(text):
    if SUBMITS.search(text):
        return True
    return bool(GENERATOR.search(text)) and ("--auto-submit" in text or "--dry-run" not in text)


def submissions(cmd, segs):
    """The item ids each submission in the command names ([] for a submission with none: site refuses it itself)."""
    if segs is None:
        text = cmdparse.scan_text(cmd)
        if re.search(r"\bsite\s+(?:submit|launch\s+add)\b", text) or (
                re.search(r"\bsite\s+run\b", text) and remote_submits(text)):
            return [re.findall(r"--item[=\s]+['\"]?([^\s'\";&|]+)", cmdparse.code_text(cmd))]
        m = re.search(r"scripts/launch_\w+\.sh\s+['\"]?([^\s'\";&|]+)", text)
        return [[m.group(1)]] if m else []
    out = []
    for toks in segs:
        if not toks:
            continue
        if os.path.basename(toks[0]) == "site" and len(toks) > 1:
            rest, verb = toks[1:], toks[1]
            if verb == "submit" or (verb == "launch" and "add" in rest[:2]):
                out.append(items_of(rest))
            elif verb == "run" and "--" in rest:
                k = rest.index("--")
                if remote_submits(" ".join(rest[k + 1:])):
                    out.append(items_of(rest[:k]))
            continue
        i = 1 if os.path.basename(toks[0]) in ("bash", "sh") else 0
        if i < len(toks) and re.search(r"(?:^|/)scripts/launch_\w+\.sh$", toks[i]) and i + 1 < len(toks):
            out.append([toks[i + 1]])
    return out


def gives_verdict(cmd):
    segs = cmdparse.all_segments(cmd)
    if segs is None:
        return bool(VERDICT.search(cmd))
    for seg in segs:
        ws = cmdparse.words(seg)
        if cmdparse.prog(ws) in cmdparse.READERS:
            continue
        k = next((i for i, w in enumerate(ws) if re.search(r"(?:^|/)plan\.py$", w)), None)
        if k is not None and any(w in ("design-pass", "design-fix") for w in ws[k + 1:]):
            return True
        if any(VERDICT.search(x) for x in cmdparse.inline_programs(seg)):
            return True
    return False


def _plan():
    sys.path.insert(0, os.path.join(HERE, "tools"))
    import plan
    return plan


def reread_ok(plan, e, st, runs):
    x = plan.last_design(e)
    if not runs or not x or not x.get("passed_at"):
        return False
    mine = {r for r, _, _ in plan.runs_of(e, st)} | set((e.get("evidence") or {}).get("runs") or [])
    return all(r in mine for r in runs)


def design(cmd, inp, subs):
    if gives_verdict(cmd) and inp.get("agent_type") != "design-reviewer":
        return ["design: plan design-pass and design-fix are the design reviewer's verdict; only the design-reviewer "
                "subagent runs them (this call comes from %s). Launch it (Agent tool, subagent_type design-reviewer) on "
                "the entry." % (inp.get("agent_type") or "the main session")]
    if not subs:
        return []
    try:
        plan = _plan()
        d, st = plan.load()
        ids = {e["id"]: e for e in d["entries"]}
        bad = []
        rereads = REREAD.findall(cmdparse.code_text(cmd))
        for x in sorted({x for s in subs for x in s if x != "*"}):
            if x in ids and not plan.design_fresh(ids[x]) and not reread_ok(plan, ids[x], st, rereads):
                last = plan.last_design(ids[x])
                old = last and plan.now() - plan.parse(last["at"]) > plan.dt.timedelta(hours=plan.DESIGN_H)
                bad.append("%s (%s)" % (x, "no design recorded" if not last else
                                        "last design recorded %s, more than %d h ago" % (last["at"][:16], plan.DESIGN_H)
                                        if old else "the design recorded %s %s" % (last["at"][:16], (
                                            "was sent back by the design reviewer: " + last.get("fix", "")
                                            if last.get("fix_at") else "waits for the design reviewer's verdict"))))
        hours = plan.DESIGN_H
    except BaseException as ex:                    # plan.py exits through sys.exit on a broken plan
        return ["design: could not check this submission's design (%s: %s). Fix the plan, then submit again."
                % (type(ex).__name__, ex)]
    if not bad:
        return []
    return ["design: a job for a plan entry needs its design recorded within %d h and passed by the design reviewer; "
            "missing for %s.\nIf the design is recorded, run the design-reviewer subagent on it (Agent tool, "
            "subagent_type design-reviewer, naming the entry); it runs plan design-pass on PASS. An eval-only re-read "
            "of a run of the entry under its passed design carries REREAD=<run id>. Otherwise record it first, every "
            "field (the check says how it was verified, on existing data, that the measure moves with the tested "
            "change; a design whose check fails is not run):\n  python3 tools/plan.py design <entry> --question "
            "\"<what the job answers>\" --measure \"<the number that answers it>\" --check \"<how it was seen to move "
            "with the change, on which data>\" --fixed \"<what is held fixed>\" --control \"<the matched control>\" "
            "--seeds \"<seeds>\" --outcome \"<what yes and what no look like>\"\nthen submit again." % (hours, ", ".join(bad))]


# ------------------------------------------------------------------ budget (new)
SUBMIT_VALUED = {"--branch", "--note", "--est-gpu-hours", "--hdr", "--env", "--item", "--expect"}


def site_submits(segs):
    """(site, project, script, item, est or None, allow_jeanzay) of every `site submit` that submits (not --check),
    its arguments read as site's own parser reads them (valued options take the next token, dashes or not)."""
    out = []
    for toks in segs or []:
        if not toks or os.path.basename(toks[0]) != "site" or toks[1:2] != ["submit"]:
            continue
        rest = cmdparse.strip_redirs(toks[2:])
        if "--" in rest:
            rest = rest[:rest.index("--")]
        pos, opts, k = [], {}, 0
        while k < len(rest):
            t = rest[k]
            name, eq, val = t.partition("=")
            if t.startswith("--") and name in SUBMIT_VALUED:
                if not eq:
                    val = rest[k + 1] if k + 1 < len(rest) else ""
                    k += 1
                opts.setdefault(name, []).append(val)
            elif t.startswith("-"):
                opts.setdefault(name, []).append(True)
            else:
                pos.append(t)
            k += 1
        if len(pos) < 3 or "--check" in opts or "--help" in opts or "-h" in opts:
            continue
        est = None
        if "--est-gpu-hours" in opts:
            try:
                est = float(opts["--est-gpu-hours"][-1])
            except ValueError:
                est = -1.0
        item = (opts.get("--item") or [None])[-1]
        out.append((pos[0], pos[1], pos[2], item, est, "--allow-jeanzay" in opts))
    return out


def asks_gpu(script):
    p = _path(script)
    try:
        return bool(GPU_ASK.search(open(p, errors="replace").read()))
    except OSError:
        return True                 # unreadable here: counted as a GPU job (the site refuses a missing script anyway)


REPEAT = re.compile(r"^SEQ:(\d+):(\d+)$|^\{(\d+)\.\.(\d+)\}$")
SEQ_PIPE = re.compile(r"\bseq\s+(?:(\d+)\s+)?(\d+)\s*\|\s*(?:xargs|parallel)\b")


def _words_count(words):
    """passes of a `for x in WORDS` list: literal words count one each, `$(seq N)`, `$(seq A B)` and {A..B} their
    range; anything else (a variable, a glob, another substitution) cannot be read: None."""
    n = 0
    for w in words:
        m = REPEAT.match(w)
        if m:
            n += (int(m.group(2)) - int(m.group(1)) + 1) if m.group(2) else int(m.group(4)) - int(m.group(3)) + 1
        elif re.search(r"[$*?`\[]", w) or w.startswith("{"):
            return None
        else:
            n += 1
    return n


def submit_multipliers(cmd):
    """[passes] for each `site submit` of the command, in order: the product of the loops that enclose it (for over a
    word list, $(seq ...), {A..B}; `seq N | xargs/parallel`); None when an enclosing loop's count cannot be read
    (while, until, a variable list, xargs over anything else). Read on the code with quoted strings blanked, so a
    --note that says "for" or "parallel" is no loop."""
    text = cmdparse.blank_quotes(cmdparse.strip_comments(cmdparse.code_text(cmd)))
    text = re.sub(r"\$\(\s*seq\s+(\d+)(?:\s+(\d+))?\s*\)",
                  lambda m: "SEQ:%s:%s" % ((m.group(1), m.group(2)) if m.group(2) else ("1", m.group(1))), text)
    toks = [t for t in re.split(r"[ \t]+", re.sub(r"([;\n|&()])", r" \1 ", text)) if t]
    stack, mults, start = [], [], True
    for k, t in enumerate(toks):
        if t in (";", "\n", "&", "&&", "|", "||", "(", ")"):
            start = True
            continue
        if start and t == "for":
            if k + 2 < len(toks) and toks[k + 2] == "in":
                words = []
                for w in toks[k + 3:]:
                    if w in (";", "\n", "do"):
                        break
                    words.append(w)
                stack.append(_words_count(words))
            else:                                   # for ((...)), or `for x` over "$@": a count it cannot read
                stack.append(None)
        elif start and t in ("while", "until"):
            stack.append(None)
        elif start and t == "done" and stack:
            stack.pop()
        elif t == "site" and toks[k + 1:k + 2] == ["submit"]:
            seg = " ".join(toks[max(0, k - 12):k])
            piped = None
            if re.search(r"\b(?:xargs|parallel)\b", seg):
                m = SEQ_PIPE.search(" ".join(toks[:k]))
                piped = (int(m.group(2)) - int(m.group(1) or 1) + 1) if m else None
                stack_here = stack + [piped]
            else:
                stack_here = stack
            m = 1
            for c in stack_here:
                m = None if (m is None or c is None) else m * c
            mults.append(m)
        start = t in ("do", "then", "else", "{") or (start and re.match(r"^[A-Za-z_]\w*=", t) is not None)
    return mults


def _registry(project, item):
    """(GPU-h of the item's submissions in the window, of them on Jean Zay, the item's recorded cost or None)."""
    recent = recent_jz = 0.0
    covered = None
    try:
        c = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True, timeout=10)
        for s, h in c.execute("SELECT site, est_gpu_hours FROM runs WHERE project=? AND item IS ? AND submitted >= ?",
                              (project, item, int(time.time()) - BUDGET_WINDOW_S)):
            recent += h or 0.0
            recent_jz += (h or 0.0) if s == "jeanzay" else 0.0
        r = c.execute("SELECT est_gpu_hours FROM items WHERE project=? AND id=?", (project, item)).fetchone()
        covered = r[0] if r else None
        c.close()
    except sqlite3.Error:
        pass
    return recent, recent_jz, covered


def budget(cmd, segs):
    """Everything the command submits counts together with the item's last 60 min: several `site submit` in one
    command are summed, and one inside a loop counts once per pass (`seq N` / {A..B}; a loop whose count cannot be
    read is refused unless the item's recorded cost covers a pass on top of the window... it cannot, so it is refused
    with the reason). Not seen here: submissions made inside a script (scripts/launch_*.sh) or by sweep_manager on a
    site; the site's submission gate counts those (site_gate.py)."""
    out = []
    if segs is None:
        if re.search(r"\bsite\s+submit\b", cmdparse.scan_text(cmd)):
            return ["budget: a `site submit` in a command that cannot be read; run it on its own line"]
        return []
    mults = submit_multipliers(cmd)
    totals = {}                                         # (project, item) -> [GPU-h, on Jean Zay]
    subs = site_submits(segs)
    if len(mults) != len(subs):          # a submit the plain reading cannot place (inside sh -c, a substitution)
        loopy = re.search(r"(?:^|[;&|(\s])(?:for|while|until)\s|\bxargs\b|\bparallel\b",
                          cmdparse.scan_text(cmd))          # the inner programs of sh -c / eval included
        mults = [None if loopy else 1] * len(subs)
    for (site, project, script, item, est, allow_jz), n in zip(subs, mults):
        if not asks_gpu(script):
            continue
        if est is None or est < 0:
            out.append("budget: `site submit %s %s %s` asks for a GPU: pass --est-gpu-hours H (wall time x GPUs; "
                       "CLAUDE.md rule 2)" % (site, project, script))
            continue
        if n is None:
            out.append("budget: a GPU `site submit` inside a loop whose count is not readable (seq N or {A..B}): "
                       "the total cannot be summed; write the count out")
            continue
        jz = site == "jeanzay" or (site == "auto" and allow_jz)
        t = totals.setdefault((project, item), [0.0, 0.0, False])
        t[0] += est * n
        t[1] += est * n if jz else 0.0
        t[2] = t[2] or jz
    for (project, item), (here, here_jz, jz) in sorted(totals.items(), key=str):
        recent, recent_jz, covered = _registry(project, item)
        total, total_jz = recent + here, recent_jz + here_jz
        over = []
        if total > BUDGET_GPU_H:
            over.append("%.1f GPU-h for item %s in the last 60 min with this command (limit %g)" % (total, item,
                                                                                               BUDGET_GPU_H))
        if jz and total_jz > BUDGET_JZ_GPU_H:
            over.append("%.1f GPU-h on Jean Zay (limit %g)" % (total_jz, BUDGET_JZ_GPU_H))
        if over and not (covered is not None and covered >= max(total, total_jz if jz else 0.0)):
            out.append("budget: %s, and the item's recorded cost (%s) does not cover it. Above the budget the user "
                       "confirms first (CLAUDE.md rule 2); with their words recorded, `site item set %s %s "
                       "--est-gpu-hours <approved total>`." % ("; ".join(over), "none" if covered is None else
                                                                 "%g GPU-h" % covered, project, item))
    return out


# ------------------------------------------------------------------ data (new, E3072)
# A training arm whose recipe's pools are not on the site fails in init_data (require_cache) after its queue wait; the
# mixture arms of 2026-10-30 went out so (FA-ccin2p3-c6d1c5ff). Every `site submit <site> FA <script> -- <arm> ...` of a
# script that picks its training recipe by arm (`TRAIN=` set before and inside a `case "$ARM" in`), and every recipe
# named in the job's arguments, is checked on that site with tools/check_site_data.py (pools prebuilt, sidecars) before
# it goes; a pass is remembered DATA_OK_S. A submission chained after the job that builds the pools passes without the
# read: --hdr=--dependency=afterok:<id>[:<id>] naming a registry run of a prebuild script (job_catalog_v4_regen.sh,
# prebuild_*.sh) on that site, not failed or cancelled, whose --note names the recipe's file name.
DATA_OK_S = 6 * 3600
DATA_CACHE = os.environ.get("SUBMIT_DATA_CACHE") or os.path.expanduser("~/.local/share/ccorch/placement/site_data_ok.json")
PREBUILD_JOB = re.compile(r"(?:^|/)(?:job_catalog_v4_regen|prebuild_\w+)\.sh$")
PARAM_ASSIGN = re.compile(r"(?:^|[\s;])([A-Z_][A-Z0-9_]*)=(\$\{[A-Za-z_][A-Za-z0-9_]*[#%]{1,2}[^}$`()]*\})(?=[\s;]|$)")


def _submit_parts(toks):
    """(script, job args after `--`, the --hdr values) of one `site submit` segment."""
    rest = cmdparse.strip_redirs(toks[2:])
    args = rest[rest.index("--") + 1:] if "--" in rest else []
    head = rest[:rest.index("--")] if "--" in rest else rest
    hdr, pos, k = [], [], 0
    while k < len(head):
        t = head[k]
        name, eq, val = t.partition("=")
        if t.startswith("--") and name in SUBMIT_VALUED:
            if not eq:
                val = head[k + 1] if k + 1 < len(head) else ""
                k += 1
            if name == "--hdr":
                hdr.append(val)
        elif not t.startswith("-"):
            pos.append(t)
        k += 1
    return (pos[2] if len(pos) > 2 else None), args, hdr


def arm_recipe(script_text, arm):
    """The training recipe a job script picks for `arm`: the last TRAIN= of the `case "$ARM" in` branch whose patterns
    match it, else the TRAIN= set before the case; $PROJECT_DIR read as the repo, parameter expansions of the branch
    (SH=${ARM#...}) evaluated; None when the script picks no recipe by arm."""
    import fnmatch, subprocess
    lines = script_text.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r'^case "\$ARM" in\s*$', l)), None)
    if start is None:
        return None
    default = None
    for l in lines[:start]:
        m = re.match(r'^TRAIN="([^"]+)"', l)
        if m:
            default = m.group(1)
    branch, inside = [], False
    for l in lines[start + 1:]:
        if re.match(r"^esac\b", l):
            break
        m = re.match(r"^    ([^\s()#][^()]*)\)", l)         # a top-level pattern line (4-space indent)
        if m:
            pats = [p.strip() for p in m.group(1).split("|")]
            if inside:
                break
            inside = any(fnmatch.fnmatchcase(arm, p) for p in pats)
        if inside:
            branch.append(l)
    text = "\n".join(branch)
    trains = re.findall(r'\bTRAIN="([^"]+)"', text)
    train = trains[-1] if trains else default
    if not train:
        return None
    assigns = PARAM_ASSIGN.findall(text)
    env = {"ARM": arm}
    if assigns and re.search(r"\$\{?(%s)\b" % "|".join(n for n, _ in assigns), train):
        prog = "; ".join("%s=%s" % (n, v) for n, v in assigns) + "; " + "; ".join(
            'printf "%%s=%%s\\n" %s "$%s"' % (n, n) for n, _ in assigns)
        out = subprocess.run(["bash", "--norc", "--noprofile", "-c", prog], env={"ARM": arm, "PATH": "/usr/bin:/bin"},
                             capture_output=True, text=True, timeout=10).stdout
        env.update(dict(l.split("=", 1) for l in out.splitlines() if "=" in l))
    for n, v in sorted(env.items(), key=lambda kv: -len(kv[0])):
        train = train.replace("${%s}" % n, v).replace("$%s" % n, v)
    for root in ("${PROJECT_DIR}/", "$PROJECT_DIR/"):
        if train.startswith(root):
            train = train[len(root):]
    return train


def _dependency_ids(hdr):
    out = []
    for h in hdr:
        m = re.match(r"^--dependency=(?:afterok|afterany):([0-9:]+)$", h)
        if m:
            out += [x for x in m.group(1).split(":") if x]
    return out


def _built_by(site, recipe, deps):
    """a dependency is a registry prebuild run on `site`, not failed or cancelled, whose note names the recipe file."""
    if not deps:
        return False
    try:
        c = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True, timeout=10)
        rows = c.execute("SELECT job, state, note FROM runs WHERE project='FA' AND site=? AND sched_id IN (%s)"
                         % ",".join("?" * len(deps)), [site] + deps).fetchall()
        c.close()
    except sqlite3.Error:
        return False
    name = os.path.basename(recipe)
    return any(PREBUILD_JOB.search(job or "") and (state or "") not in ("FAILED", "CANCELLED", "TIMEOUT")
               and name in (note or "") for job, state, note in rows)


def _site_data_ok(site, recipe):
    """(ok, what the site said): tools/check_site_data.py <recipe> on the site; passes are remembered DATA_OK_S."""
    import subprocess
    key = "%s:%s:alphas" % (site, recipe)
    try:
        cache = json.load(open(DATA_CACHE))
    except (OSError, ValueError):
        cache = {}
    if time.time() - cache.get(key, 0) < DATA_OK_S:
        return True, "passed within %d h" % (DATA_OK_S // 3600)
    try:
        p = subprocess.run(["site", "run", site, "FA", "--", "python", "tools/check_site_data.py", "--alphas", recipe],
                           capture_output=True, text=True, timeout=100)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, "the read on %s did not finish (%s)" % (site, type(e).__name__)
    said = [l.strip() for l in (p.stdout + p.stderr).splitlines()
            if l.strip().startswith((os.path.basename(recipe) + ":", "env:"))]
    if p.returncode == 0:
        cache[key] = time.time()
        try:
            os.makedirs(os.path.dirname(DATA_CACHE), exist_ok=True)
            json.dump(cache, open(DATA_CACHE, "w"))
        except OSError:
            pass
        return True, " ".join(said)
    return False, " ".join(said) or (p.stdout + p.stderr).strip()[-300:]


def data(segs):
    if os.environ.get("SUBMIT_TEST") == "1" and "data" not in (os.environ.get("SUBMIT_ONLY") or "").split(","):
        return []                       # the replay tests never read a site (tests/test_submit_data_check.py covers it)
    out = []
    for toks in segs or []:
        if not toks or os.path.basename(toks[0]) != "site" or toks[1:2] != ["submit"]:
            continue
        rest = toks[2:]
        if "--check" in rest or "--help" in rest or "-h" in rest:
            continue
        script, args, hdr = _submit_parts(toks)
        pos = [t for t in cmdparse.strip_redirs(rest[:rest.index("--")] if "--" in rest else rest)
               if not t.startswith("-")]
        if not script or len(pos) < 2 or pos[1] != "FA":
            continue
        site = pos[0]
        recipes = [a for a in args if re.match(r"^(?:\$PROJECT_DIR/)?recipes/[\w./-]+\.ya?ml$", a)]
        try:
            text = open(_path(script), errors="replace").read()
        except OSError:
            text = ""
        if args and re.search(r"^TRAIN=", text, re.M):
            r = arm_recipe(text, args[0])
            if r:
                recipes.append(r)
        for r in dict.fromkeys(recipes):
            if site == "auto":
                out.append("data: `site submit auto` of a job that trains on %s: name the site `site pick` gave, so "
                           "its pools can be checked there" % r)
                continue
            if _built_by(site, r, _dependency_ids(hdr)):
                continue
            ok, said = _site_data_ok(site, r)
            if not ok:
                out.append("data: %s trains on %s, whose pools are not all on %s (tools/check_site_data.py: %s). "
                           "Prebuild them first and chain the job on that build: --hdr=--dependency=afterok:<slurm id> "
                           "of a prebuild run (scripts/job_catalog_v4_regen.sh, prebuild_*.sh) on %s whose --note "
                           "names %s." % (args[0] if args else script, r, site, said, site, os.path.basename(r)))
    return out


SUBMITTING = re.compile(r"\bsite\s+(?:submit|run|launch)\b|(^|[;&|(]|\$\()\s*(sbatch|condor_submit)", re.M)


def expect(segs):
    """--expect KEY=VALUE is compared with the run's RUN_SUMMARY: the JSON form's keys or the plain form's k=v pairs
    (`RUN_SUMMARY lhe_probe gridpool rc=0` gives {"rc": "0"}), so a KEY with blanks never matches and every end of the
    run raises an ALERT (2026-10-31: --expect "lhe_probe gridpool rc=0" on 47 gridpool jobs). Refused here."""
    out = []
    for toks in segs or []:
        if not toks or os.path.basename(toks[0]) != "site" or toks[1:2] != ["submit"]:
            continue
        rest = cmdparse.strip_redirs(toks[2:])
        for k, t in enumerate(rest):
            v = rest[k + 1] if t == "--expect" and k + 1 < len(rest) else (t[9:] if t.startswith("--expect=") else None)
            if v is None:
                continue
            key = v.partition("=")[0]
            if not key or any(c.isspace() for c in key):
                out.append("expect: --expect %r: the key %r is not one RUN_SUMMARY gives (its k=v pairs or JSON keys; "
                           "for `RUN_SUMMARY <name> rc=0` write --expect rc=0)" % (v, key))
    return out


def problems(inp):
    """every check runs on its own: an error in one never switches off another; it refuses a command that submits
    (fails closed) and passes any other."""
    cmd = (inp.get("tool_input") or {}).get("command", "") or ""
    if not cmd:
        return []
    only = set((os.environ.get("SUBMIT_ONLY") or "").split(",")) - {""} if os.environ.get("SUBMIT_TEST") == "1" \
        else set()                                  # tests: one check at a time, to compare it with its old guard
    try:
        scan = cmdparse.scan_text(cmd)
    except Exception:
        scan = cmd
    try:
        segs = cmdparse.commands(cmd)
    except Exception:
        segs = None

    def design_check():
        try:
            subs = submissions(cmd, segs)
        except Exception:
            subs = []                               # not readable as a submission: never refused for its design
        return design(cmd, inp, subs)

    checks = (("placement", lambda: placement(cmd, scan)), ("hpo", lambda: hpo(cmd)), ("plot", lambda: plot(cmd, segs)),
              ("design", design_check), ("budget", lambda: budget(cmd, segs)), ("data", lambda: data(segs)),
              ("expect", lambda: expect(segs)))
    out = []
    for name, f in checks:
        if only and name not in only:
            continue
        try:
            out += f()
        except Exception as e:
            if SUBMITTING.search(scan):
                out.append("%s: the check failed (%s: %s) on a command that submits; refused until it can be read"
                           % (name, type(e).__name__, e))
    if not out and (not only or "placement" in only):
        # a unit of the last `site pick` is spent only by a command that goes: one refused by another check (data,
        # budget...) once spent it and left the plan short for the corrected resubmission (2026-10-30)
        try:
            for site, project in named_submissions(scan):
                if site != "auto" and not site.startswith("-"):
                    spend_unit(project, site)
        except Exception:
            pass
    return out


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    bad = problems(inp)
    if bad:
        sys.stderr.write("BLOCKED by submit_guard:\n" + "\n".join("  " + b for b in bad) + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
