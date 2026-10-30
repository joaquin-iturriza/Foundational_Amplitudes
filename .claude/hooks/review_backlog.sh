#!/usr/bin/env bash
# review_backlog.sh — BATCHED pillar review. Spaced out, not per-change.
#
# Three pillars of this repo each have a reviewer subagent:
#   CLAUDE.md          -> claudemd-keeper   (keep it an operating manual, not a notebook)
#   docs/*.tex         -> notes-editor      (author voice, figures, claims)
#   code/config/sweeps -> repo-reviewer     (correctness, structure, hygiene)
#
# Why batched: reviewing every 2-line diff at commit time burns a subagent per edit and
# reviews nothing in context. Instead each pillar carries a WATERMARK — the commit at
# which it was last reviewed, plus the size of the change that was reviewed then. The
# Stop hook accumulates everything since that watermark and only asks for a reviewer once
# the backlog is worth a pass. The reviewer then sees the WHOLE backlog at once, which is
# also a better review: it can judge a section, not a hunk.
#
# WHERE THE TEETH ARE. No Stop nudge (removed, agent redesign B4): the gate is `gate`, a
# PreToolUse(Edit|Write) hook: while a pillar is over threshold,
# EDITS TO THAT PILLAR'S FILES ARE DENIED. Passing the turn then buys nothing, because the
# next edit to the overdue file is refused until its reviewer has run. Commits, and work on
# every other pillar, stay unblocked.
#
# Modes:
#   check, bashpost       removed (no-ops until unregistered, agent redesign B4)
#   gate                  (PreToolUse(Edit|Write)) DENY edits to an over-threshold pillar
#   begin <reviewer>      reviewer takes the lock: lifts `gate` so it can apply its own fixes
#   advance <reviewer>    reviewer PASSED; resets its watermark and drops the lock
#   status                human-readable backlog table (also: what /review-now would run)
#   init                  set every watermark to "everything so far is reviewed"
#
# Escape hatch, deliberately explicit: `begin <reviewer>` lifts the gate for that pillar
# until the next `advance`. It is the reviewer's normal first step, and it is also how a
# human says "I am editing this myself, stand down" — an auditable act, not a silent bypass.
#
# Shell only — no python/jq (not guaranteed on the hook PATH). Pathspecs keep every git
# call scoped, which matters on Lustre where a full-tree diff is slow.
set -uo pipefail

# No filename globbing anywhere in this script: the pillar pathspecs (`*.py`, `docs/*.tex`) expand unquoted and must
# reach git as written, where they match at any depth. With globbing on, `*.py` became the root's own .py files and
# analysis/**/*.py never counted toward the repo-reviewer's backlog (in the line and commit counts, gate and bashpre).
set -f
FALLBACK_REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STATE_REL=".claude/.review_state"

# --- pillar table -----------------------------------------------------------
# name | line threshold | commit threshold | git pathspecs  (CLAUDE.md: every change is reviewed)
#
# The code pathspec is SOURCE only. `*.py`/`*.sh`/`*.yaml`-style globs match at any depth in
# git, so they already cover analysis/, compare_models/, etc. without listing them; the
# :(exclude) terms drop the generated companions that share those trees (result JSONs, job
# logs, run/sweep output), which would otherwise swamp the line count with data churn.
# Plan bookkeeping, generated data and generated recipe lists are not code (agent redesign B5): plan/, analysis/**/data/,
# tests/*_output.txt, the catalog recipe lists and their per-class directories. .claude/settings.json (the hook
# registration) is: the JSON exclusion skips paths under dot-directories, so it stays in. .claude/context/ is rules
# text, the CLAUDE.md keeper's.
PILLARS='claudemd-keeper|1|1|CLAUDE.md .claude/context
notes-editor|200|20|docs/*.tex
repo-reviewer|520|100000|*.py *.sh config recipes models sweep tools tests scripts IntrinsicDimDeep .claude/settings.json :(exclude,glob)[!.]*/**/*.json :(exclude,glob)*.json :(exclude)*.log :(exclude)*.out :(exclude)*.err :(exclude)runs/* :(exclude)sweeps/* :(exclude)plan/* :(exclude,glob)analysis/**/data/** :(exclude)tests/*_output.txt :(exclude)recipes/catalog_v*_*.yaml :(exclude,glob)recipes/catalog_v*/**'

pillar_field() { printf '%s\n' "$PILLARS" | awk -F'|' -v n="$1" -v f="$2" '$1==n{print $f}'; }
pillar_names() { printf '%s\n' "$PILLARS" | awk -F'|' '{print $1}'; }

REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || REPO="$FALLBACK_REPO"
[ -z "$REPO" ] && REPO="$FALLBACK_REPO"
cd "$REPO" 2>/dev/null || exit 0
git rev-parse --verify -q HEAD >/dev/null 2>&1 || exit 0
# The watermarks and locks are one per repository: inside a git worktree (agent worktrees under .claude/worktrees,
# worktrees/wt-*) they are read from the main checkout (git rev-parse --git-common-dir), never from a stale copy of
# its own (a worktree once carried a watermark 1477 commits old and the gate refused every edit there).
STATE_ROOT="$REPO"
_common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
case "$_common" in */.git) [ -d "${_common%/.git}" ] && STATE_ROOT="${_common%/.git}" ;; esac

# Count changed CONTENT lines in a diff blob (not the +++/--- file headers).
# NB: `grep -c` already prints 0 on no match and exits 1 — a `|| echo 0` here would emit
# a SECOND zero and poison the arithmetic downstream.
changed_lines() { local n; n=$(printf '%s\n' "$1" | grep -c '^[+-][^+-]' 2>/dev/null); echo "${n:-0}"; }

# Total size of the pillar's change since $1, counting untracked-but-not-ignored files in
# the pillar's paths at their full length (a brand-new module is a change, even uncommitted).
pillar_lines() {
  local since="$1" paths="$2" blob n u f
  blob=$(git diff "$since" -- $paths 2>/dev/null)
  n=$(changed_lines "$blob")
  u=0
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    [ -f "$f" ] || continue
    local w; w=$(wc -l < "$f" 2>/dev/null); u=$(( u + ${w:-0} ))
  done < <(git ls-files --others --exclude-standard -- $paths 2>/dev/null)
  echo $(( n + u ))
}

# The code pillar's size is weighted (tools/review_weight.py): each hand-written line by its area's and kind's measured
# rate of blocking errors, a generated config family once; 520 = about two blocking errors waiting, the user's call of
# 2026-10-29 [D171]. The commit count never fired the review alone, so its threshold is off (100000).
pillar_size() {
  if [ "$1" = repo-reviewer ]; then
    python3 "$REPO/tools/review_weight.py" "$2" $3 2>/dev/null || pillar_lines "$2" "$3"
  else pillar_lines "$2" "$3"; fi
}

pillar_commits() { git rev-list --count "$1"..HEAD -- $2 2>/dev/null || echo 0; }

# What a reviewer passed, as content: the files of the pillar that differ from the watermark commit $1 (committed
# since or not, untracked included), each with its blob id (or "deleted"), hashed. Committing the reviewed diff leaves
# this unchanged, so the commit is not a new backlog; any further edit (or reverting one of those files) changes it.
# Only the files changed since the watermark are read, so the repo-reviewer's wide pathspec stays cheap.
reviewed_hash() {
  local since="$1" paths="$2" f
  { git -c core.quotePath=false diff --name-only "$since" -- $paths 2>/dev/null
    git -c core.quotePath=false ls-files --others --exclude-standard -- $paths 2>/dev/null; } | LC_ALL=C sort -u |
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    if [ -f "$f" ]; then printf '%s %s\n' "$(git hash-object -- "$f" 2>/dev/null)" "$f"
    else printf 'deleted %s\n' "$f"; fi
  done | cksum | awk '{print $1"-"$2}'
}

# state file: "<sha> <baseline_lines> <baseline_commits> [<reviewed_hash>]"
state_file() { echo "$STATE_ROOT/$STATE_REL/$1"; }

read_state() {                       # $1=reviewer -> echoes "sha lines commits"
  local f; f=$(state_file "$1")
  if [ -f "$f" ]; then cat "$f"; else echo ""; fi
}

write_state() {                      # $1=reviewer $2=sha $3=lines $4=commits $5=reviewed_hash
  mkdir -p "$STATE_ROOT/$STATE_REL"
  printf '%s %s %s %s\n' "$2" "$3" "$4" "${5:-}" > "$(state_file "$1")"
}

lock_file() { echo "$STATE_ROOT/$STATE_REL/$1.lock"; }

# A lock exempts its pillar from `gate` only while fresh. A reviewer whose `advance` never ran (it was refused by the
# permission check on 2026-10-06) left both locks in place for two days, and the gate stood down the whole time.
LOCK_TTL_MIN=180
lock_fresh() { local f; f=$(lock_file "$1"); [ -f "$f" ] && [ -n "$(find "$f" -mmin -"$LOCK_TTL_MIN" 2>/dev/null)" ]; }

# Which reviewer owns a repo-relative path (empty = ungated).
pillar_for_path() {
  case "$1" in
    CLAUDE.md)                                             echo claudemd-keeper ;;
    docs/*.tex)                                            echo notes-editor ;;
    .claude/*)                                             echo "" ;;   # hook/agent config
    runs/*|sweeps/*|outputs/*|data/*|plots/*)              echo "" ;;   # generated
    *.py|*.sh|config/*|recipes/*|models/*|sweep/*|tools/*|tests/*|scripts/*|IntrinsicDimDeep/*)
                                                           echo repo-reviewer ;;
    *)                                                     echo "" ;;
  esac
}

# Set a pillar's watermark to "as of right now, nothing is owed".
reset_pillar() {
  local who="$1" paths sha lines
  paths=$(pillar_field "$who" 4)
  sha=$(git rev-parse HEAD)
  lines=$(pillar_size "$who" "$sha" "$paths")     # the uncommitted remainder just reviewed
  write_state "$who" "$sha" "$lines" 0 "$(reviewed_hash "$sha" "$paths")"
}

# Backlog for a pillar -> echoes "lines commits"; empty state self-initializes to zero owed.
backlog_for() {
  local who="$1" paths st sha base_l base_c tl tc
  paths=$(pillar_field "$who" 4)
  st=$(read_state "$who")
  if [ -z "$st" ]; then reset_pillar "$who"; echo "0 0"; return; fi
  sha=$(echo "$st" | awk '{print $1}')
  base_l=$(echo "$st" | awk '{print $2+0}')
  base_c=$(echo "$st" | awk '{print $3+0}')
  # watermark commit gone (rebase/gc) -> re-baseline rather than diff against nothing
  git cat-file -e "$sha^{commit}" 2>/dev/null || { reset_pillar "$who"; echo "0 0"; return; }
  tl=$(pillar_size "$who" "$sha" "$paths")
  tc=$(pillar_commits "$sha" "$paths")
  # nothing owed while the pillar's content is what the reviewer passed, whatever commits sit in between (committing a
  # reviewed but uncommitted diff counted as a new commit and asked for a second review of identical content)
  if [ "$tl" -gt "$base_l" ] || [ "$tc" -gt "$base_c" ]; then
    local rh; rh=$(echo "$st" | awk '{print $4}')
    [ -n "$rh" ] && [ "$rh" = "$(reviewed_hash "$sha" "$paths")" ] && { echo "0 0"; return; }
  fi
  # subtract what was already reviewed at the watermark; clamp (a revert can go negative)
  echo "$(( tl > base_l ? tl - base_l : 0 )) $(( tc > base_c ? tc - base_c : 0 ))"
}

mode="${1:-status}"
case "$mode" in

  begin)
    who="${2:-}"
    [ -z "$who" ] && { echo "usage: review_backlog.sh begin <reviewer>" >&2; exit 2; }
    pillar_field "$who" 1 | grep -q . || { echo "unknown reviewer '$who'" >&2; exit 2; }
    mkdir -p "$STATE_ROOT/$STATE_REL"; : > "$(lock_file "$who")"
    echo "[review-backlog] $who holds the lock — edits to its pillar are allowed until 'advance'"
    # the backlog to read, as commands: reviewers otherwise cat the state files and read this script to find it
    read -r bl bc <<< "$(backlog_for "$who")"
    sha=$(read_state "$who" | awk '{print $1}')
    paths=$(pillar_field "$who" 4)
    q=""; for p in $paths; do q="$q '$p'"; done
    echo "watermark: $(git log -1 --format='%h %cs %s' "$sha" 2>/dev/null) (backlog: $bl lines, $bc commits)"
    echo "the backlog (committed and uncommitted):  git diff $sha --$q"
    echo "commits in it:                            git log --oneline $sha..HEAD --$q"
    echo "untracked files in it:                    git ls-files --others --exclude-standard --$q"
    exit 0
    ;;

  advance)
    who="${2:-}"
    [ -z "$who" ] && { echo "usage: review_backlog.sh advance <reviewer>" >&2; exit 2; }
    pillar_field "$who" 1 | grep -q . || { echo "unknown reviewer '$who'" >&2; exit 2; }
    reset_pillar "$who"
    rm -f "$(lock_file "$who")"
    echo "[review-backlog] $who watermark advanced to $(git rev-parse --short HEAD) — backlog cleared"
    exit 0
    ;;

  gate)
    input="$(cat 2>/dev/null || true)"
    fp=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception: print("")' 2>/dev/null)
    [ -z "$fp" ] && exit 0
    rel=${fp#"$REPO"/}
    case "$rel" in /*) exit 0 ;; esac          # outside the repo (scratchpad etc.)
    who=$(pillar_for_path "$rel")
    [ -z "$who" ] && exit 0
    lock_fresh "$who" && exit 0                # reviewer (or human) holds a fresh lock

    lt=$(pillar_field "$who" 2); ct=$(pillar_field "$who" 3)
    set -- $(backlog_for "$who")
    l="${1:-0}"; c="${2:-0}"
    { [ "$l" -ge "$lt" ] || [ "$c" -ge "$ct" ]; } || exit 0

    st=$(read_state "$who"); sha=$(echo "$st" | awk '{print $1}')
    short=$(git rev-parse --short "$sha" 2>/dev/null || echo "$sha")
    {
      echo "BLOCKED by review-backlog: '$rel' belongs to the $who pillar, which has ~$l unreviewed"
      echo "changed lines over $c commit(s) since $short — past its $lt-line / $ct-commit threshold."
      echo "Editing it further would pile more onto a backlog nobody has read."
      echo
      echo "Run the $who subagent on the WHOLE backlog now (\`git diff $short -- $(pillar_field "$who" 4)\`)."
      echo "It takes the lock with 'review_backlog.sh begin $who', applies its fixes, and on a pass"
      echo "clears the backlog with 'review_backlog.sh advance $who' — after which this edit succeeds."
      echo "Do not run 'advance' yourself on a reviewer's behalf."
    } >&2
    exit 2
    ;;

  # `gate` sees only the Edit/Write tools; a file rewritten by a shell command passed it unseen (2026-10-06/08,
  # most of docs/results.tex). bashpre (PreToolUse(Bash)) refuses a `git commit` while an overdue, unlocked pillar has
  # changes not yet committed, whatever wrote them.
  bashpre)
    input="$(cat 2>/dev/null || true)"
    cmd=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("command",""))
except Exception: print("")' 2>/dev/null)
    mkdir -p "$REPO/$STATE_REL"
    overdue=""
    for who in $(pillar_names); do
      lock_fresh "$who" && continue
      lt=$(pillar_field "$who" 2); ct=$(pillar_field "$who" 3)
      set -- $(backlog_for "$who")
      { [ "${1:-0}" -ge "$lt" ] || [ "${2:-0}" -ge "$ct" ]; } && overdue="$overdue $who"
    done
    deny() { echo "BLOCKED by review-backlog: $1" >&2
             echo "Run the $2 subagent on its whole backlog now; it takes the lock ('review_backlog.sh begin $2')," >&2
             echo "applies its fixes and, on a pass, runs 'review_backlog.sh advance $2' itself." >&2; }
    # the code only (cmdparse.py): a heredoc or a quoted string that names a git commit runs none
    scan=$(printf '%s' "$cmd" | python3 "$(dirname "${BASH_SOURCE[0]}")/cmdparse.py" scan 2>/dev/null) || scan="$cmd"
    if printf '%s' "${scan:-$cmd}" | grep -qE '(^|[;&|(]|\s)git(\s+-C\s+\S+)?\s+commit(\s|$)'; then
      for who in $overdue; do
        paths=$(pillar_field "$who" 4)
        if [ -n "$(git status --porcelain -- $paths 2>/dev/null)" ]; then
          deny "a commit while the $who pillar is overdue and has uncommitted changes ($(git status --porcelain -- $paths | wc -l) file(s))." "$who"
          exit 2
        fi
      done
    fi
    exit 0
    ;;

  init)
    for who in $(pillar_names); do reset_pillar "$who"; done
    echo "[review-backlog] all watermarks set to $(git rev-parse --short HEAD) — clean slate"
    exit 0
    ;;

  status)
    printf '%-18s %8s %8s   %s\n' REVIEWER LINES COMMITS 'STATE'
    for who in $(pillar_names); do
      lt=$(pillar_field "$who" 2); ct=$(pillar_field "$who" 3)
      set -- $(backlog_for "$who")
      l="${1:-0}"; c="${2:-0}"
      if [ "$l" -ge "$lt" ] || [ "$c" -ge "$ct" ]; then s="DUE (>= $lt lines or $ct commits)"; else s="ok"; fi
      if [ -f "$(lock_file "$who")" ]; then lock_fresh "$who" && s="$s, locked" || s="$s, STALE lock (no longer exempts)"; fi
      printf '%-18s %8s %8s   %s\n' "$who" "$l/$lt" "$c/$ct" "$s"
    done
    exit 0
    ;;

  # The Stop nudge (`check`) and the PostToolUse write check (`bashpost`) were removed (agent redesign B4: one Stop
  # gate, plan_check.py; an overdue pillar is held by `gate`, `bashpre` and the review before a merge). Both stay
  # no-ops until their registrations leave .claude/settings.json with switch-over B, so a registered call never
  # refuses.
  check|bashpost)
    cat >/dev/null 2>&1 || true
    exit 0
    ;;

  *) echo "usage: review_backlog.sh {gate|bashpre|begin <reviewer>|advance <reviewer>|status|init}" >&2; exit 2 ;;
esac
