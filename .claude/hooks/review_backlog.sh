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
# WHERE THE TEETH ARE. A Stop hook can only ever NUDGE: the harness sets stop_hook_active on
# the next stop so a hook cannot block a turn twice, otherwise it would loop forever. So
# `check` is a reminder that can be walked past indefinitely by just ending the turn again.
# The actual gate is `gate`, a PreToolUse(Edit|Write) hook: while a pillar is over threshold,
# EDITS TO THAT PILLAR'S FILES ARE DENIED. Passing the turn then buys nothing, because the
# next edit to the overdue file is refused until its reviewer has run. Commits, and work on
# every other pillar, stay unblocked.
#
# Modes:
#   check                 (Stop hook, default) nudge once if any pillar is over threshold
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

FALLBACK_REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STATE_REL=".claude/.review_state"

# --- pillar table -----------------------------------------------------------
# name | line threshold | commit threshold | git pathspecs
#
# The code pathspec is SOURCE only. `*.py`/`*.sh`/`*.yaml`-style globs match at any depth in
# git, so they already cover analysis/, compare_models/, etc. without listing them; the
# :(exclude) terms drop the generated companions that share those trees (result JSONs, job
# logs, run/sweep output), which would otherwise swamp the line count with data churn.
PILLARS='claudemd-keeper|80|8|CLAUDE.md
notes-editor|80|8|docs/*.tex
repo-reviewer|200|12|*.py *.sh config recipes models sweep tools tests scripts IntrinsicDimDeep :(exclude)*.json :(exclude)*.log :(exclude)*.out :(exclude)*.err :(exclude)runs/* :(exclude)sweeps/*'

pillar_field() { printf '%s\n' "$PILLARS" | awk -F'|' -v n="$1" -v f="$2" '$1==n{print $f}'; }
pillar_names() { printf '%s\n' "$PILLARS" | awk -F'|' '{print $1}'; }

REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || REPO="$FALLBACK_REPO"
[ -z "$REPO" ] && REPO="$FALLBACK_REPO"
cd "$REPO" 2>/dev/null || exit 0
git rev-parse --verify -q HEAD >/dev/null 2>&1 || exit 0

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

pillar_commits() { git rev-list --count "$1"..HEAD -- $2 2>/dev/null || echo 0; }

# state file: "<sha> <baseline_lines> <baseline_commits>"
state_file() { echo "$REPO/$STATE_REL/$1"; }

read_state() {                       # $1=reviewer -> echoes "sha lines commits"
  local f; f=$(state_file "$1")
  if [ -f "$f" ]; then cat "$f"; else echo ""; fi
}

write_state() {                      # $1=reviewer $2=sha $3=lines $4=commits
  mkdir -p "$REPO/$STATE_REL"
  printf '%s %s %s\n' "$2" "$3" "$4" > "$(state_file "$1")"
}

lock_file() { echo "$REPO/$STATE_REL/$1.lock"; }

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
  lines=$(pillar_lines "$sha" "$paths")     # the uncommitted remainder just reviewed
  write_state "$who" "$sha" "$lines" 0
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
  tl=$(pillar_lines "$sha" "$paths")
  tc=$(pillar_commits "$sha" "$paths")
  # subtract what was already reviewed at the watermark; clamp (a revert can go negative)
  echo "$(( tl > base_l ? tl - base_l : 0 )) $(( tc > base_c ? tc - base_c : 0 ))"
}

mode="${1:-check}"
case "$mode" in

  begin)
    who="${2:-}"
    [ -z "$who" ] && { echo "usage: review_backlog.sh begin <reviewer>" >&2; exit 2; }
    pillar_field "$who" 1 | grep -q . || { echo "unknown reviewer '$who'" >&2; exit 2; }
    mkdir -p "$REPO/$STATE_REL"; : > "$(lock_file "$who")"
    echo "[review-backlog] $who holds the lock — edits to its pillar are allowed until 'advance'"
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
    [ -f "$(lock_file "$who")" ] && exit 0     # reviewer (or human) holds the lock

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
      printf '%-18s %8s %8s   %s\n' "$who" "$l/$lt" "$c/$ct" "$s"
    done
    exit 0
    ;;

  check)
    input="$(cat 2>/dev/null || true)"
    # Already nudged in this stop sequence -> let it through (no infinite loop; also the
    # deliberate-pause escape). The watermark is untouched, so it fires again next turn.
    case "$input" in *'"stop_hook_active"'*true*) exit 0 ;; esac

    br="$(git symbolic-ref --quiet --short HEAD 2>/dev/null)" || exit 0
    [ -z "$br" ] && exit 0
    [ "$br" = "main" ] && exit 0        # generated artifact, never authored here

    due=""
    for who in $(pillar_names); do
      lt=$(pillar_field "$who" 2); ct=$(pillar_field "$who" 3)
      set -- $(backlog_for "$who")
      l="${1:-0}"; c="${2:-0}"
      if [ "$l" -ge "$lt" ] || [ "$c" -ge "$ct" ]; then
        due="$due|$who:$l:$c"
      fi
    done
    [ -z "$due" ] && exit 0

    msg="BATCHED REVIEW DUE. Accumulated unreviewed changes have crossed the threshold for the"
    msg="$msg pillar(s) below. Run each subagent now; each reviews its ENTIRE backlog at once (not"
    msg="$msg just the last edit), and on a pass calls \`review_backlog.sh advance <name>\` itself to"
    msg="$msg clear it. Backlog and scope per reviewer:"
    IFS='|' read -ra items <<< "${due#|}"
    for it in "${items[@]}"; do
      [ -z "$it" ] && continue
      who="${it%%:*}"; rest="${it#*:}"; l="${rest%%:*}"; c="${rest##*:}"
      st=$(read_state "$who"); sha=$(echo "$st" | awk '{print $1}')
      short=$(git rev-parse --short "$sha" 2>/dev/null || echo "$sha")
      case "$who" in
        claudemd-keeper) what="CLAUDE.md stays an operating manual — no results, no session log, no bloat" ;;
        notes-editor)    what="docs/results.tex — author voice, figures earning their place, claims sound" ;;
        repo-reviewer)   what="code correctness, repo structure, committed artifacts" ;;
        *)               what="pillar review" ;;
      esac
      msg="$msg  * $who — ~$l changed lines over $c commit(s) since $short; $what. Its backlog:"
      msg="$msg \`git diff $short -- $(pillar_field "$who" 4)\`."
    done
    msg="$msg  RUN THEM NOW, in this turn, without asking the user first — a hook cannot spawn a"
    msg="$msg subagent, so this message IS the automation and you are the part that executes it."
    msg="$msg Ending the turn instead is not a pause: the PreToolUse gate will refuse your next edit"
    msg="$msg to these files until the reviewer has run. \`review_backlog.sh status\` lists all backlogs."

    esc=$(printf '%s' "$msg" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
    printf '{"decision":"block","reason":"%s"}\n' "$esc"
    exit 0
    ;;

  *) echo "usage: review_backlog.sh {check|advance <reviewer>|status|init}" >&2; exit 2 ;;
esac
