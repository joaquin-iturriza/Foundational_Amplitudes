#!/usr/bin/env bash
# PreToolUse(Bash|Write) hook — refuse to configure a RUN with plotting disabled.
#
# Why: CLAUDE.md, Conventions: "Always keep `plot: true` in sweep/experiment configs — I always
# want plots." The model (me) shipped a plotting-disable flag in SEVEN scripts and both sweep
# configs — copy-pasted from an old hand-off script, never once checking what it gated. It gated
# mse.pdf, the mu-MSE-vs-step curve: the single plot showing whether mu is descending or
# plateaued. The live investigation was *"why won't mu fit?"*. So the exact diagnostic needed was
# switched off, and ~20 GPU-hours plus six wrong hypotheses went into reconstructing it by hand.
#
# The failure mode is NOT "chose to disable plots" — it is "carried a flag forward without ever
# asking what it does". Only a hook catches that, because by construction I'm not thinking about it.
#
# Fires when a real RUN is configured with plotting off:
#   - Bash invoking run.py with such an override
#   - Bash `sbatch <script>` where <script> carries such an override
#   - Write of a .sh/.yaml/.yml whose content carries such an override
# Deliberately does NOT fire on grep/sed/rg/cat/git — searching for or REMOVING the flag
# (i.e. cleaning it up) must stay possible. Also never polices .claude/hooks/ (its own source).
#
# Escape hatch: .claude/plot_disable_allowlist.txt (one substring/path per line).
set -uo pipefail
REPO="/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
ALLOWLIST="$REPO/.claude/plot_disable_allowlist.txt"

# Pattern for "plotting turned off", assembled so this file never contains a literal match.
K1='plot'
OFF_RE="(^|[[:space:]])${K1}=(false|False)|(^|[[:space:]])${K1}:[[:space:]]*'?(false|False)|${K1}ting\.[a-zA-Z_]+=(false|False)|${K1}ting\.[a-zA-Z_]+:[[:space:]]*'?(false|False)"

input=$(cat)
tool=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_name",""))
except Exception: print("")' 2>/dev/null)

allowlisted() {
  [ -f "$ALLOWLIST" ] || return 1
  while IFS= read -r line; do
    line=$(printf '%s' "$line" | sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//')
    [ -z "$line" ] && continue
    printf '%s' "$1" | grep -qF -- "$line" && return 0
  done < "$ALLOWLIST"
  return 1
}

emit() {
  {
    echo "BLOCKED by plot_guard: this configures a RUN with PLOTTING OFF ($1)."
    echo "Offending:"
    printf '%s\n' "$2" | sed 's/^/    /'
    echo ""
    echo "CLAUDE.md (Conventions): 'Always keep plot: true in sweep/experiment configs.'"
    echo ""
    echo "ASK WHAT THE FLAG GATES before disabling it. Precedent: a plotting-disable flag was"
    echo "copy-pasted through 7 scripts + 2 sweep configs with nobody checking; it gated mse.pdf,"
    echo "the mu-MSE-vs-step curve — the exact diagnostic the then-live investigation needed."
    echo "~20 GPU-hours went into reconstructing by hand what that plot gave for free."
    echo ""
    echo "Never carry a plotting flag forward from an old script just because it was there."
    echo "If the user EXPLICITLY approved it: add a matching line to .claude/plot_disable_allowlist.txt."
  } >&2
  exit 2
}

if [ "$tool" = "Write" ]; then
  fp=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception: print("")' 2>/dev/null)
  case "$fp" in */.claude/hooks/*) exit 0 ;; esac
  case "$fp" in *.sh|*.yaml|*.yml) ;; *) exit 0 ;; esac
  content=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("content",""))
except Exception: print("")' 2>/dev/null)
  allowlisted "$fp" && exit 0
  hit=$(printf '%s' "$content" | grep -nE "$OFF_RE" | head -3 || true)
  [ -n "$hit" ] && emit "Write $fp" "$hit"
  exit 0
fi

[ "$tool" = "Bash" ] || exit 0
cmd=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("command",""))
except Exception: print("")' 2>/dev/null)
[ -z "$cmd" ] && exit 0
allowlisted "$cmd" && exit 0

if printf '%s' "$cmd" | grep -q "run\.py"; then
  hit=$(printf '%s' "$cmd" | grep -oE "$OFF_RE" | head -3 || true)
  [ -n "$hit" ] && emit "run.py invocation" "$hit"
fi

if printf '%s' "$cmd" | grep -qE '(^|[^[:alnum:]_])sbatch([^[:alnum:]_]|$)'; then
  for s in $(printf '%s' "$cmd" | tr ' \t\n' '\n\n\n' | grep -E '\.sh$' || true); do
    path="$s"; [ -f "$path" ] || path="$REPO/$s"
    [ -f "$path" ] || continue
    allowlisted "$path" && continue
    hit=$(grep -nE "$OFF_RE" "$path" | head -3 || true)
    [ -n "$hit" ] && emit "sbatch $s" "$hit"
  done
fi

exit 0
