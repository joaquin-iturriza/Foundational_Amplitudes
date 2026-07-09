#!/usr/bin/env bash
# PreToolUse(Write) hook — block creation of NEW scattered notes/report docs.
#
# Why: CLAUDE.md ground rule #3 says all guidance/results live in ONE document —
# no scattered .md / .tex / findings / report / summary dumps anywhere in the
# tree. The model (me) violated this by rationalizing "it's a report, not
# guidance", and by abusing a blanket notes/ exemption to drop a new .md there.
# This makes the rule non-negotiable at the moment of creation instead of
# relying on the model to not invent exceptions.
#
# Scope: Write (file creation) of a NEW doc file (.md/.markdown/.tex/.rst) that
# does NOT already exist. Editing/overwriting an existing file is always fine
# (that's not "scattering a new file"). Exemptions: CLAUDE.md, README*, and the
# harness plan-mode dir (*/.claude/plans/*). Nothing else — not notes/, not the
# rest of .claude/, not the scratchpad.
#
# Escape hatch (deliberate + recorded): to create an approved new doc, add its
# repo-relative or absolute path to  .claude/md_allowlist.txt  (one per line).
# That makes "the user approved this doc" an explicit, auditable act rather than
# an in-the-moment rationalization.
set -uo pipefail
REPO="/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
ALLOWLIST="$REPO/.claude/md_allowlist.txt"

input=$(cat)
fp=$(printf '%s' "$input" | python3 -c 'import sys,json;
try:
    print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception:
    print("")' 2>/dev/null)
[ -z "$fp" ] && exit 0

# Only guard note/report/document formats.
case "$fp" in
  *.md|*.markdown|*.tex|*.rst) ;;
  *) exit 0 ;;
esac

# Exemptions: CLAUDE.md and README* only, plus the harness plan-mode dir
# (~/.claude/plans/*.md are plan-mode files outside the repo, not scattered repo docs).
case "$fp" in
  */CLAUDE.md|*/README.md|*/README*.md|*/.claude/plans/*) exit 0 ;;
esac

# Editing an existing file is fine — only NEW files are "scattering".
[ -e "$fp" ] && exit 0

# User-approved new docs: allowlist match (exact path, repo-relative or absolute).
rel=${fp#"$REPO"/}
if [ -f "$ALLOWLIST" ]; then
  while IFS= read -r line; do
    line=$(printf '%s' "$line" | sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//')
    [ -z "$line" ] && continue
    if [ "$line" = "$fp" ] || [ "$line" = "$rel" ]; then exit 0; fi
  done < "$ALLOWLIST"
fi

# Block. exit 2 => tool call denied, stderr shown to the model.
{
  echo "BLOCKED by md_guard: refusing to create new doc file '$rel'."
  echo "CLAUDE.md ground rule #3: no scattered .md/.tex/notes/findings/report files anywhere (notes/ included) — results/guidance go in the ONE existing document (docs/results.tex), not a new file. 'It's a report/analysis, not guidance' is NOT an exception, and there is no notes/ loophole."
  echo "Write a section into the existing doc instead. Only if the user has EXPLICITLY approved a brand-new file: record its path in .claude/md_allowlist.txt, then retry. Otherwise ask first."
} >&2
exit 2
