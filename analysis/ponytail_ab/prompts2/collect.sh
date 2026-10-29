#!/usr/bin/env bash
# collect.sh <task> <label>: the run's whole change against the base, into analysis/ponytail_ab/diffs2/
S=/tmp/claude-1000/-home-joaquin-work-FA/71602535-0955-42ad-a58f-23dc320798dc/scratchpad/ab2
R=/home/joaquin/work/FA; t=$1; l=$2; W=$R/worktrees/zz_ab2_${t}_$l
base=$(python3 -c "import json;print(json.load(open('$S/tasks.json'))['$t']['base'])")
mkdir -p $R/analysis/ponytail_ab/diffs2
git -C $W add -A -N -- . ':(exclude).venv-driver' 2>/dev/null
git -C $W diff $base -- . ':(exclude).venv-driver' > $R/analysis/ponytail_ab/diffs2/${t}_$l.diff
echo "$t $l $(wc -l < $R/analysis/ponytail_ab/diffs2/${t}_$l.diff) diff lines, commits: $(git -C $W rev-list --count $base..HEAD)"
