#!/usr/bin/env bash
# run_arm.sh <task> <label>: one headless replay run in worktrees/zz_ab2_<task>_<label>
set -u
S=/tmp/claude-1000/-home-joaquin-work-FA/71602535-0955-42ad-a58f-23dc320798dc/scratchpad/ab2
R=/home/joaquin/work/FA
t=$1; l=$2; W=$R/worktrees/zz_ab2_${t}_$l
read on base dirs < <(python3 -c "import json;c=json.load(open('$S/tasks.json'))['$t'];print(str(c['arms']['$l']['ponytail']).lower(),c['base'],' '.join(c['dirs']))")
git -C $R worktree add --detach $W $base >/dev/null 2>&1 || { echo "$t $l worktree failed" >> $S/done.txt; exit 1; }
python3 $S/stage.py $W $base $dirs >> $S/stage.log
ln -s $R/.venv-driver $W/.venv-driver
extra=""
if [ $t = v4pp ]; then mkdir -p $W/runs; sed "s#/home/joaquin/work/FA#$W#g" $S/step_v4pp.txt > $W/runs/replay_step_prompt.txt; fi
if [ $t = ftharm ]; then mkdir -p $W/runs/zz_lowd_curves; cp -a ~/.local/share/ccorch/artifacts/FA/zz_lowd_curves/first_round $W/runs/zz_lowd_curves/; extra=" The first round's curves were already copied home from CC zz_lowd before this replay, to $W/runs/zz_lowd_curves/first_round."; fi
prompt=$(sed "s#/home/joaquin/work/FA#$W#g" $S/prompt_$t.txt)
prompt="$prompt

(Replay note, the same in every run: you work in a throwaway git worktree, $W, checked out at the commit before this work was first done. Work and write only inside it; paths in this prompt already point into it. The site tool, ssh and the clusters are not available in this replay: skip any step that needs them and say so. The gitignored data files under analysis/ that existed when this work was first done are copied into the worktree at their usual paths, and .venv-driver is linked in.$extra Commit in this worktree as asked (detached HEAD), never push. No subagents are available: do the work yourself.)"
cd $W
timeout 3600 claude -p "$prompt" --max-turns 60 --model claude-opus-5-5 --effort high \
  --output-format stream-json --verbose \
  --settings "{\"enabledPlugins\":{\"ponytail@ponytail\":$on}}" \
  --disallowedTools Agent WebFetch WebSearch "Bash(site:*)" "Bash(ssh:*)" "Bash(scp:*)" "Bash(rsync:*)" "Bash(git push:*)" \
  > $S/${t}_$l.stream.jsonl 2> $S/${t}_$l.stderr
echo "$t $l exit $?" >> $S/done.txt
