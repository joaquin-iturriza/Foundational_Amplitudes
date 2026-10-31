#!/usr/bin/env python3
"""Run a command fully detached from the calling agent; when it exits, post a `site event` on the entry so the
waiter wakes and the entry's owner is resumed.

  tools/detach.py --entry <plan entry> [--name <label>] -- <command...>
  tools/detach.py --entry <plan entry> --name <label> --each "<cmd 1>" --each "<cmd 2>" ...

--each: N shell commands started in parallel as one detached job; one DONE event when all have ended, its message
and the main log listing each command's rc (rc of the job = the largest); command i's output in <log>.<i>. Parallel
background batches go through it, so the owner (and the main session) is woken once per batch, not once per part.

Prints the log path and the PID at once. Log: ~/.local/share/ccorch/detached/<timestamp>_<label>.log.
Event: kind DONE (routing is by --item, any kind is routed), key detach:<label>:<pid>; DETACH_SITE replaces the
`site` command (tests)."""
import argparse, os, re, subprocess, time

D = os.path.expanduser("~/.local/share/ccorch/detached")

WRAP = r'''
log="$1"; label="$2"; entry="$3"; shift 3
"$@" </dev/null >>"$log" 2>&1
rc=$?
last=$(grep -v '^[[:space:]]*$' "$log" | tail -n 1 | cut -c1-200)
"${DETACH_SITE:-site}" event --project FA --kind DONE --item "$entry" --key "detach:$label:$$" \
  --msg "$label exited rc=$rc; log $log; last line: $last" >>"$log" 2>&1
'''

WRAP_EACH = r'''
log="$1"; label="$2"; entry="$3"; shift 3
i=0; pids=()
for c in "$@"; do i=$((i+1)); echo "[$i] $c" >>"$log"; bash -c "$c" </dev/null >"$log.$i" 2>&1 & pids+=($!); done
rcs=(); max=0; i=0
for p in "${pids[@]}"; do i=$((i+1)); wait "$p"; r=$?; rcs+=("$i:rc=$r"); [ "$r" -gt "$max" ] && max=$r
  echo "[$i] rc=$r; last line: $(grep -v '^[[:space:]]*$' "$log.$i" | tail -n 1 | cut -c1-160)" >>"$log"; done
"${DETACH_SITE:-site}" event --project FA --kind DONE --item "$entry" --key "detach:$label:$$" \
  --msg "$label: $i commands ended, rc=$max (${rcs[*]}); log $log (outputs $log.<i>)" >>"$log" 2>&1
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--name")
    ap.add_argument("--each", action="append", default=[], help="a shell command of a parallel batch (repeat)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if a.each and cmd:
        ap.error("--each or a command after --, not both")
    if not cmd and not a.each:
        ap.error("no command")
    if a.each and not a.name:
        ap.error("--each needs --name")
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", a.name or os.path.basename(cmd[0]))
    os.makedirs(D, exist_ok=True)
    log = os.path.join(D, "%s_%s.log" % (time.strftime("%Y%m%d-%H%M%S"), label))
    p = subprocess.Popen(["bash", "-c", WRAP_EACH if a.each else WRAP, "detach", log, label, a.entry] + (a.each or cmd),
                         stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                         cwd=os.getcwd())
    print("log %s\npid %d" % (log, p.pid))


main()
