#!/usr/bin/env python3
"""Run a command fully detached from the calling agent; when it exits, post a `site event` on the entry so the
waiter wakes and the entry's owner is resumed.

  tools/detach.py --entry <plan entry> [--name <label>] -- <command...>

Prints the log path and the PID at once. Log: ~/.local/share/ccorch/detached/<timestamp>_<label>.log.
Event: kind DONE (routing is by --item, any kind is routed), key detach:<label>:<pid>."""
import argparse, os, re, subprocess, time

D = os.path.expanduser("~/.local/share/ccorch/detached")

WRAP = r'''
log="$1"; label="$2"; entry="$3"; shift 3
"$@" </dev/null >>"$log" 2>&1
rc=$?
last=$(grep -v '^[[:space:]]*$' "$log" | tail -n 1 | cut -c1-200)
site event --project FA --kind DONE --item "$entry" --key "detach:$label:$$" \
  --msg "$label exited rc=$rc; log $log; last line: $last" >>"$log" 2>&1
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--name")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", a.name or os.path.basename(cmd[0]))
    os.makedirs(D, exist_ok=True)
    log = os.path.join(D, "%s_%s.log" % (time.strftime("%Y%m%d-%H%M%S"), label))
    p = subprocess.Popen(["bash", "-c", WRAP, "detach", log, label, a.entry] + cmd, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                         cwd=os.getcwd())
    print("log %s\npid %d" % (log, p.pid))


main()
