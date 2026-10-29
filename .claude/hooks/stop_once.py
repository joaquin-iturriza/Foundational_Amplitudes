#!/usr/bin/env python3
"""Runs one Stop hook so that it can never loop: when the turn is already continuing because a Stop hook refused
its end (Claude Code's stop_hook_active), every wrapped hook passes. Each hook can refuse a turn's end at most once
in a row, whatever its condition; a condition the turn did not clear goes to the user in that one reply, not again.
Usage in settings: python3 stop_once.py '<the hook's command>'  (2026-10-29: three Stop hooks refused some
twenty times in a row on conditions no turn could clear.)"""
import json, subprocess, sys
raw = sys.stdin.read()
try:
    active = bool(json.loads(raw or "{}").get("stop_hook_active"))
except Exception:
    active = False
if active:
    sys.exit(0)
r = subprocess.run(sys.argv[1], shell=True, input=raw, text=True, capture_output=True)
sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
sys.exit(r.returncode)
