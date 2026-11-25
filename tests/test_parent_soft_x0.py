#!/usr/bin/env python
"""job_datasaving_parent.sh's finale applies the D392 momentum rescale only below x0, neutrinos exempt (D561): its FIX
carries ++model.input_soft_x0 with ++model.input_soft_skip_neutrinos=true, and x0 lies below every non-neutrino final-state
momentum of every finale pool and of the Z+4g and u u~ g pools (analysis/selective_rescale/data/support_minima.json,
measured below 0.009 by PDG id), so no trained, validation or test momentum changes; flags=off still zeroes input_soft_eps.
Run: python tests/test_parent_soft_x0.py"""
import json, os, re, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SH = os.path.join(ROOT, "scripts", "job_datasaving_parent.sh")


def fix(*args):
    """The FIX array the script builds for the finale with `args` (its case and argument loop, run alone in bash)."""
    lines = open(SH).read().splitlines()
    a = next(i for i, l in enumerate(lines) if l.startswith("SUB=10000; FIX=()"))
    b = next(i for i, l in enumerate(lines) if l.startswith("TAG=; ARGS=()"))
    body = "\n".join(lines[a:b + 1])
    out = subprocess.run(["bash", "-c", f'PROJECT_DIR=.; WHICH=finale; set -- "$@"; {body}\nprintf "%s\\n" "${{FIX[@]}}"',
                          "x", *args], capture_output=True, text=True, check=True).stdout
    return out.split()


def main():
    on = fix()
    x0 = [float(x.split("=")[1]) for x in on if x.startswith("++model.input_soft_x0=")]
    assert len(x0) == 1, on
    assert "++model.input_soft_skip_neutrinos=true" in on, on
    m = json.load(open(os.path.join(ROOT, "analysis", "selective_rescale", "data", "support_minima.json")))
    assert len(m["finale"]) == 84 and x0[0] <= 0.009, x0   # the minima record momenta below 0.009 by PDG id
    pools = [*m["finale"].values(), *m["zng"].values(), *m["uug"].values()]
    # every final-state momentum below 0.009 is a neutrino, or the pool's softest x is above x0
    bad = {r["path"]: r["below_0.009_by_pid"] for r in pools
           if set(r["below_0.009_by_pid"]) - {"12", "-12", "14", "-14", "16", "-16"} or
           (not r["below_0.009_by_pid"] and r["min"] < x0[0])}
    assert not bad, bad
    floor = min(r["min"] for r in pools if not r["below_0.009_by_pid"])
    assert "++model.input_soft_eps=1e-8" in on
    off = fix("flags=off", "tag=t")
    assert "++model.input_soft_eps=0.0" in off and "++model.input_pair_logs=0" in off, off
    print("ok", x0[0], floor)


def test_parent_soft_x0():
    main()


if __name__ == "__main__":
    main()
