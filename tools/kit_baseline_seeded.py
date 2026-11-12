#!/usr/bin/env python3
"""Bakar's numpy src/baseline.py (KIT benchmark), unchanged except its one RNG seed (np.random.default_rng(0): the
init and the batch order), to measure his MLP's seed-to-seed spread for the D403 port gate.

    python tools/kit_baseline_seeded.py <seed> <baseline.py> <its args...>
"""
import sys

seed, src = int(sys.argv[1]), sys.argv[2]
code = open(src).read()
assert code.count("np.random.default_rng(0)") == 1, "baseline.py changed: its seed line is not the one expected"
sys.argv = [src] + sys.argv[3:]
exec(compile(code.replace("np.random.default_rng(0)", f"np.random.default_rng({seed})"), src, "exec"),
     {"__name__": "__main__"})
