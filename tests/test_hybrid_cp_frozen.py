"""data.target_physics_hybrid's constants c_p under frozen data stats (physics_scales.hybrid_cps, called from
experiment._init_data_recipes): a process whose c_p the frozen stats carry (a fine-tune's parent trained on it, a
resumed run) keeps it; any other is fitted on its own momenta, as without frozen stats. CPU only, numpy, no torch.

Run as a script (pytest is shadowed by the stray root-level py.py):
    python tests/test_hybrid_cp_frozen.py
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from physics_scales import hybrid_cp, hybrid_cps, qperp_logs  # noqa: E402


def _events(n_final, n=500, seed=0, sqrts=500.0):
    """(n, 2 + n_final, 4) momenta in GeV: two incoming beams along z and massless final particles in random
    directions with random energies (no momentum conservation needed: c_p only reads invariants)."""
    rng = np.random.default_rng(seed)
    E = sqrts / 2
    P = np.zeros((n, 2 + n_final, 4))
    P[:, 0] = [E, 0, 0, E]
    P[:, 1] = [E, 0, 0, -E]
    for k in range(n_final):
        e = rng.uniform(0.05, 0.5, n) * sqrts
        c = rng.uniform(-0.95, 0.95, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        s = np.sqrt(1 - c ** 2)
        P[:, 2 + k] = np.stack([e, e * s * np.cos(phi), e * s * np.sin(phi), e * c], 1)
    return P


def test_cp_is_the_mean_of_lq_minus_the_shat_power():
    P = _events(3)
    shat = (2 * P[0, 0, 0]) ** 2
    assert abs(hybrid_cp(P) - float(np.mean(qperp_logs(P) - 1 * np.log(shat)))) < 1e-12
    assert hybrid_cp(_events(2)) == 0.0                # 2 -> 2: the s_hat and emission-scale targets agree


def test_no_frozen_stats_fits_every_process():
    ev = {"a": _events(3, seed=1), "b": _events(4, seed=2)}
    cps, reused = hybrid_cps(["a", "b"], ev.__getitem__, frozen=None)
    assert reused == []
    assert cps == {"a": hybrid_cp(ev["a"]), "b": hybrid_cp(ev["b"])}


def test_frozen_cp_is_reused_and_events_not_read():
    """A fine-tune onto a process its parent trained on: the parent's c_p, through a data_stats.json round trip,
    whatever this run's own (smaller) train split would give."""
    parent = {"a": hybrid_cp(_events(3, n=5000, seed=3))}
    stats = json.loads(json.dumps({"target_physics_hybrid_cp": parent}))   # as _save_data_stats writes it
    own_small = _events(3, n=50, seed=4)
    assert abs(hybrid_cp(own_small) - parent["a"]) > 1e-6              # refitting would move the constant
    read = []
    cps, reused = hybrid_cps(["a"], lambda n: read.append(n) or own_small,
                             frozen=stats.get("target_physics_hybrid_cp"))
    assert cps["a"] == parent["a"] and reused == ["a"] and read == []


def test_process_not_in_frozen_stats_falls_back_to_its_own_events():
    """A fine-tune or a zero-shot read onto a process the parent never trained on: computed as without stats."""
    frozen = {"a": -4.2}
    ev = {"a": _events(3, seed=5), "c": _events(5, seed=6)}
    read = []
    cps, reused = hybrid_cps(["a", "c"], lambda n: read.append(n) or ev[n], frozen=frozen)
    assert cps["a"] == -4.2 and reused == ["a"]
    assert cps["c"] == hybrid_cp(ev["c"]) and read == ["c"]


def test_stats_written_without_the_key_fit_every_process():
    """data_stats.json of a run that predates the key, or of a non-hybrid parent (the hybrid flag itself is asserted
    equal before): stats.get gives None, every process is fitted."""
    ev = {"a": _events(3, seed=7)}
    cps, reused = hybrid_cps(["a"], ev.__getitem__, frozen={}.get("target_physics_hybrid_cp"))
    assert reused == [] and cps["a"] == hybrid_cp(ev["a"])


def test_experiment_wires_the_frozen_cp():
    """experiment.py cannot be imported without torch: check at source level that the frozen-stats path reads the key
    and the c_p fit passes it to hybrid_cps, and that _save_data_stats writes it."""
    src = open(os.path.join(ROOT, "experiment.py")).read()
    assert re.search(r'_cp_frozen\s*=\s*stats\.get\("target_physics_hybrid_cp"\)', src)
    assert re.search(r"hybrid_cps\(names,\s*_cp_events,\s*frozen=_cp_frozen\)", src)
    assert '"target_physics_hybrid_cp": getattr(self, "_phys_cp", None)' in src


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
