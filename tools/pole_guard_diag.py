"""Diagnostic: why a one-loop entry's generation is refused by the per-event pole guard while its certification passes.
Draws --n events exactly as generate_virt_dataset does (the entry's cuts, sqrt(s) measure and window) and evaluates
each once, as generated (card width) or, with --zero-width, with the top width zeroed as the certification does. Per
log sqrt(s) bin: the fraction of exceptional points (MadLoop rc hundreds digit 4), of points whose double pole is off by
more than the entry's c2_rel_tol, and the median and max relative deviation. CPU only; it writes no pool, but like
generation it edits the standalone's Cards/param_card.dat in place (beam masses; with --zero-width DECAY 6 -> 0 while it
runs, restored on exit), so never run it while that entry is being generated or certified. Plain QCD virtuals only
(no loop-induced or electroweak entry, no catalog_v3 massive-beam balancing).
    python tools/pole_guard_diag.py v4_bbx_hwpwm --sqrts-min 286 --sqrts-max 6000 [--n 1000] [--zero-width]"""
import argparse, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import nlo_virtual_pipeline as V
import mg5_pipeline_final as mg
import nlo_pole_check as PC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("process")
    ap.add_argument("--sqrts-min", type=float, required=True)
    ap.add_argument("--sqrts-max", type=float, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--zero-width", action="store_true")
    a = ap.parse_args()
    cfg = V.VIRT_PROCESSES[a.process]
    assert not (cfg.get("loopind") or cfg.get("ew")), f"{a.process}: loop-induced or electroweak, not a plain QCD virtual"
    assert not (cfg.get("_v3") and any(V.beam_masses(cfg).get(abs(int(p)), 0.0) > 0 for p in cfg["pdg_ids"][:2])), \
        f"{a.process}: catalog_v3 massive beams (generation balances the last leg; not reproduced here)"
    pdg, m_finals = list(cfg["pdg_ids"]), list(cfg["m_finals"])
    cuts = (cfg.get("fiducial_cuts") or mg.FIDUCIAL_CUTS) if mg.FIDUCIAL_CUTS_ENABLED else None
    measure, bm = cfg.get("sqrts_measure", "uniform"), V.beam_masses(cfg)
    rng = np.random.default_rng(a.seed)
    events, sqrts = mg.sample_nbody_phase_space(a.n, a.sqrts_min, a.sqrts_max, m_finals, pdg, rng=rng, cuts=cuts,
                                                sqrts_measure=measure, beam_mass=cfg.get("beam_masses"))
    sa = V.virt_standalone_dir(a.process)
    card = os.path.join(sa, "Cards", "param_card.dat")
    V.restore_stale_card(card)
    V.massive_beam_params(sa, pdg, bm)
    p0 = V.find_p0(sa)
    c2_pred = PC.double_pole(pdg, [bm.get(abs(int(p)), 0.0) for p in pdg[:2]] + m_finals)
    tol = float(cfg.get("c2_rel_tol", 1e-3))
    import contextlib
    ctx = V._zero_top_width(card) if a.zero_width else contextlib.nullcontext()
    import nlo_madloop as ML
    rows = []
    with ctx:
        get_me = ML.load(p0)
        swap = mg.row_to_slot_perm(pdg, cfg["mg5"])
        for i, (mom, _) in enumerate(events):
            r = ML.evaluate(get_me, mom[swap], alphas=0.118)
            if r.get("c0") is None:
                rows.append((sqrts[i], 400, np.inf)); continue      # generation refuses it: counted as exceptional
            dev = abs(r["c2"] - c2_pred) / max(abs(c2_pred), 1.0)
            rows.append((sqrts[i], r["rc"], dev))
    s = np.array([x[0] for x in rows]); rc = np.array([x[1] for x in rows]); dev = np.array([x[2] for x in rows])
    exc = rc // 100 == 4
    bad = exc | (dev > tol)
    print(f"DIAG {a.process} width={'zero' if a.zero_width else 'card'} n={len(s)} tol={tol} c2_pred={c2_pred:.5f} "
          f"redrawn={bad.mean():.3f} exceptional={exc.mean():.3f} off_pole={(dev > tol).mean():.3f}")
    edges = np.geomspace(a.sqrts_min, a.sqrts_max, 9)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum():
            print(f"DIAG_BIN {lo:7.0f}-{hi:7.0f} GeV n={m.sum():4d} exceptional={exc[m].mean():.3f} "
                  f"off_pole={(dev[m] > tol).mean():.3f} median_dev={np.median(dev[m]):.2e} max_dev={dev[m].max():.2e} "
                  f"rcs={sorted(set(int(x) for x in rc[m]))[:8]}")


if __name__ == "__main__":
    main()
