"""Where the hybrid's level error comes from (the preprocessing exploration on catalog_v3, D83/D129): per probe, from the
zero-shot reads of the hybrid (s_hat target plus the per-process constant c_p) and of the emission scales on the same
2000 validation events (data/short_reads/<arm>/<axis>/<probe>.npz; t, q, logF in target units, t - logF = log|M|^2,
the same events in every arm). The hybrid's mean error mean(q_h - t_h) splits exactly into
  net     mean(q_h) - mean(t_q)          the network's error on the class mean the hybrid is built to have, the emission
                                         scales' class mean on these events (t_q: the emission-scale arm's target)
  const   mean(t_q) - mean(t_h)          = c_p on these events - the c_p the run handed the probe (fitted on its own
                                         first 10000 train events); logF_h - logF_shat is that constant, read per probe
and the emission scales' own mean error mean(q_q - t_q) is the network-only reference: the same class mean, predicted
by another network. Squared, as the level in short_reads.py. Per axis: medians over the probes of the hybrid's level,
net^2, const^2 and the emission scales' level, the median per-probe ratio net^2 / level_q, and the share of probes
with const^2 > net^2.
    python3 analysis/catalog_v3/prep_hybrid_split.py   -> data/prep_hybrid_split.json, one flat JSON line last
"""
import glob, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AXES = ("H1", "H2", "H2p5", "H3", "H6", "H7E", "H7c")     # H5 (signed one-loop pools) apart, as in prep_hybrid.py
D = os.path.join(HERE, "data", "short_reads")


def probe(ax, name):
    h, q, s = (np.load(os.path.join(D, a, ax, name)) for a in ("dimensionless_hybrid", "dimensionless",
                                                                 "dimensionless_shat"))
    if not all(len(z["q"]) and len(z["logF"]) for z in (h, q, s)):
        return None
    th, tq = h["t"].astype(float), q["t"].astype(float)
    assert np.allclose(th - h["logF"], tq - q["logF"], atol=1e-3), name    # the same events, the same log|M|^2
    cp = h["logF"].astype(float) - s["logF"].astype(float)
    net = float(np.mean(h["q"]) - np.mean(tq))
    const = float(np.mean(tq) - np.mean(th))
    return {"level_h": float(np.mean(h["q"] - th) ** 2), "net2": net ** 2, "const2": const ** 2, "net": net,
            "const": const, "level_q": float(np.mean(q["q"] - tq) ** 2), "cp_handed": float(np.mean(cp)),
            "cp_spread": float(np.std(cp))}


def main():
    out, flat = {}, {}
    for ax in AXES:
        names = sorted(set(map(os.path.basename, glob.glob(os.path.join(D, "dimensionless_hybrid", ax, "*.npz"))))
                       & set(map(os.path.basename, glob.glob(os.path.join(D, "dimensionless", ax, "*.npz")))))
        rows = {n[:-4]: r for n in names if (r := probe(ax, n)) is not None}
        v = {k: np.array([r[k] for r in rows.values()]) for k in ("level_h", "net2", "const2", "level_q")}
        o = {"n": len(rows), **{f"med_{k}": float(np.median(a)) for k, a in v.items()},
             "med_net2_over_level_q": float(np.median(v["net2"] / v["level_q"])),
             "med_level_h_over_level_q": float(np.median(v["level_h"] / v["level_q"])),
             "frac_const_dominant": float(np.mean(v["const2"] > v["net2"])),
             "max_cp_spread": max(r["cp_spread"] for r in rows.values()), "probes": rows}
        out[ax] = o
        print(f"{ax:5s} n={o['n']:3d} level_h {o['med_level_h']:.3g} net^2 {o['med_net2']:.3g} "
              f"const^2 {o['med_const2']:.3g} level_q {o['med_level_q']:.3g}  per probe level_h/level_q "
              f"{o['med_level_h_over_level_q']:.2f} net^2/level_q {o['med_net2_over_level_q']:.2f}  const dominant "
              f"{o['frac_const_dominant']:.2f}")
        flat.update({f"{ax}_{k}": float(f"{o[k]:.3g}") for k in ("med_level_h", "med_net2", "med_const2", "med_level_q",
                                                                 "med_net2_over_level_q", "med_level_h_over_level_q",
                                                                 "frac_const_dominant")})
        flat[f"{ax}_n"] = o["n"]
    json.dump(out, open(os.path.join(HERE, "data", "prep_hybrid_split.json"), "w"), indent=1)
    print(json.dumps(flat))


if __name__ == "__main__":
    main()
