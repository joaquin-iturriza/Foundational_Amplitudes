#!/usr/bin/env python3
"""Default against catalog sampling on one catalog_v4 tree class, for a deck slide (plan catalog_coverage).

The class is u u~ > e+ e- g (v4_uux_epemg, Drell-Yan plus a jet), train role, seed 42, recipe catalog_v4_train.yaml:
  catalog    its catalog_v4 pool (the entry without the default share): RAMBO bulk, 0.15 forward (rotate_forward),
             0.35 flat in log|M|^2 (6x oversampled candidates, half splitter half RAMBO, adaptive sqrt(s_hat));
             each row's component from mix_compose.provenance (the bulk and forward draws replayed and matched to
             the pool's rows by their momenta; the rest is the flat part)
  default    every row of the class's default-sampled MadEvent pools (pp at 13 TeV: pp13v4_uux_epemg reused plus
             dsh_uux_epemg) in the partonic frame inside the pool's sqrt(s_hat) window and cuts
             (mix_compose._default_accepted), i.e. the generator's own sampling with pp beams
  trained    the default-share 0.25 mixture as the loader composes it (mix_compose.compose): what training reads
The flat part's splitter and RAMBO candidates are not told apart: the pool keeps no tag and the flat selection draws
from both after labelling, so only bulk / forward / flat are split.

    site run --item catalog_coverage ccin2p3 FA -- python analysis/catalog_v4/sampling_example.py hist
    python analysis/catalog_v4/sampling_example.py figs     # local, from analysis/catalog_v4/data/sampling_example.json
"""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
OUT_JSON = os.path.join(ROOT, "analysis", "catalog_v4", "data", "sampling_example.json")
FIGS = os.path.join(ROOT, "analysis", "catalog_v4", "figs", "sampling_example")
RECIPE = os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")
NAME, ROLE, SEED = "v4_uux_epemg", "train", 42
LOGM_BINS = np.linspace(-14.0, 4.0, 73)              # log10 |M|^2, 0.25 per bin; under/overflow counted apart
LOGM_XLIM = (-9.5, 1.0)                              # the figures' range: every event of the class lies inside
SQ_BINS = np.logspace(np.log10(30.0), np.log10(6000.0), 61)   # sqrt(s_hat) in GeV, the pool's window


def _sqrts(P):
    q = P[:, 0] + P[:, 1]
    return np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0, None))


def _h(me, sq):
    with np.errstate(divide="ignore"):
        lg = np.log10(np.abs(me))
    fin = np.isfinite(lg)
    return {"n": int(len(me)), "n_nonfinite": int((~fin).sum()),
            "under": int((lg[fin] < LOGM_BINS[0]).sum()), "over": int((lg[fin] > LOGM_BINS[-1]).sum()),
            "logm": np.histogram(lg[fin], LOGM_BINS)[0].tolist(), "sqrts": np.histogram(sq, SQ_BINS)[0].tolist(),
            "logm_mean": float(lg[fin].mean()), "logm_min": float(lg[fin].min()), "logm_max": float(lg[fin].max()),
            "sqrts_median": float(np.median(sq))}


def hist():
    import yaml
    import datagen
    import mg5_pipeline_final as mg
    import mix_compose as mc
    spec = yaml.safe_load(open(RECIPE))
    mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
                                   "sampling": p.get("sampling")} for p in spec["processes"]],
                                 default_sampling=spec.get("sampling"))
    p = {q["name"]: q for q in spec["processes"]}[NAME]
    lo, hi = map(float, p["sqrts"]); n = int(p["n_" + ROLE])
    cfg = mg.PROCESSES[NAME]; k = int(cfg["nfinal"]) + 2
    with mc._Swap(NAME, mc.control_config(NAME)):
        rec_c = mg.variable_energy_recipe(NAME, lo, hi, n, role=ROLE, seed=SEED)
        ctrl = mg.recipe_output_path(rec_c, datagen.dest_for_role(ROLE))
    pool = np.asarray(np.load(ctrl, mmap_mode="r"), np.float64)
    prov = mc.provenance(NAME, lo, hi, n, ROLE, SEED, pool)
    Pc = pool[:, :4 * k].reshape(len(pool), k, 4); sq_c = _sqrts(Pc); me_c = pool[:, -1]
    m_list, cuts, pol, _, _ = mc._sampler_args(cfg, ROLE)
    Pd, sq_d, me_d = mc._default_accepted(pol["default_source"], ROLE, SEED, (lo, hi, m_list, cfg["pdg_ids"], cuts))
    src_pools = [(nm, path) for nm, _, path in mg.default_share_pools(pol["default_source"], ROLE, SEED)]
    n_src = sum(int(np.load(path, mmap_mode="r").shape[0]) for _, path in src_pools)
    rows, kinds = mc.compose(NAME, lo, hi, n, ROLE, SEED, ctrl)
    Pt = rows[:, :4 * k].reshape(len(rows), k, 4); sq_t = _sqrts(Pt); me_t = rows[:, -1]
    out = {"class": NAME, "role": ROLE, "seed": SEED, "recipe": os.path.relpath(RECIPE, ROOT), "window": [lo, hi],
           "catalog_pool": ctrl, "default_source": pol["default_source"], "f_default": float(pol["f_default"]),
           "default_pools": [[nm, path] for nm, path in src_pools], "default_rows_total": n_src,
           "logm_bins": LOGM_BINS.tolist(), "sqrts_bins": SQ_BINS.tolist(), "h": {}}
    out["h"]["catalog"] = _h(me_c, sq_c)
    for kind in ("bulk", "fwd", "flat"):
        s = prov == mc.KIND[kind]
        out["h"]["catalog_" + kind] = _h(me_c[s], sq_c[s])
    out["h"]["default"] = _h(me_d, sq_d)
    out["h"]["trained"] = _h(me_t, sq_t)
    for kind in ("bulk", "fwd", "flat", "default"):
        s = kinds == mc.KIND[kind]
        out["h"]["trained_" + kind] = {"n": int(s.sum())}
    print(json.dumps(out, separators=(",", ":")))


def figs():
    import plot_style as ps
    d = json.load(open(OUT_JSON)); h = d["h"]
    lb, sb = np.array(d["logm_bins"]), np.array(d["sqrts_bins"])

    def dens(key, var):
        c = np.asarray(h[key][var], float); e = lb if var == "logm" else np.log10(sb)
        return c / (h["catalog"]["n"] * np.diff(e))      # events per unit log10, per event of the full pool

    def step(ax, key, var, **kw):
        x = lb if var == "logm" else sb
        y = dens(key, var)
        ax.stairs(y, x, **kw)

    proc = r"$u\bar u\to e^+e^-g$"
    # (a) the catalog mixture by component, stacked shares of the one pool
    fig, (axL, axR) = ps.figure(ncols=2)
    comps = [("catalog_bulk", "bulk (RAMBO)", ps.C.sky), ("catalog_fwd", "forward", ps.C.green),
             ("catalog_flat", r"flat in $\log|\mathcal{M}|^2$", ps.C.orange)]
    for ax, var in ((axL, "logm"), (axR, "sqrts")):
        x = lb if var == "logm" else sb
        base = np.zeros(len(x) - 1)
        for key, lab, col in comps:
            y = dens(key, var)
            ax.stairs(base + y, x, baseline=base, fill=True, color=col, alpha=0.6, label=lab)
            base = base + y
        step(ax, "catalog", var, color=ps.C.blue, label="catalog (sum)")
    axL.set_xlim(*LOGM_XLIM); axL.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$"); axL.set_ylabel(r"density per $\log_{10}$")
    axR.set_xscale("log"); axR.set_xlabel(r"$\sqrt{\hat s}$ [GeV]"); axR.set_ylabel(r"density per $\log_{10}$")
    ps.legend(axL, "upper left")
    ps.process_label(axR, proc)
    ps.save(fig, FIGS + "_a")
    # (b) generator default (pp 13 TeV MadEvent), catalog, and the trained 0.25 mixture
    fig, (axL, axR) = ps.figure(ncols=2)
    nd = h["default"]["n"]; nc = h["catalog"]["n"]; nt = h["trained"]["n"]
    for ax, var in ((axL, "logm"), (axR, "sqrts")):
        x = lb if var == "logm" else sb
        e = lb if var == "logm" else np.log10(sb)
        for key, lab, col, nn, ls in (("default", "default (MadEvent, pp 13 TeV)", ps.C.vermillion, nd, "-"),
                                      ("catalog", "catalog", ps.C.blue, nc, "-"),
                                      ("trained", r"trained: $0.75\,$catalog$\,+\,0.25\,$default", ps.C.grey, nt, "--")):
            y = np.asarray(h[key][var], float) / (nn * np.diff(e))
            ax.stairs(y, x, color=col, linestyle=ls, label=lab)
    axL.set_xlim(*LOGM_XLIM); axL.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$"); axL.set_ylabel(r"density per $\log_{10}$")
    axR.set_xscale("log"); axR.set_xlabel(r"$\sqrt{\hat s}$ [GeV]"); axR.set_ylabel(r"density per $\log_{10}$")
    ps.legend(axL, "upper left")
    ps.process_label(axR, proc)
    ps.save(fig, FIGS + "_b")


if __name__ == "__main__":
    {"hist": hist, "figs": figs}[sys.argv[1]]()
