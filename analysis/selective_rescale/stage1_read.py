"""Plan selective_rescale (D552) stage 1: three event-dependent rescale schemes at each cell's no-rescale best HPs
(Z+4g hp1333; u u~ g the chosen trial of ftmech_d539_uug_<D>_off_b2048), seeds 42-44, runs selres_s1_<proc>_<D>_<scheme>_s<s>
(seed_chunks/selres_stage1_all.txt), against D539's controls at the same HPs: Z+4g off / rescale = d539s.json (hp1333),
u u~ g off = d539u.json (its own optimum, the same HPs), u u~ g rescale at its own optimum (d539u.json, soft reference).

Measure (design 2026-11-25): best val MSE(log|M|^2) = min val mu-MSE x prepd_std^2; r = three-seed geometric mean over
the control, 2-sigma interval (zng4g_d1e3_read.ratio). u u~ g kit deepest-bin log10-MSE (tools/kit_mlp_readout.deepest),
seed medians; soft tolerance = the rescale arm's seed median plus its three-seed spread log10(max/min).

    python3 analysis/selective_rescale/stage1_read.py [--cached]   -> data/stage1.json, figs/stage1
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "finetune_mechanism"))
import numpy as np  # noqa: E402
from analysis.common.readers import site_python  # noqa: E402
from tools.kit_mlp_readout import deepest  # noqa: E402
from zng4g_d1e3_read import REMOTE, ratio  # noqa: E402

SITES = ("ccin2p3", "lxplus")
SCHEMES = ["softbelow_x0p1", "softbelow_x0p3", "eventgate_x0p1"]
CELLS = [(p, d) for p in ("zng", "uug") for d in ("1e4", "1e5")]
SEEDS = (42, 43, 44)
FM = os.path.join(ROOT, "analysis", "finetune_mechanism", "data")
KIT = r'''
import json, os, sys
sys.path.insert(0, os.getcwd()); import siteconf
out = {}
for r in sys.argv[1:]:
    p = os.path.join(siteconf.PROJECT_DIR, "runs", r, "kit_benchmark", "kit_result.json")
    if os.path.exists(p):
        d = json.load(open(p)); d = d.get("strata", d)
        out[r] = {k: v["logmse"] for k, v in d.items() if isinstance(v, dict) and "logmse" in v}
print("SELRESKIT " + json.dumps(out))
'''


def name(p, d, a, s):
    return f"selres_s1_{p}_{d}_{a}_s{s}"


def main():
    cache = os.path.join(HERE, "data", "stage1_raw.json")
    if "--cached" in sys.argv:
        raw = json.load(open(cache))
    else:
        runs = [name(p, d, a, s) for p, d in CELLS for a in SCHEMES for s in SEEDS]
        val, kit = {}, {}
        for site in SITES:   # runs of one batch can land on either site (site pick); keep each run's complete record
            paths = site_python('import json, os, sys; sys.path.insert(0, os.getcwd()); import siteconf; print("SITEDIRS " + json.dumps([siteconf.PROJECT_DIR, siteconf.SWEEP_DIR]))',
                                [], "SITEDIRS", site=site)
            v = site_python(REMOTE.replace("ZNG1K", "SELRES"),
                            [f"'{r}#{paths[0]}/runs/{r}#{paths[1]}/{r}/results/fixed.json'" for r in runs], "SELRES", site=site)
            k = site_python(KIT, [r for r in runs if "_uug_" in r], "SELRESKIT", site=site)
            v = {r: x | {"site": site} for r, x in v.items() if x.get("val_mse_curve")}
            for got, new in ((val, v), (kit, {r: x | {"site": site} for r, x in k.items()})):
                both = sorted(set(got) & set(new))   # a run's numbers come from the one site that ran it
                if both:
                    sys.exit(f"stage1_read: runs with records on {got[both[0]]['site']} and {site}: {both}")
                got |= new
        bad = sorted(r for r in kit if r in val and kit[r]["site"] != val[r]["site"])
        if bad:
            sys.exit(f"stage1_read: kit and validation numbers from different sites: {bad}")
        raw = {"val": val, "kit": {r: {f: x for f, x in d.items() if f != "site"} for r, d in kit.items()},
               "kit_site": {r: d["site"] for r, d in kit.items()}}
        json.dump(raw, open(cache, "w"))
    s539, u539 = (json.load(open(os.path.join(FM, f))) for f in ("d539s.json", "d539u.json"))
    k539 = json.load(open(os.path.join(FM, "d539u_kit.json")))["cells"]
    ctl = {("zng", d): {"off": s539["cells"][f"zng_{d}"]["off"]["best"], "rescale": s539["cells"][f"zng_{d}"]["eps1e-8"]["best"]}
           for d in ("1e4", "1e5")}
    ctl |= {("uug", d): {"off": u539["cells"][f"uug_{d}"]["off"]["best"], "rescale": u539["cells"][f"uug_{d}"]["eps1e-8"]["best"]}
            for d in ("1e4", "1e5")}
    res = {"missing": [], "blowups": [], "cells": {}}
    for p, d in CELLS:
        cell = {k: {"best": v} for k, v in ctl[(p, d)].items()}
        for a in SCHEMES:
            b = []
            for s in SEEDS:
                v = raw["val"].get(name(p, d, a, s), {})
                if not v.get("val_mse_curve") or "std" not in v:
                    res["missing"].append(name(p, d, a, s)); b.append(None); continue
                c = np.array(v["val_mse_curve"]) * v["std"] ** 2
                up = c > 10 * np.minimum.accumulate(c)   # val above 10x its best so far (D398)
                if up.any():
                    res["blowups"].append({"run": name(p, d, a, s), "at_pct": 100 * (int(np.argmax(up)) + 1) / len(c)})
                b.append(float(c.min()))
            cell[a] = {"best": b}
            if None not in b:
                cell[a]["over_off"] = ratio(np.array(b), np.array(cell["off"]["best"]))
                cell[a]["over_rescale"] = ratio(np.array(b), np.array(cell["rescale"]["best"]))
        if p == "uug":   # kit: seed medians and decades against both controls
            kc = k539[d]
            for a in SCHEMES:
                deep = [deepest(raw["kit"][name(p, d, a, s)]) if name(p, d, a, s) in raw["kit"] else None for s in SEEDS]
                if None in deep:
                    res["missing"] += [f"kit:{name(p, d, a, s)}" for s, x in zip(SEEDS, deep) if x is None]; continue
                fam = {f: [x[f] for x in deep] for f in deep[0]}
                cell[a]["kit"] = fam
                cell[a]["kit_dec_vs_off"] = {f: float(np.log10(np.median(v) / np.median(kc["off"][f]))) for f, v in fam.items()}
                cell[a]["kit_dec_vs_rescale"] = {f: float(np.log10(np.median(v) / np.median(kc["eps1e-8"][f]))) for f, v in fam.items()}
            rs = kc["eps1e-8"]["soft_x"]
            cell["soft_tolerance_dec"] = float(np.log10(max(rs) / min(rs)))
        res["cells"][f"{p}_{d}"] = cell
    for a in SCHEMES:   # the design's YES, per scheme
        zc = [res["cells"][f"zng_{d}"][a] for d in ("1e4", "1e5")]
        bulk_complete = all("over_off" in c for c in zc)   # the YES needs both Z+4g cells read
        bulk = bulk_complete and all(c["over_off"]["lo"] <= 1 and c["over_rescale"]["hi"] < 1 for c in zc)
        soft = [res["cells"][f"uug_{d}"][a]["kit_dec_vs_rescale"]["soft_x"] <= res["cells"][f"uug_{d}"]["soft_tolerance_dec"]
                for d in ("1e4", "1e5") if "kit_dec_vs_rescale" in res["cells"][f"uug_{d}"].get(a, {})]
        res.setdefault("yes", {})[a] = {"bulk": bulk, "soft": len(soft) == 2 and all(soft),
                                        "complete": bulk_complete and len(soft) == 2}
    json.dump(res, open(os.path.join(HERE, "data", "stage1.json"), "w"), indent=1)
    print(json.dumps(res))   # one line: verify_claims reads the last {...} line
    plot(res)


def plot(res):
    """(a)-(b) best val MSE per seed against the arm, Z+4g and u u~ g, both sizes; (c) u u~ g kit deepest bins at both sizes."""
    sys.path.insert(0, ROOT)
    import plot_style as ps
    k539 = json.load(open(os.path.join(FM, "d539u_kit.json")))["cells"]
    arms = ["off", "rescale"] + SCHEMES
    lab = {"off": "no rescale", "rescale": r"rescale, all $x$", "softbelow_x0p1": r"rescale below $x=0.1$",
           "softbelow_x0p3": r"rescale below $x=0.3$", "eventgate_x0p1": r"rescale if $\min x<0.1$"}
    cols = dict(zip(arms, [ps.C.blue, ps.C.vermillion] + ps.sequence(3)))
    blown = {b["run"] for b in res["blowups"]}
    fig, axs = ps.figure(ncols=2, nrows=2)
    (za, ua), (kk, cell) = axs
    for ax, p, nm in ((za, "zng", r"$q\bar q\to Z+4g$"), (ua, "uug", r"$e^+e^-\to u\bar ug$")):
        for i, d in enumerate(("1e4", "1e5")):
            c = res["cells"][f"{p}_{d}"]
            for j, a in enumerate(arms):
                for m, (sd, x) in enumerate(zip(SEEDS, c.get(a, {}).get("best", []))):
                    if x is not None:
                        ax.plot(i + (j - 2) * 0.15 + (m - 1) * 0.03, x, "o", color=cols[a], ms=3,
                                label=lab[a] if (i, m) == (0, 0) and p == "zng" else None,
                                mfc="none" if name(p, d, a, sd) in blown else cols[a])
        ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1]); ax.set_xticklabels([r"$10^4$", r"$10^5$"]); ax.set_yscale("log")
        ax.set_xlabel(r"$D_{\rm train}$"); ax.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$"); ps.process_label(ax, nm)
    fam = {"soft_x": "soft", "coll_y": "coll.", "bulk": "bulk"}
    groups = [(f, d) for d in ("1e4", "1e5") for f in fam]
    for j, a in enumerate(arms):
        for i, (f, d) in enumerate(groups):
            kv = {"off": k539[d]["off"], "rescale": k539[d]["eps1e-8"]}.get(a, res["cells"][f"uug_{d}"].get(a, {}).get("kit"))
            if kv:
                v = [x for x in kv[f] if x is not None]
                kk.plot(i + (j - 2) * 0.15 + (np.arange(len(v)) - 1) * 0.03, v, "o", color=cols[a], ms=3)
    kk.set_xticks(range(len(groups))); kk.set_xlim(-0.5, len(groups) - 0.5)
    kk.set_xticklabels([f"{fam[f]}\n$10^{d[-1]}$" for f, d in groups], fontsize="x-small")
    kk.set_yscale("log"); kk.set_xlabel(r"deepest kit bin, $D_{\rm train}$"); kk.set_ylabel(r"MSE$(\log_{10}|\mathcal{M}|^2)$")
    ps.process_label(kk, r"$e^+e^-\to u\bar ug$")
    if blown:
        za.plot([], [], "o", color="0.5", mfc="none", label="blew up (read\nat its best)")
    ps.legend_cell(cell, za)
    ps.save(fig, os.path.join(HERE, "figs", "stage1"))


if __name__ == "__main__":
    main()
