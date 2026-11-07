#!/usr/bin/env python3
"""Read of the regeneration of catalog_v4's 1301 hadronic tree classes with the channel-grid flat sampler (plan
catalog_coverage step 65, design of 2026-11-07): fill, cost, integrity and load-time composition of the 3903 _smixq
pools of recipes/catalog_v4_g12_regen.yaml.

    python3 analysis/catalog_v4/regen_read.py pull   on CC-IN2P3's login node through site run (read-only):
                                                     per pool its flat fill (<pool>.flatfill.json, datagen._write_flat_fill),
                                                     labels, chunk seconds, rows, finite |M|^2, recipe id, alpha_s file;
                                                     the default25 composition of the 5 step-75 classes' val pools (regeneration entry + default25's share)
                                                     -> data/regen_g12_pull.json
    python3 analysis/catalog_v4/regen_read.py read   -> data/regen_g12_read.json (verdict by the design's rules)
    python3 analysis/catalog_v4/regen_read.py fig    -> figs/regen_g12_fill.{png,pdf}

Fill of a pool: per in-range 0.25-dex log10|M|^2 bin (quota > 0), kept over quota summed over the pool's chunks; a bin
is short below FILL_BAR (0.95, step 72/75's rule); a class is FILLED when no bin of any role is short. Cost of a class:
the chunk seconds of its three pools (gen_chunk's wall time on one core, the definition of sampling_cost.py's old
per-class core-s) over the old per-class core-s.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
PULL_OUT = os.path.join(DATA, "regen_g12_pull.json")
READ_OUT = os.path.join(DATA, "regen_g12_read.json")
COST = os.path.join(DATA, "sampling_cost_grid12.json")
FIG = os.path.join(HERE, "figs", "regen_g12_fill")
FILL_BAR = 0.95            # steps 72/75: a bin is filled at >= 0.95 of its quota
COST_BAR = 2.0             # CLAUDE.md: a new method's cost over the one it replaces above 2x is a finding
CHECK5 = ["v4_uux_epemg", "v4_uux_epemvevex", "v4_uu_bbxuu", "v4_us_ddxsu", "v4_udx_csxvevex"]   # step 75's classes
STEP75 = {"lo": 0.78, "hi": 2.11}
# D269 ("there is no way "a few" events lie outside, be that 10%, 1%, or whatever, if you had done what I asked, that
# percentage would be zero"): held-out in-support pp events outside the regenerated pools' log|M|^2 range; FILLED needs
# none on every class, a class with one or more is SHORT with its count (a tolerance is Joaquin's call, never the read's)
COVER_BAR = 0     # step 75's cost range over the old, seeds 42/43 (grid_fwd_check.json)

PULL = r'''
import json, os, sys, traceback
sys.path.insert(0, os.getcwd())
import numpy as np, yaml
import mg5_pipeline_final as mg, datagen
def reg(path):
    s = yaml.safe_load(open(path))
    mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
                                   "sampling": p.get("sampling")} for p in s["processes"]], default_sampling=s.get("sampling"))
    return s["processes"]
out = {"pools": {}, "compose": {}}
for p in reg("recipes/catalog_v4_g12_regen.yaml"):
    for role in ("train", "val", "test"):
        r = mg.variable_energy_recipe(p["name"], float(p["sqrts"][0]), float(p["sqrts"][1]), int(p["n_" + role]), role=role, seed=42)
        f = mg.recipe_output_path(r, datagen.dest_for_role(role)); d = {"file": os.path.basename(f)}
        if not os.path.exists(f):
            d["missing"] = True; out["pools"][p["name"] + "/" + role] = d; continue
        try:
            a = np.load(f, mmap_mode="r"); m = np.asarray(a[:, -1])
            d.update(rows=int(a.shape[0]), want=int(p["n_" + role]), finite=bool(np.isfinite(m).all() and (m > 0).all()),
                     id_ok=json.load(open(f + ".recipe.json")).get("recipe_id") == mg.recipe_id(r),
                     alphas=os.path.exists(f[:-4] + ".alphas.npy"))
            ff = f[:-4] + ".flatfill.json"
            if os.path.exists(ff):
                ch0 = json.load(open(ff))["chunks"]; ch = [c for c in ch0 if c]
                d["chunks_without_fill"] = len(ch0) - len(ch)
                q = np.sum([c["quota"] for c in ch], axis=0); k = np.sum([c["kept_per_bin"] for c in ch], axis=0)
                d.update(edges_log10=ch[0]["edges_log10"], quota=q.tolist(), kept=k.tolist(),
                         budget_hit=sum(bool(c["budget_hit"]) for c in ch), chunks=len(ch),
                         labels=int(sum(c["n_labels"] or 0 for c in ch)), secs=float(sum(c["chunk_seconds"] for c in ch)),
                         t_label=float(sum(c["t_label"] or 0 for c in ch)), t_propose=float(sum(c["t_propose"] or 0 for c in ch)))
        except Exception as e:
            d["error"] = "%%s: %%s" %% (type(e).__name__, e)
        out["pools"][p["name"] + "/" + role] = d
# D269 / D264 coverage against held-out default-sampled pp events (never read for the range table: the PP4 probe's val
# pool and the class's dsh_ pools; multichannel_check.pp_sources / _pp, step 75's count): in-support events outside the
# regenerated val+test pools' log10|M|^2 [min, max] (every role samples one distribution; the 1e5-row train pools are
# not read on the login node), the out-of-window share, and the catalog-over-pp density per 0.25-dex bin
sys.path.insert(0, os.path.join(os.getcwd(), "analysis", "catalog_v4"))
import multichannel_check as mcc
probe = {r["class"]: r["probe"] for r in json.load(open("analysis/catalog_v4/data/pp_support_reads.json"))["rows"]}
out["coverage"] = {}
B = np.arange(-40.0, 15.0001, 0.25)
for p in reg("recipes/catalog_v4_g12_regen.yaml"):
    n = p["name"]
    try:
        lm = []
        for role in ("val", "test"):
            r = mg.variable_energy_recipe(n, float(p["sqrts"][0]), float(p["sqrts"][1]), int(p["n_" + role]), role=role, seed=42)
            f = mg.recipe_output_path(r, datagen.dest_for_role(role))
            if os.path.exists(f):
                lm.append(np.log10(np.abs(np.asarray(np.load(f, mmap_mode="r")[:, -1]))))
        if not lm:
            out["coverage"][n] = {"missing": True}; continue
        lm = np.concatenate(lm); lm = lm[np.isfinite(lm)]
        lo, hi = float(lm.min()), float(lm.max())
        # by source: the PP4 probe's val pool and the class's dsh_ pools apart (a SHORT revision may widen the range from
        # the PP4 val events and is then scored on the dsh_ events only, never on the events it was widened from)
        hs = mcc.pp_sources(n, probe[n])["heldout"]
        by = {"pp4_val": mcc._pp(hs[:1], mg.PROCESSES[n], float(p["sqrts"][0]), float(p["sqrts"][1]), n + " pp4_val"),
              "dsh": mcc._pp(hs[1:], mg.PROCESSES[n], float(p["sqrts"][0]), float(p["sqrts"][1]), n + " dsh")}
        by = {k: v for k, v in by.items() if v is not None}
        if not by:
            out["coverage"][n] = {"kept": [lo, hi], "heldout": None}; continue
        ls = np.concatenate([v["lm"] for v in by.values()]); ss = np.concatenate([v["sup"] for v in by.values()])
        h = {"outside_window": float(np.mean([v["outside_window"] for v in by.values()])),
             "fail_cuts_in_window": float(np.mean([v["fail_cuts_in_window"] for v in by.values()]))}
        src_out = {k: {"n_in_support": int(v["sup"].sum()),
                       "outside_kept": int((v["sup"] & ((v["lm"] < lo) | (v["lm"] > hi))).sum()),
                       "max_in_support": float(v["lm"][v["sup"]].max()) if v["sup"].any() else None,
                       "min_in_support": float(v["lm"][v["sup"]].min()) if v["sup"].any() else None} for k, v in by.items()}
        cat = np.histogram(np.clip(lm, B[0], B[-1] - 1e-9), B)[0] / len(lm)
        pp = np.histogram(np.clip(ls[ss], B[0], B[-1] - 1e-9), B)[0]
        m = pp >= 10
        ratio = np.where(m, cat / np.maximum(pp / max(ss.sum(), 1), 1e-300), np.inf)
        i = int(np.argmin(ratio)) if m.any() else None
        out["coverage"][n] = {"kept": [lo, hi], "n_heldout": int(len(ls)), "n_in_support": int(ss.sum()),
                              "outside_kept": int((ss & ((ls < lo) | (ls > hi))).sum()),
                              "above_kept": int((ss & (ls > hi)).sum()), "below_kept": int((ss & (ls < lo)).sum()),
                              "by_source": src_out, "outside_window_all": h["outside_window"], "fail_cuts_in_window_all": h["fail_cuts_in_window"],
                              "density_ratio_min": float(ratio[i]) if i is not None else None,
                              "density_ratio_min_bin": [float(B[i]), float(B[i + 1])] if i is not None else None,
                              "density_ratio_median": float(np.median(ratio[m])) if m.any() else None}
    except Exception as e:
        out["coverage"][n] = {"error": "%%s: %%s" %% (type(e).__name__, str(e)[:300])}
import mix_compose as mc
# the default25 mixture's share on the regenerated pool: the regeneration entry plus default25's f_default and source
# (the _g12 training copies wait on step 68), composed as the loader composes it
d25 = {p["name"]: p for p in yaml.safe_load(open("recipes/catalog_v4_default25_train.yaml"))["processes"]}
spec = {p["name"]: p for p in yaml.safe_load(open("recipes/catalog_v4_g12_regen.yaml"))["processes"]}
for n in %(check5)r:
    p = spec[n]; s25 = (d25.get(n) or {}).get("sampling") or {}
    if not float(s25.get("f_default", 0)):
        out["compose"][n] = {"skipped": "not a default-share entry of default25"}; continue
    mg.PROCESSES[n]["sampling"] = dict(mg.PROCESSES[n]["sampling"], f_default=s25["f_default"],
                                       default_source=s25["default_source"])
    try:
        cfg_c = mc.control_config(n)
        with mc._Swap(n, cfg_c):
            rc = mg.variable_energy_recipe(n, float(p["sqrts"][0]), float(p["sqrts"][1]), int(p["n_val"]), role="val", seed=42)
            ctrl = mg.recipe_output_path(rc, datagen.dest_for_role("val"))
        rows, kinds = mc.compose(n, float(p["sqrts"][0]), float(p["sqrts"][1]), int(p["n_val"]), "val", 42, ctrl)
        out["compose"][n] = {"ok": True, "rows": int(len(rows)), "kinds": np.bincount(kinds, minlength=4).tolist(),
                             "control": os.path.basename(ctrl)}
    except Exception as e:
        out["compose"][n] = {"ok": False, "error": "%%s: %%s" %% (type(e).__name__, str(e)[:300])}
print("REGEN_READ " + json.dumps(out))
'''


def cmd_pull(a):
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    d = site_python(PULL % {"check5": CHECK5}, [], "REGEN_READ", timeout=3000)
    json.dump(d, open(PULL_OUT, "w"))
    print(f"{len(d['pools'])} pools read, compose {d['compose']} -> {PULL_OUT}")


def read():
    d = json.load(open(PULL_OUT)); old = json.load(open(COST))
    by = {}
    for key, p in d["pools"].items():
        by.setdefault(key.split("/")[0], {})[key.split("/")[1]] = p
    classes, short, bad = {}, [], []
    for c, roles in sorted(by.items()):
        r = {"fill_min_share": 1.0, "short_bins": [], "secs": 0.0, "budget_hit": 0}
        for role, p in roles.items():
            if p.get("missing") or p.get("error") or not (p.get("finite") and p.get("id_ok") and p.get("alphas")
                                                          and p.get("rows") == p.get("want")):
                bad.append((c, role, {k: p.get(k) for k in ("missing", "error", "finite", "id_ok", "alphas", "rows", "want")}))
                continue
            if "quota" not in p:
                bad.append((c, role, "no flatfill.json")); continue
            q, k, e = np.asarray(p["quota"], float), np.asarray(p["kept"], float), p["edges_log10"]
            inr = q > 0; fill = np.where(inr, k / np.maximum(q, 1), 1.0)
            share = float((fill[inr] >= FILL_BAR).mean()) if inr.any() else 1.0
            r["fill_min_share"] = min(r["fill_min_share"], share)
            r["short_bins"] += [[role, round(e[i], 3), round(e[i + 1], 3), round(float(fill[i]), 3)]
                                for i in np.where(inr & (fill < FILL_BAR))[0]]
            r["secs"] += p.get("secs", 0.0); r["budget_hit"] += p.get("budget_hit", 0)
            if p.get("chunks_without_fill"):
                bad.append((c, role, f"{p['chunks_without_fill']} chunks without flat fill"))
        o = old["old_per_class_s"].get(c)
        r["cost_ratio"] = r["secs"] / o if o else None
        r["nfinal"] = old["nfinal"].get(c)
        classes[c] = r
        if r["short_bins"]:
            short.append(c)
    ratios = {n: [v["cost_ratio"] for v in classes.values() if v["nfinal"] == n and v["cost_ratio"]] for n in (2, 3, 4)}
    new_h = sum(v["secs"] for v in classes.values()) / 3600
    old_h = sum(old["old_per_class_s"][c] for c in classes if c in old["old_per_class_s"]) / 3600
    cov = d.get("coverage", {})
    over = {c: v["outside_kept"] for c, v in cov.items() if (v.get("outside_kept") or 0) > COVER_BAR}
    cov_bad = [c for c, v in cov.items() if v.get("error") or v.get("missing")]
    for c, v in cov.items():
        if c in classes:
            classes[c]["coverage"] = v
    cost_over = sorted((c for c, v in classes.items() if (v["cost_ratio"] or 0) > COST_BAR),
                       key=lambda c: -classes[c]["cost_ratio"])
    comp = d["compose"]
    comp_ok = all(v.get("ok") for v in comp.values() if "skipped" not in v)
    ctl = {c: classes.get(c, {}).get("cost_ratio") for c in CHECK5}
    verdict = ("BROKEN" if bad or cov_bad else "COMPOSE_FAILS" if not comp_ok else "SHORT" if short or over
               else "COST" if new_h > COST_BAR * old_h else "FILLED")
    return {"design": "plan catalog_coverage step 65 (design 2026-11-07)", "verdict": verdict,
            "classes_read": len(classes), "classes_short": short, "pools_bad": bad[:50], "n_pools_bad": len(bad),
            "new_core_h_sampling": round(new_h, 2), "old_core_h": round(old_h, 2),
            "estimate_core_h": old.get("total_1301"),
            "cost_ratio_by_nfinal": {str(n): {"median": float(np.median(v)), "q10": float(np.quantile(v, 0.1)),
                                              "q90": float(np.quantile(v, 0.9)), "max": float(max(v)), "n": len(v)}
                                     for n, v in ratios.items() if v},
            "control_step75_cost_ratio": ctl, "step75_range": STEP75,
            "budget_hit_chunks": sum(v["budget_hit"] for v in classes.values()),
            "cover_bar": COVER_BAR, "classes_outside_over_bar": over, "coverage_errors": cov_bad,
            "heldout_outside_kept_total": sum(v.get("outside_kept") or 0 for v in cov.values()),
            "heldout_in_support_total": sum(v.get("n_in_support") or 0 for v in cov.values()),
            "classes_with_any_outside": sum(1 for v in cov.values() if (v.get("outside_kept") or 0) > 0),
            "density_ratio_min_q10": (float(np.quantile([v["density_ratio_min"] for v in cov.values()
                                                         if v.get("density_ratio_min") is not None], 0.1))
                                      if cov else None),
            "classes_cost_over_2x": {c: round(classes[c]["cost_ratio"], 2) for c in cost_over},
            "compose": comp, "classes": classes}


def cmd_read(a):
    r = read()
    json.dump(r, open(READ_OUT, "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("verdict", "classes_read", "n_pools_bad", "new_core_h_sampling", "old_core_h",
                                        "cost_ratio_by_nfinal", "control_step75_cost_ratio", "budget_hit_chunks",
                                        "heldout_outside_kept_total", "heldout_in_support_total",
                                        "classes_with_any_outside", "classes_outside_over_bar", "density_ratio_min_q10",
                                        "classes_cost_over_2x")},
                     indent=1))
    print("short:", r["classes_short"][:20])


def cmd_fig(a):
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = json.load(open(READ_OUT)); cl = r["classes"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    s = np.sort([v["fill_min_share"] for v in cl.values()])
    ax1.plot(np.arange(len(s)), s, ".", ms=2, label="share of in-range bins at >= 0.95 of quota (worst role)")
    ax1.axhline(1.0, color="0.6", lw=0.8)
    ax1.set_xlabel("class (sorted)"); ax1.set_ylabel("share of bins filled"); ax1.legend(fontsize=8)
    for n, col in ((2, "C0"), (3, "C1"), (4, "C2")):
        v = [x["cost_ratio"] for x in cl.values() if x["nfinal"] == n and x["cost_ratio"]]
        ax2.plot(np.full(len(v), n) + np.random.default_rng(0).uniform(-0.2, 0.2, len(v)), v, ".", ms=2, color=col,
                 label=f"2->{n} ({len(v)} classes)")
    ax2.axhline(COST_BAR, color="0.4", ls="--", lw=0.8, label="2x old")
    ax2.set_yscale("log"); ax2.set_xlabel("final-state particles"); ax2.set_ylabel("core-s new / core-s old")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}.{ext}", dpi=150)
    print(f"{FIG}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("pull", "read", "fig"):
        sub.add_parser(c)
    a = ap.parse_args()
    {"pull": cmd_pull, "read": cmd_read, "fig": cmd_fig}[a.cmd](a)


if __name__ == "__main__":
    main()
