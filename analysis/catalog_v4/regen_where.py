#!/usr/bin/env python3
"""Why the step-65 regeneration's short classes fall short (plan catalog_coverage step 75): which channel of the grid
sampler (grid_both_res2f_budget12) put candidates in which 0.25-dex log10|M|^2 bin, and where the held-out pp events
of the short region sit.

    python3 analysis/catalog_v4/regen_where.py pull   on CC-IN2P3's login node through site run (read-only), for the
        classes regen_profile.py lists short: from each train pool's <pool>.flatfill.json, per channel and quota bin
        the kept events summed over the chunks (in a bin under its quota every in-range candidate is kept, so there
        they are the candidates), the quota, the channels; and multichannel_check.where of the held-out in-support pp
        events above the lower edge of the class's lowest short bin inside the held-out extent (the PP4 val pool and
        the dsh_ pools; sqrt(s_hat), pT, Delta R, |eta|, pair masses against the events below it)
        -> data/regen_g12_where_pull.json
    python3 analysis/catalog_v4/regen_where.py pull-dsh   the in-support log10|M|^2 min and max of every class's dsh_
        train pool (sqrt(s_hat) in the window, the cuts passed in the partonic frame, multichannel_check._pp)
        -> data/dsh_train_support.json (the SHORT revision's range source, tools/flat_range.py)
    python3 analysis/catalog_v4/regen_where.py read   -> data/regen_g12_where.json: per class kind (regen_profile.py:
        only short beyond the held-out extent, or short inside it) and final-state count, the channel types' share of
        the candidates in the short bins against their share in the filled bins of the top 1 dex below them, and the
        where quantiles' medians over the classes
"""
import collections, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
PULL_OUT = os.path.join(DATA, "regen_g12_where_pull.json")
READ_OUT = os.path.join(DATA, "regen_g12_where.json")
DSH_OUT = os.path.join(DATA, "dsh_train_support.json")

# the in-support log|M|^2 extent of each class's dsh_ train pool (45001 default-sampled MadEvent events, a training
# input of the default share, never scored for coverage once the range reads it): step 75's SHORT revision widens the
# range table with it (tools/flat_range.py), coverage then scored on the PP4 val and dsh_ val/test events only
DSH = r'''
import json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "analysis", "catalog_v4"))
import numpy as np, yaml
import mg5_pipeline_final as mg
import multichannel_check as mcc
s = yaml.safe_load(open("recipes/catalog_v4_g12_regen.yaml"))
mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
                               "sampling": p.get("sampling")} for p in s["processes"]], default_sampling=s.get("sampling"))
out = {}
for p in s["processes"]:
    n = p["name"]; stem = "dsh_" + n[len("v4_"):] + "_13000-13000GeV_train_lhe_amplitudes.npy"
    try:
        v = mcc._pp([stem], mg.PROCESSES[n], float(p["sqrts"][0]), float(p["sqrts"][1]), n + " dsh train")
        if v is None:
            continue
        m = v["lm"][v["sup"]]
        out[n] = {"n": int(len(v["lm"])), "n_in_support": int(len(m)), "file": v["files"][0],
                  "logm10_min_in_support": float(m.min()) if len(m) else None,
                  "logm10_max_in_support": float(m.max()) if len(m) else None}
    except Exception as e:
        out[n] = {"error": "%s: %s" % (type(e).__name__, str(e)[:200])}
print("DSH_RANGE " + json.dumps(out))
'''


PULL = r'''
import json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "analysis", "catalog_v4"))
import numpy as np, yaml
import mg5_pipeline_final as mg, datagen
import multichannel_check as mcc
want = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
s = yaml.safe_load(open("recipes/catalog_v4_g12_regen.yaml"))
mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
                               "sampling": p.get("sampling")} for p in s["processes"]], default_sampling=s.get("sampling"))
procs = [p for p in s["processes"] if p["name"] in WANT]
probe = {r["class"]: r["probe"] for r in json.load(open("analysis/catalog_v4/data/pp_support_reads.json"))["rows"]}
out = {}
# every pool path first: multichannel_check.where re-registers the class's catalog entry (mix_compose.control_config),
# after which the recipe resolves to the old _smix pool
paths = {p["name"]: mg.recipe_output_path(mg.variable_energy_recipe(p["name"], float(p["sqrts"][0]), float(p["sqrts"][1]),
                                                                    int(p["n_train"]), role="train", seed=42),
                                          datagen.dest_for_role("train")) for p in procs}
for p in procs:
    n = p["name"]; d = {}
    try:
        f = paths[n]
        ch = [c for c in json.load(open(f[:-4] + ".flatfill.json"))["chunks"] if c]
        d["channels"] = ch[0]["channels"]; d["edges_log10"] = ch[0]["edges_log10"]
        d["quota"] = np.sum([c["quota"] for c in ch], 0).tolist()
        d["kept_per_channel_bin"] = np.sum([c["kept_per_channel_bin"] for c in ch], 0).tolist()
        cut = WANT[n]
        if cut is not None:
            d["where"] = mcc.where(n, probe[n], cut)
    except Exception as e:
        d["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
    out[n] = d
print("REGEN_WHERE " + json.dumps(out))
'''


def targets():
    """{class: lower edge (log10) of its lowest short bin inside the held-out extent, or None} for the short classes"""
    R = json.load(open(os.path.join(DATA, "regen_g12_read.json")))["classes"]
    kind = json.load(open(os.path.join(DATA, "regen_g12_profile.json")))["short"]["classes"]
    out = {}
    for c, k in kind.items():
        bs = R[c]["coverage"].get("by_source", {})
        mx = max([v["max_in_support"] for v in bs.values() if v.get("max_in_support") is not None], default=None)
        mn = min([v["min_in_support"] for v in bs.values() if v.get("min_in_support") is not None], default=None)
        ins = [a for _, a, b, _ in R[c]["short_bins"] if mx is not None and a < mx and b > mn]
        out[c] = min(ins) if ins else None
    return out


def pull():
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    t = targets(); res = {}
    names = sorted(t)
    for i in range(0, len(names), 60):        # batches: one login-node call each
        part = {c: t[c] for c in names[i:i + 60]}
        res.update(site_python(PULL.replace("WANT", repr(part)), [], "REGEN_WHERE", timeout=3000))
        print(f"{len(res)} of {len(names)}", flush=True)
    json.dump(res, open(PULL_OUT, "w"))


def pull_dsh():
    sys.path.insert(0, ROOT)
    from analysis.common.readers import site_python
    json.dump(site_python(DSH, [], "DSH_RANGE", timeout=3000), open(DSH_OUT, "w"), indent=0)


def ctype(c):
    return {"res2f": "res2_fwd", "mperi": "ladder_edge" if c[1:] == ["edge"] else "ladder"}.get(c[0], c[0])


def read():
    W = json.load(open(PULL_OUT))
    R = json.load(open(os.path.join(DATA, "regen_g12_read.json")))["classes"]
    kind = json.load(open(os.path.join(DATA, "regen_g12_profile.json")))["short"]["classes"]
    grp = collections.defaultdict(lambda: {"n": 0, "short": collections.Counter(), "filled_top": collections.Counter(),
                                           "short_zero": 0, "short_bins": 0, "where": collections.defaultdict(list)})
    errs = {c: d["error"] for c, d in W.items() if "error" in d}
    for c, d in W.items():
        if "error" in d:
            continue
        g = grp[(kind[c], R[c]["nfinal"])]; g["n"] += 1
        q = np.asarray(d["quota"], float); K = np.asarray(d["kept_per_channel_bin"], float); e = np.asarray(d["edges_log10"])
        tot = K.sum(0); fill = np.where(q > 0, np.minimum(tot, q) / np.maximum(q, 1), 1.0)
        sb = np.nonzero(fill < 0.95)[0]
        if not len(sb):
            continue
        top = sb[sb >= len(q) // 2]
        lowest = top.min() if len(top) else sb.min()
        fb = [j for j in range(len(q)) if fill[j] >= 0.95 and e[j] >= e[lowest] - 1.0 and j < lowest]
        types = [ctype(x) for x in d["channels"]] if d["channels"] else ["proposal"]
        for j in sb:
            g["short_bins"] += 1; g["short_zero"] += int(tot[j] == 0)
            for i, t in enumerate(types):
                g["short"][t] += K[i, j]
        for j in fb:
            for i, t in enumerate(types):
                g["filled_top"][t] += K[i, j]
        for f, v in (d.get("where") or {}).get("quantiles_10_50_90", {}).items():
            if v.get("above") and v.get("below"):
                g["where"][f].append((v["above"][1], v["below"][1]))
    out = {"errors": errs, "groups": {}}
    for (k, nf), g in sorted(grp.items()):
        sh = sum(g["short"].values()); ft = sum(g["filled_top"].values())
        out["groups"][f"{k} / {nf} finals"] = {
            "classes": g["n"], "short_bins": g["short_bins"], "short_bins_without_any_candidate": g["short_zero"],
            "channel_share_in_short_bins": {t: round(v / sh, 3) for t, v in g["short"].most_common()} if sh else None,
            "channel_share_in_filled_bins_1dex_below": {t: round(v / ft, 3) for t, v in g["filled_top"].most_common()} if ft else None,
            "where_median_above_vs_below": {f: [round(float(np.median([a for a, _ in v])), 3), round(float(np.median([b for _, b in v])), 3)]
                                            for f, v in g["where"].items()}}
    json.dump(out, open(READ_OUT, "w"), indent=1)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    {"pull": pull, "read": read, "pull-dsh": pull_dsh}[sys.argv[1]]()
