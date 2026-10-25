"""Assemble analysis/storage/inventory.json from the raw site inventories (inv_mg5.py and inv_runs.py output, one
JSON line per entry) and the study mapping of analysis/storage/trim_plan.tsv (family glob -> study, OPEN/CONCLUDED,
what loads the models; built 2026-10-23 from docs/results.tex's hand-off and the decisions then in force), updated by
the decisions recorded since (D106/D110: catalog_v2 not used further).
    python3 analysis/storage/build_inventory.py MG5.jsonl RUNS.jsonl [RUNS.jsonl ...] [--catalog-v4-dirs N --catalog-v4-virt M]
Reads only; writes inventory.json next to itself. Sizes are allocated bytes (st_blocks), what the group quota counts.
"""
import argparse, csv, fnmatch, json, os, re, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
G = 1e9
NEED = ("need_runtime", "need_card")
KEEP_REBUILD = ("rebuild_src", "other")
BYPROD = ("objects", "html", "events", "source_tree", "madevent")


def mg_kind(n):
    for t in ("virt_standalone", "standalone", "events"):
        if n.endswith("_" + t):
            b = n[: -len(t) - 1]
            break
    else:
        return "MG5 itself and loose files", ""
    if t == "virt_standalone":
        k = "one-loop (MadLoop) standalones"
    elif b.startswith("v3_"):
        k = "catalog_v3"
    elif b.startswith("v4_"):
        k = "catalog_v4 (generating now)"
    elif b.startswith("dc_"):
        k = "Daniel's catalog (dc_)"
    elif b.startswith("pp13"):
        k = "pp at 13 TeV (pp13_)"
    elif re.match(r"(d4|dE100)", b):
        k = "2->4 catalogs (d4*, dE100)"
    elif b.startswith(("zp_", "thdm_")):
        k = "BSM (Z', 2HDM)"
    else:
        k = "catalog_v2 and older (ee_, uubar_, ...)"
    return k, t


def mg5(path, v4_dirs, v4_virt):
    recs = [json.loads(l) for l in open(path) if '"_done"' not in l]
    agg = defaultdict(lambda: Counter())
    per_dir = []
    for r in recs:
        k, t = mg_kind(r["name"])
        a = r["alloc"]
        tot = sum(a.values())
        need = sum(a.get(c, 0) for c in NEED)
        keep_rb = sum(a.get(c, 0) for c in KEEP_REBUILD)
        if t == "events":                       # madevent output: the catalogs never read it (RAMBO path)
            need, keep_rb = 0, 0
        g = agg[(k, t)]
        g["n"] += 1; g["bytes"] += tot; g["files"] += sum(r["files"].values())
        g["need"] += need; g["rebuild_src"] += keep_rb; g["byproduct"] += tot - need - keep_rb
        for c, v in a.items():
            g["cat_" + c] += v
        if t:
            per_dir.append({"dir": r["name"], "kind": k, "type": t, "GB": round(tot / G, 4),
                            "need_GB": round(need / G, 4), "rebuild_src_GB": round(keep_rb / G, 4),
                            "byproduct_GB": round((tot - need - keep_rb) / G, 4), "files": sum(r["files"].values())})
    table = []
    for (k, t), g in sorted(agg.items(), key=lambda x: -x[1]["bytes"]):
        table.append({"kind": k, "type": t or "-", "dirs": g["n"], "GB": round(g["bytes"] / G, 2), "files": g["files"],
                      "needed_GB": round(g["need"] / G, 2), "rebuild_sources_GB": round(g["rebuild_src"] / G, 2),
                      "byproducts_GB": round(g["byproduct"] / G, 2),
                      "by_category_GB": {c[4:]: round(v / G, 2) for c, v in sorted(g.items()) if c.startswith("cat_")}})
    proc = [x for x in table if x["type"] != "-"]
    tot = sum(x["GB"] for x in table)
    clean_keep_src = sum(x["byproducts_GB"] for x in proc)
    clean_min = clean_keep_src + sum(x["rebuild_sources_GB"] for x in proc)
    # catalog_v4 projection from catalog_v3's measured per-dir means (v3 has the same generator and layers)
    v3 = {x["type"]: x for x in proc if x["kind"] == "catalog_v3"}
    lp = [x for x in proc if x["type"] == "virt_standalone"]
    ev = v3["events"]["GB"] / v3["events"]["dirs"]
    sa = v3["standalone"]["GB"] / v3["standalone"]["dirs"]
    sa_need = v3["standalone"]["needed_GB"] / v3["standalone"]["dirs"]
    sa_src = v3["standalone"]["rebuild_sources_GB"] / v3["standalone"]["dirs"]
    vt = sum(x["GB"] for x in lp) / sum(x["dirs"] for x in lp)
    vt_need = sum(x["needed_GB"] for x in lp) / sum(x["dirs"] for x in lp)
    proj = {"process_dirs": v4_dirs, "one_loop_dirs": v4_virt,
            "per_dir_GB_from_catalog_v3": {"events": round(ev, 4), "standalone": round(sa, 4),
                                           "standalone_needed": round(sa_need, 4),
                                           "standalone_rebuild_sources": round(sa_src, 4),
                                           "one_loop_standalone": round(vt, 4), "one_loop_needed": round(vt_need, 4)},
            "as_generated_now_GB": round(v4_dirs * (ev + sa) + v4_virt * vt, 1),
            "without_events_dirs_GB": round(v4_dirs * sa + v4_virt * vt, 1),
            "after_byproduct_cleanup_keeping_sources_GB": round(v4_dirs * (sa_need + sa_src) + v4_virt * vt, 1),
            "runtime_only_GB": round(v4_dirs * sa_need + v4_virt * vt_need, 1)}
    return {"total_GB": round(tot, 1), "by_kind": table,
            "cleanup": {"byproducts_keeping_rebuild_sources_GB": round(clean_keep_src, 1),
                        "byproducts_and_rebuild_sources_GB": round(clean_min, 1),
                        "kept_if_sources_kept_GB": round(tot - clean_keep_src, 1),
                        "kept_runtime_only_GB": round(tot - clean_min, 1)},
            "catalog_v4_projection": proj}, per_dir


def load_map():
    rows = list(csv.DictReader(open(os.path.join(HERE, "trim_plan.tsv")), delimiter="\t"))
    for r in rows:
        if r["study"].startswith("catalog_v2"):          # D106 (2026-10-24): catalog_v2 is not used further
            r["status"] = "CONCLUDED"
            r["source"] = "decision D106/D110 (catalog_v2 not used further); docs/results.tex's catalog hand-off still lists catalog_v2 work"
        elif r["status"] == "CONCLUDED":
            r["source"] = "trim_plan.tsv (2026-10-23): " + r["reason"].split(";")[0]
        else:
            r["source"] = "trim_plan.tsv (2026-10-23): " + r["reason"].split(";")[0]
    return rows


def family_group(root, fam, rows):
    p = "%s/%s" % (root, fam)
    for r in rows:
        if fnmatch.fnmatch(p, r["family_glob"]):
            return r
    return None


def runs(paths, rows):
    now = time.time()
    prot = set()
    recs, loose = [], Counter()
    for path in paths:
        for l in open(path):
            d = json.loads(l)
            if "_protect" in d:
                prot |= set(d["_protect"])
            elif d.get("run"):
                recs.append(d)
            elif "loose" in d:
                loose[(d["root"], d["family"])] += d["loose"]
    groups = {}
    cand = {"under_1h": [], "under_5h": []}
    for d in recs:
        r = family_group(d["root"], d["family"], rows)
        if r:
            key, study, status, src, loaded = r["family_glob"], r["study"], r["status"], r["source"], r["models_loaded_by"]
        else:
            fam = d["family"]
            pre = "_".join(fam.split("_")[:2]) if fam.startswith(("tp3_", "tp2_", "zz_", "v3_")) else re.split(r"[_\d]", fam)[0] or fam
            key, study, status, loaded = "%s/%s*" % (d["root"], pre), "not in trim_plan.tsv (new since 2026-10-23)", "OPEN", ""
            src = "no study mapping: treated as open"
        g = groups.setdefault(key, {"family_glob": key, "study": study, "status": status, "status_source": src,
                                    "models_loaded_by": loaded, "runs": 0, "GB": 0.0, "model_GB": 0.0, "rest_GB": 0.0,
                                    "protected_runs": 0, "protected_model_GB": 0.0, "hours": [], "no_time": 0})
        tail = d["run"].split("/", 1)[1] if "/" in d["run"] else d["run"]
        b = d["bytes"]
        tot, mod = sum(b.values()), b.get("model", 0)
        h = d["fin_min"] / 60 if d["fin_min"] else None
        g["runs"] += 1; g["GB"] += tot / G; g["model_GB"] += mod / G; g["rest_GB"] += (tot - mod) / G
        is_prot = tail in prot or any(tail.startswith(p + "/") for p in prot)
        if is_prot:
            g["protected_runs"] += 1; g["protected_model_GB"] += mod / G
        if h is None:
            g["no_time"] += 1
        else:
            g["hours"].append(h)
        recent = d["mtime"] > now - 6 * 3600
        if status == "CONCLUDED" and not is_prot and not loaded and h is not None and mod and not recent:
            for k, lim in (("under_1h", 1), ("under_5h", 5)):
                if h < lim:
                    cand[k].append({"run": d["run"], "model_GB": round(mod / G, 4), "train_h": round(h, 2),
                                    "study": study, "status_source": src})
    out = []
    for g in sorted(groups.values(), key=lambda g: -g["GB"]):
        hs = sorted(g.pop("hours"))
        g["train_h_median"] = round(hs[len(hs) // 2], 2) if hs else None
        g["train_h_max"] = round(hs[-1], 2) if hs else None
        g["runs_over_5h"] = sum(1 for x in hs if x >= 5)
        g["runs_under_1h"] = sum(1 for x in hs if x < 1)
        for k in ("GB", "model_GB", "rest_GB", "protected_model_GB"):
            g[k] = round(g[k], 2)
        out.append(g)
    loose_t = [{"root": r, "family": f, "GB": round(v / G, 2)} for (r, f), v in loose.most_common() if v > 0.05 * G]
    sets = {}
    for k, L in cand.items():
        sets[k] = {"runs": len(L), "model_GB_freed": round(sum(x["model_GB"] for x in L), 1),
                   "by_study": dict(Counter(x["study"] for x in L)),
                   "GB_by_study": {s: round(sum(x["model_GB"] for x in L if x["study"] == s), 1)
                                   for s in sorted(set(x["study"] for x in L))},
                   "kept": "config.yaml, data_stats.json, particle_tokenizer.json, out_*.log, metrics, plots, preds",
                   "run_list": L}
    root_tot = Counter()
    for d in recs:
        root_tot[d["root"]] += sum(d["bytes"].values())
    for (r, f), v in loose.items():
        root_tot[r] += v
    return {"total_GB_by_root": {k: round(v / G, 1) for k, v in root_tot.items()},
            "model_GB_by_root": {r: round(sum(d["bytes"].get("model", 0) for d in recs if d["root"] == r) / G, 1)
                                 for r in root_tot},
            "protected_runs": len(prot), "by_study": out, "loose_files_outside_run_dirs": loose_t,
            "candidate_sets": sets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mg5")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--catalog-v4-dirs", type=int, default=2404)
    ap.add_argument("--catalog-v4-virt", type=int, default=167)
    ap.add_argument("--extra", help="JSON file of site facts read by hand (du of other dirs, prune dry-runs)")
    a = ap.parse_args()
    rows = load_map()
    m, per_dir = mg5(a.mg5, a.catalog_v4_dirs, a.catalog_v4_virt)
    inv = {"site": "ccin2p3", "when": time.strftime("%Y-%m-%d %H:%M"),
           "filesystem": "/sps/lpnhe (group quota 512 GiB, 95% used at inventory time)",
           "sizes": "allocated bytes (st_blocks); small files cost 32 KiB each",
           "mg5amcnlo": m, "other": json.load(open(a.extra)) if a.extra else None, "runs": runs(a.runs, rows), "mg5amcnlo_per_dir": per_dir}
    json.dump(inv, open(os.path.join(HERE, "inventory.json"), "w"), indent=1)
    print(json.dumps({"mg5amcnlo": {k: v for k, v in m.items() if k != "by_kind"}}, indent=1))


if __name__ == "__main__":
    main()
