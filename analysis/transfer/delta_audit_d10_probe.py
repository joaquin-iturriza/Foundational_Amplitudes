"""Read-only listing for delta_audit_d10.py: per sweep matching the 32k d9/d10 families, its result files (val_loss,
best_step, traintime) under SWEEP_DIR and RESULTS_DIR, its sweep-dir markers, and per run dir under every runs* tree
of the checkout whether config.yaml, data_stats.json (and prepd_std), particle_tokenizer.json and
models/model_run0_best.pt[.gz] exist, with the result file its config names. Prints one JSON line after __J__.
Run on a site's login node from the checkout (the source is sent inline, nothing is written there):
    site run <site> FA -- "python -c \"import base64;exec(base64.b64decode('<b64 of this file>'))\""
(delta_audit_d10.py --probe does that for ccin2p3 and lxplus; on Jean Zay the inline run hung, the listing there was
taken with ls, see delta_audit_d10_sites.json jeanzay._note.)
"""
import glob, json, os, re, sys
PD = os.environ.get("PROJECT_DIR") or os.getcwd()
sys.path.insert(0, os.getcwd())
try:
    import siteconf
    SW, RD, PD = siteconf.SWEEP_DIR, siteconf.RESULTS_DIR, siteconf.PROJECT_DIR
except Exception as e:
    SW = RD = os.path.join(PD, "sweeps")
PATS = []
for _f in ["tp3_scr32k", "tp3_uu64fte32k"] + [f"tp3_r{k}fte32k" for k in range(1, 10)] + ["tp3_finfte32k", "tp3_synfte32k"]:
    PATS += [f"{_f}_*_d9", f"{_f}_*_d9_*", f"{_f}_*_d10", f"{_f}_*_d10_*"]
PATS += ["tp3_s*fte32k_*", "tp3s_*32k*"]
out = {"_site_paths": [PD, SW, RD]}
cands = set()
for pat in PATS:
    for root in {SW, RD}:
        cands |= {os.path.basename(d) for d in glob.glob(os.path.join(root, pat))}
    for rr in glob.glob(os.path.join(PD, "runs*")):
        cands |= {os.path.basename(d) for d in glob.glob(os.path.join(rr, pat))}
def rpath(c):
    try:
        t = open(c).read()
        m = re.search(r"result_path:\s*(\S+)", t)
        return os.path.basename(m.group(1).strip("'\"")) if m else None
    except OSError:
        return None
for name in sorted(cands):
    e = {"results": {}, "dirs": {}, "sweepdir": {}}
    for root in sorted({SW, RD}):
        sd = os.path.join(root, name)
        if os.path.isdir(sd):
            e["sweepdir"][root] = sorted(x for x in os.listdir(sd) if not x.startswith("trial"))[:20]
        for rp in glob.glob(os.path.join(sd, "results", "*.json")):
            try:
                r = json.load(open(rp))
            except Exception:
                r = {}
            e["results"][os.path.basename(rp)] = {k: r.get(k) for k in ("val_loss", "best_step", "traintime_hours", "test_loss") if not isinstance(r.get(k), list)}
    for rr in glob.glob(os.path.join(PD, "runs*")):
        base = os.path.join(rr, name)
        if not os.path.isdir(base):
            continue
        ds = [base] + sorted(glob.glob(os.path.join(base, "trial_*")))
        for d in ds:
            m = os.path.join(d, "models")
            info = {"cfg": os.path.exists(os.path.join(d, "config.yaml")),
                    "ds": os.path.exists(os.path.join(d, "data_stats.json")),
                    "pt": os.path.exists(os.path.join(m, "model_run0_best.pt")),
                    "ptgz": os.path.exists(os.path.join(m, "model_run0_best.pt.gz")),
                    "tok": os.path.exists(os.path.join(d, "particle_tokenizer.json"))}
            if info["ds"]:
                try:
                    info["std"] = json.load(open(os.path.join(d, "data_stats.json")))["prepd_std"][0]
                except Exception:
                    pass
            if info["cfg"]:
                info["rp"] = rpath(os.path.join(d, "config.yaml"))
            if d == base and not any(info[k] for k in ("cfg", "ds", "pt", "ptgz")):
                continue
            e["dirs"][os.path.relpath(d, PD)] = info
    out[name] = e
print("__J__" + json.dumps(out))
