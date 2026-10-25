"""Read-only inventory of run roots on a site: one JSON line per run dir (a dir holding config.yaml) with its bytes
(allocated) split into model files (*.pt, *.pt.gz, *.ckpt, anything under models/) and the rest (plots, preds, logs,
configs, other), the training time its logs report (sum over out_*.log of "Finished training: N iterations ... in X min",
so a warm-started trial counts every increment its logs hold) and, as a fallback, the span between the first and last
timestamped log line; a "protect" line lists the run tails sweep/prune.py and tools/prune_copies.py keep whole
(fine-tune parents, job-script sources, Delta lists, trials named in the repo).
    python analysis/storage/inv_runs.py <root> [<root> ...]
"""
import json, os, re, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIN = re.compile(r"Finished training: (\d+) iterations.* in ([0-9.]+)min")
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
ITER = re.compile(r"^\s*iterations:\s*(\d+)", re.M)
PF = re.compile(r"^\s*processes_file:\s*(\S+)", re.M)
PP = re.compile(r"^\s*pretrained_path:\s*(\S+)", re.M)


def kind(rel, f):
    parts = rel.split("/")
    if "models" in parts or f.endswith((".pt", ".pt.gz", ".ckpt", ".pth")):
        return "model"
    if f.endswith((".png", ".pdf")) or any(p.startswith("plots") for p in parts):
        return "plots"
    if f.endswith((".npz", ".npy")):
        return "preds"
    if f.endswith((".log", ".out", ".err", ".gz")):
        return "logs"
    if f.endswith((".yaml", ".json", ".txt", ".bsh")):
        return "meta"
    return "other"


def span(path):
    first = last = None
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                m = TS.match(line)
                if m:
                    first = first or m.group(1)
                    last = m.group(1)
    except OSError:
        pass
    if not first:
        return 0.0
    tp = lambda s: time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
    return (tp(last) - tp(first)) / 3600


try:
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "sweep"))
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import prune
    prot = set(prune.parents()) | set(prune.delta_runs())
    try:
        import prune_copies
        prot |= set(prune_copies.referenced())
    except Exception as e:                                  # noqa: BLE001
        print(json.dumps({"_warn": "prune_copies.referenced: %r" % e}))
    print(json.dumps({"_protect": sorted(prot)}), flush=True)
except Exception as e:                                      # noqa: BLE001
    print(json.dumps({"_warn": "protect list: %r" % e}), flush=True)

for root in sys.argv[1:]:
    for fam in sorted(os.listdir(root)):
        top = os.path.join(root, fam)
        if not os.path.isdir(top) or os.path.islink(top):
            st = os.lstat(top)
            print(json.dumps({"root": root, "family": fam, "run": None, "loose": st.st_blocks * 512}))
            continue
        recs = {}
        loose = 0
        for dp, dns, fns in os.walk(top):
            rd = dp
            while rd != top and not os.path.exists(os.path.join(rd, "config.yaml")):
                rd = os.path.dirname(rd)
            if rd == top and not os.path.exists(os.path.join(top, "config.yaml")):
                rd = None
            for f in fns:
                fp = os.path.join(dp, f)
                try:
                    b = os.lstat(fp).st_blocks * 512
                except OSError:
                    continue
                if rd is None:
                    loose += b
                    continue
                r = recs.setdefault(rd, {"bytes": {}, "fin_min": 0.0, "fin_iters": 0, "span_h": 0.0})
                k = kind(os.path.relpath(fp, rd), f)
                r["bytes"][k] = r["bytes"].get(k, 0) + b
                if dp == rd and f.startswith("out_") and f.endswith(".log"):
                    try:
                        txt = open(fp, errors="replace").read()
                    except OSError:
                        txt = ""
                    for m in FIN.finditer(txt):
                        r["fin_iters"] += int(m.group(1))
                        r["fin_min"] += float(m.group(2))
                    r["span_h"] += span(fp)
        for rd, r in recs.items():
            try:
                cfg = open(os.path.join(rd, "config.yaml"), errors="replace").read()
            except OSError:
                cfg = ""
            m = ITER.search(cfg); r["iterations"] = int(m.group(1)) if m else None
            m = PF.search(cfg); r["processes_file"] = m.group(1) if m else None
            m = PP.search(cfg); r["pretrained_path"] = m.group(1) if m and m.group(1) != "null" else None
            r["mtime"] = int(os.path.getmtime(rd))
            print(json.dumps({"root": root, "family": fam, "run": os.path.relpath(rd, os.path.dirname(root.rstrip("/")) or "."), **r}))
        print(json.dumps({"root": root, "family": fam, "run": None, "loose": loose}), flush=True)
print(json.dumps({"_done": sys.argv[1:]}))
