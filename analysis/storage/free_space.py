"""Free CC's /sps/lpnhe under the user's call of 2026-10-25 ("About the storage, yes to 1, 2 under 5 h, keep the tars"):
  (1) the MadGraph events directories (<proc>_events: madevent output, which the catalogs never read; they sample phase
      space themselves and the backend checks read only <proc>_standalone) of catalog_v3, the older catalogs
      (catalog_v2 and before) and catalog_v4's first validation; the BSM (Z', 2HDM) events dirs are not in the call
  (2) the model files of concluded studies' runs that trained in under 5 h (analysis/storage/inventory.json candidate
      set under_5h), keeping config, logs, metrics, data_stats.json, the tokenizer, plots and predictions
The archive tars (tmp/zz_archive_lxera) are not touched.

    python analysis/storage/free_space.py list                  # read-only: recompute both lists, print the manifest
    python analysis/storage/free_space.py delete MANIFEST.tsv --kind events|model [--start I --count N]

`list` re-applies every exclusion now: a run stays whole when it is protected (sweep/prune.py's fine-tune parents and
job-script sources, the Delta lists, tools/prune_copies.py's repo references), named as pretrained_path by any run's
config.yaml on this site, named (its directory name) anywhere in the checkout, in a family whose models an analysis
loads (trim_plan.tsv), in a study not concluded, without a measured training time, at 5 h or more, or holding any file
written in the last 6 h. An events dir holding a file written in the last 6 h stays too.
Manifest lines (TSV): kind, path (absolute), allocated bytes, apparent bytes, mtime (int), detail.
`delete` removes exactly the manifest's paths of one kind and refuses any whose size or mtime changed since the list.
"""
import argparse, fnmatch, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from build_inventory import load_map, family_group, mg_kind          # noqa: E402

MG5 = "/sps/lpnhe/jiturrizaramirez01/mg5amcnlo"
EVENT_KINDS = {"catalog_v3", "catalog_v2 and older (ee_, uubar_, ...)", "catalog_v4 (generating now)"}
RUN_ROOTS = ("runs", "runs_from_lxplus")
WEIGHTS = (".pt", ".pt.gz", ".ckpt", ".pth")
FIN = re.compile(r"Finished training: (\d+) iterations.* in ([0-9.]+)min")
PP = re.compile(r"pretrained_path:\s*(\S+)")
RECENT = 6 * 3600
LIM_H = 5.0


def walk_dir(p, now):
    alloc = app = 0
    recent = False
    for dp, dns, fns in os.walk(p):
        for f in fns:
            try:
                st = os.lstat(os.path.join(dp, f))
            except OSError:
                continue
            alloc += st.st_blocks * 512
            app += st.st_size
            recent |= st.st_mtime > now - RECENT
    return alloc, app, recent


def list_events(now, out, skipped):
    for e in sorted(os.listdir(MG5)):
        p = os.path.join(MG5, e)
        if not e.endswith("_events") or not os.path.isdir(p) or os.path.islink(p):
            continue
        k, t = mg_kind(e)
        if t != "events":
            continue
        if k not in EVENT_KINDS:
            skipped.append(("events", p, "kind %s not in the call" % k))
            continue
        alloc, app, recent = walk_dir(p, now)
        if recent:
            skipped.append(("events", p, "a file written in the last 6 h"))
            continue
        out.append(("events", p, alloc, app, int(os.lstat(p).st_mtime), k))


def protected():
    sys.path.insert(0, os.path.join(REPO, "sweep"))
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import prune, prune_copies
    return set(prune.parents()) | set(prune.delta_runs()) | set(prune_copies.referenced())


def repo_text():
    """Every tracked text file of the checkout, for the 'named anywhere' check."""
    files = subprocess.run(["git", "-C", REPO, "ls-files"], capture_output=True, text=True).stdout.split()
    buf = []
    for f in files:
        if f.startswith("analysis/storage/") or f.endswith((".npy", ".npz", ".png", ".pdf", ".pkl", ".pt", ".gz", ".tar")):
            continue
        try:
            buf.append(open(os.path.join(REPO, f), errors="replace").read())
        except OSError:
            pass
    return "\n".join(buf)


def list_models(now, out, skipped):
    rows = load_map()
    prot = protected()
    runs = {}
    for root in RUN_ROOTS:
        top_root = os.path.join(REPO, root)
        for fam in sorted(os.listdir(top_root)):
            top = os.path.join(top_root, fam)
            if not os.path.isdir(top) or os.path.islink(top):
                continue
            for dp, dns, fns in os.walk(top):
                rd = dp
                while rd != top and not os.path.exists(os.path.join(rd, "config.yaml")):
                    rd = os.path.dirname(rd)
                if rd == top and not os.path.exists(os.path.join(top, "config.yaml")):
                    continue
                r = runs.setdefault(rd, {"root": root, "fam": fam, "fin_min": 0.0, "models": [], "recent": False})
                for f in fns:
                    fp = os.path.join(dp, f)
                    try:
                        st = os.lstat(fp)
                    except OSError:
                        continue
                    r["recent"] |= st.st_mtime > now - RECENT
                    rel = os.path.relpath(fp, rd).split("/")
                    if f.endswith(WEIGHTS) or (rel[0] == "models" and not f.endswith((".json", ".yaml", ".txt", ".log", ".bsh"))):
                        r["models"].append((fp, st.st_blocks * 512, st.st_size, int(st.st_mtime)))
                    if dp == rd and f.startswith("out_") and f.endswith(".log"):
                        for m in FIN.finditer(open(fp, errors="replace").read()):
                            r["fin_min"] += float(m.group(2))
    # every pretrained_path any run here names
    parents = set()
    for rd in runs:
        try:
            for m in PP.finditer(open(os.path.join(rd, "config.yaml"), errors="replace").read()):
                v = m.group(1).strip("'\"")
                if v not in ("null", "None", "~"):
                    parents.add(v)
        except OSError:
            pass
    text = repo_text()
    for rd, r in sorted(runs.items()):
        if not r["models"]:
            continue
        tail = os.path.relpath(rd, os.path.join(REPO, r["root"]))
        g = family_group(r["root"], r["fam"], rows)
        h = r["fin_min"] / 60 if r["fin_min"] else None
        why = None
        if g is None or g["status"] != "CONCLUDED":
            why = "study not concluded" if g else "family not in trim_plan.tsv"
        elif g["models_loaded_by"]:
            why = "an analysis loads the family's models: " + g["models_loaded_by"]
        elif tail in prot or any(tail.startswith(p + "/") for p in prot):
            why = "protected (fine-tune parent, job-script source, Delta list or repo reference)"
        elif any(p.rstrip("/").startswith(rd.rstrip("/") + "/") or p.rstrip("/") == rd.rstrip("/")
                 or ("/" + tail + "/") in p for p in parents):
            why = "named as pretrained_path by a run here"
        elif (tail if re.match(r"trial_\d+$", os.path.basename(rd)) else os.path.basename(rd)) in text:
            why = "named in the checkout"
        elif h is None:
            why = "no measured training time"
        elif h >= LIM_H:
            why = "trained %.2f h" % h
        elif r["recent"]:
            why = "a file written in the last 6 h"
        if why:
            skipped.append(("model", rd, why))
            continue
        for fp, alloc, app, mt in r["models"]:
            out.append(("model", fp, alloc, app, mt, "%s | %.2f h | %s" % (os.path.relpath(rd, REPO), h, g["study"])))


def cmd_list():
    now = time.time()
    out, skipped = [], []
    list_events(now, out, skipped)
    list_models(now, out, skipped)
    print("#kind\tpath\talloc_bytes\tapparent_bytes\tmtime\tdetail")
    for row in out:
        print("\t".join(str(x) for x in row))
    for kind, p, why in skipped:
        print("#KEEP\t%s\t%s\t%s" % (kind, p, why))
    for kind in ("events", "model"):
        rows = [r for r in out if r[0] == kind]
        units = len(set(r[1] if kind == "events" else r[5].split(" | ")[0] for r in rows))
        print("#TOTAL\t%s\t%d paths\t%d units\t%.2f GB" % (kind, len(rows), units, sum(r[2] for r in rows) / 1e9))


def cmd_delete(a):
    rows = [l.rstrip("\n").split("\t") for l in open(a.manifest) if l.strip() and not l.startswith("#")]
    rows = [r for r in rows if r[0] == a.kind][a.start:a.start + a.count if a.count else None]
    freed, refused = 0, 0
    for kind, p, alloc, app, mt, _ in rows:
        ok = (p.startswith(MG5 + "/") and p.endswith("_events")) if kind == "events" else \
             any(p.startswith(os.path.join(REPO, r) + "/") for r in RUN_ROOTS)
        if not ok or not os.path.lexists(p):
            print("SKIP\t%s\t%s" % (p, "outside the approved roots" if not ok else "gone"))
            refused += 1
            continue
        st = os.lstat(p)
        if int(st.st_mtime) != int(mt) or (kind == "model" and st.st_size != int(app)):
            print("SKIP\t%s\tchanged since the list" % p)
            refused += 1
            continue
        if kind == "events":
            shutil.rmtree(p)
        else:
            os.remove(p)
        freed += int(alloc)
        print("DEL\t%s\t%s" % (p, alloc), flush=True)
    print("#BATCH\t%s\tstart %d\t%d deleted\t%d skipped\t%.2f GB" % (a.kind, a.start, len(rows) - refused, refused, freed / 1e9))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list")
    d = sp.add_parser("delete")
    d.add_argument("manifest")
    d.add_argument("--kind", required=True, choices=("events", "model"))
    d.add_argument("--start", type=int, default=0)
    d.add_argument("--count", type=int, default=0)
    a = ap.parse_args()
    cmd_list() if a.cmd == "list" else cmd_delete(a)
