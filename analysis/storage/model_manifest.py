"""Model files of the finished small-D and transfer-study fine-tunes on CC, under the user's call of 2026-10-26 ("we can
delete the models of the runs already done, it's unlikely we will need them again, just keep the results we need to
plot stuff", D149).

Scope: $SCRATCH/zz_lowd (the small-D fine-tunes, first round) and the fine-tune runs of runs/tp3_* and
runs_from_lxplus/tp3_* (a run whose config.yaml, or an enclosing one, names a fine_tune.pretrained_path; the small-D
DyHPO round tp3_lowd_* and the grid's fine-tunes). A model file is a weights file (*.pt, *.pt.gz, *.ckpt, *.pth,
optimizer, EMA and Fisher states included) or any non-text file under a run's models/. Metrics, curves, configs, logs,
result json, data_stats, the tokenizer, base shapes, predictions (.npz) and plots are never listed.
A run stays whole when: it is named as pretrained_path by any config.yaml under runs/, runs_from_lxplus/ or $SCRATCH
(a parent: the transfer study's pretraining rungs, catalog parents, a fine-tune that is itself fine-tuned from), or by
sweep/prune.py's parents or the Delta lists (the active Delta item evaluates those fine-tunes); an active or held
work item's glob names it (CCORCH_WORK; the finished small-D items excepted); it holds a file written in the last 6 h; or a job of this site's queue names it.
Pretrainings (no pretrained_path) and runs/catalog_v3_short are out of scope and never listed.

    python analysis/storage/model_manifest.py list > manifest.tsv                 # read-only
    python analysis/storage/model_manifest.py delete manifest.tsv [--apply]       # dry run without --apply

`delete` removes only the manifest's `delete` rows, one file at a time with os.remove, and refuses (SKIP) a row whose
path is outside the scope roots, is not a model file by the rule above, is a symlink, is gone, or whose size or mtime
differ from the manifest's.
"""
import argparse, csv, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCRATCH = os.environ.get("SCRATCH", "")
LOWD = os.path.join(SCRATCH, "zz_lowd") if SCRATCH else None
RUN_ROOTS = [os.path.join(REPO, "runs"), os.path.join(REPO, "runs_from_lxplus")]
WEIGHTS = (".pt", ".pt.gz", ".ckpt", ".pth", ".ckpt.gz", ".pth.gz")
TEXT = (".json", ".yaml", ".yml", ".txt", ".log", ".bsh", ".csv", ".npz", ".npy", ".pdf", ".png")
PP = re.compile(r"pretrained_path:\s*(\S+)")
RECENT = 6 * 3600


def is_model(path, run_dir):
    f = os.path.basename(path)
    if f.endswith(WEIGHTS):
        return True
    rel = os.path.relpath(path, run_dir).split(os.sep)
    return "models" in rel[:-1] and not f.endswith(TEXT)


def in_scope(path):
    p = os.path.realpath(path)
    if LOWD and p.startswith(os.path.realpath(LOWD) + os.sep):
        return True
    for r in RUN_ROOTS:
        rr = os.path.realpath(r) + os.sep
        if p.startswith(rr) and p[len(rr):].startswith("tp3_"):
            return True
    return False


def pretrained_of(cfg):
    try:
        for m in PP.finditer(open(cfg, errors="replace").read()):
            v = m.group(1).strip("'\"")
            if v not in ("null", "None", "~", ""):
                return v
    except OSError:
        pass
    return None


def all_parents():
    """Every pretrained_path named by a config.yaml under the run roots and $SCRATCH, and sweep/prune.py's parents."""
    out = set()
    roots = RUN_ROOTS + ([SCRATCH] if SCRATCH else [])
    for root in roots:
        cmd = ["find", root, "-name", "config.yaml", "-not", "-path", "*/.*"]
        for cfg in subprocess.run(cmd, capture_output=True, text=True).stdout.split("\n"):
            if cfg:
                v = pretrained_of(cfg)
                if v:
                    out.add(v)
    try:
        sys.path.insert(0, os.path.join(REPO, "sweep"))
        import prune
        for t in set(prune.parents()) | set(prune.delta_runs()):   # <sweep>/<trial>, under either run root
            out |= {os.path.join(r, t) for r in RUN_ROOTS}
    except Exception as e:  # noqa: BLE001
        print("#WARN\tsweep/prune.py parents not read: %s" % e)
    return {os.path.realpath(p.replace(".gz", "")) for p in out}


DONE_ITEMS = ("tp3_lowd*", "zz_lowd*")    # the small-D first round and its DyHPO round: finished (lowd_ft 900/900)


def item_globs():
    """The active and held work items' globs (CCORCH_WORK, exported by `site run`), less the finished small-D ones."""
    import json
    try:
        w = json.loads(os.environ.get("CCORCH_WORK", "{}"))
    except ValueError:
        return []
    return [g for g in w.get("active", []) + w.get("held", []) if g not in DONE_ITEMS]


def item_match(rd, top, globs):
    import fnmatch
    names = [os.path.basename(top), os.path.relpath(rd, os.path.dirname(top)),
             "runs/" + os.path.relpath(rd, os.path.dirname(top))]
    return next((g for g in globs if any(fnmatch.fnmatch(n, g) for n in names)), None)


def queue_names():
    q = subprocess.run(["squeue", "--me", "-h", "-o", "%j %Z %o"], capture_output=True, text=True).stdout
    return q


def run_dir_of(path, top):
    d = os.path.dirname(path)
    while d != os.path.dirname(top) and not os.path.exists(os.path.join(d, "config.yaml")):
        d = os.path.dirname(d)
    return d if os.path.exists(os.path.join(d, "config.yaml")) else None


def is_finetune(rd, top):
    d = rd
    while True:
        if os.path.exists(os.path.join(d, "config.yaml")) and pretrained_of(os.path.join(d, "config.yaml")):
            return True
        if d == top or d == os.path.dirname(d):
            return False
        d = os.path.dirname(d)


def tops():
    if LOWD and os.path.isdir(LOWD):
        yield "lowd", LOWD
    for r in RUN_ROOTS:
        if os.path.isdir(r):
            for fam in sorted(os.listdir(r)):
                p = os.path.join(r, fam)
                if fam.startswith("tp3_") and os.path.isdir(p) and not os.path.islink(p):
                    yield "tp3", p


def cmd_list():
    now = time.time()
    parents = all_parents()
    queue = queue_names()
    globs = item_globs()
    print("#ITEM_GLOBS\t" + " ".join(globs))
    runs = {}
    for kind, top in tops():
        for dp, dns, fns in os.walk(top):
            for f in fns:
                fp = os.path.join(dp, f)
                try:
                    st = os.lstat(fp)
                except OSError:
                    continue
                rd = run_dir_of(fp, top) or top
                r = runs.setdefault(rd, {"top": top, "kind": kind, "models": [], "recent": False})
                r["recent"] |= st.st_mtime > now - RECENT
                if not os.path.islink(fp) and is_model(fp, rd):
                    r["models"].append((fp, st.st_size, st.st_blocks * 512, int(st.st_mtime)))
    print("decision\tpath\tsize_bytes\talloc_bytes\tmtime\trun\treason")
    tot = {"delete": [0, 0, set()], "keep": [0, 0, set()]}
    for rd, r in sorted(runs.items()):
        if not r["models"]:
            continue
        rrd = os.path.realpath(rd)
        why = None
        if r["kind"] == "tp3" and not is_finetune(rd, r["top"]):
            why = "a pretraining or scratch run (no pretrained_path), out of scope"
        elif any(p == rrd or p.startswith(rrd + os.sep) for p in parents):
            why = "a parent: named as pretrained_path by a run on this site"
        elif item_match(rd, r["top"], globs):
            why = "an active or held work item names it (%s)" % item_match(rd, r["top"], globs)
        elif r["recent"]:
            why = "a file written in the last 6 h"
        elif os.path.basename(r["top"]) in queue or os.path.basename(rd) in queue:
            why = "named by a job in the queue"
        dec = "keep" if why else "delete"
        reason = why or ("small-D fine-tune, first round (zz_lowd), finished" if r["kind"] == "lowd"
                         else "fine-tune, finished (not running, not a parent)")
        for fp, size, alloc, mt in r["models"]:
            print("\t".join([dec, fp, str(size), str(alloc), str(mt), rd, reason]))
            tot[dec][0] += size
            tot[dec][1] += 1
            tot[dec][2].add(rd)
    for dec, (b, n, rs) in tot.items():
        print("#TOTAL\t%s\t%d files\t%d runs\t%.2f GB apparent" % (dec, n, len(rs), b / 1e9))


def cmd_delete(a):
    rows = [r for r in csv.DictReader((l for l in open(a.manifest) if not l.startswith("#")), delimiter="\t")
            if r["decision"] == "delete"]
    done = skipped = freed = 0
    for r in rows:
        p, why = r["path"], None
        if not in_scope(p):
            why = "outside the scope roots"
        elif not is_model(p, r["run"]):
            why = "not a model file"
        elif os.path.islink(p):
            why = "symlink"
        elif not os.path.lexists(p):
            why = "gone"
        else:
            st = os.lstat(p)
            if st.st_size != int(r["size_bytes"]) or int(st.st_mtime) != int(r["mtime"]):
                why = "changed since the list"
        if why:
            skipped += 1
            print("SKIP %s: %s" % (why, p))
            continue
        if a.apply:
            try:
                os.remove(p)
            except OSError as e:
                skipped += 1
                print("SKIP %s: %s" % (e, p))
                continue
        done += 1
        freed += int(r["size_bytes"])
    print("SUMMARY %s %d files, %.2f GB apparent; skipped %d" % ("removed" if a.apply else "would remove", done,
                                                                 freed / 1e9, skipped))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list")
    d = sp.add_parser("delete")
    d.add_argument("manifest")
    d.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    cmd_list() if a.cmd == "list" else cmd_delete(a)
