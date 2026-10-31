#!/usr/bin/env python3
"""Collect finished runs home in one call: the mechanical step of a plan entry (read the log, copy the reads, count,
ack, record), so a step agent spends one tool call on it instead of dozens.

    python3 tools/collect_run.py <run> [<run> ...] --entry <plan entry> [--dest DIR] [--axes H1,H5,...]
                                 [--prefix-filter auto|none|<prefix>] [--arm ARM] [--dry-run]

Per run:
  1. one `site logs` read (the whole log, counted as read for `site ack`), filtered here to the lines that matter:
     `<axis> ZERO_SHOT {json}` (tools/zero_shot_parent.py; per axis the probes read, errored and non-finite),
     V3_RECIPE_CHECK / V3_AXIS_SKIPPED, the done marker V3_SHORT_DONE, Traceback, non-finite gradient skips and
     "Training diverged";
  2. one `site run` that sends home, as tar | base64, the run's prediction files $SCRATCH/zz_<cat>_short_<arm>/<axis>/
     *.npz and <axis>_signed.txt, plus the run's result.json (best val_loss_no_reg and its step, the checkpoint
     selection); kept are only the axis recipe's probes with the run's own catalog prefix (--prefix-filter auto: the
     arm's catalog `<cat>_` on its own axes, v3_ / v4_ / pp13_ on the cross reads XV3 / XV4 / PP, the rule of
     scripts/job_catalog_v3_short.sh and analysis/catalog_v3/short_reads.py), and of <axis>_signed.txt only the lines
     of those pools; written into <dest>/<axis>/ (default dest analysis/catalog_v3/data/short_reads/<arm>, what
     short_reads.py reads). The site folder holds the latest job's reads of an axis (the job moves earlier ones to
     <axis>.prev_<job>): collect a run before a later job of the same arm re-reads that axis;
  3. counts against the axis recipes (recipes/catalog_<cat>_heldout_<axis>.yaml, the shared held-out recipes, the pp
     test recipe);
  4. if nothing is off: `site ack` of the run's open events with a one-line summary, and
     `plan evidence <entry> --run <run> --path <dest>`.
Physics (tools/pool_physics.py zs_status): every scored probe's ZERO_SHOT record carries its physics check; a probe read
with a mismatch on purpose is named with its reason in the summary, a probe refused for a mismatch is an errored probe,
and a probe read without the check (code before it) is off unless --allow-physics-unchecked "<reason>".
One compact summary per run. Exit 1 if anything is off (an expected probe without a prediction, an errored or
non-finite probe, an axis skipped, a Traceback, a non-finite step or a divergence, no done marker, the run not
finished); such a run is neither acked nor recorded: the step stops and reports.

Covers every arm of scripts/job_catalog_v3_short.sh (training, mode eval with V3_AXES, both catalogs, the cross reads;
mode smoke gets the log checks only, nothing copied), and any run whose log carries no ZERO_SHOT lines (a plain
training or prep job): there only the log checks, the result.json when the log names a run_dir, the ack and the
evidence. --dry-run reads the log and fetches into a temporary folder that is removed, so nothing is written, acked or
recorded.
"""
import argparse, base64, io, json, math, os, re, shlex, shutil, subprocess, sys, tarfile, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
import pool_physics  # noqa: E402  (standard library only at import)
SITE = shlex.split(os.environ.get("PLAN_SITE", "site"))
PLAN = [sys.executable, os.path.join(HERE, "tools", "plan.py")]
CROSS_PREFIX = {"XV3": "v3_", "XV4": "v4_", "PP": "pp13_", "PP4": ("pp13v4_", "pp13_"), "EE4": "ee4_",
                "HD4": "hd_"}  # the job's rule
# EE4 entries excluded by decision (pool_physics.DROPPED: ee4_epem_cgsxwm, its 3 TeV block dropped, D193); the job
# leaves them out and lists (V3_AXIS_PARTIAL) every other entry without a pool, so any other one missing is still
# reported here
EE4_DROPPED = set(pool_physics.DROPPED)
# the role each axis is scored on, as scripts/job_catalog_v3_short.sh sets SPLIT (test on PP4 and TR, val elsewhere)
AXIS_SPLIT = {"PP4": "test", "TR": "test", "EE4": "test", "HD4": "test"}
FINISHED = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "DONE", "REMOVED"}


def sh(cmd, check=False):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit("collect_run: %s failed: %s" % (" ".join(cmd[:4]), (r.stderr or r.stdout).strip()[-400:]))
    return r


def axis_recipe(axis, cat):
    if axis == "XV3":
        return "recipes/catalog_v3_shared_heldout.yaml"
    if axis == "XV4":
        return "recipes/catalog_v4_shared_heldout.yaml"
    if axis == "PP":
        return "recipes/catalog_v3_pp_test.yaml"
    if axis == "PP4":
        return "recipes/catalog_v4_pp_test.yaml"
    if axis == "EE4":
        return "recipes/catalog_v4_ee_test.yaml"
    if axis == "HD4":
        return "recipes/catalog_v4_heldout_default_test.yaml"
    if axis == "TR":   # the in-distribution read of the job's axis TR: the controls' training recipe (other arms' differ)
        return "recipes/catalog_%s_train.yaml" % cat
    return "recipes/catalog_%s_heldout_%s.yaml" % (cat, axis)


def expected(axis, cat, prefix):
    import yaml
    f = os.path.join(HERE, axis_recipe(axis, cat))
    if not os.path.exists(f):
        return None
    names = [p["name"] for p in yaml.safe_load(open(f))["processes"]]
    if axis == "EE4":
        names = [n for n in names if n not in EE4_DROPPED]
    return sorted(n for n in names if not prefix or n.startswith(prefix))


def finite(x):
    return not isinstance(x, float) or math.isfinite(x)


def parse_log(text):
    """the lines that matter, and what they say"""
    p = {"axes_checked": [], "skipped": [], "done": None, "tracebacks": [], "nonfinite_steps": 0, "diverged": None,
         "zs": {}, "parent": None, "smoke": False}
    lines = text.splitlines()
    for i, l in enumerate(lines):
        m = re.match(r"^(\S+) ZERO_SHOT (\{.*)$", l)
        if m:
            try:
                rec = json.loads(m.group(2))
            except ValueError:
                continue
            ax = p["zs"].setdefault(m.group(1), {"ok": set(), "err": {}, "nonfinite": set()})
            if "error" not in rec:   # the role scored (records before --split were all val)
                ax.setdefault("splits", set()).add(rec.get("split", "val"))
            name = rec.get("probe") or os.path.basename(str(rec.get("recipe", "")))[:-len(".yaml")]
            p["parent"] = p["parent"] or rec.get("parent")
            st = pool_physics.zs_status(rec)
            if st == "unchecked":
                ax.setdefault("physics_unchecked", set()).add(name)
            elif st == "allowed":
                ax.setdefault("physics_allowed", {})[name] = (rec.get("physics") or {}).get("reason", "")
            if "error" in rec:
                ax["err"][name] = rec["error"][:120]
            elif not all(finite(v) for v in rec.values()):
                ax["nonfinite"].add(name)
            else:
                ax["ok"].add(name)
            continue
        if l.startswith("V3_RECIPE_CHECK ") and "FAIL" not in l:
            p["axes_checked"].append(l.split()[1].rstrip(":"))
        elif l.startswith("V3_RECIPE_CHECK FAIL") or l.startswith("V3_SIDECARS missing"):
            p["tracebacks"].append(l[:200])
        elif l.startswith("V3_AXIS_SKIPPED"):
            p["skipped"].append(l[:120])
        elif l.startswith("V3_SHORT_DONE"):
            p["done"] = l.split()[1] if len(l.split()) > 1 else "?"
        elif l.startswith("Traceback"):
            last = next((x for x in lines[i + 1:i + 200] if x and not x.startswith((" ", "Traceback"))), "")
            p["tracebacks"].append(last[:200])
        elif "non-finite gradient norm" in l:
            p["nonfinite_steps"] += 1
        elif "Training diverged" in l:
            p["diverged"] = l[l.find("Training diverged"):][:160]
        if p["parent"] is None:
            m = re.search(r"run_dir=(\S+)", l) or re.search(r'"run": "([^"]+)"', l)
            if m:
                p["parent"] = m.group(1)
        if "zz_v3_smoke" in l or "zz_v4_smoke" in l:
            p["smoke"] = True
    return p


def fetch(site, item, pred, files_by_axis, signed_axes, result_json):
    """one site run: the result.json as one line, then the selected files as tar | base64"""
    parts = ["set -o pipefail; echo __RESULT__"]
    if result_json:
        parts.append("python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))' %s 2>/dev/null "
                     "|| echo null" % shlex.quote(result_json))
    else:
        parts.append("echo null")
    if files_by_axis or signed_axes:
        # a tuple prefix (PP4: pp13v4_ or pp13_) globs on its common stem; the kept set is still the recipe's names
        globs = " ".join("%s/%s*.npz" % (shlex.quote(a), shlex.quote(pfx if isinstance(pfx, str) else
                                                                     os.path.commonprefix(list(pfx))))
                         for a, pfx in files_by_axis)
        signed = " ".join("%s_signed.txt" % shlex.quote(a) for a in signed_axes)
        parts.append('cd "%s" && echo __B64__ && { shopt -s nullglob; ls -d %s %s 2>/dev/null; } | '
                     "tar czf - -T - | base64 -w0; echo" % (pred, globs, signed))
    r = sh(SITE + ["run", "--item", item, site, "FA", "--", "; ".join(parts)])
    # site run echoes the command first, markers included: a marker counts only as a line of its own
    out = r.stdout.splitlines()
    after = lambda mark: next((out[i + 1].strip() for i, l in enumerate(out[:-1]) if l.strip() == mark), "")
    try:
        res = json.loads(after("__RESULT__") or "null")
    except ValueError:
        res = None
    b = after("__B64__")
    blob = base64.b64decode(b) if b else b""
    return res, blob, (r.stderr.strip()[-300:] if r.returncode else "")


def extract(blob, into):
    """members that are plain files under <axis>/ or <axis>_signed.txt only"""
    os.makedirs(into, exist_ok=True)
    if not blob:
        return
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
        for m in t.getmembers():
            if not m.isfile() or m.name.startswith("/") or ".." in m.name.split("/") or m.name.count("/") > 1:
                continue
            t.extract(m, into, filter="data")


def open_events(run):
    r = sh(SITE + ["events", "--project", "FA"])
    return [l.split()[0] for l in r.stdout.splitlines() if run in l and re.match(r"^\s*E\d+", l)]


def collect(run, a):
    off, notes = [], []
    r = sh(SITE + ["runs", "--id", run])
    row = r.stdout.split()
    if r.returncode or len(row) < 4:
        return ["not registered"], "", None
    site, state = row[2], row[3]
    if state not in FINISHED:
        return ["state %s, not finished" % state], "", None
    if state != "COMPLETED":
        off.append("state " + state)
    log = sh(SITE + ["logs", "-n", "100000000", run]).stdout
    p = parse_log(log)
    arm = a.arm or p["done"] or (os.path.basename(p["parent"].rstrip("/")) if p["parent"] else None)
    m = re.search(r"runs/catalog_(v\d+)_short/", p["parent"] or "")
    cat = m.group(1) if m else (arm.split("_", 1)[0] if arm and re.match(r"v\d+_", arm) else "v3")
    if p["done"] is None and row[-1].endswith("job_catalog_v3_short.sh"):
        off.append("no V3_SHORT_DONE marker")
    off += ["Traceback: " + t for t in p["tracebacks"]]
    off += p["skipped"]
    if p["diverged"]:
        off.append(p["diverged"])
    if p["nonfinite_steps"]:
        off.append("%d non-finite gradient steps skipped" % p["nonfinite_steps"])
    axes = sorted(set(p["axes_checked"]) | set(p["zs"]))
    if a.axes:
        axes = [x for x in a.axes.split(",") if x]
    plan_axes, lines = [], []
    for ax in axes:
        pfx = (CROSS_PREFIX.get(ax, cat + "_") if a.prefix_filter == "auto"
               else "" if a.prefix_filter == "none" else a.prefix_filter)
        exp = expected(ax, cat, pfx)
        z = p["zs"].get(ax, {"ok": set(), "err": {}, "nonfinite": set()})
        if exp is None:
            off.append("%s: no recipe %s" % (ax, axis_recipe(ax, cat)))
            continue
        miss = sorted(set(exp) - z["ok"])
        want = AXIS_SPLIT.get(ax, "val")
        if z.get("splits") and z["splits"] != {want}:
            off.append("%s: scored on %s, the axis is read on %s" % (ax, "+".join(sorted(z["splits"])), want))
        lines.append("%s %d/%d%s%s" % (ax, len(set(exp) & z["ok"]), len(exp),
                                       " err %d" % len(z["err"]) if z["err"] else "",
                                       " nonfinite %d" % len(z["nonfinite"]) if z["nonfinite"] else ""))
        if miss and not p["smoke"]:
            off.append("%s: %d expected probes without a prediction (%s)" % (ax, len(miss), ", ".join(miss[:5])))
        if z["err"]:
            k, v = next(iter(z["err"].items()))
            off.append("%s: %d probes errored (%s: %s)" % (ax, len(z["err"]), k, v))
        if z["nonfinite"]:
            off.append("%s: %d probes non-finite (%s)" % (ax, len(z["nonfinite"]), ", ".join(sorted(z["nonfinite"])[:5])))
        unchecked = sorted(z.get("physics_unchecked", ()))
        if unchecked and not a.allow_physics_unchecked:
            off.append("%s: %d probes read without the physics check (%s); --allow-physics-unchecked \"<reason>\" to "
                       "collect them" % (ax, len(unchecked), ", ".join(unchecked[:5])))
        elif unchecked:
            notes.append("%s: %d probes read without the physics check, collected: %s"
                         % (ax, len(unchecked), a.allow_physics_unchecked))
        allowed = z.get("physics_allowed", {})
        if allowed:
            notes.append("%s: %d probes read with a physics mismatch on purpose (%s)"
                         % (ax, len(allowed), "; ".join(sorted(set(allowed.values())))[:200]))
        plan_axes.append((ax, pfx, set(exp)))
    dest = a.dest or os.path.join("analysis", "catalog_v3", "data", "short_reads", arm or run)
    res, kept = None, {}
    do_fetch = plan_axes and not p["smoke"] and arm
    rj = os.path.join(p["parent"], "result.json") if p["parent"] else None
    if do_fetch or rj:
        pred = "$SCRATCH/zz_%s_short_%s" % (cat, arm)
        res, blob, err = fetch(site, a.entry, pred, [(x, pf) for x, pf, _ in plan_axes] if do_fetch else [],
                               [x for x, _, _ in plan_axes] if do_fetch else [], rj)
        if err:
            off.append("fetch: " + err)
        if do_fetch:
            base = os.path.join(HERE, os.path.dirname(dest.rstrip("/")) or ".")
            os.makedirs(base, exist_ok=True)
            tmp = tempfile.mkdtemp(prefix=".collect_%s_" % run, dir=base)
            try:
                extract(blob, tmp)
                for ax, pfx, exp in plan_axes:
                    got = {f[:-4] for f in os.listdir(os.path.join(tmp, ax))} if os.path.isdir(os.path.join(tmp, ax)) else set()
                    keep = got & exp
                    kept[ax] = (len(keep), len(got - exp))
                    if exp - got:
                        off.append("%s: %d expected probes without a .npz on the site" % (ax, len(exp - got)))
                    sf = os.path.join(tmp, ax + "_signed.txt")
                    signed = []
                    if os.path.exists(sf):
                        for l in open(sf):
                            m = re.search(r'"pool": "([^"]+)"', l)
                            if not m or m.group(1).startswith(pfx):
                                signed.append(l)
                    if a.dry_run:
                        continue
                    d = os.path.join(HERE, dest, ax)
                    os.makedirs(d, exist_ok=True)
                    for n in keep:
                        shutil.copy2(os.path.join(tmp, ax, n + ".npz"), os.path.join(d, n + ".npz"))
                    if signed:
                        open(os.path.join(HERE, dest, ax + "_signed.txt"), "w").writelines(signed)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    best = ""
    if isinstance(res, dict) and "val_loss" in res:
        best = " best val_loss_no_reg %.4g at step %s" % (res["val_loss"], res.get("best_step"))
        if not all(finite(v) for v in [res["val_loss"]] + list(res.get("val_curve") or [])):
            off.append("non-finite validation loss in result.json")
    fetched = " | home " + ", ".join("%s %d%s" % (x, k, " (dropped %d foreign)" % f if f else "")
                                      for x, (k, f) in kept.items()) if kept else ""
    summary = "%s %s (%s, arm %s, %s): %s%s%s%s" % (run, state, site, arm, cat, "; ".join(lines) or "no zero-shot reads",
                                                    best, fetched, (" | physics: " + "; ".join(notes)) if notes else "")
    return off, summary, dest if kept else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--entry", required=True, help="the plan entry (and site work item) the runs belong to")
    ap.add_argument("--dest", help="home folder of the reads (default analysis/catalog_v3/data/short_reads/<arm>)")
    ap.add_argument("--axes", help="comma list; default every axis the log read")
    ap.add_argument("--prefix-filter", default="auto",
                    help="auto (the run's catalog per axis), none, or one prefix for every axis")
    ap.add_argument("--arm", help="the arm, when the log does not name it (no done marker)")
    ap.add_argument("--dry-run", action="store_true", help="read and count only: nothing written, acked or recorded")
    ap.add_argument("--allow-physics-unchecked", metavar="REASON",
                    help="collect probes read before the physics check existed (the reason goes into the ack note)")
    a = ap.parse_args()
    bad = 0
    for run in a.runs:
        off, summary, dest = collect(run, a)
        print(summary or run)
        for o in off:
            print("  OFF " + o)
        if off:
            bad = 1
            print("  not acked, not recorded")
            continue
        evs = open_events(run)
        if a.dry_run:
            print("  dry run: would ack %s and record --run %s%s" % (",".join(evs) or "no open event", run,
                                                                    " --path " + dest if dest else ""))
            continue
        for e in evs:
            sh(SITE + ["ack", e, "--note", "collected by tools/collect_run.py: " + summary[:600]], check=True)
        sh(PLAN + ["evidence", a.entry, "--run", run] + (["--path", dest] if dest else []), check=True)
        print("  acked %s; evidence recorded on %s" % (",".join(evs) or "no open event", a.entry))
    sys.exit(bad)


if __name__ == "__main__":
    main()
