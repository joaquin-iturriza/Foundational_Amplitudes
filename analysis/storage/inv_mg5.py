"""Read-only inventory of the MadGraph work dir on a site: per top-level entry, bytes (allocated and apparent) and file
counts split into what computing amplitudes again needs and the build byproducts. One JSON line per entry on stdout.
    python analysis/storage/inv_mg5.py <mg5_dir> [glob ...]
Categories, by path inside a process dir:
  need_runtime  driver_v2, libME_*.so, matrix2py*.so, MadLoop5_resources/ (what the C++ driver / MadLoop call at run time)
  need_card     Cards/param_card.dat
  rebuild_src   CPPProcess.cc/.h, src/*.cc/.h, wrapper_*.cc, driver.cpp, SubProcesses/P*/*.f/*.inc, Cards/* (to recompile)
  objects       *.o, *.a, *.mod, other *.so, compiled executables (check, check_sa, gensym, madevent)
  html          HTML/, *.html, *.ps, *.eps, *.jpg, *.png, *.gif
  events        Events/, *.lhe*, *_events dirs' run output
  source_tree   Source/ (MadEvent Fortran tree)
  madevent      bin/, SubProcesses/P*/G*/ channel dirs, ajob*, *.log, madevent leftovers
  other
A whole <proc>_events dir is the madevent output, used only by the LHE path (the catalogs sample phase space
themselves); its categories are reported as for any dir and flagged events_dir.
"""
import fnmatch, json, os, re, sys, time

root = sys.argv[1]
globs = sys.argv[2:] or ["*"]
EXE = {"check", "check_sa", "gensym", "madevent", "madevent_forhel", "check_poles", "ProcessCheck"}


def cat(rel, name):
    parts = rel.split("/")
    if name == "driver_v2" or re.match(r"libME_.*\.so$", name) or re.match(r"matrix2py.*\.so$", name):
        return "need_runtime"
    if "MadLoop5_resources" in parts:
        return "need_runtime"
    if rel == "Cards/param_card.dat":
        return "need_card"
    if name.endswith((".o", ".a", ".mod")) or name in EXE:
        return "objects"
    if "HTML" in parts or name.endswith((".html", ".ps", ".eps", ".jpg", ".png", ".gif")):
        return "html"
    if "Events" in parts or ".lhe" in name:
        return "events"
    if parts[0] == "Source":
        return "source_tree"
    if parts[0] == "bin" or name.startswith("ajob") or name.endswith(".log") or \
            (len(parts) >= 4 and parts[0] == "SubProcesses" and re.match(r"G[0-9.]+$", parts[2])):
        return "madevent"
    if name.endswith(".so"):
        return "objects"
    if name.startswith(("CPPProcess.", "wrapper_")) or name == "driver.cpp" or parts[0] in ("src", "Cards") or \
            (parts[0] == "SubProcesses" and name.endswith((".f", ".inc", ".h", ".cc"))):
        return "rebuild_src"
    return "other"


t0 = time.time()
for e in sorted(os.listdir(root)):
    if not any(fnmatch.fnmatch(e, g) for g in globs):
        continue
    p = os.path.join(root, e)
    rec = {"name": e, "alloc": {}, "apparent": {}, "files": {}}
    if os.path.isdir(p) and not os.path.islink(p):
        for dp, dns, fns in os.walk(p):
            for f in fns:
                fp = os.path.join(dp, f)
                try:
                    st = os.lstat(fp)
                except OSError:
                    continue
                c = cat(os.path.relpath(fp, p), f)
                rec["alloc"][c] = rec["alloc"].get(c, 0) + st.st_blocks * 512
                rec["apparent"][c] = rec["apparent"].get(c, 0) + st.st_size
                rec["files"][c] = rec["files"].get(c, 0) + 1
            rec["dirs"] = rec.get("dirs", 0) + len(dns)
        rec["mtime"] = int(os.lstat(p).st_mtime)
    else:
        st = os.lstat(p)
        rec["alloc"]["other"] = st.st_blocks * 512
        rec["apparent"]["other"] = st.st_size
        rec["files"]["other"] = 1
    print(json.dumps(rec), flush=True)
print(json.dumps({"_done": globs, "secs": round(time.time() - t0)}), flush=True)
