#!/usr/bin/env python
"""Which width MadGraph actually used where a param card's DECAY line was ignored (plan catalog_coverage).

MadGraph's card readers skip a DECAY line that does not start in column 1 (the C++ standalone's read_slha.cc tests
line.find("decay ") == 0; MadLoop's lha_read.f the same), and the parameter keeps the default compiled into the
standalone at output time. Our patchers wrote those lines with a leading blank until 7b00eb21. This reads the audit of
the production standalones (one line per standalone: <dir> <pdg>=<width written> ..., the widths on ignored lines) and,
per standalone, the compiled default of each such width:
  C++ (trees)        src/Parameters_<model>.cc   slha.get_block_entry("decay", <pdg>, <default>)
  Fortran (MadLoop)  Source/MODEL/param_read.inc  LHA_GET_REAL(...,'mdl_W<x>',MDL_W<x>,<default>), the name from the
                     card's "DECAY <pdg> <w> # W<x>" comment
and groups them by pool family (catalog, tree or one-loop, layer, model): width intended, width used, standalones.
Read-only; runs on the site that holds the standalones:
    python analysis/catalog_v4/card_width_audit.py <decay_audit.txt> [--mg-dir $WORK/mg5amcnlo] > out.json
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

LAYER_PDG = {6: "WT", 23: "WZ", 24: "WW", 25: "WH", 5000001: "WZp (Y1)"}


def defaults(d):
    """{pdg: compiled default width} and the model name of one standalone dir"""
    out, model = {}, None
    cc = glob.glob(os.path.join(d, "src", "Parameters_*.cc"))
    if cc:
        model = os.path.basename(cc[0])[len("Parameters_"):-3]
        for pdg, v in re.findall(r'get_block_entry\("decay",\s*(-?\d+),\s*([^)\s]+)\s*\)', open(cc[0]).read()):
            out[int(pdg)] = float(v)
        return out, model, "tree (C++)"
    pr = os.path.join(d, "Source", "MODEL", "param_read.inc")
    pc = os.path.join(d, "Cards", "param_card.dat")
    if os.path.exists(pr) and os.path.exists(pc):
        txt = open(pr).read().replace("\n     $ ", "")
        comp = {n.lower(): float(v.replace("D", "e").replace("d", "e"))
                for n, v in re.findall(r"LHA_GET_REAL\(NPARA,PARAM,VALUE,'mdl_(\w+)',MDL_\w+,([0-9.EeDd+-]+)\)", txt)}
        for line in open(pc):
            m = re.match(r"\s*DECAY\s+(-?\d+)\s+\S+\s*#\s*(\w+)", line, re.I)
            if m and m.group(2).lower() in comp:
                out[int(m.group(1))] = comp[m.group(2).lower()]
        pcard = os.path.join(d, "Cards", "proc_card_mg5.dat")
        if os.path.exists(pcard):
            m = re.search(r"^\s*import model\s+(\S+)", open(pcard).read(), re.M)
            model = m.group(1) if m else None
        return out, model, "one-loop (MadLoop)"
    return out, model, "unknown"


def family(name):
    """(catalog, layer) of a standalone dir name"""
    n = name[:-len("_standalone")] if name.endswith("_standalone") else name
    cat = n.split("_", 1)[0] if re.match(r"v\d_", n) else ("thdm" if n.startswith("thdm") else "other")
    m = re.search(r"__mt(\d+)", n)
    if m:
        layer = f"heavy quark m_Q={m.group(1)}"
    elif "_zp_" in n or n.startswith("zp_"):
        layer = "Z' (MY1)"
    elif cat == "thdm" or "thdm" in n or "_2hdm" in n:
        layer = "2HDM"
    else:
        layer = "SM"
    return cat, layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audit")
    ap.add_argument("--mg-dir", default=os.path.join(os.environ.get("WORK", ""), "mg5amcnlo"))
    a = ap.parse_args()
    groups, missing = defaultdict(int), []
    n_lines = 0
    for line in open(a.audit):
        parts = line.split()
        if not parts or not parts[0].endswith("_standalone"):
            continue
        n_lines += 1
        d = os.path.join(a.mg_dir, parts[0])
        dflt, model, kind = defaults(d)
        cat, layer = family(parts[0])
        for kv in parts[1:]:
            pdg, w = kv.split("=")
            pdg, w = int(pdg), float(w)
            used = dflt.get(pdg)
            if used is None:
                missing.append(f"{parts[0]} {pdg}")
            groups[(cat, kind, layer, model, pdg, w, used)] += 1
    rows = [dict(catalog=c, kind=k, layer=l, model=m, pdg=p, width=LAYER_PDG.get(p, str(p)), intended=w, used=u,
                 harmless=(u is not None and abs(u - w) <= 1e-6 * max(abs(w), 1e-12)), standalones=n)
            for (c, k, l, m, p, w, u), n in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][4],
                                                                                    x[0][5], str(x[0][3])))]
    json.dump({"audit": os.path.abspath(a.audit), "standalones_in_audit": n_lines,
               "default_not_found": missing[:50], "n_default_not_found": len(missing), "rows": rows},
              sys.stdout, indent=1)
    print()
    for r in rows:
        if not r["harmless"]:
            print(f"# {r['catalog']:5s} {r['kind']:19s} {r['layer']:22s} {str(r['model']):26s} {r['width']:9s} "
                  f"intended {r['intended']:.6g} used {r['used']} : {r['standalones']}", file=sys.stderr)


if __name__ == "__main__":
    main()
