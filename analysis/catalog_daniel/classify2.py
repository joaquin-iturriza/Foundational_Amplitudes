"""Classes under a given MG diagram set (default orders = what Daniel's events carry, or all
orders = what his diagram JSONs/counts carry). Adds a diagram-informed flavour abstraction to
classify.py: massless quark flavours are interchangeable as far as the vertices present allow
  - no EW vertex on a light quark (pure QCD on quark lines): flavour-blind (keep which legs share flavour)
  - EW vertices on light quarks but no W: only the up/down type matters (u==c, d==s individually)
  - W on light quarks: Lagrangian generation swap only (already in classify.orbit)."""
import json, sys, itertools
from collections import defaultdict, Counter
sys.path.insert(0, sys.argv[1]); import classify as C
cat = json.load(open(sys.argv[2])); split = json.load(open(sys.argv[3]))
diag = json.load(open(sys.argv[4])); out_path = sys.argv[5]
LIGHTPDG = {1, 2, 3, 4}

def flags(folder, light=LIGHTPDG):
    v = diag[folder]["vertices"]
    LIGHTPDG_ = light
    qew = any(any(abs(p) in LIGHTPDG_ for p in t) and any(abs(p) in (22, 23, 24, 25) for p in t) for t in v)
    qw = any(any(abs(p) in LIGHTPDG_ for p in t) and any(abs(p) == 24 for p in t) for t in v)
    return ("qcd" if not qew else ("ew_noW" if not qw else "W"))

TYPE = {"u": "U", "c": "U", "d": "D", "s": "D"}
def abstract(ini, fin, mode):
    if mode == "W":
        return C.form(ini, fin)
    fl = sorted({p.rstrip("~") for p in ini + fin if p.rstrip("~") in TYPE})
    best = None
    for perm in itertools.permutations(range(1, len(fl) + 1)):
        lab = {}
        for f, k in zip(fl, perm):
            lab[f] = ("Q" if mode == "qcd" else TYPE[f]) + str(k)
        def m(p):
            b = p.rstrip("~")
            return lab[b] + ("~" if p.endswith("~") else "") if b in lab else p
        k = C.form([m(p) for p in ini], [m(p) for p in fin])
        best = k if best is None or k < best else best
    return best

def key(ini, fin, mode, use_cp=True):
    best = None
    for (i2, f2) in C.orbit(list(ini), list(fin), use_cp):
        i3, f3, _ = C.accidental(list(i2), list(f2))
        for (i4, f4) in C.orbit(i3, f3, use_cp):
            k = abstract(list(i4), list(f4), mode)
            best = k if best is None or k < best else best
    return " ".join(best[0]) + " > " + " ".join(best[1])

test = {f: (k, g["group"]) for k, gs in split.items() for g in gs for f in g["folders"]}
recs = []
for x in cat:
    ini, fin = x["incoming"], x["outgoing"]; mode = flags(x["folder"])
    wi, wf, tw = C.weak_map(list(ini), list(fin))
    recs.append(dict(folder=x["folder"], process=x["madgraph_process"], n_out=len(fin), mode=mode,
                     n_diagrams=diag[x["folder"]]["n"], orders=diag[x["folder"]]["orders"],
                     exact_key=key(ini, fin, mode), nocp_key=key(ini, fin, mode, False),
                     weak_key=key(wi, wf, flags(x["folder"], LIGHTPDG | ({5} if any(t.startswith("b->d") for t in tw) else set()))), weak_tags=tw,
                     split=test.get(x["folder"], ("train",))[0], test_group=test.get(x["folder"], (None, None))[1]))
res = {}
for kn in ["exact_key", "nocp_key", "weak_key"]:
    d = defaultdict(list)
    for r in recs: d[r[kn]].append(r)
    res[kn] = d
    print(f"{kn}: {len(d)} classes; sizes {dict(sorted(Counter(len(v) for v in d.values()).items()))}; "
          f"by n_out {dict(sorted(Counter(v[0]['n_out'] for v in d.values()).items()))}")
print("modes", Counter(r["mode"] for r in recs))
tr = {r["exact_key"] for r in recs if r["split"] == "train"}; trw = {r["weak_key"] for r in recs if r["split"] == "train"}
pg = defaultdict(lambda: [0, 0, 0])
for r in recs:
    if r["split"] == "train": continue
    g = pg[(r["split"], r["test_group"])]; g[0] += 1; g[1] += r["exact_key"] in tr; g[2] += r["weak_key"] in trw
for (k, g), v in pg.items(): print(f"  {k:13s} {g[:62]:62s} n={v[0]:3d} exact-twin-in-train={v[1]:3d} weak={v[2]:3d}")
print("  total", [sum(v[i] for v in pg.values()) for i in range(3)])
E = res["exact_key"]
json.dump(dict(n_classes={k: len(v) for k, v in res.items()},
               exact_classes=[dict(key=k, n=len(v), n_out=v[0]["n_out"], mode=v[0]["mode"],
                                   n_train=sum(r["split"] == "train" for r in v),
                                   members=[r["process"] for r in v], folders=[r["folder"] for r in v])
                              for k, v in sorted(E.items(), key=lambda kv: -len(kv[1]))],
               weak_classes=[dict(key=k, n=len(v), n_exact_classes=len({r["exact_key"] for r in v}),
                                  tags=sorted({t for r in v for t in r["weak_tags"]}), members=[r["process"] for r in v])
                             for k, v in sorted(res["weak_key"].items(), key=lambda kv: -len(kv[1]))
                             if len({r["exact_key"] for r in v}) > 1],
               processes=recs), open(out_path, "w"), indent=1)
print("wrote", out_path)
