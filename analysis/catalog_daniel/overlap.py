import json, sys
sys.path.insert(0, sys.argv[1])
import classify as C
S = sys.argv[2]
ours = json.load(open(f"{S}/our_processes.json"))
dan = json.load(open(f"{S}/daniel_catalog_classes.json"))
PDG = {1:"d",2:"u",3:"s",4:"c",5:"b",6:"t",11:"e-",13:"mu-",15:"ta-",12:"ve",14:"vm",16:"vt",21:"g",22:"a",23:"z",24:"w+",25:"h"}
def name(i):
    if i in PDG: return PDG[i]
    n = PDG[-i]
    return C.ANTI[n]
ex = {}; wk = {}
for c in dan["processes"]:
    ex.setdefault(c["exact_key"], []).append(c); wk.setdefault(c["weak_key"], []).append(c)
def keys(ini, fin):
    ke, _ = C.key_exact(ini, fin)
    wi, wf, _ = C.weak_map(list(ini), list(fin)); kw, _ = C.key_exact(wi, wf)
    f = lambda k: " ".join(k[0]) + " > " + " ".join(k[1])
    return f(ke), f(kw)
rows = {}
for n, v in ours["procs"].items():
    if not v["tree"] or v["pdg"] is None: continue
    p = [name(i) for i in v["pdg"]]
    ini, fin = p[:2], p[2:]
    ke, kw = keys(ini, fin)
    rows[n] = dict(proc=" ".join(ini) + " > " + " ".join(fin), exact=ke, weak=kw,
                   dan_exact=[c["process"] for c in ex.get(ke, [])][:4],
                   dan_weak=[c["process"] for c in wk.get(kw, [])][:4])
def report(label, names):
    names = [n for n in names if n in rows]
    hitE = {rows[n]["exact"] for n in names if rows[n]["dan_exact"]}
    hitW = {rows[n]["weak"] for n in names if rows[n]["dan_weak"]}
    print(f"{label}: {len(names)} tree entries; {sum(bool(rows[n]['dan_exact']) for n in names)} entries in a Daniel exact class "
          f"-> {len(hitE)}/{len(ex)} Daniel exact classes covered; weak: {len(hitW)}/{len(wk)}")
    return hitE, hitW
per = ours["per_recipe"]
strip = lambda ns: [n.split("__")[0] for n in ns]
report("catalog_v2_train", strip(per["recipes/catalog_v2_train.yaml"]))
report("catalog_v2_holdout", strip(per["recipes/catalog_v2_holdout.yaml"]))
report("catalog_v2 train+holdout", strip(per["recipes/catalog_v2_train.yaml"] + per["recipes/catalog_v2_holdout.yaml"]))
allE, allW = report("all recipes", list(rows))
pp = [n for n in rows if not any(x in rows[n]["proc"].split(">")[0] for x in ["e-","e+"])]
print("our pp-parton-initial tree processes:", len(pp))
hits = sorted((n, rows[n]["proc"], rows[n]["dan_exact"][:2]) for n in rows if rows[n]["dan_exact"])
json.dump(dict(rows=rows, covered_exact_classes=sorted(allE), covered_weak_classes=sorted(allW)),
          open(f"{S}/overlap_ours_vs_daniel.json", "w"), indent=1)
for h in hits: print("  ", h)
