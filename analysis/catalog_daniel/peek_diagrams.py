"""Print the structure of one catalog diagram sidecar (data/diagrams/dc_<p>.diagrams.json), to read its vertices."""
import json, sys
d = json.load(open(sys.argv[1]))
sp = d["subprocesses"][0]
print("n_diagrams", sp["n_diagrams"], "external", sp["external"])
dg = sp["diagrams"][0]
print("diagram keys", list(dg) if isinstance(dg, dict) else type(dg))
print(json.dumps(dg)[:2500])
