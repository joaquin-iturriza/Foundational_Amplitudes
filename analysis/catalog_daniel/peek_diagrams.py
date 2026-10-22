"""Print the structure of one catalog diagram sidecar (data/diagrams/dc_<p>.diagrams.json), to read its vertices."""
import json, sys
d = json.load(open(sys.argv[1]))
print({k: (v if k != "subprocesses" else "...") for k, v in d.items()})
sp = d["subprocesses"][0]
print("subprocess keys:", list(sp))
for k, v in sp.items():
    if isinstance(v, list) and v and isinstance(v[0], dict):
        print(k, len(v), "item keys", list(v[0]))
        x = v[0]
        for kk, vv in x.items():
            print("  ", kk, json.dumps(vv)[:600])
        break
