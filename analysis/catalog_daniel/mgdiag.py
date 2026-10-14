"""Per catalog process, MG5 (local ~/mg5amcnlo) diagram content under a given order suffix."""
import sys, json, os, time, logging
MG = os.path.expanduser("~/mg5amcnlo")
sys.path.insert(0, MG)
cat_path, out_path, suffix = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
logging.disable(logging.CRITICAL)
import madgraph.interface.master_interface as mi
cmd = mi.MasterCmd()
cmd.exec_cmd("set automatic_html_opening False", printcmd=False)
cmd.exec_cmd("import model sm", printcmd=False)
model = cmd._curr_model
cat = json.load(open(cat_path))
res = {}
if os.path.exists(out_path):
    res = json.load(open(out_path))
t0 = time.time()
for i, x in enumerate(cat):
    f = x["folder"]
    if f in res: continue
    try:
        cmd.exec_cmd(f"generate {x['madgraph_process']} {suffix}".strip(), printcmd=False)
        amps = cmd._curr_amps
        diags = []; orders = set(); verts = set(); props = set()
        for amp in amps:
            for d in amp.get("diagrams"):
                orders.add(tuple(sorted(d.get("orders").items())))
                vs = []
                for v in d.get("vertices"):
                    if v.get("id") == 0: continue
                    inter = model.get("interaction_dict")[v.get("id")]
                    pd = tuple(sorted(p.get_pdg_code() for p in inter.get("particles")))
                    verts.add(pd); vs.append(list(pd))
                    # last leg of a vertex (except final vertex) is the propagator
                    props.add(abs(v.get("legs")[-1].get("id")))
                diags.append(vs)
        res[f] = dict(n=len(diags), orders=[dict(o) for o in orders], vertices=sorted(map(list, verts)), props=sorted(props))
    except Exception as e:
        res[f] = dict(error=str(e)[:200])
    if i % 50 == 0:
        json.dump(res, open(out_path, "w")); print(i, f, res[f].get("n"), round(time.time() - t0), flush=True)
json.dump(res, open(out_path, "w"))
print("done", len(res), round(time.time() - t0))
