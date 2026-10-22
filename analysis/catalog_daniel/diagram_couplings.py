"""Per catalog process, the coupling product of each tree diagram, from its sidecar (data/diagrams/dc_<p>.diagrams.json),
for the statistics map g (stats_net.py). Why (2026-10-22): counting features cannot carry a suppression to a class with
no relative in training (q q~ > b b~ h sits 8 below g's mean: its Higgs attaches through the b Yukawa coupling). Each
vertex's coupling is read off the particles on its lines with the MG5 sm model's values (alpha = 1/132.5, alpha_s =
0.118, sin^2 theta_W = 0.222, v = 246 GeV; the generator's masses, particle_ids.py). Per diagram: the sum over vertices
of log c^2 (dimensionful couplings made dimensionless with v: H W W = g m_W / v and the like), and its massive internal lines (t, W, Z, H).
Writes data/diagram_couplings.json {process: {"logc2": [...per diagram], "massive_internal": [...], "unmatched": n}}.
CPU, on the site holding data/diagrams (CC):
    python analysis/catalog_daniel/diagram_couplings.py
"""
import glob, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
E = math.sqrt(4 * math.pi / 132.5)
GS = math.sqrt(4 * math.pi * 0.118)
SW2 = 0.222
SW, CW = math.sqrt(SW2), math.sqrt(1 - SW2)
G = E / SW
GZ = G / CW
V = 246.0
M = {5: 4.7, 6: 172.5, 15: 1.777, 23: 91.19, 24: 80.4, 25: 125.0}
Q = {1: -1 / 3, 2: 2 / 3, 3: -1 / 3, 4: 2 / 3, 5: -1 / 3, 6: 2 / 3, 11: -1, 12: 0, 13: -1, 14: 0, 15: -1, 16: 0}
T3 = {1: -.5, 2: .5, 3: -.5, 4: .5, 5: -.5, 6: .5, 11: -.5, 12: .5, 13: -.5, 14: .5, 15: -.5, 16: .5}
FERM = set(Q)


def vertex_coupling(pdgs, orders):
    """|coupling| of one vertex from the |pdg| of its lines; None if not recognised."""
    a = sorted(abs(p) for p in pdgs)
    f = [p for p in a if p in FERM]
    b = [p for p in a if p not in FERM]
    if len(f) == 2 and len(b) == 1:
        x, bos = f[0], b[0]
        if bos == 21:
            return GS
        if bos == 22:
            return E * abs(Q[x])
        if bos == 23:
            gv, ga = T3[x] / 2 - Q[x] * SW2, T3[x] / 2
            return GZ * math.sqrt(gv ** 2 + ga ** 2)
        if bos == 24:
            return G / math.sqrt(2)
        if bos == 25:
            return M.get(x, 0.0) / V
    if not f:
        n = {p: b.count(p) for p in set(b)}
        if b == [21, 21, 21]:
            return GS
        if b == [21, 21, 21, 21]:
            return GS ** 2
        if n.get(25) == 1 and n.get(24) == 2:
            return G * M[24] / V
        if n.get(25) == 1 and n.get(23) == 2:
            return GZ * M[23] / V
        if b == [25, 25, 25]:
            return 3 * M[25] ** 2 / V ** 2
        if n.get(25) == 2 and n.get(24) == 2:
            return G ** 2 / 2
        if n.get(25) == 2 and n.get(23) == 2:
            return GZ ** 2 / 2
        if b == [25, 25, 25, 25]:
            return 3 * M[25] ** 2 / V ** 2
        if n.get(24) == 2 and len(b) == 3:
            return E if 22 in n else G * CW
        if n.get(24) == 2 and len(b) == 4:
            return G ** 2 if n.get(24) == 4 or (n.get(24) == 2 and 23 in n) else E ** 2
        if n.get(24) == 4:
            return G ** 2
    return None


def main():
    out = {}
    for fn in sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "diagrams", "dc_*.diagrams.json"))):
        name = os.path.basename(fn).split(".")[0]
        d = json.load(open(fn))
        rec = {"logc2": [], "massive_internal": [], "unmatched": 0}
        for sp in d["subprocesses"]:
            for dg in sp["diagrams"]:
                nodes, edges = dg["nodes"], dg["edges"]
                lines = {i: [] for i, n in enumerate(nodes) if n["kind"] == "vertex"}
                for e in edges:
                    for k in (e["u"], e["v"]):
                        if k in lines:
                            lines[k].append(e["pdg"])
                s = 0.0
                for i, pd in lines.items():
                    c = vertex_coupling(pd, nodes[i].get("orders", {}))
                    if c is None:
                        rec["unmatched"] += 1
                        c = GS if "QCD" in nodes[i].get("orders", {}) else E
                    s += 2 * math.log(max(c, 1e-12))
                rec["logc2"].append(s)
                rec["massive_internal"].append(sum(abs(e["pdg"]) in (6, 23, 24, 25) for e in edges if not e["external"]))
        out[name] = rec
    json.dump(out, open(os.path.join(HERE, "data", "diagram_couplings.json"), "w"))
    um = sum(r["unmatched"] for r in out.values())
    print(f"{len(out)} processes, {sum(len(r['logc2']) for r in out.values())} diagrams, {um} unmatched vertices")
    for n in ("dc_uux_bbxh", "dc_uux_ddx", "dc_gd_da", "dc_uux_btxwp"):
        if n in out:
            print(n, [round(x, 2) for x in out[n]["logc2"]], out[n]["massive_internal"])


if __name__ == "__main__":
    main()
