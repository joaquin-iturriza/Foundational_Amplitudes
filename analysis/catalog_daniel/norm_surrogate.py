"""D83 exploration (the user, 2026-10-22: physics priors that give a process's statistics without true data): per catalog
process, on the momenta of the first --n events of its train pool (the phase-space generator's own sampling and cuts;
the labels are read only to score), cheap surrogates log|M~|^2 per event, and per process their mean, std and
correlation with the true log|M|^2 on the same events. Surrogates (constants: the leading diagram's coupling product
log C_p and MadGraph's IDEN and final colour sum, from the sidecar's external legs):
  dim     (4 - N)(L - log s0) + log C_p                                 (A's factor as launched; L = log s_hat)
  dimid   dim - log IDEN + sum log(final colour)                        (A + IDEN, colour)
  eik     dimid + sum over final gluons and photons i of log(w_i sum_{a<b} s_ab / (s_ai s_ib)), the pairs over the
          other coloured legs (gluon i, w = C_A) or charged legs (photon i, w = 1); s_xy = 2|p_x.p_y|
  pt      dimid + Parke-Taylor for pure-QCD processes (all gluons, or one quark pair and gluons):
          log(sum_{i<j} s_ij^4) + log(sum over orderings with leg 1 fixed of 1/prod of adjacent s); else eik
  prop    dimid with the dimension factor replaced by the diagrams: log sum_d C_d prod_{internal S} 1/|D_S|^k,
          D_S = s_S - M^2 + i M Gamma, k = 2 for a massive line, 1 for a massless one (the synthetic generator's rule),
          s_S from the leg set on one side of the internal edge (data/diagrams/dc_<p>.diagrams.json)
  propeik prop + the eik term
  catani  (one-loop pools) prop + log|C alpha_s/(2 pi) (-l^2 - 3 l - 8 + pi^2)|, l = log(mu^2 / s_hat), mu = M_Z,
          alpha_s = 0.118, C = C_F per quark line (the finite part of the Catani operator on a q qbar Born, a shape prior
          for the virtual correction relative to the Born); a signed pool is read as log|value|, its sign fraction kept
Login node, CPU only:
    python analysis/catalog_daniel/norm_surrogate.py <out.json> [--n 2000]
    python analysis/catalog_daniel/norm_surrogate.py <out.json> --pools uubar_Zgg=uubar_Zgg_96-1000GeV_train_smix_amplitudes.npy ...
"""
import argparse, itertools, json, math, os, sys, time
from collections import Counter

import numpy as np
import yaml

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--recipes", nargs="+", default=["recipes/catalog_daniel_train.yaml", "recipes/catalog_daniel_heldout.yaml"])
ap.add_argument("--cache", default=None)
ap.add_argument("--pools", nargs="*", default=None, help="name=pool file in the cache (the sidecar is data/diagrams/<name>.diagrams.json)")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
cache = a.cache or os.path.join(os.environ["SCRATCH"], "amp_data_cache")
files = sorted(os.listdir(cache))
dcoup = json.load(open(os.path.join(HERE, "data", "diagram_couplings.json")))
LS0 = math.log(289.0)
SPIN = {21: 2, 22: 2, 23: 3, 24: 3, 25: 1}
COLOR = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 21: 8}
CHARGE = {1: -1 / 3, 2: 2 / 3, 3: -1 / 3, 4: 2 / 3, 5: -1 / 3, 6: 2 / 3, 11: -1, 13: -1, 15: -1, 24: 1}
MW = {23: (91.188, 2.4414), 24: (80.419, 2.0476), 25: (125.0, 0.00407), 6: (173.0, 1.4915), 5: (4.7, 0.0)}
QUARKS = {1, 2, 3, 4, 5, 6}


def mdot(x, y):
    return x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(-1)


def iden(pin, pout):
    v = 1.0
    for p in pin:
        v *= SPIN.get(abs(p), 2) * COLOR.get(abs(p), 1)
    for c in Counter(pout).values():
        v *= math.factorial(c)
    return v


def edge_sides(diag, nleg):
    """per internal edge: the external leg numbers on one side (the tree split at that edge)"""
    nodes = diag["nodes"]; edges = diag["edges"]
    adj = {i: [] for i in range(len(nodes))}
    for k, e in enumerate(edges):
        adj[e["u"]].append((e["v"], k)); adj[e["v"]].append((e["u"], k))
    out = []
    for k, e in enumerate(edges):
        if e["external"]:
            continue
        seen, stack, legs = {e["u"]}, [e["u"]], []
        while stack:
            x = stack.pop()
            if nodes[x]["kind"] == "external":
                legs.append(nodes[x]["leg_number"])
            for y, kk in adj[x]:
                if kk != k and y not in seen:
                    seen.add(y); stack.append(y)
        out.append((abs(e["pdg"]), legs))
    return out


out, t0 = {}, time.time()
if a.pools:
    todo = [(kv.split("=")[0], [kv.split("=")[1]]) for kv in a.pools]
else:
    todo = []
    for rf in a.recipes:
        for p in yaml.safe_load(open(rf))["processes"]:
            todo.append((p["name"], [x for x in files if x.startswith(p["name"] + "_") and "_train_" in x
                                     and x.endswith("amplitudes.npy")]))
for name, g in todo:
        sc = os.path.join(ROOT, "data", "diagrams", name + ".diagrams.json")
        if not g or not os.path.exists(sc):
            continue
        x = np.asarray(np.load(os.path.join(cache, g[0]), mmap_mode="r")[: a.n], dtype=np.float64)
        n = (x.shape[1] - 1) // 5
        P = x[:, : 4 * n].reshape(len(x), n, 4)
        pdg = x[0, 4 * n: 5 * n].astype(int)
        sign_frac = float(np.mean(x[:, -1] > 0))
        y = np.log(np.abs(x[:, -1]))
        sd = json.load(open(sc))
        sub = [s for s in sd["subprocesses"] if [l["pdg"] for l in sorted(s["external"], key=lambda l: l["number"])]
               == list(pdg)]
        if not sub:
            out[name] = {"error": "no subprocess with the pool's leg order"}
            continue
        # couplings of that subprocess's diagrams: diagram_couplings.json lists every subprocess's in order
        i0 = 0
        for s in sd["subprocesses"]:
            if s is sub[0]:
                break
            i0 += len(s["diagrams"])
        if name in dcoup:
            logc = np.array(dcoup[name]["logc2"][i0: i0 + len(sub[0]["diagrams"])])
        else:                                        # outside the catalog: equal couplings (the level map absorbs them)
            logc = np.zeros(sum(1 for d in sub[0]["diagrams"] if d.get("diagram_type", "tree") == "tree") or 1)
        pin, pout = list(pdg[:2]), list(pdg[2:])
        const = float(logc.max()) - math.log(iden(pin, pout)) + sum(math.log(COLOR.get(abs(q), 1)) for q in pout)
        # all-outgoing momenta
        Q = P.copy(); Q[:, :2] *= -1
        s = mdot(P[:, 0] + P[:, 1], P[:, 0] + P[:, 1]); L = np.log(s)
        sij = lambda i, j: np.maximum(2 * np.abs(mdot(P[:, i], P[:, j])), 1e-12 * s)
        sur = {"dim": (4 - n) * (L - LS0) + float(logc.max())}
        sur["dimid"] = sur["dim"] - math.log(iden(pin, pout)) + sum(math.log(COLOR.get(abs(q), 1)) for q in pout)
        col = [k for k in range(n) if abs(pdg[k]) in QUARKS or pdg[k] == 21]
        chg = [k for k in range(n) if abs(pdg[k]) in CHARGE]
        eik = np.zeros(len(x))
        for i in range(2, n):
            if pdg[i] == 21:
                part, w = [k for k in col if k != i], 3.0
            elif pdg[i] == 22:
                part, w = [k for k in chg if k != i], 1.0
            else:
                continue
            if len(part) < 2:
                continue
            ant = sum(sij(a_, b_) / (sij(a_, i) * sij(b_, i)) for a_, b_ in itertools.combinations(part, 2))
            eik += np.log(w * ant)
        sur["eik"] = sur["dimid"] + eik
        pure = all(abs(q) in QUARKS or q == 21 for q in pdg) and sum(abs(q) in QUARKS for q in pdg) in (0, 2) \
            and sum(1 for d in sub[0]["diagrams"] for nd in d["nodes"] if nd.get("orders", {}).get("QED")) == 0
        if pure and n >= 4:
            qs = [k for k in range(n) if abs(pdg[k]) in QUARKS]
            if qs:
                num = sum(sij(qs[0], i) ** 3 * sij(qs[1], i) + sij(qs[0], i) * sij(qs[1], i) ** 3
                          for i in range(n) if i not in qs) if n > 4 else sij(qs[0], qs[1]) ** 0 * sum(
                    sij(qs[0], i) ** 3 * sij(qs[1], i) + sij(qs[0], i) * sij(qs[1], i) ** 3 for i in range(n) if i not in qs)
            else:
                num = sum(sij(i, j) ** 4 for i, j in itertools.combinations(range(n), 2))
            den = sum(1.0 / np.prod([sij(o[k], o[(k + 1) % n]) for k in range(n)], 0)
                      for o in ([0] + list(r) for r in itertools.permutations(range(1, n))))
            sur["pt"] = sur["dimid"] - (4 - n) * (L - LS0) + np.log(num) + np.log(den)
        else:
            sur["pt"] = sur["eik"]
        # diagrams: sum_d C_d prod 1/|D_S|^k
        tot = np.zeros(len(x))
        for d, lc in zip([d for d in sub[0]["diagrams"] if d.get("diagram_type", "tree") == "tree"], logc):
            term = np.full(len(x), math.exp(lc - logc.max()))
            for pid, legs in edge_sides(d, n):
                k = np.zeros((len(x), 4))
                for l in legs:
                    k += Q[:, l - 1]
                sS = mdot(k, k)
                if pid in MW:
                    M, G = MW[pid]
                    term = term / ((sS - M * M) ** 2 + (M * G) ** 2)
                else:
                    term = term / np.maximum(np.abs(sS), 1e-12 * s)
            tot += term
        sur["prop"] = sur["dimid"] - (4 - n) * (L - LS0) + np.log(np.maximum(tot, 1e-300))
        sur["propeik"] = sur["prop"] + eik
        nq = sum(abs(q) in QUARKS for q in pdg) // 2
        l_ = np.log(91.188 ** 2 / s)
        sur["catani"] = sur["prop"] + np.log(np.abs(nq * (4 / 3) * 0.118 / (2 * math.pi) * (-l_ ** 2 - 3 * l_ - 8 + math.pi ** 2)) + 1e-300)
        rec = {"N": int(n), "n": int(len(y)), "y_mean": float(y.mean()), "y_std": float(y.std()), "const": const,
               "sign_frac": sign_frac}
        for k_, v in sur.items():
            v = np.asarray(v, float)
            rec[k_] = [float(v.mean()), float(v.std()), float(np.corrcoef(v, y)[0, 1]) if v.std() > 0 else 0.0]
        out[name] = rec
        if len(out) % 200 == 0:
            print(f"{len(out)} processes, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
json.dump(out, open(a.out, "w"))
print(f"wrote {a.out}: {len(out)} processes in {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
