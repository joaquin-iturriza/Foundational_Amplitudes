#!/usr/bin/env python3
"""Does a cheap propagator surrogate predict log|M|^2 well enough to decide, before the matrix element is computed,
which phase-space candidates to label? (plan catalog_coverage; Joaquin 2026-11-06 22:31, D317, refused the quota sampler's
~100x generation time; the <= 2x bound is the step-69 design's; the waste
today is the matrix elements of candidates the flat selection rejects.)

Surrogate, per event, from the class's diagram sidecar (data/diagrams/<class>.diagrams.json, the subprocess with the
pool's beams and finals, legs mapped to slots as mg5_pipeline_final.resonance_channels does), every tree diagram d and
every internal line k of it with momentum q_k (all-outgoing signed sum of its legs' slots) and s_k = q_k^2:
    D_k = (s_k - M_k^2)^2 + M_k^2 G_k^2   for Z, W, H, t (mg5_pipeline_final.MC_RES masses and widths)
    D_k = s_k^2                            for a massless line
    S_sum = ln sum_d prod_k 1/D_k     (the single-diagram-squared channel weight)
    S_max = ln max_d prod_k 1/D_k
No numerator, no coupling, no colour: microseconds per event in numpy, against ~0.2 ms for the matrix element.

Read on the five step-57 classes (multichannel_check.CLASSES), two sources each: the catalog_v4 train pool (the old
sampling) and the default-share MadEvent train pool dsh_<x> (13 TeV, the events at the top of the range). Per class
and source: Spearman rank correlation of S with log10|M|^2; the conditional spread, the median over 40 equal-count
bins of S of the within-bin standard deviation of log10|M|^2 (dex); the share of events whose 0.25-dex bin of
log10|M|^2 lies within +-1 bin of the median of their S bin (the prescreen's hit rate); the same on the top decile
of log10|M|^2 of the source.

    python3 analysis/catalog_v4/prescreen_check.py run     (on CC: site run ccin2p3 FA -- python ...; prints one JSON
                                                            line PRESCREEN_JSON {...})
    python3 analysis/catalog_v4/prescreen_check.py save <log>   local: the JSON line -> data/prescreen_check.json
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "data", "prescreen_check.json")


def lines(diagrams, pdg_ids):
    """[(diagram index, signed slot row (n,), |pdg|)] of every internal line of every tree diagram"""
    from diagram_graphs import propagator_leg_sets, leg_to_slot
    pdg = [int(x) for x in pdg_ids]; k = len(pdg)
    want = (sorted(pdg[:2]), sorted(pdg[2:]))
    st = lambda e: e.get("state", "in" if e["number"] <= 2 else "out")
    subs = [sp for sp in diagrams.get("subprocesses", [])
            if (sorted(int(e["pdg"]) for e in sp["external"] if st(e) == "in"),
                sorted(int(e["pdg"]) for e in sp["external"] if st(e) == "out")) == want]
    if not subs:
        raise ValueError(f"no subprocess with beams {want[0]} and finals {want[1]}")
    ext = [dict(e, state=st(e)) for e in subs[0]["external"]]
    slot = leg_to_slot(ext, pdg, 2)
    out = []
    tree = [g for g in subs[0]["diagrams"] if g.get("diagram_type", "tree") == "tree"]
    for d, g in enumerate(tree):
        for _u, _v, legs, ap in propagator_leg_sets(g):
            row = np.zeros(k)
            for L in legs:
                s = slot[L]
                row[s] = -1.0 if s < 2 else 1.0
            out.append((d, row, abs(int(ap))))
    return out, len(tree)


def surrogate(P, props, n_diag):
    """(S_sum, S_max) per event of momenta P (N, n, 4)"""
    import mg5_pipeline_final as mg
    lnw = np.zeros((len(P), n_diag))
    for d, row, ap in props:
        q = np.einsum("s,nsc->nc", row, P)
        s = q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1)
        if ap in mg.MC_RES:
            M, G = mg.MC_RES[ap]
            D = (s - M * M) ** 2 + (M * G) ** 2
        else:
            D = np.maximum(s * s, 1e-30)
        lnw[:, d] -= np.log(D)
    m = lnw.max(1)
    return m + np.log(np.exp(lnw - m[:, None]).sum(1)), m


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def read(S, y, nb=40):
    """spread and hit rate of y (log10|M|^2) given S"""
    o = np.argsort(S); edges = np.array_split(o, nb)
    sd, hit = [], np.zeros(len(y), bool)
    for idx in edges:
        yy = y[idx]
        sd.append(float(np.std(yy)))
        med = np.median(yy)
        hit[idx] = np.abs(np.floor(yy / 0.25) - np.floor(med / 0.25)) <= 1
    top = y >= np.quantile(y, 0.9)
    return {"spearman": round(spearman(S, y), 4), "spread_dex_median": round(float(np.median(sd)), 3),
            "spread_dex_q90": round(float(np.quantile(sd, 0.9)), 3), "hit_pm1bin": round(float(hit.mean()), 3),
            "hit_pm1bin_top_decile": round(float(hit[top].mean()), 3),
            "spearman_top_decile": round(spearman(S[top], y[top]), 4), "n": int(len(y))}


def run():
    import multichannel_check as mcc
    from analysis.common.readers import find_pool, split_pool
    w = "13000-13000GeV"
    res = {}
    for name in mcc.CLASSES:
        ent = mcc._entry(name)
        diag = json.load(open(os.path.join(ROOT, "data", "diagrams", f"{ent.get('base', name)}.diagrams.json")))
        srcs = {"catalog_train": find_pool(name, role="train"),
                "default_dsh_train": mcc._find(f"dsh_{name[len('v4_'):]}_{w}_train_lhe_amplitudes.npy")}
        res[name] = {}
        for src, f in srcs.items():
            if f is None:
                res[name][src] = {"missing": True}
                continue
            rows = np.asarray(np.load(f, mmap_mode="r"), dtype=np.float64)
            n = (rows.shape[1] - 1) // 5
            P, ids, amp = split_pool(rows, n)
            props, nd = lines(diag, ids[0])
            ok = np.isfinite(amp) & (amp > 0)
            y = np.log10(amp[ok])
            S_sum, S_max = surrogate(P[ok], props, nd)
            res[name][src] = {"file": os.path.basename(f), "n_diagrams": nd, "n_lines": len(props),
                              "S_sum": read(S_sum, y), "S_max": read(S_max, y)}
    print("PRESCREEN_JSON " + json.dumps(res))


def save(log):
    for line in open(log, errors="replace"):
        if line.startswith("PRESCREEN_JSON "):
            d = json.loads(line[len("PRESCREEN_JSON "):])
            json.dump({"tool": "analysis/catalog_v4/prescreen_check.py", "classes": d}, open(OUT, "w"), indent=1)
            print(json.dumps(d, indent=1))
            return
    sys.exit("no PRESCREEN_JSON line")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run()
    else:
        save(sys.argv[2])
