"""Which catalog entries are mixed-order, read from the generated diagrams, and how far the alpha_s scans of the
mixed ones are from a single coupling monomial (plan catalog_coverage; the dimensionless target's coupling factor,
experiment._physics_logF, data.target_physics_orders).

A tree |M|^2 = sum_ij A_i A_j* with A_i of QCD order a_i (the sum of its vertices' QCD powers, from the diagram
sidecar data/diagrams/<name>.diagrams.json, tools/dump_diagrams.py): its alpha_s powers are (a_i + a_j)/2, the highest
max(a_i), the lowest min(a_i). One value of a_i: one monomial, the coupling factor exact. Several: a polynomial in
alpha_s / alpha of fixed total degree. A one-loop QCD virtual's Born orders are its loop diagrams' QCD orders minus 2.

Part 1 (--recipes): per entry the set of amplitude QCD orders, against the order vector the target divides by
(mg5_pipeline_final.order_vector: [L_QCD, L_EW, alpha_s max, alpha max]).
Part 2 (--scans): for each alpha_s scan process (pools <name>__as090/__as130/__as150 next to the nominal one, every
event labelled at its own alpha_s, <pool>.alphas.npy), y = log|M|^2 - k log alpha_s(event) with k the highest amplitude
QCD order; for a single monomial y is the same function of the momenta in every pool. Bins of (log sqrt(s_hat), log
of each final pair's invariant mass, the pair with the largest spread), means compared with the nominal pool's: the
weighted mean and the largest bin difference, and a pure process of the same scan as the control.

python tools/mixed_order_audit.py --recipes recipes/catalog_v4_train.yaml ... --scans --out <json>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mg5_pipeline_final as mg  # noqa: E402


def _base(name):
    return name if name in mg.PROCESSES else name.split("__")[0]


def amp_qcd_orders(name, diagrams_dir):
    """The set of amplitude QCD orders of a process's Born (from its sidecar), or None if no sidecar."""
    p = os.path.join(diagrams_dir, f"{name}.diagrams.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = set()
    for sp in d["subprocesses"]:
        for dg in sp["diagrams"]:
            a = sum(int(n.get("orders", {}).get("QCD", 0)) for n in dg["nodes"] if n["kind"] == "vertex")
            out.add(a - 2 if dg.get("diagram_type") == "loop" else a)
    return sorted(out)


def part1(recipes, diagrams_dir):
    import yaml
    rows = {}
    for rf in recipes:
        for p in yaml.safe_load(open(rf))["processes"]:
            n = _base(p["name"])
            if n in rows:
                rows[n]["recipes"].append(os.path.basename(rf))
                continue
            pc = mg.PROCESSES[n]
            o = mg.order_vector(pc)
            virt = pc.get("kind") == "virt" and pc.get("order") is None
            orders = amp_qcd_orders(n, diagrams_dir)
            rows[n] = {"order_vector": o, "virt": virt, "explicit_order": pc.get("order") is not None,
                       "nfinal": int(pc["nfinal"]), "amp_qcd_orders": orders, "recipes": [os.path.basename(rf)],
                       "mixed": None if orders is None else len(orders) > 1,
                       # the factor the target divides by, against the highest alpha_s power the diagrams have
                       "k_vector": int(o[2]) - (1 if (virt and pc.get("alphas_prefactor")) else 0),
                       "k_max": None if orders is None else max(orders)}
    summ = {}
    for rf in recipes:
        b = os.path.basename(rf)
        rr = [r for r in rows.values() if b in r["recipes"] and not r["explicit_order"]]
        def m(c):
            return sum(1 for r in rr if c(r))
        summ[b] = {"entries": len(rr), "no_sidecar": m(lambda r: r["mixed"] is None),
                   "mixed": m(lambda r: r["mixed"]), "pure": m(lambda r: r["mixed"] is False),
                   "k_vector_ne_k_max": m(lambda r: r["k_max"] is not None and r["k_vector"] != r["k_max"]),
                   "by_multiplicity": {}}
        for nf in sorted({r["nfinal"] for r in rr}):
            for lp in (False, True):
                sel = [r for r in rr if r["nfinal"] == nf and r["virt"] == lp]
                if sel:
                    summ[b]["by_multiplicity"][f"2->{nf}{' one-loop' if lp else ''}"] = {
                        "entries": len(sel), "mixed": sum(1 for r in sel if r["mixed"]),
                        "no_sidecar": sum(1 for r in sel if r["mixed"] is None)}
    return rows, summ


def _pool(data_dir, name, role="train"):
    c = sorted(glob.glob(os.path.join(data_dir, f"{name}_[0-9]*GeV_{role}_smix_amplitudes.npy")))
    return c[0] if c else None


def _load(path):
    a = np.load(path, mmap_mode="r")
    pdg_cols = (a.shape[1] - 1) // 5
    n = pdg_cols
    mom = np.asarray(a[:, : 4 * n], dtype=np.float64).reshape(-1, n, 4)
    amp = np.asarray(a[:, -1], dtype=np.float64)
    als = np.load(path[: -len(".npy")] + ".alphas.npy")
    return mom, amp, np.asarray(als, dtype=np.float64)


def _m2(p):
    return p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(-1)


def part2(data_dir, rows, nbins=12, min_n=30):
    scans = sorted({os.path.basename(f).split("__as")[0] for f in glob.glob(os.path.join(data_dir, "v4_*__as1[35]0_*"))
                    if "_train_" in f or "_test_" in f})
    out = {}
    for base in scans:
        nom = _pool(data_dir, base)
        if nom is None:
            continue
        orders = amp_qcd_orders(base, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "data", "diagrams"))
        k = max(orders) if orders else int(mg.order_vector(mg.PROCESSES[base])[2])
        mom0, amp0, as0 = _load(nom)
        n = mom0.shape[1]
        # the binning: log sqrt(s_hat) and the log invariant mass of the final pair whose mass varies most
        pairs = [(i, j) for i in range(2, n) for j in range(i + 1, n)]
        lm = {pr: 0.5 * np.log(np.clip(_m2(mom0[:, pr[0]] + mom0[:, pr[1]]), 1e-6, None)) for pr in pairs}
        pr = max(pairs, key=lambda q: lm[q].std())

        def feats(mom):
            s = 0.5 * np.log(np.clip(_m2(mom[:, 0] + mom[:, 1]), 1e-6, None))
            m = 0.5 * np.log(np.clip(_m2(mom[:, pr[0]] + mom[:, pr[1]]), 1e-6, None))
            return s, m
        s0, m0 = feats(mom0)
        es = np.quantile(s0, np.linspace(0, 1, nbins + 1))
        em = np.quantile(m0, np.linspace(0, 1, nbins + 1))

        def binned(mom, amp, als):
            s, m = feats(mom)
            y = np.log(np.abs(amp)) - k * np.log(als)
            ix = np.clip(np.searchsorted(es, s) - 1, 0, nbins - 1)
            jx = np.clip(np.searchsorted(em, m) - 1, 0, nbins - 1)
            b = ix * nbins + jx
            cnt = np.bincount(b, minlength=nbins * nbins)
            sm = np.bincount(b, weights=y, minlength=nbins * nbins)
            sq = np.bincount(b, weights=y * y, minlength=nbins * nbins)
            mean = sm / np.maximum(cnt, 1)
            var = sq / np.maximum(cnt, 1) - mean ** 2
            return cnt, mean, var
        c0, y0, v0 = binned(mom0, amp0, as0)
        res = {"amp_qcd_orders": orders, "k": k, "pair_legs": list(pr), "nominal": os.path.basename(nom),
               "alphas_nominal_median": float(np.median(as0)), "points": {}}
        for tag in ("as090", "as130", "as150"):
            f = _pool(data_dir, f"{base}__{tag}") or _pool(data_dir, f"{base}__{tag}", "test")
            if f is None:
                continue
            mom, amp, als = _load(f)
            c, y, v = binned(mom, amp, als)
            ok = (c >= min_n) & (c0 >= min_n)
            d = (y - y0)[ok]
            se = np.sqrt(v[ok] / c[ok] + v0[ok] / c0[ok])
            w = 1 / np.maximum(se, 1e-9) ** 2
            res["points"][tag] = {"pool": os.path.basename(f), "alphas_median": float(np.median(als)),
                                  "bins": int(ok.sum()), "mean_diff": float((w * d).sum() / w.sum()),
                                  "rms_diff": float(np.sqrt(np.mean(d ** 2))), "max_abs_diff": float(np.abs(d).max()),
                                  "median_se": float(np.median(se)),
                                  "chi2_per_bin": float(np.mean((d / se) ** 2))}
        res["mixed"] = rows.get(base, {}).get("mixed")
        out[base] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="*", default=[])
    ap.add_argument("--diagrams", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                         "data", "diagrams"))
    ap.add_argument("--scans", action="store_true")
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, summ = part1(a.recipes, a.diagrams)
    res = {"summary": summ, "entries": rows}
    if a.scans:
        res["scans"] = part2(a.data_dir, rows)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print("MIXED_SUMMARY " + json.dumps(summ))
    for b, r in res.get("scans", {}).items():
        print("SCAN", b, "mixed" if r["mixed"] else "pure", r["amp_qcd_orders"],
              json.dumps({t: {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items() if k != "pool"}
                          for t, p in r["points"].items()}))
    bad = [n for n, r in rows.items() if r["k_max"] is not None and r["k_vector"] != r["k_max"]]
    print("K_MISMATCH", len(bad), bad[:40])


if __name__ == "__main__":
    main()
