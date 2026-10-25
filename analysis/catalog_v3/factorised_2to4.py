"""The factorised 2->4 head's data check on catalog_v3 (tools/factorised_2to4.py --v3; its npz copied home into
data/factorised_2to4/<probe>.npz, its log's V3_SKIP lines into data/factorised_2to4_skipped.jsonl): every catalog_v3
2->4 class built as a 2->3 class plus one gluon, photon, Z or H, against the parent's exact |M|^2 at the event mapped
by the Catani-Seymour initial-initial map (for a Z or H, x = K^2/s); on b beams, whose rescaled beam would leave its
mass shell, by the beam-recoil map (the other finals kept, the beams absorbing k's recoil on their mass shells), with the
b-beam probes summarised apart. The same three numbers per probe as Daniel's
catalog's check (analysis/catalog_daniel/factorised_2to4.py), in log|M|^2 with the per-probe mean removed:
  shape     var(log t4), the probe's own spread
  parent    var(r), r = log t4 - log t3(mapped): what is left after the parent's exact |M|^2
  learned   var(r - h), h a gradient-boosted regression of r on the splitting variables (log s_ak/s, log s_bk/s, the
            nearest final's log s_jk/s, log pT_k^2/s, y_k, log(1-x), log s; the added particle; the nearest emitter's
            kind), fitted on probes of OTHER parents and scored on this one
The learned part is fitted in 10 folds grouped by parent (every probe of one parent in one fold), in place of the
leave-one-probe-out of Daniel's 40 probes: about 900 probes make 900 fits too slow, and grouping by parent keeps a
probe's siblings (the same parent plus another particle) out of its own fit. Each fit sees 400 events per probe.
Medians per added particle; every skipped class is counted by its reason.
    .venv-driver/bin/python analysis/catalog_v3/factorised_2to4.py -> data/factorised_2to4.json, figs/factorised_2to4_*
"""
import collections, glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
KINDS = {"g": "a gluon", "a": "a photon", "z": "a Z", "h": "an H"}


def emitter_kind(pdg):
    a = abs(int(pdg))
    return 0 if a <= 6 else 1 if a == 21 else 2 if a in (11, 13, 15) else 3 if a == 22 else 4


def load():
    import mg5_pipeline_final as mg
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "data", "factorised_2to4", "*.npz"))):
        p = os.path.basename(f)[:-4]
        why = str(mg.PROCESSES[p].get("why", ""))
        par, kind = why.rsplit("+", 1)
        z = dict(np.load(f))
        ok = (z["t3"] > 0) & np.isfinite(z["t3"]) & (z["t4"] > 0) & np.all(np.isfinite(z["feats"]), 1)
        r = np.log(z["t4"][ok]) - np.log(z["t3"][ok])
        X = np.column_stack([z["feats"][ok], np.full(ok.sum(), float(list(KINDS).index(kind))),
                             np.array([emitter_kind(e) for e in z["emitter"][ok]], float)])
        out[p] = dict(b_beam=bool(z.get("b_beam", False)), parent=par, kind=kind, lt4=np.log(z["t4"][ok]), r=r, X=X, dropped=int((~ok).sum()))
    return out


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    d = load()
    pars = sorted({v["parent"] for v in d.values()})
    fold = {par: i % 10 for i, par in enumerate(np.random.default_rng(0).permutation(pars))}
    rng = np.random.default_rng(0)
    sub = {p: rng.permutation(len(v["r"]))[:400] for p, v in d.items()}
    res = {}
    for k in range(10):
        tr = [p for p, v in d.items() if fold[v["parent"]] != k]
        Xtr = np.concatenate([d[p]["X"][sub[p]] for p in tr])
        ytr = np.concatenate([(d[p]["r"] - d[p]["r"].mean())[sub[p]] for p in tr])
        h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features=[7, 8],
                                          random_state=0).fit(Xtr, ytr)
        for p, v in d.items():
            if fold[v["parent"]] != k:
                continue
            e = v["r"] - h.predict(v["X"])
            res[p] = dict(kind=v["kind"], b_beam=v["b_beam"], parent=v["parent"], shape=float(v["lt4"].var()), parent_var=float(v["r"].var()),
                          learned=float(e.var()), dropped=v["dropped"], n=int(len(v["r"])))
        print(f"fold {k}: {sum(fold[v['parent']] == k for v in d.values())} probes", flush=True)
    skipped = collections.Counter()
    sf = os.path.join(HERE, "data", "factorised_2to4_skipped.jsonl")
    if os.path.exists(sf):
        for line in open(sf):
            s = json.loads(line)
            skipped[f"{s['kind']}: {s['why']}"] += 1
    summ = {}
    for kind in KINDS:
        for beams, bb in (("light", False), ("b", True)):
            rs = [r for r in res.values() if r["kind"] == kind and r["b_beam"] == bb]
            if not rs:
                continue
            t = f"{kind} {beams} beams"
            summ[t] = {key: float(np.median([r[key] for r in rs])) for key in ("shape", "parent_var", "learned")}
            summ[t]["n"] = len(rs)
            summ[t]["dropped_events"] = int(sum(r["dropped"] for r in rs))
            print("MEDIAN", t, json.dumps(summ[t]))
    print("SKIPPED", json.dumps(dict(skipped)))
    json.dump({"summary": summ, "skipped": dict(skipped), "probes": res},
              open(os.path.join(HERE, "data", "factorised_2to4.json"), "w"), indent=1)
    plot(res)


def plot(res):
    import plot_style as ps
    figs = ps.panels(4)
    for (fig, ax), kind in zip(figs, KINDS):
        ps_ = sorted((p for p, r in res.items() if r["kind"] == kind), key=lambda p: res[p]["shape"])
        x = np.arange(len(ps_))
        ax.plot(x, [res[p]["shape"] for p in ps_], "o", mfc="none", color=ps.C.grey, label=r"$\log|\mathcal{M}|^2$")
        ax.plot(x, [res[p]["parent_var"] for p in ps_], "s", color=ps.C.blue, label="after the parent")
        ax.plot(x, [res[p]["learned"] for p in ps_], "^", color=ps.C.vermillion, label="and learned")
        ax.set_yscale("log")
        ax.set_xlabel("probe (by its spread)"); ax.set_ylabel(r"variance (nat$^2$)")
        ps.process_label(ax, r"$2\to3$ plus " + KINDS[kind])
        ps.legend(ax, "upper left")
        ps.make_room(ax, max_growth=4)
    ps.save_panels(figs, os.path.join(HERE, "figs", "factorised_2to4"))


if __name__ == "__main__":
    main()
