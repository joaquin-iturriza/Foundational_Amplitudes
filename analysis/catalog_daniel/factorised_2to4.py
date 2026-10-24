"""The level item's design (ii) checked on the data (tools/factorised_2to4.py's npz, data/factorised_2to4/<probe>.npz):
how much of a 2->4 probe's shape does its 2->3 parent's exact |M|^2 at the mapped event carry, and how much of the rest
does a correction learned on OTHER probes remove? Per probe, in log|M|^2 (nat), with the per-probe mean removed (the
level is the offset's business, Fig. kshot_level; this is the shape):
  shape        var(log t4), the probe's own spread
  parent       var(r), r = log t4 - log t3(mapped): the shape left after the parent's exact |M|^2
  learned      var(r - h), h a gradient-boosted regression of r on the splitting variables (log s_ak/s, log s_bk/s, the
               nearest final's log s_jk/s, log pT_k^2/s, y_k, log(1-x), log s; gluon or photon; the nearest emitter's
               kind), fitted on the other 39 probes (leave-one-probe-out, each probe's r centred), scored on this one
Against these, the network's own 2->4 shape floor (the variance of its zero-shot error, kshot_context_cmp.json:
1.34 with a gluon, 1.37 with a photon): the head of design (ii) is worth building when parent + learned sits well under
it, since the network's 2->3 error on the mapped events (in distribution) adds to it.
    .venv-driver/bin/python analysis/catalog_daniel/factorised_2to4.py   -> data/factorised_2to4.json, figs/factorised_2to4
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def kind(pdg):
    a = abs(int(pdg))
    return 0 if a <= 6 else 1 if a == 21 else 2 if a in (11, 13, 15) else 3 if a == 22 else 4


def load():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "data", "factorised_2to4", "*.npz"))):
        z = dict(np.load(f))
        ok = (z["t3"] > 0) & np.isfinite(z["t3"]) & (z["t4"] > 0) & np.all(np.isfinite(z["feats"]), 1)
        p = os.path.basename(f)[:-4]
        r = np.log(z["t4"][ok]) - np.log(z["t3"][ok])
        X = np.column_stack([z["feats"][ok], np.full(ok.sum(), float(p.endswith("_g"))),
                             np.array([kind(e) for e in z["emitter"][ok]], float)])
        out[p] = dict(lt4=np.log(z["t4"][ok]), r=r, X=X, dropped=int((~ok).sum()))
    return out


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    d = load()
    res = {}
    for p, v in d.items():
        tr = [q for q in d if q != p]
        Xtr = np.concatenate([d[q]["X"] for q in tr])
        ytr = np.concatenate([d[q]["r"] - d[q]["r"].mean() for q in tr])
        h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features=[7, 8],
                                          random_state=0).fit(Xtr, ytr)
        e = v["r"] - h.predict(v["X"])
        res[p] = dict(shape=float(v["lt4"].var()), parent=float(v["r"].var()), learned=float(e.var()),
                      dropped=v["dropped"], n=int(len(v["r"])))
        print(p, json.dumps(res[p]))
    summ = {}
    for tag, sel in (("+g", "_g"), ("+a", "_a")):
        rs = [r for p, r in res.items() if p.endswith(sel)]
        summ[tag] = {k: float(np.median([r[k] for r in rs])) for k in ("shape", "parent", "learned")}
        summ[tag]["n"] = len(rs)
        print("MEDIAN", tag, json.dumps(summ[tag]))
    json.dump({"summary": summ, "probes": res}, open(os.path.join(HERE, "data", "factorised_2to4.json"), "w"), indent=1)
    plot(res)


def plot(res):
    import plot_style as ps
    ref = json.load(open(os.path.join(HERE, "data", "kshot_context_cmp.json")))
    fig, axs = ps.figure(ncols=2)
    for ax, sel, key, lab in ((axs[0], "_g", "2to4 +g", r"$2\to4$, $+g$"), (axs[1], "_a", "2to4 +a", r"$2\to4$, $+\gamma$")):
        ps_ = sorted((p for p in res if p.endswith(sel)), key=lambda p: res[p]["shape"])
        x = np.arange(len(ps_))
        ax.plot(x, [res[p]["shape"] for p in ps_], "o", mfc="none", color=ps.C.grey, label=r"$\log|\mathcal{M}|^2$")
        ax.plot(x, [res[p]["parent"] for p in ps_], "s", color=ps.C.blue, label="after the parent")
        ax.plot(x, [res[p]["learned"] for p in ps_], "^", color=ps.C.vermillion, label="and learned")
        ax.plot([0, len(ps_) - 1], [ref[key]["floor"]] * 2, ls="--", color=ps.C.green, label="network floor")
        ax.set_yscale("log")
        top = max(max(res[p][k] for k in ("shape", "parent", "learned")) for p in ps_)
        ax.set_ylim(top=top * 10)                        # clears the process label over the highest point
        ax.set_xlabel("probe (by its spread)"); ax.set_ylabel(r"variance (nat$^2$)")
        ps.process_label(ax, lab)
    ps.legend(axs[0], "upper left")
    for ax in axs:
        ps.make_room(ax, max_growth=4)
    ps.save(fig, os.path.join(HERE, "figs", "factorised_2to4"))


if __name__ == "__main__":
    main()
