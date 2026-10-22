"""Weight interpolation between the finale and an add-W run (D31 item 1; item interp): theta(a) = (1 - a) theta_finale +
a theta_W, with theta_W the best checkpoint on W of the naive add-W run at 1k, 2k and 4k steps (tp3_waddm_naive*,
analysis/transfer/waddm_tradeoff.py's "best on W"), a = 0, 0.1, ..., 1, scored on the add-W run's own data
(tools/run_data_tools.py interp): W = the geometric mean of the W+4g, W+5g validation MSE (preprocessed), forgetting =
the 28 monitors' geometric mean over the finale's own (as waddm_tradeoff.py). Drawn against the Pareto fronts of the
naive run's own validation points and of the 10% replay buffer (waddm_curves.json).
Inputs: interp_data/zz_interp/naive{1k,2k,4k}.json (from CC's zz_interp/), waddm_curves.json.
    python analysis/transfer/interp_tradeoff.py   -> figs/interp_tradeoff
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402

GM = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))


def front(p):
    out, fmin = [], float("inf")
    for w, f in sorted(p):
        if f < fmin:
            out.append((w, f)); fmin = f
    return out


def main():
    D = json.load(open(os.path.join(HERE, "waddm_curves.json")))
    old = set(D["finale_old28"]); F0 = GM(D["finale_old28"].values())
    curves, res = {}, {}
    for h in ("1k", "2k", "4k"):
        d = json.load(open(os.path.join(HERE, "interp_data", "zz_interp", f"naive{h}.json")))
        pts = []
        for pt in d["points"]:
            m = pt["val_mse_prepd"]
            pts.append((pt["alpha"], GM([v for k, v in m.items() if k not in old and k != "combined"]),
                        GM([m[k] for k in old]) / F0))
        curves[h] = pts
        res[h] = [dict(alpha=a, W=w, forgetting=f) for a, w, f in pts]
        print(h, " ".join(f"a={a:.1f}: W {w:.3g}, forgetting {f:.3g};" for a, w, f in pts))
    traj = {}
    for r in D["runs"]:
        m = r["sweep"][len("tp3_waddm_"):-2]
        if m in ("naive", "buf10pc"):
            n = len(r["W"])
            traj.setdefault(m, []).extend((r["W"][i], GM([c[i] for c in r["old28"].values()]) / F0) for i in range(n))
    json.dump(res, open(os.path.join(HERE, "interp_tradeoff.json"), "w"), indent=1)

    fig, ax = ps.figure()
    for m, lab, col in (("naive", "naive run, its validations", ps.C.grey),
                        ("buf10pc", "replay 10%, its validations", ps.C.vermillion)):
        fr = front(traj[m])
        ax.plot([x[0] for x in fr], [x[1] for x in fr], color=col, drawstyle="steps-post", label=lab)
    for h, mk in (("1k", "o"), ("2k", "s"), ("4k", "^")):
        ax.plot([p[1] for p in curves[h]], [p[2] for p in curves[h]], mk + "-", color=ps.C.blue, mfc="none",
                label=f"interpolation, naive {h}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$W$: MSE on $W{+}4g$, $W{+}5g$")
    ax.set_ylabel("forgetting: old 28 / finale")
    ps.legend(ax, "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "interp_tradeoff"))


if __name__ == "__main__":
    main()
