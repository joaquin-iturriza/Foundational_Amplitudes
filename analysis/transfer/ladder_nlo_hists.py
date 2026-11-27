"""The ladder study's one-loop probes e+e- -> d d~ and b b~ (ee_dd_nlo, ee_bb_nlo; D598): true against predicted
distributions of log|M|^2 and of the phase-space variables sqrt(s_hat) and cos(theta*) (each event weighted by its
true or by its predicted |M|^2), and the relative error Delta = |M|^2_pred / |M|^2_true - 1 beside MSE(log|M|^2).

Runs: every ladder arm of the two probes at D = 10^4 (8k grid, analysis/transfer/delta_audit.json, the cells of
fig:ladder and fig:finale_synth), per-event predictions on the test split from tools/eval_delta.py (best checkpoint,
EMA as validated) in $SCRATCH/zz_delta/<sweep>__trial_<hp> on the site that evaluated it (originals only: a "_rerun"
is used only where the original was pruned, delta_plots.pick). The momenta come from the probe's test pool, the one whose |M|^2 at the
saved pool_row equals the saved truth on every event.

    python analysis/transfer/ladder_nlo_hists.py collect   # read-only site runs -> data/ladder_nlo_hists.npz
    python analysis/transfer/ladder_nlo_hists.py plot      # -> figs/ladder_nlo_dist_<probe>, ladder_nlo_rel_<probe>, ladder_nlo_rank
"""
import base64, io, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, HERE)
NPZ = os.path.join(HERE, "data", "ladder_nlo_hists.npz")
PROBES = {"ee_dd_nlo": r"$e^+e^-\to d\bar d$, one loop", "ee_bb_nlo": r"$e^+e^-\to b\bar b$, one loop"}
ARMS = ["scratch"] + [f"r{i}" for i in range(10)] + ["synthetic", "finale"]
HIST_ARMS = ["scratch", "r9", "synthetic", "finale"]          # the draft's finale_synth set
NAME = {"scratch": "from scratch", "r9": "rung 9", "synthetic": "synthetic", "finale": "finale"}


def cells():
    """{(probe, arm): (tag, recorded val MSE(log|M|^2))} from delta_audit.json's 8k cells."""
    out = {}
    for r in json.load(open(os.path.join(HERE, "delta_audit.json")))["rows"]:
        p = r.get("paper_8k")
        if r["probe"] in PROBES and r["arm"] in ARMS and p:
            t = "__".join(os.path.normpath(p["run_dir"]).split(os.sep)[-2:])
            out[r["probe"], r["arm"]] = (t, p)
    return out


REMOTE = r'''
import base64, glob, io, json, os, sys
import numpy as np
root = os.path.join(os.environ["SCRATCH"], "zz_delta")
out, pools = {}, {}
for t in sys.argv[1:]:
    d = os.path.join(root, t)
    if not os.path.exists(os.path.join(d, "preds_test.npz")):
        continue
    rec = {"eval": json.load(open(os.path.join(d, "eval_delta.json"))) if os.path.exists(os.path.join(d, "eval_delta.json")) else None}
    b = io.BytesIO(); arr = {}
    for s in ("test", "val"):
        z = np.load(os.path.join(d, f"preds_{s}.npz"))
        arr[s + "_true"], arr[s + "_pred"] = z["raw_truth"].ravel(), z["raw_pred"].ravel().astype(np.float64)
        arr[s + "_row"] = z["pool_row"]
    probe = "ee_dd_nlo" if "ee_dd_nlo" in t else "ee_bb_nlo"
    # the test pool whose |M|^2 at pool_row is the saved truth on every event
    hit = None
    for f in sorted(glob.glob(os.path.join(os.environ["DATA_DIR"], probe + "_*_test_*amplitudes.npy"))):
        a = np.load(f, mmap_mode="r")
        if a.shape[0] > arr["test_row"].max() and a.shape[1] == 21 and np.allclose(a[arr["test_row"], -1], arr["test_true"], rtol=1e-6, atol=0):
            hit = f; break
    rec["pool"] = hit
    if hit and hit not in pools:
        a = np.load(hit)
        pools[hit] = base64.b64encode(a.tobytes()).decode()
    np.savez_compressed(b, **arr)
    rec["arrays"] = base64.b64encode(b.getvalue()).decode()
    out[t] = rec
print("LADNLO " + json.dumps({"runs": out, "pools": pools}))
'''


def collect():
    from common.readers import site_python
    want = {t for t, _ in cells().values()}
    have, pools, where = {}, {}, {}
    for site in ("jeanzay", "lxplus", "ccin2p3"):
        ask = sorted(want - set(where))
        if not ask:
            break
        got = site_python(REMOTE, ask, "LADNLO", site=site)
        for f, s in got["pools"].items():
            pools[os.path.basename(f)] = np.frombuffer(base64.b64decode(s), np.float64).reshape(-1, 21)
        for t, rec in got["runs"].items():
            z = np.load(io.BytesIO(base64.b64decode(rec["arrays"])))
            for k in z.files:
                have[f"{t}/{k}"] = z[k]
            have[f"{t}/pool"] = np.array(os.path.basename(rec["pool"]) if rec["pool"] else "")
            where[t] = site
        print(f"{site}: {len(got['runs'])} of {len(ask)} asked")
    for f, a in pools.items():
        have[f"pool/{f}"] = a
    have["_where"] = np.array(json.dumps(where))
    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez_compressed(NPZ, **have)
    print("missing:", sorted(want - set(where)))


def kin(pool, rows):
    """sqrt(s_hat) and cos(theta*) of the outgoing quark (PDG id > 0 among the finals) to the incoming e- (PDG 11),
    in the partonic frame (e+e- beams: the pool frame is it)."""
    p = pool[rows, :16].reshape(-1, 4, 4)
    ids = pool[rows, 16:20].astype(int)
    s = (p[:, 0] + p[:, 1])
    sq = np.sqrt(s[:, 0] ** 2 - (s[:, 1:] ** 2).sum(1))
    em = np.where(ids[:, :1] == 11, p[:, 0], p[:, 1])
    q = np.where(ids[:, 2:3] > 0, p[:, 2], p[:, 3])
    c = (em[:, 1:] * q[:, 1:]).sum(1) / np.linalg.norm(em[:, 1:], axis=1) / np.linalg.norm(q[:, 1:], axis=1)
    return sq, c


def load():
    """Each arm's test-split truth and prediction. Jean Zay's zz_delta outputs are gone (its scratch purges), so every
    arm is read as the probe's test truth (collected on CC) times 1 + Delta, Delta the arm's test array of
    delta_data/<site>.npz (collect_delta.py, the same eval_delta outputs): checked on the 9 arms CC still holds, whose
    pool_row is the same for every arm of a probe (the split's seed-42 order over the one recipe pool) and whose Delta
    equals preds_test's to 6e-8. The val split (the reproduction check against the recorded MSE) only for those 9."""
    from delta_plots import load as delta_load, pick
    z = np.load(NPZ)
    dl = delta_load()
    ref = {}
    for (probe, arm), (t, rec) in cells().items():
        if f"{t}/test_true" in z.files:
            ref.setdefault(probe, t)
    out = {}
    for (probe, arm), (t, rec) in cells().items():
        rt = ref[probe]
        key = pick(dl, rec)   # the shared rule: own or rerun, at the precision that reproduces the recorded loss
        if key not in dl:
            print(f"ARM {probe} {arm} {t}: no Delta on record", file=sys.stderr)
            continue
        own = f"{t}/test_true" in z.files
        if own:
            assert np.array_equal(z[f"{t}/test_row"], z[f"{rt}/test_row"]), t
        tr = z[f"{rt}/test_true"]
        r = dict(test_true=tr, test_pred=tr * (1 + dl[key]), test_row=z[f"{rt}/test_row"], tag=t, delta_key=key,
                 recorded=rec["mse_logM2"], pool=str(z[f"{rt}/pool"]))
        if own:
            r["val_true"], r["val_pred"] = z[f"{t}/val_true"], z[f"{t}/val_pred"]
        pf = r["pool"]
        r["sqrts"], r["cos"] = kin(z[f"pool/{pf}"], r["test_row"])
        out[probe, arm] = r
    return out


def stats(r, s="test"):
    t, q = r[s + "_true"], r[s + "_pred"]
    lr = np.log(q) - np.log(t)
    d = q / t - 1
    return {"mse": float(np.mean(lr ** 2)), "med_abs_delta": float(np.median(np.abs(d))),
            "mean_abs_delta": float(np.mean(np.abs(d))), "q90_abs_delta": float(np.quantile(np.abs(d), 0.9))}


def report(R):
    for (probe, arm), r in sorted(R.items(), key=lambda x: (x[0][0], ARMS.index(x[0][1]))):
        v, te = (stats(r, "val") if "val_true" in r else None), stats(r, "test")
        # where the MSE(log|M|^2) sits: the share of it from the events with the 10% largest true |M|^2
        lr2 = (np.log(r["test_pred"]) - np.log(r["test_true"])) ** 2
        top = r["test_true"] >= np.quantile(r["test_true"], 0.9)
        ad = np.abs(r["test_pred"] / r["test_true"] - 1)
        zw = (r["sqrts"] > 80) & (r["sqrts"] < 100)
        print(f"ARM {probe} {arm:9s} {r['delta_key']}: "
              + (f"val MSE(ln|M|^2) {v['mse']:.3g} (recorded {r['recorded']:.3g}, ratio {v['mse'] / r['recorded']:.2f}); "
                 if v else f"recorded val MSE {r['recorded']:.3g}; ") + f"test MSE {te['mse']:.3g}, median|Delta| {te['med_abs_delta']:.3g}, "
              f"mean|Delta| {te['mean_abs_delta']:.3g}, q90|Delta| {te['q90_abs_delta']:.3g}; top-10% |M|^2 share of "
              f"MSE {lr2[top].sum() / lr2.sum():.2f}; median|Delta| top-10% {np.median(ad[top]):.3g}, rest "
              f"{np.median(ad[~top]):.3g}; Z window 80-100 GeV share of events {np.mean(zw):.3f}, of MSE "
              f"{lr2[zw].sum() / lr2.sum():.2f}")


def _binned_median(v, ad, nb=10):
    """Median |Delta| in nb equal-count bins of v: (bin centres, medians)."""
    e = np.quantile(v, np.linspace(0, 1, nb + 1))
    k = np.clip(np.digitize(v, e[1:-1]), 0, nb - 1)
    return 0.5 * (e[1:] + e[:-1]), np.array([np.median(ad[k == i]) for i in range(nb)])


def plot(R):
    import matplotlib.pyplot as plt
    import plot_style as ps
    from scipy.stats import spearmanr
    style = {a: dict(color=ps.CYCLE[i]) for i, a in enumerate(HIST_ARMS)}
    out = os.path.join(HERE, "figs")
    LM = r"$\log_{10}|\mathcal{M}|^2$"
    for probe, lab in PROBES.items():
        arms = [a for a in HIST_ARMS if (probe, a) in R]
        t0 = R[probe, arms[0]]
        # true against predicted distributions on the same test events: log|M|^2, and sqrt(s_hat), cos(theta*) with each
        # event weighted by its true / predicted |M|^2
        fig, axs = ps.figure(2, 2)
        axs = axs.ravel()
        lt = np.log10(t0["test_true"])
        lq = [np.log10(R[probe, a]["test_pred"]) for a in arms]
        el = np.linspace(min([lt.min()] + [x.min() for x in lq]), max([lt.max()] + [x.max() for x in lq]), 41)
        for ax, (x, xl, e) in zip(axs[:3], [
                ("lm", LM, el),
                ("sqrts", r"$\sqrt{\hat s}$ [GeV]", np.linspace(t0["sqrts"].min(), t0["sqrts"].max(), 41)),
                ("cos", r"$\cos\theta^*$ (quark to $e^-$)", np.linspace(-1, 1, 41))]):
            models = []
            for a in arms:
                r = R[probe, a]
                if x == "lm":
                    ht, hq = np.histogram(lt, e)[0], np.histogram(np.log10(r["test_pred"]), e)[0]
                else:
                    ht = np.histogram(r[x], e, weights=r["test_true"])[0]
                    hq = np.histogram(r[x], e, weights=r["test_pred"])[0]
                models.append(dict(label=NAME[a], edges=e, pred=hq, truth=ht, style=style[a]))
            ps.true_pred_hist(ax, models, xlabel=xl)
            if x != "lm":
                ax.set_ylabel(r"$\sum|\mathcal{M}|^2$ per bin")
        ps.process_label(axs[0], lab)
        ps.legend_cell(axs[3], axs[0])
        ps.save(fig, os.path.join(out, f"ladder_nlo_dist_{probe}"))
        plt.close(fig)
        # the relative error Delta = |M|^2_pred / |M|^2_true - 1: its distribution, and its median against
        # sqrt(s_hat), cos(theta*) and the true log10|M|^2 (is the error larger where |M|^2 is large?)
        fig, axs = ps.figure(2, 2)
        axs = axs.ravel()
        b = np.logspace(-7, 1, 49)
        for a in arms:
            r = R[probe, a]
            ad = np.abs(r["test_pred"] / r["test_true"] - 1)
            axs[0].stairs(np.histogram(ad, b)[0], b, label=NAME[a], **style[a])
            for ax, v in zip(axs[1:], (r["sqrts"], r["cos"], np.log10(r["test_true"]))):
                ax.plot(*_binned_median(v, ad), marker="o", **style[a])
        axs[0].set_xscale("log"); axs[0].set_xlabel(r"$|\Delta|$, $\Delta=|\mathcal{M}|^2_{\rm pred}/|\mathcal{M}|^2_{\rm true}-1$")
        axs[0].set_ylabel("events")
        for ax, xl in zip(axs[1:], (r"$\sqrt{\hat s}$ [GeV]", r"$\cos\theta^*$ (quark to $e^-$)", LM + r" (true)")):
            ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel(r"median $|\Delta|$")
        ps.process_label(axs[1], lab)
        ps.legend(axs[0], loc="upper left")
        ps.save(fig, os.path.join(out, f"ladder_nlo_rel_{probe}"))
        plt.close(fig)
    # every ladder arm: MSE(log|M|^2) against median |Delta|, both on the test split
    fig, ax = ps.figure()
    for probe, mk, c in zip(PROBES, ("o", "s"), (ps.C.blue, ps.C.vermillion)):
        arms = [a for a in ARMS if (probe, a) in R]
        m = np.array([np.mean(np.log(R[probe, a]["test_pred"] / R[probe, a]["test_true"]) ** 2) for a in arms])
        d = np.array([stats(R[probe, a])["med_abs_delta"] for a in arms])
        rho = spearmanr(m, d).correlation
        print(f"RANK {probe}: Spearman(MSE(ln|M|^2), median|Delta|) over {len(arms)} arms = {rho:.2f}; "
              f"MSE order {[arms[i] for i in np.argsort(m)]}; median|Delta| order {[arms[i] for i in np.argsort(d)]}")
        ax.scatter(m, d, marker=mk, color=c, label=PROBES[probe])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"MSE$(\ln|\mathcal{M}|^2)$"); ax.set_ylabel(r"median $|\Delta|$")
    ps.legend(ax, loc="upper left")
    ps.save(fig, os.path.join(out, "ladder_nlo_rank"))


if __name__ == "__main__":
    if sys.argv[1:] == ["collect"]:
        collect()
    else:
        R = load()
        report(R)
        plot(R)
