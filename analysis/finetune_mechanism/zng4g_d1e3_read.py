"""Plan finetune_mechanism step 1/2 (D519/D520/D529; designs passed 2026-11-24 05:35 and 07:57): the D=1e3, 3125-step,
batch-128 arms on q q~ -> Z+4g. S1k from scratch, Fn1k fine-tuned from the 28-process parent without the two settings
(FA-ccin2p3-9a73698a), Fg1k fine-tuned from the gluon/Z parent (when its search exists). Each arm: a driven search,
16 trials on [1.9e-3, 1.9e-2] then 8 on [6e-4, 6e-2] (extend_sweep.py), and seeds 43/44 of its chosen trial.

Measure: best val MSE(log|M|^2) = min val mu-MSE x prepd_std^2 (each run's own data_stats.json). The chosen trial is
the search's lowest val_loss; a seed run belongs to the trial whose lr its config.yaml carries, and an arm is read
from the seeds of its current chosen trial only. Ratio r = three-seed geometric means, log noise
sigma_r = sqrt((s_a^2 + s_b^2) / 3), s = an arm's log sd over its seeds; 2-sigma interval exp(log r +- 2 sigma_r);
the 9-pairing 10-90% range beside. Fn1k/S1k: hi < 1 gain, lo > 1 cost, else unresolved. Fg1k/Fn1k: hi < 1 dilution
yes, lo > 0.65 and hi >= 1 no effect of the needed size, lo > 1 gluon worse, else unresolved.

    python3 analysis/finetune_mechanism/zng4g_d1e3_read.py [--cached]
        -> data/zng4g_d1e3.json, figs/zng4g_d1e3
"""
import glob
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from analysis.common.readers import site_python  # noqa: E402
from zng4g_readout import driven, CC  # noqa: E402

ARMS = {"S1k": "ftmech_zng4g_d1e3_scr_b128", "Fn1k": "ftmech_zng4g_d1e3_ftnoset_b128",
        "Fg1k": "ftmech_zng4g_d1e3_ftglu_b128"}
NARROW, WIDE, N_FIRST = (1.9e-3, 1.9e-2), (6e-4, 6e-2), 16

REMOTE = r'''
import glob, json, os, sys, yaml
out = {}
for a in sys.argv[1:]:
    lab, d, res = a.split("#")
    r = {}
    try:
        r["lr"] = yaml.safe_load(open(os.path.join(d, "config.yaml")))["training"]["lr"]
    except Exception as e:
        r["lr_error"] = repr(e)[:100]
    try:
        r["std"] = json.load(open(os.path.join(d, "data_stats.json")))["prepd_std"][0]
    except Exception as e:
        r["std_error"] = repr(e)[:100]
    try:
        res = sorted(glob.glob(res))[0] if "*" in res else res
        j = json.load(open(res)); r.update(val_loss=j.get("val_loss"), val_mse_curve=j.get("val_mse_curve"))
    except Exception as e:
        r["res_error"] = repr(e)[:100]
    out[lab] = r
print("ZNG1K " + json.dumps(out))
'''


def q(xs):
    return [float(x) for x in np.quantile(xs, [0.1, 0.5, 0.9])]


def ratio(a, b):
    la, lb = np.log(a), np.log(b)
    lr = la.mean() - lb.mean()
    sig = np.sqrt((la.std(ddof=1) ** 2 + lb.std(ddof=1) ** 2) / 3)
    return {"r": float(np.exp(lr)), "lo": float(np.exp(lr - 2 * sig)), "hi": float(np.exp(lr + 2 * sig)),
            "sigma_r": float(sig), "q10_50_90": q([x / y for x, y in itertools.product(a, b)])}


def collect():
    items = {"ccin2p3": [], "lxplus": [], "jeanzay": []}
    for arm, name in ARMS.items():
        if not os.path.isdir(os.path.join(ROOT, "sweeps_local", name)):
            continue
        for ti, hp, v, site, rd, rp in driven(name):
            items[site].append(f"'{arm}|t{ti}|hp{hp}#{rd}#{rp}'")
        # seed runs: <sweep>_s<s> (first pick) and <sweep>_hp<idx>_s<s> (re-picks), all on CC
        for s in (43, 44):
            for n in [f"{name}_s{s}"] + [f"{name}_hp{t[1]:04d}_s{s}" for t in driven(name)]:
                items["ccin2p3"].append(f"'{arm}|s{s}|{n}#{CC}/runs/{n}#{CC}/sweeps/{n}/results/fixed.json'")
    raw = {}
    for site, args in items.items():
        if args:
            raw.update({k: dict(v, site=site) for k, v in site_python(REMOTE, args, "ZNG1K", site=site).items()})
    return raw


def main():
    cache = os.path.join(HERE, "data", "zng4g_d1e3_raw.json")
    raw = json.load(open(cache)) if "--cached" in sys.argv else collect()
    json.dump(raw, open(cache, "w"))
    res, curves = {"arms": {}}, {}
    for arm in ARMS:
        ts = {k: v for k, v in raw.items() if k.startswith(arm + "|t")}
        if not ts:
            continue
        rows = sorted((int(k.split("|")[1][1:]), int(k.split("|")[2][2:]), v) for k, v in ts.items())
        bad = [r[:2] for r in rows if r[2].get("val_loss") is None or "std" not in r[2] or "lr" not in r[2]]
        ok = [r for r in rows if r[:2] not in bad]
        for _, _, v in ok:
            assert abs(min(v["val_mse_curve"]) / v["val_loss"] - 1) < 1e-6
        std = ok[0][2]["std"]
        assert all(abs(v["std"] / std - 1) < 1e-9 for _, _, v in ok), "one pool per arm"
        k2 = std ** 2
        first = ok[:N_FIRST]   # rows are in suggestion order
        ch16 = min(first, key=lambda r: r[2]["val_loss"])
        ch = min(ok, key=lambda r: r[2]["val_loss"])
        lr = ch[2]["lr"]
        seeds = {}
        for k, v in raw.items():   # the seed runs of the chosen trial: same lr
            if k.startswith(arm + "|s") and v.get("val_mse_curve") and abs(v.get("lr", 0) / lr - 1) < 1e-9:
                assert abs(v["std"] / std - 1) < 1e-9
                seeds[int(k.split("|")[1][1:])] = v["val_mse_curve"]
        cs = {42: ch[2]["val_mse_curve"], **seeds}
        curves[arm] = {s: np.array(c) * k2 for s, c in sorted(cs.items())}
        pos = (np.log(lr) - np.log(WIDE[0])) / (np.log(WIDE[1]) - np.log(WIDE[0]))
        res["arms"][arm] = {
            "sweep": ARMS[arm], "n_trials": len(rows), "n_failed": len(bad), "std": std,
            "chosen": {"trial": ch[0], "hp": ch[1], "lr": lr, "val": ch[2]["val_loss"] * k2,
                       "log_pos_wide": float(pos), "dec_from_edge": float(min(np.log10(lr / WIDE[0]), np.log10(WIDE[1] / lr)))},
            "chosen16": {"trial": ch16[0], "hp": ch16[1], "lr": ch16[2]["lr"], "val": ch16[2]["val_loss"] * k2},
            "moved": ch[1] != ch16[1], "seeds": sorted(cs),
            "best": [float(c.min()) for c in curves[arm].values()],
            "best_at": [(int(np.argmin(c)) + 1) / len(c) for c in curves[arm].values()],
            "blowups": [s for s, c in curves[arm].items() if np.any(c > 10 * np.minimum.accumulate(c))],
            "trials": [(r[2]["lr"], r[2]["val_loss"] * k2, r[0], r[1]) for r in ok]}
    A = res["arms"]
    full = [a for a in A if len(A[a]["best"]) == 3]
    for num, den in (("Fn1k", "S1k"), ("Fg1k", "Fn1k"), ("Fg1k", "S1k")):
        if num in full and den in full:
            res[f"{num}/{den}"] = ratio(np.array(A[num]["best"]), np.array(A[den]["best"]))
    if "Fn1k/S1k" in res:
        x = res["Fn1k/S1k"]
        res["Fn1k/S1k"]["read"] = "gain" if x["hi"] < 1 else "cost" if x["lo"] > 1 else "unresolved"
    if "Fg1k/Fn1k" in res:
        x = res["Fg1k/Fn1k"]
        res["Fg1k/Fn1k"]["read"] = ("dilution yes" if x["hi"] < 1 else "gluon worse" if x["lo"] > 1 else
                                    "no effect of the needed size" if x["lo"] > 0.65 else "unresolved")
    json.dump(res, open(os.path.join(HERE, "data", "zng4g_d1e3.json"), "w"), indent=1)
    print(json.dumps({k: ({a: {kk: vv for kk, vv in v.items() if kk != "trials"} for a, v in A.items()} if k == "arms" else v)
                      for k, v in res.items()}))   # one line: verify_claims reads the last {...} line
    plot(res, curves)


def plot(res, curves):
    import plot_style as ps
    from matplotlib.ticker import NullFormatter
    sty = {"S1k": ("from scratch", ps.C.blue, "o"), "Fn1k": ("fine-tuned,\n28-process parent", ps.C.green, "^"),
           "Fg1k": ("fine-tuned,\ngluon parent", ps.C.vermillion, "s")}
    fig, axs = ps.figure(ncols=2, nrows=2)
    a, l, c, leg = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
    l.axvspan(*WIDE, color=ps.C.grey, alpha=0.12, lw=0, label="wide lr window")
    l.axvspan(*NARROW, color=ps.C.grey, alpha=0.25, lw=0, label="narrow lr window")
    for i, (arm, r) in enumerate(res["arms"].items()):
        lab, col, m = sty[arm]
        v = r["best"]
        a.plot(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, m, color=col, label=lab)
        for cv in curves[arm].values():
            c.plot((np.arange(len(cv)) + 1) / len(cv), cv, color=col)
        l.plot([t[0] for t in r["trials"]], [t[1] for t in r["trials"]], m, color=col, mfc="none")
    a.set_xticks([]); a.set_xlim(-0.5, len(res["arms"]) - 0.5)   # arms named by the legend
    a.set_yscale("log"); a.set_xlabel("arm, three seeds"); a.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    l.set_xscale("log"); l.set_yscale("log"); l.xaxis.set_minor_formatter(NullFormatter())
    l.set_xlabel("learning rate"); l.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    c.set_yscale("log"); c.set_xlabel("fraction of the run")
    c.set_ylabel(r"validation MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(a, r"$q\bar q\to Z+4g$")
    ps.legend_cell(leg, [a, l])
    ps.save(fig, os.path.join(HERE, "figs", "zng4g_d1e3"))


if __name__ == "__main__":
    main()
