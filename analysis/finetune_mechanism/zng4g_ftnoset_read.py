"""Plan finetune_mechanism step 5 (design passed 2026-11-23T15:32, re-recorded 2026-11-24): the fine-tune Fn of the W
parent retrained WITHOUT input_soft_eps / input_pair_logs (FA-ccin2p3-9a73698a) on q q~ -> Z+4g at 1e4 events, read
against scratch S and the old fine-tune F (parent with both settings), per batch; plus the b128 share re-read against
the extended scratch search's hp0078 (zng_uq_benchmark finding 2026-11-24T03:53).

Measure: best val MSE(log|M|^2) = min val mu-MSE x std^2 (std 1.11978). Fn: the 16-trial search's chosen trial (seed 42)
and its seeds 43, 44 (runs/ftmech_zng4g_ftnoset_b<B>_s<s> on CC, missing ones skipped and counted). Ratios of three-seed
means with the 10-90% range over the seed pairings. S at b2048: the D507 chosen trial's three seeds; at b128: hp0078's
three seeds (analysis/zng_uq/data/d507_v2.json, d507_search_v2.json), the 8-trial trial_0950 kept as 'S8'.
Sf* and F from zng4g_r1_b<B>.json.

    python3 analysis/finetune_mechanism/zng4g_ftnoset_read.py 2048|128 [--cached]
        -> data/zng4g_ftnoset_b<B>.json, figs/zng4g_ftnoset_b<B>
"""
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
from zng4g_readout import analyse, driven, CC  # noqa: E402
from zng4g_r1_read import REMOTE  # noqa: E402

YES_BAR, PARTIAL_BAR = 1.19, 0.79   # design outcome: Fn/S <= 1.19 yes; Fn/F < 0.79 partial


def q(xs):
    return [float(x) for x in np.quantile(xs, [0.1, 0.5, 0.9])]


def ratio(num, den):
    return {"mean": float(np.mean(num) / np.mean(den)), "q10_50_90": q([a / b for a, b in itertools.product(num, den)])}


def main():
    b = sys.argv[1]
    name = f"ftmech_zng4g_ftnoset_b{b}"
    cache = os.path.join(HERE, "data", f"zng4g_ftnoset_raw_b{b}.json")
    if "--cached" in sys.argv:
        raw = json.load(open(cache))
    else:
        items = {"ccin2p3": [], "lxplus": []}
        for ti, hp, v, site, rd, rp in driven(name):
            items[site].append(f"'Fn|{b}|t{ti}|hp{hp}#{rd}#{rp}'")
        for s in (43, 44):
            n = f"{name}_s{s}"
            items["ccin2p3"].append(f"'Fn|{b}|s{s}#{CC}/runs/{n}#{CC}/sweeps/{n}/results/fixed.json'")
        raw = {}
        for site, args in items.items():
            if args:
                raw.update({k: dict(v, site=site) for k, v in site_python(REMOTE, args, "ZNGR1", site=site).items()})
        json.dump(raw, open(cache, "w"))
    base = analyse(json.load(open(os.path.join(HERE, "data", "zng4g_readout_raw.json"))))[b]
    k2 = base["std"] ** 2
    r1 = json.load(open(os.path.join(HERE, "data", f"zng4g_r1_b{b}.json")))
    r1raw = json.load(open(os.path.join(HERE, "data", f"zng4g_r1_raw_b{b}.json")))
    ts = [(int(k.split("|")[2][1:]), v) for k, v in raw.items() if k.startswith(f"Fn|{b}|t")]
    assert len(ts) == 16 and all(isinstance(v["lr"], float) and v["val_loss"] is not None for _, v in ts), len(ts)
    ti, ch = min(ts, key=lambda t: t[1]["val_loss"])
    assert abs(min(ch["val_mse_curve"]) / ch["val_loss"] - 1) < 1e-6
    fn_seeds = [42] + [s for s in (43, 44) if raw.get(f"Fn|{b}|s{s}", {}).get("val_mse_curve")]
    fn_curves = [ch["val_mse_curve"]] + [raw[f"Fn|{b}|s{s}"]["val_mse_curve"] for s in fn_seeds[1:]]
    curves = {"Fn": [np.array(c) * k2 for c in fn_curves], "F": [np.array(c) for c in base["curves"]["F"]]}
    best = {"Fn": [float(c.min()) for c in curves["Fn"]], "F": r1["best"]["F"], "Sf": r1["best"]["Sf"]}
    if b == "128":
        d = json.load(open(os.path.join(ROOT, "analysis", "zng_uq", "data", "d507_v2.json")))
        srch = json.load(open(os.path.join(ROOT, "analysis", "zng_uq", "data", "d507_search_v2.json")))
        t42 = next(t for t in srch["tp_zng_d1e4_muon_b128_zgggg"]["trials"] if t["run_dir"].endswith("trial_0078"))
        sc = [t42["val"], d["muon_b128|43"]["val_mse"], d["muon_b128|44"]["val_mse"]]
        assert all(abs(float(d[f"muon_b128|{s}"]["std"]) / base["std"] - 1) < 1e-9 for s in (43, 44))
        curves["S"] = [np.array(c) * k2 for c in sc]
        best["S8"] = r1["best"]["S"]
    else:
        curves["S"] = [np.array(c) for c in base["curves"]["S"]]
    best["S"] = [float(c.min()) for c in curves["S"]]
    mean = {a: float(np.mean(v)) for a, v in best.items()}
    res = {"batch": b, "std": base["std"], "best": best, "mean": mean, "n_seeds_Fn": len(fn_curves),
           "Fn_chosen": {"trial": ti, "hp": next(k.split("|")[3] for k, v in raw.items() if v is ch), "lr": ch["lr"]},
           "Fn_over_S": ratio(best["Fn"], best["S"]), "Fn_over_F": ratio(best["Fn"], best["F"]),
           "F_over_S": ratio(best["F"], best["S"]), "Sf_over_S": ratio(best["Sf"], best["S"])}
    sh = [np.log(a / s) / np.log(f / s) for a, s, f in itertools.product(best["Sf"], best["S"], best["F"])]
    res["share"] = {"mean": float(np.log(mean["Sf"] / mean["S"]) / np.log(mean["F"] / mean["S"])), "q10_50_90": q(sh)}
    if b == "128":
        res["Fn_over_S8"] = ratio(best["Fn"], best["S8"])
    fs, ff = res["Fn_over_S"]["mean"], res["Fn_over_F"]["mean"]
    res["branch"] = ("yes" if fs <= YES_BAR else "partial" if ff < PARTIAL_BAR else "unresolved") + " (this batch)"
    seeds = {a: fn_seeds if a == "Fn" else [42, 43, 44] for a in curves}   # each run's real seed
    assert all(len(seeds[a]) == len(cs) for a, cs in curves.items()), {a: len(cs) for a, cs in curves.items()}
    res["blowups"] = {a: [seeds[a][i] for i, c in enumerate(cs) if np.any(c > 10 * np.minimum.accumulate(c))]
                      for a, cs in curves.items()}
    res["best_at"] = {a: [(int(np.argmin(c)) + 1) / len(c) for c in cs] for a, cs in curves.items()}
    res["trials"] = {"Fn": sorted((v["lr"], v["val_loss"] * k2, i) for i, v in ts),
                     "F": sorted((t[0], t[1], t[2]) for t in r1["trials"]["Fx"] if t[1] and t[2] < 16)}
    json.dump(res, open(os.path.join(HERE, "data", f"zng4g_ftnoset_b{b}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "trials"}))   # one line: verify_claims reads the last {...} line
    plot(res, curves)


def plot(res, curves):
    import plot_style as ps
    from matplotlib.ticker import NullFormatter
    b = res["batch"]
    arms = [("S", "from scratch", ps.C.blue, "o"), ("Sf", "from scratch +\nboth settings", ps.C.grey, "D"),
            ("F", "fine-tuned,\nparent with settings", ps.C.vermillion, "s"),
            ("Fn", "fine-tuned,\nparent without settings", ps.C.green, "^")]
    if b == "128":
        arms.insert(1, ("S8", "from scratch,\n8-trial search", ps.C.blue, "o"))
    fig, axs = ps.figure(ncols=2, nrows=2)
    a, l, c, leg = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
    for i, (k, lab, col, m) in enumerate(arms):
        v = res["best"][k]
        a.plot(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, m, color=col, label=lab,
               mfc="none" if k == "S8" else col)
        if k in curves:
            for cv in curves[k]:
                c.plot((np.arange(len(cv)) + 1) / len(cv), cv, color=col)
        if k in res["trials"]:
            l.plot([t[0] for t in res["trials"][k]], [t[1] for t in res["trials"][k]], m, color=col)
    a.set_xticks([]); a.set_xlim(-0.5, len(arms) - 0.5)   # arms named by the legend
    a.set_yscale("log"); a.set_xlabel(f"arm, batch {b}"); a.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    l.set_xscale("log"); l.set_yscale("log"); l.xaxis.set_minor_formatter(NullFormatter())
    l.set_xlabel("learning rate"); l.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    c.set_yscale("log"); c.set_xlabel(f"fraction of the run, batch {b}")
    c.set_ylabel(r"validation MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(a, r"$q\bar q\to Z+4g$")
    ps.legend_cell(leg, [a])
    ps.save(fig, os.path.join(HERE, "figs", f"zng4g_ftnoset_b{b}"))


if __name__ == "__main__":
    main()
