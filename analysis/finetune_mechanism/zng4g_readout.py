"""Plan finetune_mechanism step 1 readout on q q~ -> Z+4g (D515 design, passed 2026-11-20T16:10).

Arms per batch (2048, 128): S = the D507 Muon/WSD scratch cell's chosen trial + seeds 43, 44 (analysis/zng_uq/data/
d507_search.json, d507.json); S+flags = scratch + the parent's two input flags at S's HPs, seeds 42-44
(runs/ftmech_zng4g_scrflags_b<B>_s<s>); F = the template fine-tune search's chosen trial (seed 42) + its seeds 43, 44
(runs/ftmech_zng4g_ft_b<B>_s<s>); Sg = the sigma-head-reset search (its best trial, one seed).
Measure: best val MSE(log|M|^2) = min of the val mu-MSE curve x std^2, std = data_stats.json prepd_std of each run
(checked equal across arms). F/S = three-seed means, with the range over the 9 seed pairings; the head start = the
step fraction at which an arm's three-seed mean best-so-far curve first reaches 1.2 x S's mean best; step 0 = the
first train_steps.npz record (het. NLL before any update, lr 0) and the max pre-clip gradient norm over steps 0-40,
per arm the median over its runs. A batch is read only when its F has three seeds and its search has ended.

    python3 analysis/finetune_mechanism/zng4g_readout.py [--cached]   -> data/zng4g_readout.json, figs/zng4g_*
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from analysis.common.readers import site_python  # noqa: E402

CC = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
S_TRIAL = {"2048": "trial_1333", "128": "trial_0950"}
N_FT = 16
FIRST = 41

REMOTE = r'''
import json, os, sys
import numpy as np
out = {}
for a in sys.argv[1:]:
    lab, d, res = a.split("|")
    r = {"run_dir": d}
    try:
        r["std"] = json.load(open(os.path.join(d, "data_stats.json")))["prepd_std"][0]
    except Exception as e:
        r["std_error"] = repr(e)[:120]
    try:
        z = np.load(os.path.join(d, "train_steps.npz"))
        r["loss"] = z["loss"][:%d].tolist(); r["grad_norm"] = z["grad_norm"][:%d].tolist()
    except Exception as e:
        r["steps_error"] = repr(e)[:120]
    if res:
        try:
            j = json.load(open(res))
            r.update({k: j.get(k) for k in ("val_loss", "val_mse_curve", "validate_every", "best_step", "val_loss_step0")})
        except Exception as e:
            r["res_error"] = repr(e)[:120]
    out[lab] = r
print("ZNG4G " + json.dumps(out))
''' % (FIRST, FIRST)


def driven(name):
    """A driven search's trials: (trial_idx, hp_idx, val_loss, site, run_dir, result_path) in suggestion order."""
    st = json.load(open(os.path.join(ROOT, "sweeps_local", name, "driver_state.json")))["trials"]
    site = {v["hp_idx"]: v["site"] for v in st.values() if v.get("status") == "observed"}
    rows = []
    for f in glob.glob(os.path.join(ROOT, "sweeps_local", name, "results", "*.json")):
        t = json.load(open(f))
        rows.append((t["trial_idx"], t["hp_idx"], t["val_loss"], site.get(t["hp_idx"], "ccin2p3"), t["run_dir"],
                     t["result_path"]))
    return sorted(rows)


def collect():
    items = {"ccin2p3": [], "lxplus": []}
    meta = {}
    for b in ("2048", "128"):
        for arm, name in (("F", f"ftmech_zng4g_ft_b{b}"), ("Sg", f"ftmech_zng4g_sigreset_b{b}")):
            for ti, hp, v, site, rd, rp in driven(name):
                lab = f"{arm}|{b}|t{ti}"
                items[site].append(f"'{lab}|{rd}|{rp}'")
                meta[lab] = {"hp": hp, "trial": ti, "val_loss": v}
        for s in (43, 44):
            n = f"ftmech_zng4g_ft_b{b}_s{s}"
            items["ccin2p3"].append(f"'F|{b}|s{s}|{CC}/runs/{n}|{CC}/sweeps/{n}/results/fixed.json'")
        for s in (42, 43, 44):
            n = f"ftmech_zng4g_scrflags_b{b}_s{s}"
            items["ccin2p3"].append(f"'Sf|{b}|s{s}|{CC}/runs/{n}|{CC}/sweeps/{n}/results/fixed.json'")
        sw = f"tp_zng_d1e4_muon_b{b}_zgggg"
        items["ccin2p3"].append(f"'S|{b}|s42|{CC}/runs/{sw}/{S_TRIAL[b]}|'")
        for s in (43, 44):
            items["ccin2p3"].append(f"'S|{b}|s{s}|{CC}/runs/{sw}_best_s{s}|'")
    # labels with "|" inside: re-join the first fields (arm|batch|tag) on the site side
    out = {}
    for site, args in items.items():
        if not args:
            continue
        args = [a.replace("|", "#", 2) for a in args]
        r = site_python(REMOTE, args, "ZNG4G", site=site)
        out.update({k.replace("#", "|"): dict(v, site=site, **meta.get(k.replace("#", "|"), {})) for k, v in r.items()})
    return out


def arm_runs(raw, arm, b):
    return {k.split("|")[2]: v for k, v in raw.items() if k.startswith(f"{arm}|{b}|")}


def analyse(raw):
    d507 = json.load(open(os.path.join(ROOT, "analysis", "zng_uq", "data", "d507.json")))
    srch = json.load(open(os.path.join(ROOT, "analysis", "zng_uq", "data", "d507_search.json")))
    res = {}
    for b in ("2048", "128"):
        c = f"muon_b{b}"
        stds = {f"{k}": v["std"] for k, v in raw.items() if k.split("|")[1] == b and "std" in v}
        std = d507[f"{c}|43"]["std"]
        dev = max(abs(v / std - 1) for v in stds.values())
        # S curves: seed 42 the searched trial, seeds 43, 44 the D509 repeats (standardised mu-MSE)
        t42 = next(t for t in srch[f"tp_zng_d1e4_{c}_zgggg"]["trials"] if t["run_dir"].endswith(S_TRIAL[b]))
        curves = {"S": [t42["val"], d507[f"{c}|43"]["val_mse"], d507[f"{c}|44"]["val_mse"]]}
        ft = arm_runs(raw, "F", b)
        trials = sorted((v for k, v in ft.items() if k.startswith("t")), key=lambda v: v["trial"])
        done = [v for v in trials if v.get("val_loss") is not None]
        chosen = min(done, key=lambda v: v["val_loss"])
        seeds = [ft.get("s43"), ft.get("s44")]
        f_ready = len(trials) == N_FT and all(s and s.get("val_mse_curve") for s in seeds)
        curves["F"] = [chosen["val_mse_curve"]] + [s["val_mse_curve"] for s in seeds if s and s.get("val_mse_curve")]
        sf = arm_runs(raw, "Sf", b)
        curves["Sf"] = [sf[f"s{s}"]["val_mse_curve"] for s in (42, 43, 44)]
        sg = sorted(arm_runs(raw, "Sg", b).values(), key=lambda v: v["val_loss"])
        curves["Sg"] = [sg[0]["val_mse_curve"]]
        best = {a: [min(cv) * std ** 2 for cv in cs] for a, cs in curves.items()}
        mean = {a: float(np.mean(v)) for a, v in best.items()}
        bar = 1.2 * mean["S"]

        def reach(cs):
            m = np.mean([np.minimum.accumulate(np.array(cv) * std ** 2) for cv in cs], axis=0)
            i = np.nonzero(m <= bar)[0]
            return float((i[0] + 1) / len(m)) if len(i) else None

        r = {"std": std, "std_max_rel_dev": dev, "n_std": len(stds), "f_ready": f_ready,
             "F_chosen": {"hp": chosen["hp"], "trial": chosen["trial"]}, "n_ft_trials": len(done),
             "best": best, "mean": mean, "reach_1p2S": {a: reach(cs) for a, cs in curves.items()},
             "curves": {a: [list(np.array(cv) * std ** 2) for cv in cs] for a, cs in curves.items()}}
        if len(best["F"]) == 3:
            r["F_over_S"] = {"mean": mean["F"] / mean["S"], "min": min(best["F"]) / max(best["S"]),
                             "max": max(best["F"]) / min(best["S"])}
            r["F_over_Sf"] = {"mean": mean["F"] / mean["Sf"], "min": min(best["F"]) / max(best["Sf"]),
                              "max": max(best["F"]) / min(best["Sf"])}
            ctl = "Sf" if mean["Sf"] <= mean["S"] else "S"
            fr = r["F_over_Sf"] if ctl == "Sf" else r["F_over_S"]
            r["control"] = ctl
            r["verdict"] = "HIGHER" if fr["min"] > 1 else ("LOWER" if fr["max"] < 1 else "SAME")
        # step 0: het. NLL at the first record (lr 0, before any update) and the max grad norm over steps 0-40
        st0 = {}
        for a in ("S", "F", "Sg", "Sf"):
            rs = [v for v in arm_runs(raw, a, b).values() if v.get("loss")]
            st0[a] = {"nll0": float(np.median([v["loss"][0] for v in rs])),
                      "nll0_range": [float(min(v["loss"][0] for v in rs)), float(max(v["loss"][0] for v in rs))],
                      "gmax": float(np.median([max(v["grad_norm"]) for v in rs])), "n": len(rs)}
        r["step0"] = st0
        res[b] = r
    return res


def plot(res):
    import plot_style as ps
    names = {"S": ("from scratch", ps.C.blue, "-", "o"), "Sf": ("from scratch + input flags", ps.C.grey, ":", "D"),
             "F": ("fine-tuned", ps.C.vermillion, "-", "s"), "Sg": ("fine-tuned, sigma head reset", ps.C.green, "--", "^")}
    fig, axs = ps.figure(ncols=2, nrows=2)
    for j, b in enumerate(("2048", "128")):
        r, ax, bx = res[b], axs[0][j], axs[1][j]
        for i, a in enumerate(("S", "Sf", "F", "Sg")):
            lab, col, ls, m = names[a]
            cs = r["curves"][a]
            mcur = np.mean([np.minimum.accumulate(cv) for cv in cs], axis=0)
            x = (np.arange(len(mcur)) + 1) / len(mcur)
            ax.plot(x, mcur, color=col, linestyle=ls, label=lab)
            v = r["best"][a]
            bx.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, m, color=col)
        ax.set_yscale("log"); bx.set_yscale("log")
        ax.set_xlabel(f"fraction of the run, batch {b}")
        ax.set_ylabel(r"best-so-far MSE$(\log|\mathcal{M}|^2)$")
        bx.set_xticks([0, 1, 2, 3]); bx.set_xticklabels(["scratch", "+flags", "fine-tuned", "reset"]); bx.set_xlim(-0.5, 3.5)
        bx.set_xlabel(f"arm, batch {b}")
        bx.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(axs[0][0], r"$q\bar q\to Z+4g$")
    ps.legend(axs[0][1], "upper right")
    ps.save(fig, os.path.join(HERE, "figs", "zng4g_ft_vs_scratch"))

    fig, (l, rr) = ps.figure(ncols=2)
    for k, (b, m) in enumerate((("2048", "o"), ("128", "s"))):
        for i, a in enumerate(("S", "Sf", "F", "Sg")):
            s = res[b]["step0"][a]
            col = names[a][1]
            l.plot([i + 0.1 * (2 * k - 1)], [s["nll0"]], m, color=col)
            rr.plot([i + 0.1 * (2 * k - 1)], [s["gmax"]], m, color=col)
        l.plot([], [], m, color="black", label=f"batch {b}")
    for ax in (l, rr):
        ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["scratch", "+flags", "fine-tuned", "reset"]); ax.set_xlim(-0.5, 3.5)
        ax.set_xlabel("arm")
    l.set_yscale("log")   # every step-0 NLL here is positive (smallest 0.69)
    rr.set_yscale("log")
    ps.headroom(l); ps.headroom(rr)
    l.set_ylabel("training NLL at step 0")
    rr.set_ylabel("max gradient norm, steps 0-40")
    ps.legend(l, "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "zng4g_step0"))


def main():
    cache = os.path.join(HERE, "data", "zng4g_readout_raw.json")
    if "--cached" in sys.argv:
        raw = json.load(open(cache))
    else:
        raw = collect()
        json.dump(raw, open(cache, "w"))
    errs = {k: {e: v[e] for e in ("std_error", "steps_error", "res_error") if e in v} for k, v in raw.items()}
    errs = {k: v for k, v in errs.items() if v}
    if errs:
        print("read errors:", json.dumps(errs, indent=1))
    res = analyse(raw)
    json.dump({b: {k: v for k, v in r.items() if k != "curves"} for b, r in res.items()},
              open(os.path.join(HERE, "data", "zng4g_readout.json"), "w"), indent=1)
    for b, r in res.items():
        print(f"b{b}: std {r['std']:.5f} (max rel dev over {r['n_std']} runs {r['std_max_rel_dev']:.1e}); "
              f"F ready {r['f_ready']} (chosen hp{r['F_chosen']['hp']}, {r['n_ft_trials']} trials)")
        print("  best:", {a: [round(x, 5) for x in v] for a, v in r["best"].items()})
        print("  F/S:", r.get("F_over_S"), " F/Sf:", r.get("F_over_Sf"), " control:", r.get("control"),
              " verdict:", r.get("verdict"))
        print("  reach 1.2 S:", r["reach_1p2S"])
        print("  step0:", r["step0"])
    plot(res)


if __name__ == "__main__":
    main()
