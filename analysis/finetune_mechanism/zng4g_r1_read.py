"""Plan finetune_mechanism R1 + R3 readout on q q~ -> Z+4g (design passed 2026-11-22T03:10), one batch at a time.

Arms: S = the D507 Muon/WSD scratch search (16 trials) and its chosen trial's three seeds; Sf* = the searched scratch
with the parent's two input flags (sweeps_local/ftmech_zng4g_scrflagsrch_b<B>, 16 trials) and its chosen trial's seeds
43, 44 (runs/ftmech_zng4g_scrflagsrch_b<B>_s<s>); F = the 16-trial fine-tune search's chosen trial and its seeds
(zng4g_readout.analyse on its cached raw data); R3 = the same fine-tune search extended to 24 trials with a wider lr
window (read for H3 only: its chosen trial's lr and val, no ratio). Measure: best val MSE(log|M|^2) = min val mu-MSE x
std^2 (std 1.11978, shared). share = log(mean Sf*/mean S) / log(mean F/mean S), with its 10-90% range over the seed
triples (one seed each of Sf*, S, F). Each trial's lr is read from its run's saved config.yaml.

    python3 analysis/finetune_mechanism/zng4g_r1_read.py 2048 [--cached]  -> data/zng4g_r1_b<B>.json, figs/zng4g_r1_b<B>
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
from zng4g_readout import analyse, driven, CC, S_TRIAL  # noqa: E402

OLD_FT_WINDOW = {"2048": (3.1e-3, 4.1e-2), "128": (2.7e-4, 1.1e-2)}   # sweep_config_ftmech_zng4g_ft_b<B>.yaml @ab57ac74^
SF_WINDOW = {"2048": (3.1e-3, 4.1e-2), "128": (2.7e-4, 1.1e-2)}      # sweep_config_ftmech_zng4g_scrflagsrch_b<B>.yaml
NEW_FT_WINDOW = {"2048": (1.0e-3, 4.1e-2), "128": (2.7e-4, 3.3e-2)}   # same configs @ab57ac74
SEED_BAR = 1.19   # R3's seed spread bar (design outcome)

REMOTE = r'''
import glob, json, os, sys, yaml
out = {}
def lr(d):
    try:
        return yaml.safe_load(open(os.path.join(d, "config.yaml")))["training"]["lr"]
    except Exception as e:
        return repr(e)[:100]
for a in sys.argv[1:]:
    lab, d, res = a.split("#")
    if lab.startswith("GLOB"):     # a classic sweep on this site: res is its results dir, d its runs dir
        for f in sorted(glob.glob(os.path.join(res, "hp*_t*.json"))):
            hp = int(os.path.basename(f)[2:6]); j = json.load(open(f))
            out[f"{lab[4:]}|t{hp}"] = {"hp": hp, "val_loss": j.get("val_loss"), "lr": lr(os.path.join(d, "trial_%04d" % hp))}
        continue
    r = {"lr": lr(d)}
    try:
        res = sorted(glob.glob(res))[0] if "*" in res else res
        j = json.load(open(res)); r.update(val_loss=j.get("val_loss"), val_mse_curve=j.get("val_mse_curve"),
                                           train_mse_curve=j.get("train_eval_mse_curve"))
    except Exception as e:
        r["res_error"] = repr(e)[:100]
    out[lab] = r
print("ZNGR1 " + json.dumps(out))
'''


def collect(b):
    items = {"ccin2p3": [], "lxplus": []}
    for arm, name in (("Sfx", f"ftmech_zng4g_scrflagsrch_b{b}"), ("Fx", f"ftmech_zng4g_ft_b{b}")):
        for ti, hp, v, site, rd, rp in driven(name):
            items[site].append(f"'{arm}|{b}|t{ti}|hp{hp}#{rd}#{rp}'")
    for arm, name in (("Sfx", f"ftmech_zng4g_scrflagsrch_b{b}"), ("Fx", f"ftmech_zng4g_ftx24_b{b}")):
        for s in (43, 44):
            n = f"{name}_s{s}"
            items["ccin2p3"].append(f"'{arm}|{b}|s{s}#{CC}/runs/{n}#{CC}/sweeps/{n}/results/fixed.json'")
    sw = f"tp_zng_d1e4_muon_b{b}_zgggg"
    # train-vs-val curves of S's and F's three seeds (Sf*'s come with its trial and seed entries above)
    hp = int(S_TRIAL[b].split("_")[1])
    items["ccin2p3"].append(f"'TVS|{b}|s42#{CC}/runs/{sw}/{S_TRIAL[b]}#{CC}/sweeps/{sw}/results/hp{hp:04d}_t*.json'")
    fsite, frd, frp = next((site, rd, rp) for ti, h, v, site, rd, rp in driven(f"ftmech_zng4g_ft_b{b}") if ti < 16 and
                           v == min(x[2] for x in driven(f"ftmech_zng4g_ft_b{b}") if x[0] < 16 and x[2] is not None))
    items[fsite].append(f"'TVF|{b}|s42#{frd}#{frp}'")
    for s in (43, 44):
        items["ccin2p3"].append(f"'TVS|{b}|s{s}#{CC}/runs/{sw}_best_s{s}#{CC}/sweeps/{sw}_best_s{s}/results/fixed.json'")
        n = f"ftmech_zng4g_ft_b{b}_s{s}"
        items["ccin2p3"].append(f"'TVF|{b}|s{s}#{CC}/runs/{n}#{CC}/sweeps/{n}/results/fixed.json'")
    items["ccin2p3"].append(f"'GLOBS|{b}#{CC}/runs/{sw}#{CC}/sweeps/{sw}/results'")
    out = {}
    for site, args in items.items():
        out.update({k: dict(v, site=site) for k, v in site_python(REMOTE, args, "ZNGR1", site=site).items()})
    return out


def triples_share(sf, s, f):
    sh = [np.log(a / b_) / np.log(c / b_) for a, b_, c in itertools.product(sf, s, f)]
    return [float(x) for x in np.quantile(sh, [0.1, 0.5, 0.9])], len(sh)


def h3(b):
    """R3 alone (the lr-window extension of the fine-tune search): its chosen trial's lr and val against the 16-trial
    best; for a batch whose scratch+flags search was dropped (b128, check 0, 2026-11-22)."""
    cache = os.path.join(HERE, "data", f"zng4g_r3_raw_b{b}.json")
    raw = json.load(open(cache)) if "--cached" in sys.argv else collect(b)
    json.dump(raw, open(cache, "w"))
    k2 = analyse(json.load(open(os.path.join(HERE, "data", "zng4g_readout_raw.json"))))[b]["std"] ** 2
    ts = [(int(k.split("|")[2][1:]), v) for k, v in raw.items() if k.startswith(f"Fx|{b}|t")]
    assert len(ts) == 24 and all(isinstance(v["lr"], float) and v["val_loss"] is not None for _, v in ts), len(ts)
    ti, fx = min(ts, key=lambda x: x[1]["val_loss"])
    f16 = min(v["val_loss"] for i, v in ts if i < 16)
    ow, nw = OLD_FT_WINDOW[b], NEW_FT_WINDOW[b]
    inside = ow[0] <= fx["lr"] <= ow[1]
    edge = min(np.log10(fx["lr"] / nw[0]), np.log10(nw[1] / fx["lr"])) < 0.1
    r = {"batch": b, "chosen_trial": ti, "chosen_lr": fx["lr"], "chosen_val": fx["val_loss"] * k2, "f16_best": f16 * k2,
         "f16_over_new": f16 / fx["val_loss"], "inside_old_window": inside, "at_new_edge": bool(edge),
         "H3": "unresolved (new edge)" if (edge and not inside) else
               ("yes" if (not inside and f16 / fx["val_loss"] > SEED_BAR) else "ruled out"),
         "trials": sorted((v["lr"], v["val_loss"] * k2, i) for i, v in ts)}
    json.dump(r, open(os.path.join(HERE, "data", f"zng4g_r3_b{b}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "trials"}, indent=1))
    import plot_style as ps
    fig, ax = ps.figure()
    old = [t for t in r["trials"] if t[2] < 16]; new = [t for t in r["trials"] if t[2] >= 16]
    ax.plot([t[0] for t in old], [t[1] for t in old], "s", color=ps.C.vermillion, label="first 16 trials")
    ax.plot([t[0] for t in new], [t[1] for t in new], "s", color=ps.C.vermillion, mfc="none", label="8 added trials")
    for e in ow:
        ax.axvline(e, color=ps.C.vermillion, linestyle=":")
    for e in nw:
        ax.axvline(e, color=ps.C.grey, linestyle="--")
    ax.plot([], [], color=ps.C.vermillion, linestyle=":", label="window, first 16")
    ax.plot([], [], color=ps.C.grey, linestyle="--", label="window, extended")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"learning rate, batch {b}"); ax.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, r"$q\bar q\to Z+4g$")
    ps.legend(ax, "lower left")
    ps.save(fig, os.path.join(HERE, "figs", f"zng4g_r3_b{b}"))


def main():
    b = sys.argv[1]
    if "--h3" in sys.argv:
        return h3(b)
    cache = os.path.join(HERE, "data", f"zng4g_r1_raw_b{b}.json")
    raw = json.load(open(cache)) if "--cached" in sys.argv else collect(b)
    json.dump(raw, open(cache, "w"))
    base = analyse(json.load(open(os.path.join(HERE, "data", "zng4g_readout_raw.json"))))[b]
    std = base["std"]
    k2 = std ** 2
    trials = {a: sorted((v for k, v in raw.items() if k.startswith(f"{a}|{b}|t")), key=lambda v: v.get("val_loss") or 9)
              for a in ("Sfx", "Fx", "S")}
    for a, ts in trials.items():
        assert all(isinstance(t["lr"], float) for t in ts), [t["lr"] for t in ts if not isinstance(t["lr"], float)]
    assert len(trials["S"]) == 16 and len(trials["Sfx"]) == 16, {a: len(t) for a, t in trials.items()}
    chosen = trials["Sfx"][0]
    # the cached S seed-42 best is the S search's best trial
    assert abs(trials["S"][0]["val_loss"] * k2 / base["best"]["S"][0] - 1) < 1e-6
    sf_curves = [chosen["val_mse_curve"]] + [raw[f"Sfx|{b}|s{s}"]["val_mse_curve"] for s in (43, 44)
                                             if raw.get(f"Sfx|{b}|s{s}", {}).get("val_mse_curve")]
    best = {"S": base["best"]["S"], "Sf": [min(c) * k2 for c in sf_curves], "F": base["best"]["F"]}
    mean = {a: float(np.mean(v)) for a, v in best.items()}
    share = np.log(mean["Sf"] / mean["S"]) / np.log(mean["F"] / mean["S"])
    q, nt = triples_share(best["Sf"], best["S"], best["F"])
    lo, hi = np.log10(chosen["lr"] / SF_WINDOW[b][0]), np.log10(SF_WINDOW[b][1] / chosen["lr"])
    f16 = min(t["val_loss"] for t in trials["Fx"] if int(t_idx(t, raw)) < 16)
    fx = trials["Fx"][0]
    ow = OLD_FT_WINDOW[b]
    r3 = {"chosen_lr": fx["lr"], "chosen_val": fx["val_loss"] * k2, "f16_best": f16 * k2,
          "f16_over_new": f16 / fx["val_loss"], "inside_old_window": ow[0] <= fx["lr"] <= ow[1],
          "seeds": {s: (min(raw[f"Fx|{b}|s{s}"]["val_mse_curve"]) * k2 if raw.get(f"Fx|{b}|s{s}", {}).get("val_mse_curve") else None)
                    for s in (43, 44)}}
    r3["H3"] = ("yes" if (not r3["inside_old_window"] and r3["f16_over_new"] > SEED_BAR) else "ruled out")
    res = {"batch": b, "std": std, "best": best, "mean": mean, "n_seeds_Sf": len(sf_curves), "share": float(share),
           "share_q10_50_90": q, "n_triples": nt, "Sf_over_S": mean["Sf"] / mean["S"], "F_over_Sf": mean["F"] / mean["Sf"],
           "F_over_S": mean["F"] / mean["S"],
           "Sf_chosen": {"hp": chosen_hp(chosen, raw), "lr": chosen["lr"], "decades_to_low": float(lo), "decades_to_high": float(hi),
                         "edge_limited": bool(min(lo, hi) < 0.1)},
           "R3": r3,
           "trials": {a: [(t["lr"], t["val_loss"] * k2 if t["val_loss"] is not None else None, t_idx(t, raw)) for t in ts]
                      for a, ts in trials.items()}}
    json.dump(res, open(os.path.join(HERE, "data", f"zng4g_r1_b{b}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "trials"}, indent=1))
    plot(res, base["curves"], [np.array(c) * k2 for c in sf_curves], ow)
    # train against val, per arm the mean over its seeds; end/best of val and train per seed
    tv = {"S": [raw[f"TVS|{b}|s{s}"] for s in (42, 43, 44)], "F": [raw[f"TVF|{b}|s{s}"] for s in (42, 43, 44)],
          "Sf": [chosen] + [raw[f"Sfx|{b}|s{s}"] for s in (43, 44) if raw.get(f"Sfx|{b}|s{s}", {}).get("val_mse_curve")]}
    assert abs(min(tv["F"][0]["val_mse_curve"]) * k2 / base["best"]["F"][0] - 1) < 1e-6
    shape = {a: [{"val_best_at": (int(np.argmin(r["val_mse_curve"])) + 1) / len(r["val_mse_curve"]),
                  "val_end_over_best": r["val_mse_curve"][-1] / min(r["val_mse_curve"]),
                  "train_end_over_at_val_best": r["train_mse_curve"][-1] / r["train_mse_curve"][int(np.argmin(r["val_mse_curve"]))],
                  "val_over_train_end": r["val_mse_curve"][-1] / r["train_mse_curve"][-1]} for r in rs] for a, rs in tv.items()}
    res["shape"] = shape
    json.dump(res, open(os.path.join(HERE, "data", f"zng4g_r1_b{b}.json"), "w"), indent=1)
    print(json.dumps(shape, indent=1))
    plot_tv(b, {a: [(np.array(r["train_mse_curve"]) * k2, np.array(r["val_mse_curve"]) * k2) for r in rs] for a, rs in tv.items()})


def t_idx(t, raw):
    return next(int(k.split("|")[2][1:]) if k.split("|")[2].startswith("t") else -1 for k, v in raw.items() if v is t)


def chosen_hp(t, raw):
    return next(k.split("|")[3] for k, v in raw.items() if v is t)


def plot(res, curves, sf_curves, ow):
    import plot_style as ps
    b = res["batch"]
    arms = (("S", "from scratch", ps.C.blue, "o"), ("Sf", "from scratch + input flags,\nsearched", ps.C.grey, "D"),
            ("F", "fine-tuned", ps.C.vermillion, "s"))
    fig, axs = ps.figure(ncols=2, nrows=2)
    a, l, c, leg = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
    for i, (k, lab, col, m) in enumerate(arms):
        v = res["best"][k]
        a.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, m, color=col, label=lab)
        cs = curves[k] if k != "Sf" else sf_curves
        for cv in cs:   # each seed's raw validation curve: a blow-up shows (a best-so-far mean hides it)
            c.plot((np.arange(len(cv)) + 1) / len(cv), cv, color=col)
    a.set_xticks(range(3)); a.set_xticklabels(["scratch", "+flags", "fine-tuned"]); a.set_xlim(-0.5, 2.5)
    a.set_yscale("log"); a.set_xlabel(f"arm, batch {b}"); a.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    tr = res["trials"]
    for k, col, m in (("S", ps.C.blue, "o"), ("Sfx", ps.C.grey, "D")):
        x = [t[0] for t in tr[k] if t[1]]; y = [t[1] for t in tr[k] if t[1]]
        l.plot(x, y, m, color=col)
    f_old = [t for t in tr["Fx"] if t[1] and t[2] < 16]
    f_new = [t for t in tr["Fx"] if t[1] and t[2] >= 16]
    l.plot([t[0] for t in f_old], [t[1] for t in f_old], "s", color=ps.C.vermillion)
    l.plot([t[0] for t in f_new], [t[1] for t in f_new], "s", color=ps.C.vermillion, mfc="none",
           label="fine-tuned, 8 added trials")
    for e in ow:
        l.axvline(e, color=ps.C.vermillion, linestyle=":")
    l.plot([], [], color=ps.C.vermillion, linestyle=":", label="fine-tune lr window\nbefore the extension")
    l.set_xscale("log"); l.set_yscale("log")
    l.set_xlabel("learning rate"); l.set_ylabel(r"best MSE$(\log|\mathcal{M}|^2)$")
    c.set_yscale("log"); c.set_xlabel(f"fraction of the run, batch {b}")
    c.set_ylabel(r"validation MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(a, r"$q\bar q\to Z+4g$")
    ps.legend_cell(leg, [a, l])
    ps.save(fig, os.path.join(HERE, "figs", f"zng4g_r1_b{b}"))


def plot_tv(b, tv):
    import plot_style as ps
    fig, (tr, va) = ps.figure(ncols=2)
    for k, lab, col in (("S", "from scratch", ps.C.blue), ("Sf", "from scratch + input flags,\nsearched", ps.C.grey),
                        ("F", "fine-tuned", ps.C.vermillion)):
        for ax, j in ((tr, 0), (va, 1)):
            for i, c in enumerate(tv[k]):   # per seed: a blow-up would dominate a mean
                ax.plot((np.arange(len(c[j])) + 1) / len(c[j]), c[j], color=col, label=lab if i == 0 else None)
    for ax, q in ((tr, "training"), (va, "validation")):
        ax.set_yscale("log"); ax.set_xlabel(f"fraction of the run, batch {b}")
        ax.set_ylabel(q + r" MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(va, r"$q\bar q\to Z+4g$")
    ps.legend(tr, "upper right")
    ps.save(fig, os.path.join(HERE, "figs", f"zng4g_r1_trainval_b{b}"))


if __name__ == "__main__":
    main()
