#!/usr/bin/env python3
"""Why the het-loss arms read about 12x the MSE-trained share-0 control on the 28 held-out one-loop classes (HD5,
recipes/catalog_v4_heldout_default_H5_test.yaml; slide 3 of the Nov 2026 deck, fig_shares_bound.py merged,
data/shares_deck_mse_ref.json "hetctl" held_default H5 R 12.0). Joaquin 2026-11-04 (plan catalog_coverage step 45).

    python3 analysis/catalog_v4/het_oneloop.py pull     one read-only `site run` on CC-IN2P3 -> data/het_oneloop_reads.json
    python3 analysis/catalog_v4/het_oneloop.py signhead_pull   step 56: logs of its runs -> data/reads/<run>_{axis,val,sign}.json
    python3 analysis/catalog_v4/het_oneloop.py signhead        step 56's read -> data/het_signhead.json
    python3 analysis/catalog_v4/het_oneloop.py          the summary -> data/het_oneloop.json, figs/het_oneloop.{png,pdf}

pull reads, for every share arm of both losses (control, 0.50, 0.25; seeds 42 and 43), the zero-shot predictions the
step-20 eval-only REREADs saved for HD5 (tools/zero_shot_parent.py --save-dir: $SCRATCH/zz_v4_short_<arm>/HD5/<probe>.npz;
the het arms read from their sigma stage, mu frozen at the best checkpoint), per probe:
    n, t_md5     events and an md5 of the truth array (same pool rows and same target on both losses when equal)
    t_mean, t_var, q_mean, q_std, bias = mean(q - t), mse = mean((q - t)^2)    (log|M|^2; abslog on a sign-changing pool)
    frac_at_floor, mu_sd   the abslog floor share and the probe's own (mean, std) of its target
    s_med, rho_s_err       het only: median sigma in log|M|^2 and Spearman of sigma against |q - t| over the events
    pool_nonpos            share of the scored pool's events (its test pool, every row) with |M|^2 label <= 0
and, per training run, the transform each one-loop training pool trained on (config.yaml data.dataset with
data_stats.json amp_trafos_pp) and, for the het runs, from sigma_stage/preds_val.npz (200 val events per pool, mu frozen),
per one-loop training pool the median sigma and the MSE, both in the parent's standardized units.

The summary checks the reading first (same events and truth on both losses: equal n, pool, and truth mean and variance
to 1e-4 per probe, the md5s differing at float32 round-off; the read files'
mse reproduced from the .npz), then per class the het-over-MSE ratio of the MSE of log|M|^2 (geometric mean over the
two seeds, as mixture_ab.side), split by the probe's pool sign (pool_nonpos > 0: sign-changing), the bias against the
spread, and sigma against the error.
"""
import json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
READS = os.path.join(DATA, "reads")
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

# arm (the job's ARM, its prediction dir) -> (loss, share, seed, the read run whose HD5 file it is)
ARMS = {"v4_het_control": ("het", "0", 42, "FA-ccin2p3-87aeabca"),
        "v4_het_control_seed2": ("het", "0", 43, "FA-ccin2p3-53b9ed56"),
        "v4_het_default50": ("het", "0.50", 42, "FA-ccin2p3-ead54b75"),
        "v4_het_default50_seed2": ("het", "0.50", 43, "FA-ccin2p3-7c131715"),
        "v4_het_default25": ("het", "0.25", 42, "FA-ccin2p3-9538e607"),
        "v4_het_default25_seed2": ("het", "0.25", 43, "FA-ccin2p3-ed96b552"),
        "v4_dimensionless_muon": ("mse", "0", 42, "FA-ccin2p3-c37a4b26"),
        "v4_dimensionless_muon_seed2": ("mse", "0", 43, "FA-ccin2p3-a1ea15aa"),
        "v4_dimensionless_muon_default50": ("mse", "0.50", 42, "FA-ccin2p3-58631ca5"),
        "v4_dimensionless_muon_default50_seed2": ("mse", "0.50", 43, "FA-ccin2p3-698c1a0b"),
        "v4_dimensionless_muon_default25": ("mse", "0.25", 42, "FA-ccin2p3-4dca09e5"),
        "v4_dimensionless_muon_default25_seed2": ("mse", "0.25", 43, "FA-ccin2p3-926dea24")}

REMOTE = r'''
import glob, hashlib, json, os, sys
import numpy as np
from scipy.stats import spearmanr
S = os.environ["SCRATCH"]
arms = sys.argv[1:]
out = {"probes": {}, "train": {}}
nonpos = {}
for arm in arms:
    d = os.path.join(S, f"zz_v4_short_{arm}", "HD5")
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        p = os.path.basename(f)[:-4]
        t, q, s = z["t"].astype(float), z["q"].astype(float), z["s"].astype(float)
        e = q - t
        r = {"n": int(len(t)), "t_md5": hashlib.md5(z["t"].tobytes()).hexdigest()[:12], "t_mean": float(t.mean()),
             "t_var": float(t.var()), "q_mean": float(q.mean()), "q_std": float(q.std()), "bias": float(e.mean()),
             "mse": float(np.mean(e ** 2)), "frac_at_floor": float(np.mean(z["at_floor"])) if len(z["at_floor"]) else 0.0,
             "mu_sd": [float(x) for x in z["mu_sd"]], "physics": json.loads(str(z["physics_check"])).get("status")}
        if len(s) == len(t):
            r["s_med"] = float(np.median(s)); r["rho_s_err"] = float(spearmanr(s, np.abs(e))[0])
        pool = str(z["pool"])
        if pool not in nonpos:
            a = np.load(pool, mmap_mode="r")[:, -1]
            nonpos[pool] = [float(np.mean(a <= 0)), int(len(a))]
        r["pool"] = os.path.basename(pool); r["pool_nonpos"], r["pool_n"] = nonpos[pool]
        pr = z["pool_row"]
        if len(pr) == len(t):                  # the scored events' own label sign, from their pool rows
            neg = np.asarray(np.load(pool, mmap_mode="r")[np.sort(pr), -1])[np.argsort(np.argsort(pr))] <= 0
            r["n_neg"] = int(neg.sum())
            for k, m in (("neg", neg), ("pos", ~neg)):
                if m.any():
                    r[f"mse_{k}"] = float(np.mean(e[m] ** 2)); r[f"bias_{k}"] = float(e[m].mean())
        rows[p] = r
    out["probes"][arm] = rows
    rd = os.path.join("runs/catalog_v4_short", arm)
    import yaml
    cfg = yaml.safe_load(open(os.path.join(rd, "config.yaml")))
    st = json.load(open(os.path.join(rd, "data_stats.json")))
    names = [str(x) for x in cfg["data"]["dataset"]]
    tr = st["amp_trafos_pp"]
    loop = {n: str(tr[i][0]) for i, n in enumerate(names) if n.endswith(("_nlo", "_loop"))}
    t = {"n_datasets": len(names), "n_trafos": len(tr), "oneloop_transform": loop}
    pv = os.path.join(rd, "sigma_stage", "preds_val.npz")
    if os.path.exists(pv):
        z = np.load(pv)
        nm = [str(x) for x in z["names"]]; pid = z["process_id"]
        pr, tu = z["pred"].ravel(), z["truth"].ravel()
        sg = z["sigma"].ravel() if "sigma" in z.files else None
        per = {}
        for i in np.unique(pid):
            m = pid == i
            e = pr[m] - tu[m]
            per[nm[i]] = {"mse_std": float(np.mean(e ** 2)), "truth_std": float(np.std(tu[m])),
                          "sigma_med": float(np.median(sg[m])) if sg is not None else None,
                          "rho": float(spearmanr(sg[m], np.abs(e))[0]) if sg is not None else None}
        t["val"] = per
    out["train"][arm] = t
print("HETLOOP " + json.dumps(out))
'''


def pull():
    cmd = (f"F=$(mktemp /tmp/zz_probe_hetloop_XXXXXX.py); cat > $F <<'PYEOF'\n{REMOTE}PYEOF\n"
           f"python3 $F {' '.join(ARMS)}; rm -f $F")
    res = subprocess.run(["site", "--timeout", "900", "run", "--item", "catalog_coverage", "ccin2p3", "FA", "--", cmd],
                         capture_output=True, text=True, timeout=1000)
    line = [l for l in res.stdout.splitlines() if l.startswith("HETLOOP ")]
    if not line:
        sys.exit("pull: no HETLOOP line\n" + res.stdout[-3000:] + res.stderr[-2000:])
    d = json.loads(line[-1][len("HETLOOP "):])
    d["arms"] = {a: list(v) for a, v in ARMS.items()}
    json.dump(d, open(os.path.join(DATA, "het_oneloop_reads.json"), "w"), indent=0)
    print({a: len(v) for a, v in d["probes"].items()})


def gm(x):
    x = np.asarray(x, float)
    return float(np.exp(np.mean(np.log(x))))


def summary():
    from mixture_ab import class_of
    d = json.load(open(os.path.join(DATA, "het_oneloop_reads.json")))
    P = d["probes"]
    cls = class_of()
    out = {"source": "data/het_oneloop_reads.json (het_oneloop.py pull)", "checks": {}}
    # 1. the reading: the same events and truth on both losses; the read files reproduced
    probes = sorted(P["v4_het_control"])
    # the truth arrays agree to float32 round-off (their md5 differ in the last bits): compared by mean and variance
    same_t = {p: max(abs(P[a][p]["t_mean"] - P["v4_het_control"][p]["t_mean"]) for a in ARMS) < 1e-4 and
              max(abs(P[a][p]["t_var"] / P["v4_het_control"][p]["t_var"] - 1) for a in ARMS) < 1e-4 and
              len({P[a][p]["n"] for a in ARMS}) == 1 and len({P[a][p]["pool"] for a in ARMS}) == 1 for p in probes}
    rep = []
    for a, (_, _, _, run) in ARMS.items():
        f = json.load(open(os.path.join(READS, f"{run}_HD5.json")))["mse"]
        rep += [abs(P[a][p]["mse"] / f[p] - 1) for p in probes]
    out["checks"] = {"n_probes": len(probes), "truth_same_all_12_arms": int(sum(same_t.values())),
                     "read_files_reproduced_max_rel_diff": float(max(rep)),
                     "physics_status": sorted({P[a][p]["physics"] for a in ARMS for p in probes}),
                     "mu_sd_identical_1e-4": int(sum(len({tuple(np.round(P[a][p]["mu_sd"], 4)) for a in ARMS}) == 1
                                                 for p in probes))}
    # 2. the training transforms of the one-loop pools per loss
    tr = {}
    for a in ARMS:
        lo = d["train"][a]["oneloop_transform"]
        tr[a] = {"n_oneloop": len(lo), "n_signedlog": sum(v.startswith("signedlog") for v in lo.values()),
                 "n_abslog": sum(v.startswith("abslog") for v in lo.values()),
                 "n_log": sum(v == "log" for v in lo.values())}
    out["train_transforms"] = tr
    # 3. per class: het over MSE at each share, seeds by geometric mean
    rows = {}
    for p in probes:
        c = cls(p)
        r = {"probe": p, "class": c, "pool_nonpos": P["v4_het_control"][p]["pool_nonpos"],
             "frac_at_floor": P["v4_het_control"][p]["frac_at_floor"], "t_var": P["v4_het_control"][p]["t_var"]}
        for loss in ("het", "mse"):
            for sh in ("0", "0.50", "0.25"):
                arms = [a for a, v in ARMS.items() if v[0] == loss and v[1] == sh]
                r[f"{loss}_{sh}_mse"] = gm([P[a][p]["mse"] for a in arms])
                r[f"{loss}_{sh}_bias"] = float(np.mean([P[a][p]["bias"] for a in arms]))
                r[f"{loss}_{sh}_qstd"] = float(np.mean([P[a][p]["q_std"] for a in arms]))
                if loss == "het":
                    r[f"het_{sh}_s_med"] = gm([P[a][p]["s_med"] for a in arms])
                    r[f"het_{sh}_rho"] = float(np.mean([P[a][p]["rho_s_err"] for a in arms]))
            a0 = [a for a, v in ARMS.items() if v[0] == loss and v[1] == "0"]
            for k in ("neg", "pos"):
                if all(f"mse_{k}" in P[a][p] for a in a0):
                    r[f"{loss}_0_mse_{k}"] = gm([P[a][p][f"mse_{k}"] for a in a0])
                    r[f"{loss}_0_bias_{k}"] = float(np.mean([P[a][p][f"bias_{k}"] for a in a0]))
            r["n_neg"] = P["v4_het_control"][p].get("n_neg")
            # approximate share of the MSE in the mean offset: the seeds' mean bias squared over their geometric-mean MSE
            r[f"{loss}_0_bias2_share"] = r[f"{loss}_0_bias"] ** 2 / r[f"{loss}_0_mse"]
        r["R_0"] = r["het_0_mse"] / r["mse_0_mse"]
        rows[p] = r
    out["per_probe"] = rows
    R = np.array([r["R_0"] for r in rows.values()])
    sg = np.array([r["pool_nonpos"] > 0 for r in rows.values()])

    def grp(m):
        rr = [r for r, k in zip(rows.values(), m) if k]
        if not rr:
            return None
        return {"n": len(rr), "R_median": float(np.median([r["R_0"] for r in rr])),
                "het_mse_median": float(np.median([r["het_0_mse"] for r in rr])),
                "mse_mse_median": float(np.median([r["mse_0_mse"] for r in rr])),
                "het_bias2_share_median": float(np.median([r["het_0_bias2_share"] for r in rr])),
                "mse_bias2_share_median": float(np.median([r["mse_0_bias2_share"] for r in rr])),
                "het_bias_median": float(np.median([r["het_0_bias"] for r in rr])),
                "mse_bias_median": float(np.median([r["mse_0_bias"] for r in rr])),
                "het_s_med_median": float(np.median([r["het_0_s_med"] for r in rr])),
                "het_rho_median": float(np.median([r["het_0_rho"] for r in rr])),
                "het_rmse_over_s_median": float(np.median([np.sqrt(r["het_0_mse"]) / r["het_0_s_med"] for r in rr]))}
    out["share0"] = {"all": grp(np.ones(len(R), bool)), "sign_changing": grp(sg), "positive": grp(~sg),
                     "n_R_above_1": int((R > 1).sum()), "R_quantiles_10_50_90": np.quantile(R, [.1, .5, .9]).tolist()}
    # inside the sign-changing pools: the events whose label is negative against the positive ones
    both = [r for r in rows.values() if "het_0_mse_neg" in r and "het_0_mse_pos" in r and r["n_neg"] >= 20]
    out["sign_split"] = {"n_probes": len(both), "rule": "sign-changing probes with >= 20 negative-label scored events",
                         **{f"{l}_{k}_{m}_median": float(np.median([r[f"{l}_0_{m}_{k}"] for r in both]))
                            for l in ("het", "mse") for k in ("neg", "pos") for m in ("mse", "bias")}}
    out["shares"] = {sh: float(np.median([r[f"het_{sh}_mse"] / r[f"mse_{sh}_mse"] for r in rows.values()]))
                     for sh in ("0", "0.50", "0.25")}
    # 4. sigma against error: across classes (Spearman of the class median sigma against its RMSE)
    from scipy.stats import spearmanr
    out["sigma_across_classes_spearman"] = float(spearmanr([r["het_0_s_med"] for r in rows.values()],
                                                           [np.sqrt(r["het_0_mse"]) for r in rows.values()])[0])
    # 5. the training one-loop pools of the het control: sigma and error by transform
    tv = {}
    for a in ("v4_het_control", "v4_het_control_seed2"):
        lo, val = d["train"][a]["oneloop_transform"], d["train"][a].get("val", {})
        for kind in ("signedlog", "log"):
            ps = [p for p, v in lo.items() if v.split(":")[0] == kind and p in val]
            tv.setdefault(kind, []).append({"n": len(ps), "sigma_med": float(np.median([val[p]["sigma_med"] for p in ps])),
                                            "rmse_std_med": float(np.median([np.sqrt(val[p]["mse_std"]) for p in ps])),
                                            "rho_med": float(np.median([val[p]["rho"] for p in ps]))})
        trees = [p for p in val if p not in lo]
        tv.setdefault("tree", []).append({"n": len(trees), "sigma_med": float(np.median([val[p]["sigma_med"] for p in trees])),
                                          "rmse_std_med": float(np.median([np.sqrt(val[p]["mse_std"]) for p in trees])),
                                          "rho_med": float(np.median([val[p]["rho"] for p in trees]))})
    out["het_train_val_by_kind"] = tv
    json.dump(out, open(os.path.join(DATA, "het_oneloop.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_probe"}, indent=1))
    figure(rows)


MAG = ("FA-ccin2p3-5abe4a0e", "FA-ccin2p3-92207f6a")          # v4_het_control_magnitude[_seed2], step 47
HET_CTL = {"HD5": ("FA-ccin2p3-87aeabca", "FA-ccin2p3-53b9ed56"), "HD4": ("FA-ccin2p3-87aeabca", "FA-ccin2p3-53b9ed56"),
           "other": ("FA-ccin2p3-87ea24d0", "FA-ccin2p3-321a3de2")}
MSE_CTL = ("FA-ccin2p3-c37a4b26", "FA-ccin2p3-a1ea15aa")


def magnitude():
    """Step 47 (design passed 2026-11-04T18:29): the het control with only data.signed_magnitude_target (the 63
    sign-changing one-loop training pools on log max(|x|,s), no sign output) over (a) the het control and (b) the MSE
    control, on HD5 (28 classes, abslog scoring, mse_logm2_real); per probe the geometric mean over seeds 42/43,
    per class the geometric mean of the ratios, R the median over classes, 95% class bootstrap widened by the het
    pair's seed ratio (mixture_ab.r_with_interval); each seed's ratio beside; sign-changing (pool share of label <= 0
    above 0, het_oneloop_reads.json) and positive pools apart; PP4, EE4, H1-H7c, HD4 against the het control beside.
    -> data/het_magnitude.json"""
    import math
    from mixture_ab import class_of, load, class_ratios, r_with_interval, blowups
    cls = class_of()
    nonpos = {p: v["pool_nonpos"] for p, v in
              json.load(open(os.path.join(DATA, "het_oneloop_reads.json")))["probes"]["v4_het_control"].items()}
    sd = lambda pair, ax: [load(r, ax) for r in pair]
    side = lambda xs, ks: {k: gm([x[k] for x in xs]) for k in ks}
    med = lambda num, den, ks: float(np.median(list(class_ratios(num, den, ks, cls).values())))
    out = {"design": "plan catalog_coverage step 47 (2026-11-04T18:29)", "arm": MAG, "het_control": HET_CTL,
           "mse_control": MSE_CTL, "reads": {}}
    for ax in ("HD5", "HD4", "PP4", "EE4", "H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c"):
        a = sd(MAG, ax)
        h = sd(HET_CTL.get(ax, HET_CTL["other"]), ax)
        refs = {"het_control": h}
        if ax == "HD5":
            refs["mse_control"] = sd(MSE_CTL, ax)
        keys_all = sorted(set.intersection(*[set(x) for x in a + h]))
        r_seed = med(h[1], h[0], keys_all)
        parts = {"all": keys_all}
        if ax == "HD5":
            parts.update(sign_changing=[k for k in keys_all if nonpos[k] > 0],
                         positive=[k for k in keys_all if nonpos[k] <= 0])
        res = {"R_seed_het_pair": round(r_seed, 4)}
        for ref, rr in refs.items():
            res[ref] = {}
            for n, ks in parts.items():
                ks = [k for k in ks if all(k in x for x in rr)]
                x = r_with_interval(class_ratios(side(a, ks), side(rr, ks), ks, cls), r_seed)
                x["per_seed"] = {s: round(med(a[i], rr[i], ks), 4) for i, s in enumerate(("seed42", "seed43"))}
                x["n_probes"] = len(ks)
                res[ref][n] = x
        out["reads"][ax] = res
    A = out["reads"]["HD5"]["het_control"]["all"]
    B = out["reads"]["HD5"]["mse_control"]["all"]
    if A["R"] <= 0.25 and A["widened"][1] < 0.5:
        va = "YES: the signed pools' signedlog target is the main cause"
    elif A["widened"][0] <= 1 <= A["widened"][1] or A["R"] >= 0.7:
        va = "NO: the target is not the cause"
    else:
        va = "PARTIAL: the target is a partial cause (0.25 < R < 0.7)"
    if B["R"] <= 1.5:
        vb = "the target is all of it"
    elif B["R"] < 3:
        vb = "the target costs about what it cost under MSE; the het loss or the missing sign cross-entropy the rest"
    else:
        vb = "both matter" if A["R"] <= 0.25 else ">= 3 with (a) > 0.25: the het loss or the missing sign cross-entropy dominates"
    out["blowups"] = blowups(list(MAG))
    out["verdict_on_reads"] = {"a": va, "b": vb, "note": "(b) differs in the loss and the sign cross-entropy: never the loss alone"}
    out["verdict"] = None if out["blowups"]["flagged"] else out["verdict_on_reads"]   # withheld while a run blew up
    json.dump(out, open(os.path.join(DATA, "het_magnitude.json"), "w"), indent=1)
    for ax, res in out["reads"].items():
        for ref in ("het_control", "mse_control"):
            for n, x in res.get(ref, {}).items():
                print(ax, ref, n, x["R"], x["widened"], x["per_seed"], x["n_classes"])
    print(out["verdict"], out["blowups"]["flagged"])


def figure(rows):
    import plot_style as ps
    fig, (a, b) = ps.figure(ncols=2)
    rr = list(rows.values())
    sty = lambda r: dict(marker="o", color=ps.C.vermillion, linestyle="none") if r["pool_nonpos"] > 0 else \
        dict(marker="s", color=ps.C.sky, mfc="none", linestyle="none")
    for r in rr:
        a.plot(r["mse_0_mse"], r["het_0_mse"], **sty(r))
        b.plot(r["het_0_s_med"], np.sqrt(r["het_0_mse"]), **sty(r))
    v = [r["mse_0_mse"] for r in rr] + [r["het_0_mse"] for r in rr]
    lim = [min(v) / 2, max(v) * 2]
    a.plot(lim, lim, "k--", label="equal")
    a.plot([], [], "o", color=ps.C.vermillion, label="sign-changing")
    a.plot([], [], "s", color=ps.C.sky, mfc="none", label="positive")
    a.set(xscale="log", yscale="log", xlim=lim, ylim=lim)
    a.set_xlabel(r"MSE loss: MSE$(\log|M|^2)$")
    a.set_ylabel(r"het. loss: MSE$(\log|M|^2)$")
    ps.legend(a, "upper left")
    w = [r["het_0_s_med"] for r in rr] + [np.sqrt(r["het_0_mse"]) for r in rr]
    lb = [min(w) / 2, max(w) * 2]
    b.plot(lb, lb, "k--", label=r"RMSE $=\sigma$")
    b.set(xscale="log", yscale="log", xlim=lb, ylim=lb)
    b.set_xlabel(r"het. loss: median $\sigma$")
    b.set_ylabel(r"het. loss: RMSE$(\log|M|^2)$")
    ps.legend(b, "upper left")
    ps.save(fig, os.path.join(FIGS, "het_oneloop"))


# step 56 (D287/D288; design passed 2026-11-05T03:55): het + sign head against the het magnitude reference, seed-paired
SIGNHEAD = {"arm": ("FA-ccin2p3-18b98b55", "FA-ccin2p3-b2538139", "FA-ccin2p3-66c83df6", "FA-ccin2p3-69001455"),
            "ref": MAG + ("FA-ccin2p3-0208896e", "FA-ccin2p3-8ddc490b"),     # v4_het_control_magnitude[_seed2-4]
            # the MSE control's training runs (their per-validation sign accuracy; REREAD sources of mse_hd5)
            "mse_train": ("FA-ccin2p3-339ec100", "FA-ccin2p3-380842ed", "FA-ccin2p3-67ca3313", "FA-ccin2p3-e3506d57"),
            "mse_hd5": ("FA-ccin2p3-dc9feba1", "FA-ccin2p3-af09cad3", "FA-ccin2p3-8833a131", "FA-ccin2p3-481ad196"),
            # the MSE control's H5_signed reads (analysis/catalog_v3/data/short_reads/<arm>/H5_signed.txt, signext_compare)
            "mse_h5": ("v4_dimensionless_muon", "v4_dimensionless_muon_seed2", "v4_dimensionless_muon_seed3",
                       "v4_dimensionless_muon_seed4")}
SH_SEEDS = (42, 43, 44, 45)
# step 61 (D287/D296; designs passed 2026-11-06T05:06:47 and 05:15:07), seeds 42-45 in order: the matched sign weights
# (v4_het_signhead_matched_event: per-event sign BCE x the het NLL's detached mu factor; _const: sign_loss_weight 75) and
# the same-code rerun of step 56's arm (v4_het_control_signhead[_seedK]_tagkeyed, SHORT_TAG=keyed), all three after the
# keyed-draw augmentation (d18e44f1); event at 92322402, const and rerun at 7b50861c (the training code between the two
# differs only in default-off warm-start options: base_experiment/optim/run.py fine_tune.*_warmup, load_optimizer_state)
STEP61 = {"matched_event": ("FA-ccin2p3-9079850b", "FA-ccin2p3-ebe4c4ed", "FA-ccin2p3-297c4366", "FA-ccin2p3-dce345d8"),
          "matched_const": ("FA-ccin2p3-e90fea9a", "FA-ccin2p3-14c988b0", "FA-ccin2p3-45b2d041", "FA-ccin2p3-04d27cd1"),
          "rerun": ("FA-ccin2p3-ac8d6fca", "FA-ccin2p3-752344ce", "FA-ccin2p3-b13f456c", "FA-ccin2p3-83dc9690")}
# plan step 73 (step 61's NO branch): the sign logit from its own detached copy of the last block (model.net.sign_branch),
# read by step 61's rules against the same weight-1 rerun and MSE control; runs filled in at submission
STEP73 = {"branch": (), "rerun": STEP61["rerun"]}
SH_AXES = ("HD5", "H5", "PP4", "EE4", "HD4", "H1", "H2", "H2p5", "H3", "H6", "H7E", "H7c")
PAIR_GATE = {"HD5": 0.43, "H5": 0.40}     # absolute bound on the sd over classes of the paired ln ratio (design check 3)
TOL = 1.25                                # magnitude no-worse tolerance on the widened upper end


def signhead_pull(runs=None):
    """Per run of SIGNHEAD (or `runs`, {kind: runs}): its job log's ZERO_SHOT, SIGNED_AB and validation lines (site
    logs), the zero-shot reads per axis (mixture_ab extract -> data/reads/<run>_<axis>.json), the validation curve
    (mixture_ab val), and the sign reads: HD5 from the ZERO_SHOT records (wrong_sign, wrong_sign_near, neg_share; test
    split), H5 from the SIGNED_AB lines (signed_ab_score --arm head, val split), the main block's pre-clip gradient norm
    -> data/reads/<run>_sign.json"""
    import re, tempfile
    from mixture_ab import extract, val
    tags = "|".join(SH_AXES)
    pat = re.compile(rf"^(({tags}) ZERO_SHOT |SIGNED_AB |RUN_SUMMARY )|Starting to train for|Val loss \(combined\)|"
                     r"sign_head: sign accuracy on the|Grad-norm \(pre-clip\)")
    acc_re = re.compile(r"sign accuracy on the (\d+) signed pools: median ([0-9.]+), min ([0-9.]+)")
    gn_re = re.compile(r"Grad-norm \(pre-clip\): median=([0-9.e+-]+) p90=([0-9.e+-]+) max=([0-9.e+-]+) \| clip=\S+ "
                       r"bound ([0-9.]+)% of steps")
    h5_pools = set()
    for kind, kind_runs in (runs or {k: SIGNHEAD[k] for k in ("arm", "ref", "mse_hd5", "mse_train")}).items():
        for run in kind_runs:
            if kind == "ref" and run in MAG and os.path.exists(os.path.join(READS, f"{run}_H5.json")):
                continue                         # seeds 42/43 of the reference were read at step 47
            txt = subprocess.run(["site", "logs", run, "-n", "3000000"], capture_output=True, text=True,
                                 timeout=1800).stdout
            lines = [l for l in txt.splitlines() if pat.search(l)]
            with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
                f.write("\n".join(lines) + "\n")
            axes = {"mse_hd5": ("HD5",), "mse_train": ()}.get(kind, SH_AXES)
            for ax in axes:
                extract(run, f.name, ax)
            if kind not in ("mse_hd5", "mse_train"):
                val(run, f.name)
            gn = [m.groups() for m in (gn_re.search(l) for l in lines) if m]
            # per validation of the training (main block, then the sigma stage): [n signed pools, median, min] accuracy
            sg = {"HD5": {}, "H5": {}, "val_sign_acc": [[int(m.group(1)), float(m.group(2)), float(m.group(3))]
                                                        for m in (acc_re.search(l) for l in lines) if m],
                  # [median, p90, max, % of steps the clip bounds] of the main block (the first Grad-norm line)
                  "grad_norm_main": [float(x) for x in gn[0]] if gn else None}
            for l in lines:
                if l.startswith("HD5 ZERO_SHOT "):
                    d = json.loads(l[len("HD5 ZERO_SHOT "):])
                    if d.get("wrong_sign") is not None:
                        sg["HD5"][d["probe"]] = {k: d.get(k) for k in ("wrong_sign", "wrong_sign_near", "neg_share", "n_val")}
                elif l.startswith("SIGNED_AB "):
                    d = json.loads(l[len("SIGNED_AB "):])
                    if "wrong_sign" in d:
                        sg["H5"][d["pool"]] = {k: d.get(k) for k in ("wrong_sign", "wrong_sign_near", "neg_share")}
            json.dump({"run": run, "source": "site logs ZERO_SHOT/SIGNED_AB lines", **sg},
                      open(os.path.join(READS, f"{run}_sign.json"), "w"), indent=0, sort_keys=True)
            os.unlink(f.name)
            print(run, kind, {k: len(v) for k, v in sg.items() if isinstance(v, (dict, list))}, flush=True)
            if kind not in ("ref", "mse_hd5") and not sg["val_sign_acc"]:
                print(f"WARNING {run}: no 'sign_head: sign accuracy' line in its log")
            if kind not in ("ref", "mse_hd5", "mse_train"):
                h5_pools |= set(sg["H5"])
    if runs is None:
        h5_neg_pull(h5_pools)
    else:
        known = set(json.load(open(os.path.join(DATA, "het_signhead_h5_neg.json")))["neg"])
        if h5_pools - known:
            print("WARNING H5 pools without a negative share (rerun h5_neg_pull):", sorted(h5_pools - known))


NEG_REMOTE = r'''
import glob, json, os, sys
import numpy as np
out = {}
for pool in sys.argv[1].split(","):
    g = sorted(glob.glob(os.path.join(os.environ["DATA_DIR"], f"{pool}_val_*amplitudes.npy")))
    if g:
        lab = np.asarray(np.load(g[0], mmap_mode="r")[:2000, -1])   # the rows signed_ab_score reads (eval_subsample 2000: [:cap])
        out[pool] = [float(np.mean(lab < 0)), int(len(lab)), os.path.basename(g[0])]
print("H5_NEG " + json.dumps(out))
'''


def h5_neg_pull(pools):
    """the negative share of each H5 pool's scored validation events (signed_ab_score: the first 2000 rows, experiment
    cap_for / eval_subsample): what an
    all-positive sign is wrong on, which SIGNED_AB records do not carry -> data/het_signhead_h5_neg.json"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    from analysis.common import site_python
    d = site_python(NEG_REMOTE, [",".join(sorted(pools))], "H5_NEG", timeout=600)
    miss = sorted(set(pools) - set(d))
    json.dump({"source": "label < 0 share of each H5 pool's val file on ccin2p3 ($DATA_DIR)", "neg": d, "missing": miss},
              open(os.path.join(DATA, "het_signhead_h5_neg.json"), "w"), indent=1)
    print("h5 neg", len(d), "missing", miss)


def _sign_read(run):
    return json.load(open(os.path.join(READS, f"{run}_sign.json")))


def _mse_h5_sign(arm):
    from signext_compare import signed
    return signed(arm, ("wrong_sign", "wrong_sign_near", "neg_share"))


def _sign_test(k, n):
    """one-sided sign test: P(at least k of n) under p = 1/2"""
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def signhead():
    """Step 56's read (design passed 2026-11-05T03:55) -> data/het_signhead.json.
    (1) Magnitude, seed-paired (D288): per probe the ratio arm seed k / reference seed k (mse_logm2_real), per class the
    geometric mean over its probes, per class the geometric mean over the four pairs, R the median over classes; 95%
    class bootstrap widened on each side by the standard error over the pairs of the per-pair class-median ln ratio.
    Pairing gate, on the first pair (seed 42): the sd over classes of the paired ln ratio <= PAIR_GATE (absolute) on
    HD5 and H5; failed, the read falls back to the unpaired 4-vs-4 contrast (each side's geometric mean over seeds,
    widened by the reference pair's seed-43-over-42 ratio) and a result inside the tolerance band is UNRESOLVED.
    (2) Sign: per pool the seed-mean wrong-sign share (whole pool; within a factor 10 of zero beside), arm vs the MSE
    control paired per pool, one-sided sign test for the arm worse (ties out), and the arm against all-positive (wrong
    on the pool's negative share) one-sided for the arm better; H5 21 sign-changing pools (val), HD5 its sign-changing
    classes (test). (3) Beside: PP4, EE4, HD4, H1-H7c paired as (1); blow-ups (mixture_ab.blowups)."""
    import math
    from mixture_ab import class_of, load, class_ratios, r_with_interval, blowups
    cls = class_of()
    arm, ref = SIGNHEAD["arm"], SIGNHEAD["ref"]
    out = {"design": "plan catalog_coverage step 56 (D287, D288; passed 2026-11-05T03:55)", "runs": SIGNHEAD,
           "seeds": SH_SEEDS, "magnitude": {}, "sign": {}}
    med = lambda num, den, ks: float(np.median(list(class_ratios(num, den, ks, cls).values())))
    for ax in SH_AXES:
        a = [load(r, ax) for r in arm]; b = [load(r, ax) for r in ref]
        ks = sorted(set.intersection(*[set(x) for x in a + b]))
        per_pair = [class_ratios(a[i], b[i], ks, cls) for i in range(len(SH_SEEDS))]
        classes = sorted(per_pair[0])
        lnr = np.array([[math.log(pp[c]) for c in classes] for pp in per_pair])      # pairs x classes
        paired = {c: math.exp(float(lnr[:, j].mean())) for j, c in enumerate(classes)}
        pair_meds = [float(np.median(row)) for row in lnr]
        se = float(np.std(pair_meds, ddof=1) / math.sqrt(len(pair_meds)))
        x = r_with_interval(paired, math.exp(se))
        x.update(per_pair_R={str(s): round(math.exp(m), 4) for s, m in zip(SH_SEEDS, pair_meds)},
                 se_ln_pairs=round(se, 4), n_probes=len(ks),
                 sd_paired_ln_ratio={str(s): round(float(np.std(lnr[i])), 4) for i, s in enumerate(SH_SEEDS)})
        side = lambda xs: {k: gm([v[k] for v in xs]) for k in ks}
        seed_r = med(b[1], b[0], ks)
        x["unpaired"] = r_with_interval(class_ratios(side(a), side(b), ks, cls), seed_r)
        x["unpaired"]["R_seed_ref"] = round(seed_r, 4)
        lnu = [math.log(b[1][k] / b[0][k]) for k in ks]
        x["ref_seed_ratio_sd_ln_per_class"] = round(float(np.std([math.log(v) for v in
                                                                  class_ratios(b[1], b[0], ks, cls).values()])), 4)
        out["magnitude"][ax] = x
    gate = {ax: out["magnitude"][ax]["sd_paired_ln_ratio"]["42"] <= PAIR_GATE[ax] for ax in PAIR_GATE}
    out["pairing_gate"] = {"bound": PAIR_GATE, "first_pair_sd": {ax: out["magnitude"][ax]["sd_paired_ln_ratio"]["42"]
                                                                 for ax in PAIR_GATE}, "passed": all(gate.values())}
    # sign
    sa = [_sign_read(r) for r in arm]
    ctl = {"HD5": [_sign_read(r)["HD5"] for r in SIGNHEAD["mse_hd5"]], "H5": [_mse_h5_sign(m) for m in SIGNHEAD["mse_h5"]]}
    h5neg = json.load(open(os.path.join(DATA, "het_signhead_h5_neg.json")))["neg"]
    for ax in ("H5", "HD5"):
        A = [x[ax] for x in sa]; C = ctl[ax]
        if ax == "H5":       # SIGNED_AB carries no negative share: the val pool's (h5_neg_pull)
            A = [{p: dict(v, neg_share=h5neg[p][0]) for p, v in x.items() if p in h5neg} for x in A]
        pools = sorted(set.intersection(*[set(x) for x in A + C]))
        res = {"n_pools": len(pools)}
        for which in ("wrong_sign", "wrong_sign_near"):
            ps = [p for p in pools if all(x[p].get(which) is not None for x in A + C)]
            am = np.array([np.mean([x[p][which] for x in A]) for p in ps])
            cm = np.array([np.mean([x[p][which] for x in C]) for p in ps])
            neg = np.array([np.mean([x[p]["neg_share"] for x in A]) for p in ps])
            worse, better = int((am > cm).sum()), int((am < cm).sum())
            r = {"n": len(ps), "arm_worse": worse, "arm_better": better, "ties": len(ps) - worse - better,
                 "p_arm_worse": round(_sign_test(worse, worse + better), 4),
                 "median_arm_pct": round(100 * float(np.median(am)), 2) if len(ps) else None,
                 "median_mse_ctl_pct": round(100 * float(np.median(cm)), 2) if len(ps) else None}
            if which == "wrong_sign":
                ap = neg      # all-positive is wrong on the pool's negative events
                ab, aw = int((am < ap).sum()), int((am > ap).sum())
                r.update(all_positive_median_pct=round(100 * float(np.median(ap)), 2), arm_better_than_allpos=ab,
                         arm_worse_than_allpos=aw, p_arm_better_than_allpos=round(_sign_test(ab, ab + aw), 4))
            r["per_pool_pct"] = {p: [round(100 * float(u), 2), round(100 * float(v), 2)] for p, u, v in zip(ps, am, cm)}
            res[which] = r
        res["n_pools_design"] = {"H5": 21, "HD5": 19}[ax]
        if res["n_pools"] != res["n_pools_design"]:
            print(f"WARNING sign {ax}: {res['n_pools']} pools read, the design names {res['n_pools_design']}")
        out["sign"][ax] = res
    # per-validation median sign accuracy on the signed training pools (measure 3; the NO-SIGN report)
    acc = lambda runs: [[v[1] for v in _sign_read(r)["val_sign_acc"]] for r in runs]
    out["val_sign_accuracy"] = {"arm": acc(arm), "mse_control": acc(SIGNHEAD["mse_train"]),
                                "note": "per run (seeds 42-45), the median over the signed training pools per "
                                        "validation (experiment.py sign_head line), main block then sigma stage"}
    out["pairing_gate"]["pair_used"] = "seed 42 (the design: the first pair to finish, seed 42 or 43)"
    out["blowups"] = blowups(list(arm) + list(ref[2:]))
    # outcome (design 2026-11-05T03:54)
    s5, sd5 = out["sign"]["H5"]["wrong_sign"], out["sign"]["HD5"]["wrong_sign"]
    sign_p = [s5["p_arm_worse"], sd5["p_arm_worse"]]
    allpos_p = [s5["p_arm_better_than_allpos"], sd5["p_arm_better_than_allpos"]]
    m5, md5 = out["magnitude"]["H5"], out["magnitude"]["HD5"]
    use = (lambda m: m) if out["pairing_gate"]["passed"] else (lambda m: m["unpaired"])
    hi = [use(m)["widened"][1] for m in (md5, m5)]; lo = [use(m)["widened"][0] for m in (md5, m5)]
    beside_ok = all(out["magnitude"][ax]["widened"][0] <= 1 for ax in ("PP4", "EE4", "H1", "H2", "H2p5", "H3", "H6",
                                                                        "H7E", "H7c"))
    v = []
    if any(p < 0.05 for p in sign_p) or any(p >= 0.05 for p in allpos_p):
        v.append("NO-SIGN")
    if any(l > 1 for l in lo):
        v.append("COST")
    if any(0.05 <= p < 0.2 for p in sign_p) or any(h > TOL and l <= 1 for h, l in zip(hi, lo)) or \
            (not out["pairing_gate"]["passed"] and any(l <= 1 <= h or h <= TOL for h, l in zip(hi, lo)) and "COST" not in v):
        v.append("UNRESOLVED")
    if not v:
        v.append("YES" if all(p >= 0.2 for p in sign_p) and all(p < 0.05 for p in allpos_p) and all(h <= TOL for h in hi)
                 and beside_ok else "UNRESOLVED")
    out["outcome_inputs"] = {"sign_p_arm_worse": sign_p, "p_better_than_allpos": allpos_p, "mag_widened_hi": hi,
                             "mag_widened_lo": lo, "paired": out["pairing_gate"]["passed"], "beside_contain_1_or_below": beside_ok}
    out["verdict_on_reads"] = v
    out["verdict"] = None if out["blowups"]["flagged"] else v
    json.dump(out, open(os.path.join(DATA, "het_signhead.json"), "w"), indent=1)
    print(json.dumps({"pairing": out["pairing_gate"], "outcome": out["outcome_inputs"], "verdict": out["verdict"],
                      "mag": {ax: [m["R"], m["widened"], m["per_pair_R"]] for ax, m in out["magnitude"].items()},
                      "sign": {ax: {k: v for k, v in r["wrong_sign"].items() if k != "per_pool_pct"}
                               for ax, r in out["sign"].items()}}))


def _paired_mag(arm, ref, ax, cls):
    """R = arm over ref on one read, seed-paired as signhead (1): per class the geometric mean over the pairs of the
    class's geometric-mean probe ratio, R the median over classes, the class bootstrap widened by the se over the pairs
    of the per-pair class-median ln ratio"""
    import math
    from mixture_ab import load, class_ratios, r_with_interval
    a = [load(r, ax) for r in arm]; b = [load(r, ax) for r in ref]
    ks = sorted(set.intersection(*[set(x) for x in a + b]))
    per_pair = [class_ratios(a[i], b[i], ks, cls) for i in range(len(arm))]
    classes = sorted(per_pair[0])
    lnr = np.array([[math.log(pp[c]) for c in classes] for pp in per_pair])
    pair_meds = [float(np.median(row)) for row in lnr]
    se = float(np.std(pair_meds, ddof=1) / math.sqrt(len(pair_meds)))
    x = r_with_interval({c: math.exp(float(lnr[:, j].mean())) for j, c in enumerate(classes)}, math.exp(se))
    x.update(per_pair_R=[round(math.exp(m), 4) for m in pair_meds], se_ln_pairs=round(se, 4), n_probes=len(ks))
    # the compared quantity itself: per side the median over classes of the class's seed-geometric-mean MSE
    side = lambda xs: float(np.median(list(class_ratios({k: gm([v[k] for v in xs]) for k in ks},
                                                        {k: 1.0 for k in ks}, ks, cls).values())))
    x.update(mse_class_median_arm=round(side(a), 6), mse_class_median_ref=round(side(b), 6))
    return x


def _sign_pair(A, C, h5neg, ax, test="arm_worse"):
    """per pool the seed-mean wrong-sign share of runs A against runs C ({pool: record} per run): counts, the one-sided
    sign test of `test` (arm_worse: A above C; arm_better: A below C), a two-sided p, the medians and their ratio"""
    if ax == "H5":           # SIGNED_AB carries no negative share: the val pool's (h5_neg_pull)
        A = [{p: dict(v, neg_share=h5neg[p][0]) for p, v in x.items() if p in h5neg} for x in A]
    pools = sorted(set.intersection(*[set(x) for x in A + C]))
    ps = [p for p in pools if all(x[p].get("wrong_sign") is not None for x in A + C)]
    am = np.array([np.mean([x[p]["wrong_sign"] for x in A]) for p in ps])
    cm = np.array([np.mean([x[p]["wrong_sign"] for x in C]) for p in ps])
    worse, better = int((am > cm).sum()), int((am < cm).sum())
    n = worse + better
    r = {"n": len(ps), "a_above": worse, "a_below": better, "ties": len(ps) - n,
         "p_a_above": round(_sign_test(worse, n), 4), "p_a_below": round(_sign_test(better, n), 4),
         "p_two_sided": round(min(1.0, 2 * min(_sign_test(worse, n), _sign_test(better, n))), 4),
         "median_a_pct": round(100 * float(np.median(am)), 2), "median_c_pct": round(100 * float(np.median(cm)), 2)}
    r["median_ratio_a_over_c"] = round(float(np.median(am)) / float(np.median(cm)), 3) if np.median(cm) > 0 else None
    if all("neg_share" in x[p] for x in A for p in ps):
        r["all_positive_median_pct"] = round(100 * float(np.median([np.mean([x[p]["neg_share"] for x in A])
                                                                    for p in ps])), 2)
    r["per_pool_pct"] = {p: [round(100 * float(u), 2), round(100 * float(v), 2)] for p, u, v in zip(ps, am, cm)}
    return r


GATE_MAG = ("PP4", "EE4")                     # upper end of the widened interval <= MAG_UP (design 05:06)
SHOWN_AXES = ("HD4", "H1", "H2", "H2p5", "H3", "H6", "H7E", "H7c")   # hurt only if lower > 1 and R > MAG_UP
MAG_UP = 1.10
SIGN_TOL = 1.2                                # median wrong-sign within 1.2x the MSE control's; re-seed band [1/1.2, 1.2]


def step61(table=None, arms=("matched_event", "matched_const"), out_name="het_signhead61.json",
           design="plan catalog_coverage step 61 (D287, D296; passed 2026-11-06T05:06:47 and 05:15:07)"):
    """Step 61's read (designs passed 2026-11-06T05:06:47 and 05:15:07) -> data/het_signhead61.json.
    Re-seed rule first: the step-56 arm (old code) against its same-code rerun on H5 and HD5: per-pool two-sided sign
    test p < 0.05 or a median wrong-sign ratio outside [1/1.2, 1.2] -> the MSE control (old code) is rerun at the same
    code before any YES or NO against it. Per matched arm: sign not worse than the MSE control on H5 and HD5 (one-sided
    p_arm_worse > 0.05 and median within 1.2x), better than the rerun on H5 (one-sided p < 0.05); magnitude R = arm over
    the rerun, seed-paired: PP4 and EE4 widened upper <= 1.10, no read of HD4, H1-H7c shown worse (widened lower > 1 and
    R > 1.10). YES / SIGN YES MAGNITUDE HURT / PARTIAL (better than the rerun, still worse than MSE) / NOT BETTER;
    the step is NO when neither arm beats the rerun. Beside: the rerun against the old het magnitude reference and the
    old step-56 arm (magnitude), the rerun against the MSE control (sign), pre-clip gradient norms, the training-pool
    sign accuracy, blow-ups."""
    from mixture_ab import class_of, blowups
    cls = class_of()
    table = table or STEP61
    rr, old = table["rerun"], SIGNHEAD["arm"]
    h5neg = json.load(open(os.path.join(DATA, "het_signhead_h5_neg.json")))["neg"]
    sr = lambda runs, ax: [_sign_read(r)[ax] for r in runs]
    ctl = {"HD5": sr(SIGNHEAD["mse_hd5"], "HD5"), "H5": [_mse_h5_sign(m) for m in SIGNHEAD["mse_h5"]]}
    out = {"design": design,
           "runs": table, "old_step56_arm": old, "mse_control": {k: SIGNHEAD[k] for k in ("mse_train", "mse_hd5", "mse_h5")},
           "het_magnitude_reference_old": SIGNHEAD["ref"], "seeds": SH_SEEDS}
    # re-seed rule
    rs = {ax: _sign_pair(sr(rr, ax), sr(old, ax), h5neg, ax) for ax in ("H5", "HD5")}
    bad = [ax for ax, r in rs.items() if r["p_two_sided"] < 0.05 or not (1 / SIGN_TOL <= r["median_ratio_a_over_c"] <= SIGN_TOL)]
    out["reseed"] = {"rule": "rerun (a) vs old step-56 arm (c): two-sided p < 0.05 or median ratio outside [1/1.2, 1.2] "
                             "-> the MSE control is rerun at the same code before a verdict against it",
                     "H5": rs["H5"], "HD5": rs["HD5"], "failed_on": bad, "mse_control_stands": not bad}
    out["rerun_vs_mse"] = {ax: _sign_pair(sr(rr, ax), ctl[ax], h5neg, ax) for ax in ("H5", "HD5")}
    out["rerun_magnitude"] = {"vs_old_het_reference": {ax: _paired_mag(rr, SIGNHEAD["ref"], ax, cls) for ax in SH_AXES},
                              "vs_old_step56_arm": {ax: _paired_mag(rr, old, ax, cls) for ax in SH_AXES}}
    out["arms"], better_any = {}, False
    for arm in arms:
        runs = table[arm]
        a = {"sign_vs_mse": {ax: _sign_pair(sr(runs, ax), ctl[ax], h5neg, ax) for ax in ("H5", "HD5")},
             "sign_vs_rerun": {ax: _sign_pair(sr(runs, ax), sr(rr, ax), h5neg, ax) for ax in ("H5", "HD5")},
             "magnitude_vs_rerun": {ax: _paired_mag(runs, rr, ax, cls) for ax in SH_AXES}}
        not_worse = {ax: a["sign_vs_mse"][ax]["p_a_above"] > 0.05 and
                     a["sign_vs_mse"][ax]["median_a_pct"] <= SIGN_TOL * a["sign_vs_mse"][ax]["median_c_pct"]
                     for ax in ("H5", "HD5")}
        better = a["sign_vs_rerun"]["H5"]["p_a_below"] < 0.05
        m = a["magnitude_vs_rerun"]
        gate = {ax: m[ax]["widened"][1] <= MAG_UP for ax in GATE_MAG}
        shown = [ax for ax in SHOWN_AXES if m[ax]["widened"][0] > 1 and m[ax]["R"] > MAG_UP]
        sign_yes = all(not_worse.values()) and better
        mag_ok = all(gate.values()) and not shown
        v = ("YES" if sign_yes and mag_ok else "SIGN YES, MAGNITUDE HURT" if sign_yes else
             "PARTIAL" if better else "NOT BETTER THAN THE RERUN")
        if not out["reseed"]["mse_control_stands"] and v in ("YES", "SIGN YES, MAGNITUDE HURT", "PARTIAL"):
            v += " (pending: MSE control rerun at the same code, re-seed rule)"
        better_any |= better
        a["checks"] = {"sign_not_worse_than_mse": not_worse, "better_than_rerun_H5": better,
                       "magnitude_gate_upper_le_1.10": gate, "magnitude_shown_worse": shown}
        a["verdict"] = v
        acc = [[x[1] for x in _sign_read(r)["val_sign_acc"]] for r in runs]
        a["val_sign_accuracy_last_main"] = [x[50] if len(x) > 50 else x[-1] for x in acc]
        a["grad_norm_main"] = [_sign_read(r)["grad_norm_main"] for r in runs]
        out["arms"][arm] = a
    out["rerun_grad_norm_main"] = [_sign_read(r)["grad_norm_main"] for r in rr]
    out["rerun_val_sign_accuracy_last_main"] = [[x[1] for x in _sign_read(r)["val_sign_acc"]][50] for r in rr]
    out["step_verdict"] = "NO (neither arm beats the rerun on H5)" if not better_any else \
        "; ".join(f"{k}: {v['verdict']}" for k, v in out["arms"].items())
    out["blowups"] = blowups([r for k in table for r in table[k]])
    if out["blowups"]["flagged"]:
        out["step_verdict"] = None
    json.dump(out, open(os.path.join(DATA, out_name), "w"), indent=1)
    brief = lambda r: {k: v for k, v in r.items() if k != "per_pool_pct"}
    print(json.dumps({"reseed": {ax: brief(out["reseed"][ax]) for ax in ("H5", "HD5")},
                      "rerun_vs_mse": {ax: brief(r) for ax, r in out["rerun_vs_mse"].items()},
                      "arms": {k: {"verdict": a["verdict"], "checks": a["checks"],
                                   "sign_vs_mse": {ax: brief(r) for ax, r in a["sign_vs_mse"].items()},
                                   "sign_vs_rerun": {ax: brief(r) for ax, r in a["sign_vs_rerun"].items()},
                                   "mag": {ax: [m["R"], m["widened"]] for ax, m in a["magnitude_vs_rerun"].items()},
                                   "acc": a["val_sign_accuracy_last_main"], "gn": [g[0] for g in a["grad_norm_main"]]}
                               for k, a in out["arms"].items()},
                      "rerun_mag_vs_old_ref": {ax: [m["R"], m["widened"]] for ax, m in
                                               out["rerun_magnitude"]["vs_old_het_reference"].items()},
                      "rerun_gn": [g[0] for g in out["rerun_grad_norm_main"]],
                      "blowups": out["blowups"]["flagged"], "spikes": out["blowups"]["spiked_recovered"],
                      "verdict": out["step_verdict"]}, indent=0))


def step61_fig():
    """figs/het_signhead61: (a) H5 and (b) HD5 per pool the seed-mean wrong-sign share of the rerun (weight 1), the two
    matched arms, the MSE control and an all-positive sign, pools sorted by the MSE control; (c) per read the median over
    classes of the class's seed-geometric-mean MSE of log|M|^2 for the rerun and the two arms; (d) the validation
    val_loss_no_reg of the main block, seed median, per arm. From data/het_signhead61.json and data/reads/<run>_val.json."""
    import plot_style as ps
    d = json.load(open(os.path.join(DATA, "het_signhead61.json")))
    fig, axs = ps.figure(ncols=2, nrows=2)
    a, b, c, e = axs.ravel()
    col = {"rerun": ps.C.blue, "matched_event": ps.C.vermillion, "matched_const": ps.C.green}
    lab = {"rerun": "het, sign weight 1", "matched_event": "het, matched per event", "matched_const": "het, sign weight 75"}
    # (c): one point per read (the 12 of SH_AXES), the dashed line is the weight-1 rerun itself (equal MSE)
    for ax, axn in ((a, "H5"), (b, "HD5")):
        ctl = d["rerun_vs_mse"][axn]["per_pool_pct"]
        pools = sorted(ctl, key=lambda p: ctl[p][1])
        x = np.arange(len(pools))
        ax.plot(x, [ctl[p][1] for p in pools], "k_", label="MSE loss control")
        ax.plot(x, [ctl[p][0] for p in pools], "o", color=col["rerun"], mfc="none", label=lab["rerun"])
        for arm, mk in (("matched_event", "^"), ("matched_const", "s")):
            pp = d["arms"][arm]["sign_vs_mse"][axn]["per_pool_pct"]
            ax.plot(x, [pp[p][0] for p in pools], mk, color=col[arm], mfc="none", label=lab[arm])
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel({"H5": "catalog-sampled one-loop pool", "HD5": "default-sampled one-loop class"}[axn])
        ax.set_ylabel("wrong-sign share (%)")
        ax.set_xticks([])
    m0 = d["arms"]["matched_event"]["magnitude_vs_rerun"]
    ref = [m0[r]["mse_class_median_ref"] for r in SH_AXES]
    for arm, mk in (("matched_event", "^"), ("matched_const", "s")):
        m = d["arms"][arm]["magnitude_vs_rerun"]
        c.plot(ref, [m[r]["mse_class_median_arm"] for r in SH_AXES], mk, color=col[arm], mfc="none")
    lim = [min(ref) / 1.5, max(ref) * 2.5]
    c.plot(lim, lim, color=col["rerun"], linestyle="--", label="equal to weight 1")
    c.set(xscale="log", yscale="log", xlim=lim, ylim=lim)
    c.set_xlabel(r"weight 1: class median MSE$(\log|M|^2)$")
    c.set_ylabel(r"matched: class median MSE")
    for arm in ("rerun", "matched_event", "matched_const"):
        cur = [json.load(open(os.path.join(READS, f"{r}_val.json")))["blocks"][0]["vals"] for r in STEP61[arm]]
        n = min(len(v) for v in cur)
        # the median over the four seeds: single-seed spikes (up to 2e10, all recovered) would hide the curves
        e.plot([v[0] for v in cur[0][:n]], np.median([[v[1] for v in x[:n]] for x in cur], axis=0), color=col[arm])
    e.set_yscale("symlog", linthresh=1)
    e.set_xlabel("training step")
    e.set_ylabel("validation het NLL")
    h = a.get_legend_handles_labels()[0] + c.get_legend_handles_labels()[0]
    ps.legend(c, "upper left", handles=h)
    ps.save(fig, os.path.join(FIGS, "het_signhead61"))


if __name__ == "__main__":
    if sys.argv[1:] == ["magnitude"]:
        magnitude()
    elif sys.argv[1:] == ["signhead_pull"]:
        signhead_pull()
    elif sys.argv[1:] == ["step61_pull"]:
        signhead_pull(STEP61)
    elif sys.argv[1:] == ["step61"]:
        step61()
    elif sys.argv[1:] == ["step73"]:
        step61(STEP73, ("branch",), "het_signhead73.json", "plan catalog_coverage step 73 (D296, D315)")
    elif sys.argv[1:] == ["step61_fig"]:
        step61_fig()
    elif sys.argv[1:] == ["signhead"]:
        signhead()
    else:
        pull() if sys.argv[1:] == ["pull"] else summary()
