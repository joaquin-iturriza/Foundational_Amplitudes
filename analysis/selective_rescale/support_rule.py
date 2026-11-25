"""Plan selective_rescale (D552, D561): the rescale applied only outside the trained range. Rule: the D539 log-rescale
on momenta with x_i = 2 p_i.Q/s below x0 (model.input_soft_x0), x0 at or below the softest x of every training pool,
so every trained momentum is unchanged; training under the rule is then the no-rescale training, and the rule is scored
on the existing no-rescale runs with the rule switched on at evaluation (kit scoring, CPU).

    python3 analysis/selective_rescale/support_rule.py minima [--cached]   softest x per pool (train/val/test), u u~ g and Z+4g
    python3 analysis/selective_rescale/support_rule.py score [--cached]    kit scores of the 12 no-rescale runs (u u~ g, Z+4g;
        1e4, 1e5; chosen trial + seeds 43/44), each without and with the rule (REBUILD_SET, tools/rebuild_run.py): reads
        the SRKIT line of each run's CPU job (scripts/job_support_rule.sh, its run id in data/support_jobs.json) from its
        log; Z+4g read earlier through site_python  -> data/support_kit.json
    python3 analysis/selective_rescale/support_rule.py jobs          the job_support_rule.sh arguments of each missing run
    python analysis/selective_rescale/support_rule.py remote '<lab>#<dir>' <rule>   (on the site, in the job) one run's SCORE
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from analysis.common.readers import site_python  # noqa: E402

LOGS = {"uug": "ftmech_d539_uug_1e5_off_b2048/trial_0020/out_0.log",
        "zng": "selres_zng_1e5_off_b2048/trial_1333/out_0.log",   # pools named in a 1e5 no-rescale run's log, under the site's runs/
        "finale": "data_saving/dsave_parent_finale_zng_uq_w_lr1p89e-3/out_0.log"}   # the D392 setup's 28-process parent

MINIMA = r'''
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.getcwd()); import siteconf
eta = np.array([1., -1., -1., -1.]); out = {}
for a in sys.argv[1:]:
    lab, log = a.split("#"); r = {}
    log = os.path.join(siteconf.PROJECT_DIR, "runs", log)
    for role, path in re.findall(r"\[(train|val|test)\] \S+: \d+ events.*? from (\S+\.npy)", open(log).read()):
        rows = np.load(path, mmap_mode="r"); n = (rows.shape[1] - 1) // 5; xs = []
        for i in range(0, len(rows), 50000):
            p = np.asarray(rows[i:i + 50000, :4 * n], dtype=np.float64).reshape(-1, n, 4)
            Q = p[:, 0] + p[:, 1]; s = (Q * Q * eta).sum(-1)
            xs.append(np.abs(2 * (p * Q[:, None] * eta).sum(-1) / s[:, None])[:, 2:].min(1))
        x = np.concatenate(xs)
        r[role] = {"path": path, "n": int(len(x)), "min": float(x.min()), "q1e-4": float(np.quantile(x, 1e-4)),
                   "below_0.01": int((x < 0.01).sum())}
    out[lab] = r
print("SRMIN " + json.dumps(out))
'''


X0 = 0.009   # below every pool's softest x (data/support_minima.json: Z+4g train 0.00923, u u~ g train 0.01107)
RULE = f"model.input_soft_eps=1e-8,model.input_soft_x0={X0}"
SEEDS = (42, 43, 44)
RUNS = {**{f"uug_{d}_{s}": f"ftmech_d539_uug_{d}_off_b2048/{t}" if s == 42 else f"ftmech_d539_uug_{d}_off_b2048_s{s}"
           for d, t in (("1e4", "trial_1559"), ("1e5", "trial_0020")) for s in SEEDS},
        **{f"zng_1e4_{s}": f"ftmech_d539s_zng_1e4_off_s{s}" for s in SEEDS},
        **{f"zng_1e5_{s}": f"selres_zng_1e5_off_b2048/trial_1333" if s == 42 else f"selres_s2_zng_1e5_off_s{s}"
           for s in SEEDS}}   # no-rescale controls of D539 / stage 2 (Z+4g 1e4 at hp1333, its search's choice)

SCORE = r'''
import json, os, shutil, subprocess, sys
sys.path.insert(0, os.getcwd()); import siteconf
out = {}
lab, d = sys.argv[1].split("#")
d = os.path.join(siteconf.PROJECT_DIR, "runs", d)
import numpy as np
from tools import kit_hardps as kh
from tools.kit_predict import run_target
x0 = float(sys.argv[2].split("input_soft_x0=")[1].split(",")[0]); eta = np.array([1., -1., -1., -1.])
stem, target, _ = kh.parse(run_target(d)); out["changed"] = {}
for sp in ("hard_test", "bulk_test"):   # kit events with any final-state momentum below x0, per stratum
    for st in kh.strata(stem, sp):
        rows = kh.stratum_rows(stem, target, sp, st); n = (rows.shape[1] - 1) // 5
        p = rows[:, :4 * n].astype(np.float64).reshape(-1, n, 4); Q = p[:, 0] + p[:, 1]; s = (Q * Q * eta).sum(-1)
        x = np.abs(2 * (p * Q[:, None] * eta).sum(-1) / s[:, None])[:, 2:]
        out["changed"][f"{sp}/{st}"] = [int((x.min(1) < x0).sum()), int(len(x))]
if not any(c for c, _ in out["changed"].values()):
    out["skipped"] = "no kit event below x0: the rule changes nothing"
    print("SRKIT " + json.dumps(out)); sys.exit(0)
for arm, env in (("off", {}), ("rule", {"REBUILD_SET": sys.argv[2]})):
    o = os.path.join(os.environ["SCRATCH"], "sr_support", f"{lab}_{arm}")
    shutil.rmtree(o, ignore_errors=True)   # never read an earlier pass's result
    r = subprocess.run([sys.executable, "tools/kit_predict.py", d, "--out", o], env={**os.environ, **env},
                       capture_output=True, text=True)
    try:
        assert r.returncode == 0, f"kit_predict rc={r.returncode}"
        k = json.load(open(os.path.join(o, "kit_result.json"))); k = k.get("strata", k)
        out[arm] = {s: v["logmse"] for s, v in k.items() if isinstance(v, dict) and "logmse" in v}
        m = [l for l in r.stdout.splitlines() if l.startswith("REBUILD_SET_MODEL ")]
        if env:
            assert m, "no REBUILD_SET_MODEL line: the rule did not reach the model"
            out["rule_model"] = json.loads(m[0].split(" ", 1)[1])
            want = dict(kv.split("=") for kv in sys.argv[2].split(","))
            assert out["rule_model"]["input_soft_x0"] == float(want["model.input_soft_x0"]) and \
                out["rule_model"]["input_soft_eps"] == float(want["model.input_soft_eps"]), f"rule not on the model: {m[0]}"
    except Exception as e:
        out[arm] = {"error": repr(e)[:200], "tail": (r.stdout + r.stderr)[-600:]}
print("SRKIT " + json.dumps(out))
'''


def score(cached):
    f = os.path.join(HERE, "data", "support_kit.json")
    res = json.load(open(f)) if os.path.exists(f) else {}
    if not cached:
        for lab, d in RUNS.items():
            if lab not in res:
                try:
                    v = job_result(lab)
                    if v is None:
                        continue
                    if any(isinstance(a, dict) and "error" in a for a in v.values()):   # kept out: listed missing, its error shown
                        print(f"{lab}: error " + json.dumps({k: a for k, a in v.items() if k != "changed"})[:1500])
                    else:
                        res[lab] = v
                except SystemExit as e:   # submitted or still running: read on the next call
                    print(f"{lab}: {str(e).splitlines()[0][:120]}")
        json.dump(res, open(f, "w"), indent=1)
    missing = [k for k in RUNS if k not in res]
    print(json.dumps({"missing": missing, "read": sorted(res)}))
    return not missing


def job_result(lab):
    """The SRKIT dict of `lab`'s job_support_rule.sh job (its run id in data/support_jobs.json), None while it has none.
    site_python's `site run --job` gives 2G (the CPU block's mem_per_cpu, one CPU): kit_predict was killed there."""
    import subprocess
    jobs = json.load(open(os.path.join(HERE, "data", "support_jobs.json")))
    if lab not in jobs:
        print(f"{lab}: no job submitted (support_rule.py jobs)"); return None
    r = subprocess.run(["site", "logs", jobs[lab], "-n", "100000"], capture_output=True, text=True, timeout=300)
    line = next((l for l in r.stdout.splitlines() if l.startswith("SRKIT ")), None)
    if line is None:
        print(f"{lab}: {jobs[lab]} has no SRKIT line yet: " + (r.stdout + r.stderr)[-300:].replace("\n", " | "))
        return None
    return json.loads(line.split(" ", 1)[1])


def jobs():
    for lab, d in RUNS.items():
        print(f"{lab}\t{lab}#{d}\t{RULE}")


def remote(arg, rule):
    sys.argv = [sys.argv[0], arg, rule]
    exec(SCORE, {"__name__": "__main__"})


RAW = os.path.join(ROOT, "analysis", "finetune_mechanism", "data", "d539u_kit_raw.json")   # stored kit scores, same runs


def read():
    """The passed design's readout (D561): pipeline checks, per kit stratum the seed medians of both arms and no rescale's
    own seed spread log10(max/min) from the stored scores, YES / NO-gain / NO-cost  -> data/support_read.json, figure"""
    import numpy as np
    from tools.kit_mlp_readout import deepest
    kit = json.load(open(os.path.join(HERE, "data", "support_kit.json"))); raw = json.load(open(RAW))
    out = {"pipeline": [], "cells": {}}
    for d in ("1e4", "1e5"):
        labs = [f"uug_{d}_{s}" for s in SEEDS]
        stored = [raw[f"ftmech_d539u_uug_{d}_off_s{s}"] for s in SEEDS]
        for lab, st in zip(labs, stored):
            k = kit[lab]
            if k["rule_model"] != {"input_soft_eps": 1e-8, "input_soft_x0": X0, "input_soft_gate": k["rule_model"].get("input_soft_gate")}:
                out["pipeline"].append(f"{lab}: rule_model {k['rule_model']}")
            for sk, v in k["off"].items():   # the off arm reproduces the stored no-rescale scores
                if abs(np.log10(v / st[sk])) >= 0.01:
                    out["pipeline"].append(f"{lab} off {sk}: {v:.4g} vs stored {st[sk]:.4g}")
            for sk in k["off"]:
                n = k["changed"]["/".join(sk.split("/")[:2])][0]
                dl = abs(np.log10(k["rule"][sk] / k["off"][sk]))
                if (n == 0 and dl >= 0.01) or (n > 0 and dl == 0):
                    out["pipeline"].append(f"{lab} {sk}: {n} changed, |log10 rule/off| {dl:.3g}")
        k0 = kit[labs[0]]; cell = {}
        for sk in sorted(k0["off"]):
            off = [kit[l]["off"][sk] for l in labs]; rule = [kit[l]["rule"][sk] for l in labs]
            spread = float(np.log10(max(t[sk] for t in stored) / min(t[sk] for t in stored)))
            cell[sk] = {"changed": k0["changed"]["/".join(sk.split("/")[:2])], "off": off, "rule": rule,
                        "dec": float(np.log10(np.median(rule) / np.median(off))), "spread": spread}
        dp = deepest({sk: sk for sk in cell})   # the deepest stratum's key per family
        deep = cell[dp["soft_x"]]
        cost = {sk: c["dec"] for sk, c in cell.items() if c["changed"][0] > 0 and c["dec"] > c["spread"]}
        gain = deep["dec"] < -deep["spread"]
        out["cells"][d] = {"strata": cell, "deepest_soft": dp["soft_x"], "gain": gain, "cost_strata": cost,
                           "outcome": "YES" if gain and not cost else ("NO-cost" if cost else "NO-gain")}
    out["outcome"] = "BROKEN" if out["pipeline"] else (
        "YES" if all(c["outcome"] == "YES" for c in out["cells"].values()) else
        ",".join(f"{d}:{c['outcome']}" for d, c in out["cells"].items()))
    json.dump(out, open(os.path.join(HERE, "data", "support_read.json"), "w"), indent=1)
    for d, c in out["cells"].items():
        print(d, c["outcome"], "deepest", c["deepest_soft"], "cost", c["cost_strata"])
        for sk, v in c["strata"].items():
            print(f"  {sk:45s} changed {v['changed'][0]:6d}/{v['changed'][1]:6d}  off {np.median(v['off']):10.4g}"
                  f"  rule {np.median(v['rule']):10.4g}  dec {v['dec']:+.3f}  spread {v['spread']:.3f}")
    print("pipeline", out["pipeline"] or "ok", "outcome", out["outcome"])
    plot(out)
    full = json.load(open(RAW.replace("_raw", "")))["cells"]   # D539's full rescale (eps1e-8), same seeds
    summ = {"outcome": out["outcome"], "pipeline_problems": len(out["pipeline"])}
    for d, c in out["cells"].items():
        dp = c["strata"][c["deepest_soft"]]; ch = {sk: v for sk, v in c["strata"].items() if v["changed"][0] > 0}
        summ[d] = {"deep_off": float(np.median(dp["off"])), "deep_rule": float(np.median(dp["rule"])),
                   "deep_dec": dp["dec"], "deep_spread": dp["spread"], "deep_full": float(np.median(full[d]["eps1e-8"]["soft_x"])),
                   "n_changed_strata": len(ch), "changed_dec_max": max(v["dec"] for v in ch.values()),
                   "changed_dec_min": min(v["dec"] for v in ch.values()),
                   "shallow_changed": c["strata"]["hard_test/soft_x_0.001_0.01/log10_me2"]["changed"][0],
                   "shallow_dec": c["strata"]["hard_test/soft_x_0.001_0.01/log10_me2"]["dec"],
                   "shallow_spread": c["strata"]["hard_test/soft_x_0.001_0.01/log10_me2"]["spread"],
                   "bulk_off": float(np.median(c["strata"]["bulk_test/bulk/log10_me2"]["off"])),
                   "bulk_dec": c["strata"]["bulk_test/bulk/log10_me2"]["dec"]}
    print(json.dumps(summ))


def plot(out):
    """MSE(log10|M|^2) of each soft kit stratum against its x range, both arms (seed median, bars min to max)"""
    import numpy as np
    sys.path.insert(0, ROOT)
    import plot_style as ps
    from tools.kit_mlp_readout import edges
    fig, axs = ps.figure(ncols=2)
    for ax, (d, c) in zip(axs, out["cells"].items()):
        soft = sorted((sk for sk in c["strata"] if "/soft_x_" in sk), key=lambda sk: edges(sk.split("/")[1])[0])
        x = [np.log10(np.sqrt(np.prod(edges(sk.split("/")[1])))) for sk in soft]   # the stratum's log-centre
        for arm, col, lab in (("off", ps.C.blue, "no rescale"), ("rule", ps.C.vermillion, "rescale below $x=0.009$")):
            v = np.array([c["strata"][sk][arm] for sk in soft]); m = np.median(v, 1)
            ax.errorbar(x, m, yerr=[m - v.min(1), v.max(1) - m], color=col, marker="o", label=lab)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10} x_{\min}$, $x_i=2p_i\cdot Q/s$")
        ax.set_ylabel(r"MSE$(\log_{10}|\mathcal{M}|^2)$")
        ps.process_label(ax, r"$e^+e^-\to u\bar ug$")
    ps.legend(axs[0], "upper left")
    ps.save(fig, os.path.join(HERE, "figs", "support_rule"))


def minima(cached):
    f = os.path.join(HERE, "data", "support_minima.json")
    res = json.load(open(f)) if cached else site_python(MINIMA, [f"'{k}#{v}'" for k, v in LOGS.items()], "SRMIN", timeout=5400)
    json.dump(res, open(f, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    if sys.argv[1] == "remote":
        remote(*sys.argv[2:4]); sys.exit(0)
    if sys.argv[1] == "read":
        read(); sys.exit(0)
    if sys.argv[1] == "jobs":
        jobs(); sys.exit(0)
    ok = {"minima": minima, "score": score}[sys.argv[1]]("--cached" in sys.argv)
    sys.exit(0 if ok is not False else 75)
