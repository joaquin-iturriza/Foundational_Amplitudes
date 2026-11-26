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
        "finale": "recipe:recipes/transfer_finale.yaml"}   # the D392 setup's 28 processes (its parent's log lists no pools)

MINIMA = r'''
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.getcwd()); import siteconf
import torch, wrappers
from particle_ids import GLOBAL_PDG_IDX
eta = np.array([1., -1., -1., -1.]); out = {}
for a in sys.argv[1:]:
    lab, log = a.split("#"); r = {}
    if log.startswith("recipe:"):   # every pool the recipe trains, validates and tests on, resolved as training resolves it
        import subprocess, yaml   # (train pools sit in the train cache, not DATA_DIR; mixture pools carry _smix)
        n = len(yaml.safe_load(open(log[7:]))["processes"])
        ps = subprocess.run([sys.executable, "tools/check_site_data.py", "--paths", log[7:]], capture_output=True, text=True).stdout
        pools = [(re.search(r"GeV_(train|val|test)", q).group(1), q) for q in
                 (l[5:] for l in ps.splitlines() if l.startswith("POOL ") and l.endswith(".npy"))]
        assert len(pools) == 3 * n and all(os.path.exists(q) for _, q in pools), (n, [q for _, q in pools if not os.path.exists(q)], ps[-2000:])
    else:
        pools = re.findall(r"\[(train|val|test)\] \S+: \d+ events.*? from (\S+\.npy)",
                           open(os.path.join(siteconf.PROJECT_DIR, "runs", log)).read())
    for role, path in pools:
        rows = np.load(path, mmap_mode="r"); n = (rows.shape[1] - 1) // 5; xs = []; pid_min = (np.inf, 0); soft = {}
        moved = 0   # momenta the shipped finale rule changes, through wrappers._soft_rescale itself (D561)
        for i in range(0, len(rows), 50000):
            p = np.asarray(rows[i:i + 50000, :4 * n], dtype=np.float64).reshape(-1, n, 4)
            ids = np.asarray(rows[i:i + 50000, 4 * n:5 * n]).astype(np.int64)[:, 2:]
            Q = p[:, 0] + p[:, 1]; s = (Q * Q * eta).sum(-1)
            xi = np.abs(2 * (p * Q[:, None] * eta).sum(-1) / s[:, None])[:, 2:]
            xs.append(xi.min(1)); k = np.unravel_index(xi.argmin(), xi.shape)
            pid_min = min(pid_min, (float(xi[k]), int(ids[k])))
            t = torch.from_numpy(p.reshape(-1, 4)); ptr = torch.arange(0, len(p) * n + 1, n)
            idx = torch.tensor([GLOBAL_PDG_IDX[int(q)] for q in np.asarray(rows[i:i + 50000, 4 * n:5 * n]).reshape(-1)])
            r_ = wrappers._soft_rescale(t, ptr, 1e-8, x0=0.009, keep=wrappers._neutrinos(idx))
            moved += int((r_ != t).any(-1).sum())
            for q in ids[xi < 0.009]:   # the final-state momenta a rule at x0 = 0.009 would change, by PDG id
                soft[str(q)] = soft.get(str(q), 0) + 1
        x = np.concatenate(xs)
        r[role if not log.startswith("recipe:") else role + "/" + os.path.basename(path)] = {"path": path, "n": int(len(x)), "min": float(x.min()), "q1e-4": float(np.quantile(x, 1e-4)),
                   "below_0.01": int((x < 0.01).sum()), "pid_min": pid_min[1], "below_0.009_by_pid": soft,
                   "moved_x0_0.009_nu_exempt": moved}
    out[lab] = r
print("SRMIN " + json.dumps(out))
'''


X0 = 0.009   # below every pool's softest x (data/support_minima.json: Z+4g train 0.00923, u u~ g train 0.01107)
RULE = f"model.input_soft_eps=1e-8,model.input_soft_x0={X0}"
RULE_NU = RULE + ",model.input_soft_skip_neutrinos=true"   # the finale's shipped rule (D561 step 4)
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
                out["rule_model"]["input_soft_eps"] == float(want["model.input_soft_eps"]) and \
                out["rule_model"].get("input_soft_skip_neutrinos") in (None, want.get("model.input_soft_skip_neutrinos") == "true"), \
                f"rule not on the model: {m[0]}"
    except Exception as e:
        out[arm] = {"error": repr(e)[:200], "tail": (r.stdout + r.stderr)[-600:]}
print("SRKIT " + json.dumps(out))
'''


RUNS_NU = {f"{k}_nu": v for k, v in RUNS.items() if k.startswith("uug_")}   # u u~ g has no neutrino: must equal RULE


def score(cached, nu=False):
    f = os.path.join(HERE, "data", "support_kit_nu.json" if nu else "support_kit.json")
    runs = RUNS_NU if nu else RUNS
    res = json.load(open(f)) if os.path.exists(f) else {}
    if not cached:
        for lab, d in runs.items():
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
    missing = [k for k in runs if k not in res]
    print(json.dumps({"missing": missing, "read": sorted(res)}))
    return not missing


def read_nu():
    """D561 step 4: per u u~ g run and kit stratum, the neutrino-exempt rule against the x0 = 0.009 rule scored before
    (data/support_kit.json, same runs and kit_predict at KIT_THREADS 1): u u~ g has no neutrino, so both arms must agree
    to the pipeline tolerance 0.01 dec, and the off arm must reproduce the earlier off arm -> data/support_nu.json"""
    import numpy as np
    old = json.load(open(os.path.join(HERE, "data", "support_kit.json")))
    new = json.load(open(os.path.join(HERE, "data", "support_kit_nu.json")))
    out = {"max_abs_dec": {}, "problems": []}
    for lab in RUNS_NU:
        k, o = new[lab], old[lab[:-3]]
        if k["rule_model"].get("input_soft_skip_neutrinos") is not True:
            out["problems"].append(f"{lab}: rule_model {k['rule_model']}")
        if k["changed"] != o["changed"]:
            out["problems"].append(f"{lab}: changed counts differ")
        for arm in ("off", "rule"):
            d = max(abs(np.log10(k[arm][sk] / o[arm][sk])) for sk in o[arm])
            out["max_abs_dec"][f"{lab}/{arm}"] = float(d)
            if d >= 0.01:
                out["problems"].append(f"{lab} {arm}: max |log10 new/old| {d:.3g}")
    out["max_dec"] = max(out["max_abs_dec"].values())
    out["outcome"] = "SAME" if not out["problems"] else "DIFFERENT"
    json.dump(out, open(os.path.join(HERE, "data", "support_nu.json"), "w"), indent=1)
    print(json.dumps(out))


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


def jobs(nu=False):
    for lab, d in (RUNS_NU if nu else RUNS).items():
        print(f"{lab}\t{lab}#{d}\t{RULE_NU if nu else RULE}")


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
    import math
    fin = res["finale"].values(); soft = [v for v in fin if v["below_0.009_by_pid"]]
    # where a momentum at x enters the model under the rule at x0: x0 log1p(x/eps) / log1p(x0/eps) (wrappers._soft_rescale)
    enter = {f"enter_1e-7_x0_{x0}": x0 * math.log1p(1e-7 / 1e-8) / math.log1p(x0 / 1e-8) for x0 in (X0, 1e-5)}
    print(json.dumps({"finale_min": min(v["min"] for v in fin), "finale_min_pid": min(fin, key=lambda v: v["min"])["pid_min"],
                      "finale_below_0.009": sum(n for v in soft for n in v["below_0.009_by_pid"].values()),
                      "finale_below_0.009_pids": sorted({q for v in soft for q in v["below_0.009_by_pid"]}),
                      "finale_min_others": min(v["min"] for v in fin if not v["below_0.009_by_pid"]),
                      "zng_train_min": res["zng"]["train"]["min"],
                      # momenta wrappers._soft_rescale changes at x0=0.009 with neutrinos kept, over every pool
                      "n_pools": sum(len(res[g]) for g in ("finale", "zng", "uug")),
                      "moved_nu_exempt": sum(v["moved_x0_0.009_nu_exempt"] for g in ("finale", "zng", "uug") for v in res[g].values()),
                      **enter}))


if __name__ == "__main__":
    if sys.argv[1] == "remote":
        remote(*sys.argv[2:4]); sys.exit(0)
    if sys.argv[1] == "read":
        read(); sys.exit(0)
    if sys.argv[1] == "jobs":
        jobs("--nu" in sys.argv); sys.exit(0)
    if sys.argv[1] == "read_nu":
        read_nu(); sys.exit(0)
    if sys.argv[1] == "score":
        ok = score("--cached" in sys.argv, "--nu" in sys.argv)
    else:
        ok = minima("--cached" in sys.argv)
    sys.exit(0 if ok is not False else 75)
