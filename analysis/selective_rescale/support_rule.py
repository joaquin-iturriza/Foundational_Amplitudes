"""Plan selective_rescale (D552, D561): the rescale applied only outside the trained range. Rule: the D539 log-rescale
on momenta with x_i = 2 p_i.Q/s below x0 (model.input_soft_x0), x0 at or below the softest x of every training pool,
so every trained momentum is unchanged; training under the rule is then the no-rescale training, and the rule is scored
on the existing no-rescale runs with the rule switched on at evaluation (kit scoring, CPU).

    python3 analysis/selective_rescale/support_rule.py minima [--cached]   softest x per pool (train/val/test), u u~ g and Z+4g
    python3 analysis/selective_rescale/support_rule.py score [--cached]    kit scores of the 12 no-rescale runs (u u~ g, Z+4g;
        1e4, 1e5; chosen trial + seeds 43/44), each without and with the rule (REBUILD_SET, tools/rebuild_run.py), CPU jobs
        on CC (a call submits the missing ones, a later call reads them)  -> data/support_kit.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from analysis.common.readers import site_python  # noqa: E402

LOGS = {"uug": "ftmech_d539_uug_1e5_off_b2048/trial_0020/out_0.log",
        "zng": "selres_zng_1e5_off_b2048/trial_1333/out_0.log"}   # pools named in a 1e5 no-rescale run's log, under the site's runs/

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
                    v = site_python(SCORE, [f"'{lab}#{d}'", RULE], "SRKIT", timeout=3600)
                    if any(isinstance(a, dict) and "error" in a for a in v.values()):   # kept out: listed missing, its error shown
                        print(f"{lab}: error {json.dumps(v)[:400]}")
                    else:
                        res[lab] = v
                except SystemExit as e:   # submitted or still running: read on the next call
                    print(f"{lab}: {str(e).splitlines()[0][:120]}")
        json.dump(res, open(f, "w"), indent=1)
    missing = [k for k in RUNS if k not in res]
    print(json.dumps({"missing": missing, "read": sorted(res)}))
    return not missing


def minima(cached):
    f = os.path.join(HERE, "data", "support_minima.json")
    res = json.load(open(f)) if cached else site_python(MINIMA, [f"'{k}#{v}'" for k, v in LOGS.items()], "SRMIN")
    json.dump(res, open(f, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    ok = {"minima": minima, "score": score}[sys.argv[1]]("--cached" in sys.argv)
    sys.exit(0 if ok is not False else 75)
