#!/usr/bin/env python3
"""What the new flat-in-log|M|^2 sampling costs against the old, per class and over the 1301 hadronic tree classes
(plan catalog_coverage, Joaquin 2026-11-06 ~19:50: "how much more expensive does this make the sampling compared to
before").

Cost = CPU core-seconds of building a class's tree pools (train 1e5, val 1e4, test 1e4): phase-space sampling and the
matrix-element labelling of every candidate together (datagen.gen_chunk's wall time on one core per chunk; labelling is
~0.97 of it, coverage_check.py @27e8edd). The default-share MadEvent events (dsh_ pools) are the same under both and are
not counted here.

  old  measured: the [TIME] chunk lines of catalog_v4's own generation jobs (scripts/job_catalog_v4_regen.sh, stages
       1-4 in 8 shards and stage 5 in 4 shards, 2026-10-26, OLD_RUNS), every role, summed per class.
  new  measured on the five test classes only: the train-pool replay of multichannel_check.py with both ladders at
       sampling.flat_budget 750 (policy quota_ladder_both_res2_budget750, seeds 42/43, mean), scaled to the three
       roles by events (x1.2: val and test sampled the same way, 1e4 events each).
  ratio new / old per test class (old from the generation logs: other nodes, other day; the plain multichannel replay
       of step 65, same work as the old sampling, is printed beside it as the node check).
  total an ESTIMATE: each of the 1301 classes' measured old cost times a ratio set by its final-state count: nfinal
       <= 3 the 2->3 test class's ratio (uux_epemg; no 2->2 class was measured, assumed the same), nfinal >= 4 the
       four 2->4 test classes' median, with the bracket min..max of the four.

    python3 analysis/catalog_v4/sampling_cost.py pull      the generation logs (site logs) -> data/sampling_cost_gen.json
    python3 analysis/catalog_v4/sampling_cost.py summary   -> data/sampling_cost.json
    python3 analysis/catalog_v4/sampling_cost.py fig       -> figs/sampling_cost.{png,pdf}
    python3 analysis/catalog_v4/sampling_cost.py ee        the e+e- classes (plan step (f)) -> data/sampling_cost_ee.json
  ee   per e+e- class of the range table: old = its generation-log core-s (every role); new/old measured on the four
       e+e- check classes in the same job (policy NEW_POLICY seconds over the old sampling's, 'current', same seed,
       node and commit; seeds' mean); the total an UPPER-SIDE ESTIMATE: the four are the classes whose default events
       lie farthest above the catalog's (data/ee_support_reads.json), so the largest measured ratio of a class with
       the same final count is applied (2 and 5+ finals, none measured: the largest of all four)
    python3 analysis/catalog_v4/sampling_cost.py ee_fig    -> figs/sampling_cost_ee.{png,pdf}
"""
import json, os, re, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
DATA = os.path.join(HERE, "data")
GEN = os.path.join(DATA, "sampling_cost_gen.json")
OUT = os.path.join(DATA, "sampling_cost.json")
READS = os.path.join(DATA, "multichannel_check_reads")
# catalog_v4's generation (2026-10-26): stages 1-4 in 8 shards, stage 5 (the electroweak-order classes) in 4
OLD_RUNS = ["FA-ccin2p3-2424a38f", "FA-ccin2p3-6bd04ec8", "FA-ccin2p3-b67b49d6", "FA-ccin2p3-7a70bae4",
            "FA-ccin2p3-05f408b4", "FA-ccin2p3-23fe1dcb", "FA-ccin2p3-0ef20da7", "FA-ccin2p3-b6969e26",
            "FA-ccin2p3-5826d401", "FA-ccin2p3-e1bd8668", "FA-ccin2p3-16686dbf", "FA-ccin2p3-f5a04d70"]
NEW_POLICY = "quota_ladder_both_res2_budget750"
NEW_FILE = "{c}_quota_ladder_res2+quota_ladder_edge_res2+quota_ladder_both_res2+quota_ladder_both_res2_budget750_{s}.json"
PLAIN_FILE = "{c}_multichannel+quota_ladder_res2+quota_ladder_res2_budget500_{s}.json"
TEST = ("v4_uux_epemg", "v4_uux_epemvevex", "v4_udx_csxvevex", "v4_us_ddxsu", "v4_uu_bbxuu")
EE_PREFIX = "v4_epem_"
EE_TEST = ("v4_epem_taptamvevex", "v4_epem_bbxvevex", "v4_epem_ahvevex", "v4_epem_vmvmxz")
EE_FILE = "{c}_current+multichannel+quota_ladder_both_res2+quota_ladder_both_res2_budget750_{s}.json"
# step 67's e+e- forward check: the forward-free policy beside current and budget 750 in one job
EE_FILE_67 = "{c}_current+quota_ladder_both_res2_budget750+quota_ladder_both_res2_fwdfree_budget750_{s}.json"
OUT_EE = os.path.join(DATA, "sampling_cost_ee.json")
ROLE_SCALE = 1.2          # (1e5 + 1e4 + 1e4) / 1e5: val and test sampled as train
TIME_RE = re.compile(r"^\[TIME\] (\S+)\s+(train|val|test)\s+chunk\s+(\d+):\s+(\d+) events in\s+([\d.]+)s")


def parse(text):
    """{class: {role: {idx: (events, seconds)}}} from a log's [TIME] chunk lines (reused chunks have no such line)"""
    out = {}
    for line in text.splitlines():
        m = TIME_RE.match(line.strip())
        if m:
            c, role, idx, n, s = m.groups()
            out.setdefault(c, {}).setdefault(role, {})[int(idx)] = (int(n), float(s))
    return out


def pull():
    classes = {}
    for run in OLD_RUNS:
        txt = subprocess.run(["site", "logs", run, "-n", "300000"], capture_output=True, text=True, timeout=600).stdout
        for c, roles in parse(txt).items():
            for role, ch in roles.items():
                classes.setdefault(c, {}).setdefault(role, {}).update(ch)   # a later run's chunk wins
        print(run, len(txt.splitlines()), "lines", flush=True)
    rows = {c: {role: {"chunks": len(ch), "events": sum(v[0] for v in ch.values()),
                       "seconds": round(sum(v[1] for v in ch.values()), 1)} for role, ch in roles.items()}
            for c, roles in classes.items()}
    json.dump({"source": "site logs of OLD_RUNS, [TIME] chunk lines", "runs": OLD_RUNS, "classes": rows},
              open(GEN, "w"), indent=1)
    print(GEN, len(rows), "classes")


def _nfinal(names):
    import yaml
    import mg5_pipeline_final as mg
    recipes = ["catalog_v4_base_train.yaml"] + [f"catalog_v4_heldout_{h}.yaml"
                                                for h in ("H1", "H2", "H2p5", "H3", "H6", "H7E", "H7c")]
    for r in recipes:
        spec = yaml.safe_load(open(os.path.join(ROOT, "recipes", r)))
        mg.register_recipe_processes([{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
                                       "sampling": p.get("sampling")} for p in spec["processes"]],
                                     default_sampling=spec.get("sampling"))
    return {c: int(mg.PROCESSES[c]["nfinal"]) for c in names}


def _seconds(c, fname, policy):
    xs = []
    for s in (42, 43):
        p = os.path.join(READS, fname.format(c=c, s=s))
        if os.path.exists(p):
            r = json.load(open(p))["runs"].get(f"{policy}_{s}")
            if r and r.get("seconds"):
                xs.append(float(r["seconds"]))
    return xs


# plan step 75's sampler (channel grids + forward joint resonance, at 12 labels per flat event), read the same way:
# `summary grid12` -> data/sampling_cost_grid12.json
STEP75_FILE = ("{c}_current+quota_ladder_both_res2+quota_ladder_both_res2_budget12+grid_both_res2f_budget12+"
               "grid_both_res2f_budget8_{s}.json")
POLICIES = {"grid12": (STEP75_FILE, "grid_both_res2f_budget12"), "grid8": (STEP75_FILE, "grid_both_res2f_budget8")}


def summary(policy=None):
    """without `policy` the budget-750 quota sampler (NEW_POLICY) -> OUT; with one of POLICIES that policy's seconds
    from its own check jobs -> OUT with _<policy> before .json"""
    new_file, new_policy = POLICIES[policy] if policy else (NEW_FILE, NEW_POLICY)
    gen = json.load(open(GEN))["classes"]
    fr = json.load(open(os.path.join(ROOT, "recipes", "catalog_v4_flat_range.json")))
    names = sorted(c for c in fr if not c.startswith(EE_PREFIX))     # the 1301 hadronic classes
    missing = [c for c in names if c not in gen or not all(r in gen[c] for r in ("train", "val", "test"))]
    nf = _nfinal(names)
    old = {c: sum(gen[c][r]["seconds"] for r in gen[c]) for c in names if c not in missing}
    test = {}
    for c in TEST:
        new = _seconds(c, new_file, new_policy)
        plain = _seconds(c, PLAIN_FILE, "multichannel")
        o_train = gen[c]["train"]["seconds"]
        test[c] = {"nfinal": nf[c], "old_train_s": o_train, "old_all_s": round(old[c], 1),
                   "plain_multichannel_train_s": plain, "new_train_s": new,
                   "new_all_s": round(float(np.mean(new)) * ROLE_SCALE, 1),
                   "ratio": round(float(np.mean(new)) / o_train, 1),
                   "ratio_vs_plain_replay": round(float(np.mean(new)) / float(np.mean(plain)), 1)}
    r3 = test["v4_uux_epemg"]["ratio"]
    r4 = [test[c]["ratio"] for c in TEST if test[c]["nfinal"] >= 4]
    rat = {"central": (r3, float(np.median(r4))), "low": (r3, min(r4)), "high": (r3, max(r4))}
    by_n = {}
    for c, s in old.items():
        b = by_n.setdefault(nf[c], {"classes": 0, "old_core_h": 0.0})
        b["classes"] += 1; b["old_core_h"] += s / 3600
    tot = {"old_core_h": round(sum(old.values()) / 3600, 1)}
    for k, (a, b) in rat.items():
        tot[f"new_core_h_{k}"] = round(sum(s * (a if nf[c] <= 3 else b) for c, s in old.items()) / 3600)
    for b in by_n.values():
        b["old_core_h"] = round(b["old_core_h"], 2)
    res = {"definition": __doc__.split("\n\n")[1], "policy": new_policy, "missing_old": missing, "test_classes": test,
           "ratio_nfinal_le3": r3, "ratio_nfinal_ge4": {"median": float(np.median(r4)), "min": min(r4), "max": max(r4)},
           "by_nfinal": {str(k): by_n[k] for k in sorted(by_n)}, "total_1301": tot,
           "old_per_class_s": {c: round(s, 1) for c, s in old.items()}, "nfinal": nf}
    json.dump(res, open(OUT.replace(".json", f"_{policy}.json") if policy else OUT, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("test_classes", "ratio_nfinal_ge4", "by_nfinal", "total_1301")}, indent=1))
    print("missing", len(missing), missing[:10])


def fig():
    """left: per test class, core-hours of its three pools, old against new (the new/old ratio above each pair);
    right: the 1301 hadronic classes' total, old measured, new estimated (bar: median 2->4 ratio, whiskers: the
    four 2->4 test classes' min and max ratio)"""
    import plot_style as ps
    sys.path.insert(0, HERE)
    from fig_flat_quota import LABEL
    d = json.load(open(OUT))
    t = d["test_classes"]
    names = list(TEST)
    fig, (ax, ax2) = ps.figure(ncols=2)
    x = np.arange(len(names))
    old = np.array([t[c]["old_all_s"] for c in names]) / 3600
    new = np.array([t[c]["new_all_s"] for c in names]) / 3600
    ax.bar(x - 0.2, old, 0.4, color=ps.C.grey, label="old sampling (as built)")
    ax.bar(x + 0.2, new, 0.4, color=ps.C.vermillion, label="new: flat quota, both ladders, budget 750")
    for i, c in enumerate(names):
        ax.text(x[i] + 0.2, new[i] * 1.3, f"{t[c]['ratio']:.0f}x", ha="center", fontsize=6)
    ax.set_yscale("log")
    ax.set_ylim(top=new.max() * 6)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(c, c) for c in names], fontsize=6, rotation=20)
    ax.set_ylabel("core-hours per class")
    ps.legend(ax, loc="upper left", fontsize=6)
    tot = d["total_1301"]
    ax2.bar([0], [tot["old_core_h"]], 0.5, color=ps.C.grey)
    c, lo, hi = tot["new_core_h_central"], tot["new_core_h_low"], tot["new_core_h_high"]
    ax2.bar([1], [c], 0.5, color=ps.C.vermillion, yerr=[[c - lo], [hi - c]], capsize=3)
    ax2.set_yscale("log")
    ax2.set_ylim(10, hi * 3)
    for xx, v in ((0, tot["old_core_h"]), (1, c)):
        ax2.text(xx, v * 1.4 if xx == 0 else hi * 1.3, f"{v:.0f}", ha="center", fontsize=7)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["old (measured)", "new (estimate)"], fontsize=7)
    ax2.set_ylabel("core-hours, 1301 hadronic classes")
    ps.save(fig, os.path.join(HERE, "figs", "sampling_cost"))


def ee(policy=None):
    """policy: the quota policy timed (default NEW_POLICY); another policy writes OUT_EE with _<policy> before .json"""
    pol = policy or NEW_POLICY
    gen = json.load(open(GEN))["classes"]
    fr = json.load(open(os.path.join(ROOT, "recipes", "catalog_v4_flat_range.json")))
    names = sorted(c for c in fr if c.startswith(EE_PREFIX))
    nf = _nfinal(names)
    old = {c: sum(gen[c][r]["seconds"] for r in gen[c]) for c in names}
    test = {}
    for c in EE_TEST:
        fn = EE_FILE if pol == NEW_POLICY else EE_FILE_67
        new, cur = _seconds(c, fn, pol), _seconds(c, fn, "current")
        r = [a / b for a, b in zip(new, cur)]
        test[c] = {"nfinal": nf[c], "old_all_s": round(old[c], 1), "old_train_gen_log_s": gen[c]["train"]["seconds"],
                   "current_train_replay_s": cur, "new_train_s": new, "ratio_same_job": [round(x, 1) for x in r],
                   "ratio": round(float(np.mean(r)), 1), "new_all_s": round(old[c] * float(np.mean(r)), 1)}
    by_meas = {}
    for c, t in test.items():
        by_meas[t["nfinal"]] = max(by_meas.get(t["nfinal"], 0.0), t["ratio"])
    top = max(by_meas.values())
    by_n = {}
    for c, s_ in old.items():
        k = nf[c]
        b = by_n.setdefault(k, {"classes": 0, "old_core_h": 0.0, "ratio_used": by_meas.get(k, top),
                                "ratio_measured": k in by_meas})
        b["classes"] += 1; b["old_core_h"] += s_ / 3600
    for b in by_n.values():
        b["new_core_h_upper"] = round(b["old_core_h"] * b["ratio_used"], 1); b["old_core_h"] = round(b["old_core_h"], 2)
    tot = {"classes": len(old), "old_core_h": round(sum(old.values()) / 3600, 2),
           "new_core_h_upper": round(sum(b["new_core_h_upper"] for b in by_n.values()), 1)}
    res = {"definition": __doc__.split("  ee   ")[1].split("\n    python3")[0], "policy": pol,
           "test_classes": test, "ratio_by_nfinal_measured": {str(k): v for k, v in sorted(by_meas.items())},
           "by_nfinal": {str(k): by_n[k] for k in sorted(by_n)}, "total": tot,
           "old_per_class_s": {c: round(s_, 1) for c, s_ in old.items()}, "nfinal": nf}
    json.dump(res, open(OUT_EE if not policy else OUT_EE.replace(".json", f"_{policy}.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("test_classes", "by_nfinal", "total")}, indent=1))


def ee_fig():
    """left: per e+e- check class, core-hours of its three pools, old (generation logs) against new (old x the
    same-job ratio, written above each pair); right: the e+e- classes by final count, old measured against new
    upper-side estimate (hatched where the count's ratio was not measured)"""
    import plot_style as ps
    sys.path.insert(0, HERE)
    from fig_quota_fill import EE_LABEL
    d = json.load(open(OUT_EE))
    t = d["test_classes"]
    fig, (ax, ax2) = ps.figure(ncols=2)
    x = np.arange(len(EE_TEST))
    old = np.array([t[c]["old_all_s"] for c in EE_TEST]) / 3600
    new = np.array([t[c]["new_all_s"] for c in EE_TEST]) / 3600
    ax.bar(x - 0.2, old, 0.4, color=ps.C.grey, label="old sampling (as built)")
    ax.bar(x + 0.2, new, 0.4, color=ps.C.vermillion, label="flat quota, budget 750, estimated:\nnew = old $\\times$ (budget-750 replay time / old replay time)")
    for i, c in enumerate(EE_TEST):
        ax.text(x[i] + 0.2, new[i] * 1.3, f"{t[c]['ratio']:.0f}x", ha="center", fontsize=6)
    ax.set_yscale("log"); ax.set_ylim(top=new.max() * 8)
    ax.set_xticks(x); ax.set_xticklabels([EE_LABEL[c] for c in EE_TEST], fontsize=6, rotation=20)
    ax.set_ylabel("core-hours per class")
    ps.legend(ax, loc="upper left", fontsize=6)
    b = d["by_nfinal"]; ks = list(b)
    x2 = np.arange(len(ks))
    o2 = np.array([b[k]["old_core_h"] for k in ks]); n2 = np.array([b[k]["new_core_h_upper"] for k in ks])
    ax2.bar(x2 - 0.2, o2, 0.4, color=ps.C.grey)
    for i, k in enumerate(ks):
        ax2.bar(x2[i] + 0.2, n2[i], 0.4, color=ps.C.vermillion, hatch=None if b[k]["ratio_measured"] else "///",
                edgecolor="white" if b[k]["ratio_measured"] else ps.C.vermillion,
                alpha=1.0 if b[k]["ratio_measured"] else 0.5)
        ax2.text(x2[i] + 0.2, n2[i] * 1.3, f"{n2[i]:.0f}", ha="center", fontsize=6)
    ax2.set_yscale("log"); ax2.set_ylim(top=n2.max() * 60)
    ax2.set_xticks(x2); ax2.set_xticklabels([f"{k} finals\n({b[k]['classes']})" for k in ks], fontsize=6)
    ax2.set_ylabel("core-hours, e+e- classes")
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor=ps.C.grey, label="old (measured)"),
                        Patch(facecolor=ps.C.vermillion,
                              label="new, upper-side estimate: new = old $\\times$ the largest\n(budget-750 replay time / old replay time) at that final count"),
                        Patch(facecolor=ps.C.vermillion, alpha=0.5, hatch="///", edgecolor=ps.C.vermillion,
                              label="final count not measured: the largest of the four classes")], loc="upper left", fontsize=5)
    ps.save(fig, os.path.join(HERE, "figs", "sampling_cost_ee"))


if __name__ == "__main__":
    {"pull": pull, "summary": summary, "fig": fig, "ee": ee, "ee_fig": ee_fig}[sys.argv[1]](*sys.argv[2:])
