#!/usr/bin/env python3
"""Steps 11 and 14 of plan entry catalog_coverage under the design passed 2026-10-30T17:29 (plan show
catalog_coverage): the pure-default bound (step 11) and the mixture A/B (step 14), read on per-probe zero-shot MSEs.

  extract <run id> <log> [<axis>]   the axis's ZERO_SHOT records of a job log (site logs -n 3000000 <run>: a
                                    het run's log is ~506k lines) ->
                                    data/reads/<run>_<axis>.json {probe: mse_logm2_real}; a record with an error, a
                                    non-finite MSE or a physics status other than ok/allowed is listed apart
  val <run id> <log>                a training run's validation curves -> data/reads/<run>_val.json, read by
                                    the blow-up check (blowups) every result below carries; a verdict is withheld
                                    while a run it reads blew up or is unchecked
  bound                             step 11 -> data/pp_bound.json
  heldout_default                   step 14's held-out classes on default-sampled events (PP4/EE4 probes of the
                                    held-out classes) -> data/mixture_heldout_default.json
  mixture [--partial]               step 14 -> data/mixture_ab.json, only when every read of the design is in;
                                    --partial prints what the reads in hand give (a missing seed left out, no
                                    verdict, nothing written)
  mixture --het                     step 15 (design passed 2026-10-31T04:56, question 1): the same comparison on the
                                    het-loss runs (MIX_HET, control v4_het_control) -> data/mixture_ab_het.json,
                                    written with what is read: every read or probe not in hand is listed as missing
                                    (never dropped), the verdict only when nothing is missing; beside it het_over_mse,
                                    the het control pair over the MSE control pair on each read (what the het loss
                                    costs mu, information)
  heldout_default [--het]           with --het the het runs, missing reads listed -> data/mixture_heldout_default_het.json

The measure (the design's): per probe the MSE of log|M|^2 at the best checkpoint (tools/zero_shot_parent.py,
mse_logm2_real); each side the geometric mean of its two seeds (42, 43); per class (analysis/catalog_v4/
census_classes.json, a probe's class is its catalog_v4 tree's, pp13v4_X and pp13_X both v4_X) the geometric mean of
its probes' arm/control ratios; R = the median over classes; its 95% class bootstrap interval (10000 resamples, seed 0)
widened on each side by |ln R_seed|, R_seed the median over the same classes of control seed 43 over seed 42.
Step 11, on PP4 (recipes/catalog_v4_pp_test.yaml, test split), the classes the bound arm trains on pp events
(a probe of recipes/catalog_v4_pp_mixed_train.yaml in the class) apart from the others:
  YES  R <= 0.90 with the widened interval below 1;  NO  R >= 0.97 or the widened interval contains 1;
  otherwise a small gain, with its interval. The held-out classes' R is reported (expected: a cost or nothing).
Step 14, per default share (0.50, 0.25), arm over the catalog_v4 control on three reads: PP4 on all its 1301 classes
(the classes catalog_v4 trains, recipes/catalog_v4_train.yaml, 797, and the others also reported apart: the design says
"on PP4", reporting the parts apart), EE4 on its 169 probes (split the same way), and the held-out H1..H7c pooled (catalog_v4 probes only, v4_*; each axis
reported as information), with R_seed taken per read:
  GAIN     R <= 0.90 with the widened interval below 1, on PP4 and on EE4
  NO LOSS  the pooled held-out widened interval's upper end below 1.05
  LOSS     the pooled held-out R >= 1.05 with the widened interval above 1;  otherwise the cost is undecided
A share WINS on GAIN and NO LOSS. Verdict: one winner -> it; both -> 0.25 unless 0.50's PP4 and EE4 widened intervals
both lie below 0.25's R on the same read (then 0.50); no winner -> 'keep the current mixture' when no share has GAIN,
else 'trade-off' (a GAIN with a LOSS or an undecided cost: the numbers go to Joaquin with the step-11 bound beside them).
"""
import argparse, json, math, os, re, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
READS = os.path.join(DATA, "reads")
# the runs of step 11 (plan catalog_coverage): PP4 reads at the best checkpoint
BOUND = {"arm": ("FA-ccin2p3-a152d7d0", "FA-ccin2p3-a01ad665"),        # REREAD afc96598 (seed 42), d26f13aa (seed 43)
         "control": ("FA-ccin2p3-04253ff7", "FA-ccin2p3-e7e32847")}    # control 339ec100 (42), 380842ed (43) read in eval
YES, NO = 0.90, 0.97
# the runs of step 14, (seed 42, seed 43) per read; PP4 and EE4 from the eval re-reads (the arms were cut at their
# 2.5 h limit before those reads), the held-out axes from the arms' and the control's own jobs
HELD = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7E", "H7c")
MIX = {"default50": {"PP4": ("FA-ccin2p3-8e1cc529", "FA-ccin2p3-52efacaf"),   # REREAD 6c246603, 1d46b938
                     "EE4": ("FA-ccin2p3-8e1cc529", "FA-ccin2p3-52efacaf"),
                     "held": ("FA-ccin2p3-6c246603", "FA-ccin2p3-1d46b938")},
       "default25": {"PP4": ("FA-ccin2p3-7a6a1967", "FA-ccin2p3-897aa5d4"),   # REREAD 11ad944a, 9c6fc7ab
                     "EE4": ("FA-ccin2p3-7a6a1967", "FA-ccin2p3-897aa5d4"),
                     "held": ("FA-ccin2p3-11ad944a", "FA-ccin2p3-9c6fc7ab")},
       "control": {"PP4": ("FA-ccin2p3-04253ff7", "FA-ccin2p3-e7e32847"),     # REREAD 339ec100, 380842ed
                   "EE4": ("FA-ccin2p3-c1863adf", "FA-ccin2p3-def11c5c"),     # REREAD 339ec100, 380842ed
                   "held": ("FA-ccin2p3-339ec100", "FA-ccin2p3-380842ed")}}
GAIN, LOSS = 0.90, 1.05
# the runs of step 15 (design passed 2026-10-31T04:56): the het-loss twins of MIX, every read from the run's own job
# (its sigma_stage parent; mu frozen there, so mu is the best checkpoint by val_loss_no_reg). The four share arms were
# cut by their limit inside PP4 (772/1055 of 1301 for 466656b2/f0156d88, 908/1129 for 9897833d/df858d33) before EE4:
# those reads wait on their eval-only REREAD (plan step 20) and are reported missing until then
# Their PP4 and EE4 come from the step-20 eval-only REREADs (ead54b75 7c131715, 9538e607 ed96b552: 1301 and 169 probes)
MIX_HET = {"default50": {"PP4": ("FA-ccin2p3-ead54b75", "FA-ccin2p3-7c131715"),     # REREAD 466656b2, 9897833d
                         "EE4": ("FA-ccin2p3-ead54b75", "FA-ccin2p3-7c131715"),
                         "held": ("FA-ccin2p3-466656b2", "FA-ccin2p3-9897833d")},
           "default25": {"PP4": ("FA-ccin2p3-9538e607", "FA-ccin2p3-ed96b552"),     # REREAD f0156d88, df858d33
                         "EE4": ("FA-ccin2p3-9538e607", "FA-ccin2p3-ed96b552"),
                         "held": ("FA-ccin2p3-f0156d88", "FA-ccin2p3-df858d33")},
           "control": {r: ("FA-ccin2p3-87ea24d0", "FA-ccin2p3-321a3de2") for r in ("PP4", "EE4", "held")}}
# the default-sampled held-out pools beyond H1-H3's PP4/EE4 probes (plan steps 18-20; D227: the held-out check is on
# events generated the default way): HD4, the 114 tree classes of H2p5 38, H6 8, H7E 34, H7c 34
# (recipes/catalog_v4_heldout_default_test.yaml), and HD5, the 28 one-loop classes of H5 (MadEvent Born events labelled
# by the class's MadLoop, recipes/catalog_v4_heldout_default_H5_test.yaml); the step-20 eval-only REREADs, (seed 42,
# seed 43) per arm. The MSE control's HD5 is its HD5-only re-read after the H5 physics-check fix 311380bf (its first
# step-20 read refused all 28)
HD4_HET = {"default50": ("FA-ccin2p3-ead54b75", "FA-ccin2p3-7c131715"),
           "default25": ("FA-ccin2p3-9538e607", "FA-ccin2p3-ed96b552"),
           "control": ("FA-ccin2p3-87aeabca", "FA-ccin2p3-53b9ed56")}
HD5_HET = dict(HD4_HET)
HD4_MSE = {"default50": ("FA-ccin2p3-58631ca5", "FA-ccin2p3-698c1a0b"),
           "default25": ("FA-ccin2p3-4dca09e5", "FA-ccin2p3-926dea24"),
           "control": ("FA-ccin2p3-9512cac2", "FA-ccin2p3-783892e6")}
HD5_MSE = dict(HD4_MSE, control=("FA-ccin2p3-c37a4b26", "FA-ccin2p3-a1ea15aa"))
DEFAULT_EXTRA = {False: {"HD4": HD4_MSE, "HD5": HD5_MSE}, True: {"HD4": HD4_HET, "HD5": HD5_HET}}
# The blow-up check (CLAUDE.md: a run that blows up is listed apart with where it blew up, never dropped). Every read
# above is made at its training run's best checkpoint; an eval-only REREAD reads its parent's, so the check is on the
# training run (TRAINED_BY; a run not listed trained itself). The validation curve (val_loss_no_reg of each "Val loss"
# line, step = every * k, the last at the last step) is extracted from the training log by `val` into
# data/reads/<run>_val.json. A training block blows up at its first validation that is non-finite (anywhere), or after
# the reference step REF_STEP above BLOWUP_FACTOR times the value at the reference (the first validation at or after
# REF_STEP; a block shorter than REF_STEP, the het runs' 250-step sigma stage, takes its first validation). Validations
# before the reference are the warm-up transient (the het runs' second validation, step 170, is ~1e9 in all six) and
# are reported as early_max, not judged. The het loss (loss HETEROSC: a Gaussian NLL, negative once fitted) has no
# ratio; there the same factor is read on the variance scale: at the sigma that minimises it the NLL is
# 0.5 ln(MSE) + const, so an MSE BLOWUP_FACTOR times larger raises it by 0.5 ln(BLOWUP_FACTOR) = 1.151; the het rule
# is an NLL above the reference + 0.5 ln(BLOWUP_FACTOR). Such a validation is a spike; the run blew up only when its
# best checkpoint (the read's) is before it, as analysis/catalog_v3/muon_compare.py; a recovered spike is listed.
BLOWUP_FACTOR = 10      # the repo's blow-up factor (analysis/catalog_v3/pair_head_blowup.py BLOWUP_FACTOR)
REF_STEP = 500          # the passed step-15 design's outcome rule (val_loss_no_reg NaN or above 10x its step-500 value)
HET_RISE = 0.5 * math.log(BLOWUP_FACTOR)
TRAINED_BY = {"FA-ccin2p3-a152d7d0": "FA-ccin2p3-afc96598", "FA-ccin2p3-a01ad665": "FA-ccin2p3-d26f13aa",
              "FA-ccin2p3-8e1cc529": "FA-ccin2p3-6c246603", "FA-ccin2p3-52efacaf": "FA-ccin2p3-1d46b938",
              "FA-ccin2p3-7a6a1967": "FA-ccin2p3-11ad944a", "FA-ccin2p3-897aa5d4": "FA-ccin2p3-9c6fc7ab",
              "FA-ccin2p3-04253ff7": "FA-ccin2p3-339ec100", "FA-ccin2p3-e7e32847": "FA-ccin2p3-380842ed",
              "FA-ccin2p3-c1863adf": "FA-ccin2p3-339ec100", "FA-ccin2p3-def11c5c": "FA-ccin2p3-380842ed",
              # the step-20 REREADs
              "FA-ccin2p3-9512cac2": "FA-ccin2p3-339ec100", "FA-ccin2p3-783892e6": "FA-ccin2p3-380842ed",
              "FA-ccin2p3-c37a4b26": "FA-ccin2p3-339ec100", "FA-ccin2p3-a1ea15aa": "FA-ccin2p3-380842ed",
              "FA-ccin2p3-58631ca5": "FA-ccin2p3-6c246603", "FA-ccin2p3-698c1a0b": "FA-ccin2p3-1d46b938",
              "FA-ccin2p3-4dca09e5": "FA-ccin2p3-11ad944a", "FA-ccin2p3-926dea24": "FA-ccin2p3-9c6fc7ab",
              "FA-ccin2p3-87aeabca": "FA-ccin2p3-87ea24d0", "FA-ccin2p3-53b9ed56": "FA-ccin2p3-321a3de2",
              "FA-ccin2p3-ead54b75": "FA-ccin2p3-466656b2", "FA-ccin2p3-7c131715": "FA-ccin2p3-9897833d",
              "FA-ccin2p3-9538e607": "FA-ccin2p3-f0156d88", "FA-ccin2p3-ed96b552": "FA-ccin2p3-df858d33"}


def val(run, log):
    """The training run's validation curves from its log (one block per 'Starting to train', the het runs' sigma
    stage the second) -> data/reads/<run>_val.json"""
    blocks, loss = [], None
    for line in open(log, errors="replace"):
        if line.startswith("RUN_SUMMARY ") and loss is None:
            m = re.search(r'"loss": "(\w+)"', line)
            loss = m.group(1) if m else None
        elif "Starting to train for" in line:
            it = int(re.search(r"Starting to train for (\d+) iterations", line).group(1))
            ev = int(re.search(r"validating every (\d+) steps", line).group(1))
            blocks.append({"iterations": it, "every": ev, "vals": []})
        elif "Val loss (combined)" in line and blocks:
            b = blocks[-1]
            x = float(re.search(r"val_loss_no_reg: ([0-9.eE+-]+|nan|inf|-inf)", line).group(1))
            b["vals"].append([min(b["every"] * (len(b["vals"]) + 1), b["iterations"]), x])
    if not blocks or not blocks[0]["vals"]:
        sys.exit(f"val: no validation in {log}")
    os.makedirs(READS, exist_ok=True)
    out = os.path.join(READS, f"{run}_val.json")
    json.dump({"run": run, "loss": loss, "log": os.path.basename(log), "blocks": blocks}, open(out, "w"), indent=0)
    print(json.dumps({"out": out, "loss": loss, "blocks": [len(b["vals"]) for b in blocks]}))


def blowup_block(vals, het):
    """{ref_step, ref, early_max, blew_up_at, at_value} of one block's [[step, val_loss_no_reg], ...]"""
    bad = next((s for s, x in vals if not math.isfinite(x)), None)
    i = next((k for k, (s, _) in enumerate(vals) if s >= REF_STEP), 0)
    if vals[-1][0] < REF_STEP:
        i = 0
    ref_step, ref = vals[i]
    early = [x for _, x in vals[:i]]
    over = (lambda x: x > ref + HET_RISE) if het else (lambda x: x > BLOWUP_FACTOR * ref)
    hit = next(((s, x) for s, x in vals[i + 1:] if math.isfinite(x) and over(x)), None)
    at = min(z for z in (bad, hit and hit[0]) if z is not None) if (bad is not None or hit) else None
    fin = [(s, x) for s, x in vals if math.isfinite(x)]
    best = min(fin, key=lambda z: z[1])[0] if fin else None
    # as analysis/catalog_v3/muon_compare.py: a spike after which the block reached its best checkpoint recovered
    # (the read, at that best, is after it; listed as spike_at); it blew up when its best is before the spike
    blew = at if at is not None and (best is None or best < at) else None
    return {"ref_step": ref_step, "ref": ref, "early_max": max(early) if early else None, "best_step": best,
            "spike_at": at, "blew_up_at": blew,
            "at_value": None if at is None else next(x for s, x in vals if s == at),
            "max_after_ref": max((x for _, x in vals[i:] if math.isfinite(x)), default=None)}


def blowup_of(run):
    """The blow-up check of the training run behind a read run; 'unchecked' when its curve is not extracted"""
    t = TRAINED_BY.get(run, run)
    f = os.path.join(READS, f"{t}_val.json")
    if not os.path.exists(f):
        return {"train_run": t, "unchecked": f"no {os.path.relpath(f, ROOT)} (mixture_ab.py val {t} <log>)"}
    d = json.load(open(f))
    het = d["loss"] == "HETEROSC"
    blocks = [blowup_block(b["vals"], het) for b in d["blocks"]]
    first = next((k for k, b in enumerate(blocks) if b["blew_up_at"] is not None), None)
    return {"train_run": t, "loss": d["loss"], "blocks": blocks,
            "spiked_recovered": [{"block": k, "step": b["spike_at"]} for k, b in enumerate(blocks)
                                 if b["spike_at"] is not None and b["blew_up_at"] is None],
            "blew_up": None if first is None else {"block": first, "step": blocks[first]["blew_up_at"]}}


def blowups(runs):
    """{run: check} over the runs a result reads, and the runs that blew up or are unchecked (a verdict is withheld
    while that list is not empty; the run's reads stay in the result, listed apart here)"""
    chk = {r: blowup_of(r) for r in sorted(set(runs))}
    flagged = sorted(f"{r}: " + (c["unchecked"] if "unchecked" in c else
                                 f"training run {c['train_run']} blew up at step {c['blew_up']['step']} "
                                 f"(block {c['blew_up']['block']})")
                     for r, c in chk.items() if "unchecked" in c or c["blew_up"])
    return {"rule": f"non-finite val_loss_no_reg, or after step {REF_STEP} (a block shorter than that: after its "
                    f"first validation) above {BLOWUP_FACTOR}x the reference validation; het NLL: above the "
                    f"reference + 0.5 ln {BLOWUP_FACTOR} = {HET_RISE:.3f}; blown up when the block's best checkpoint is "
                    f"before that validation, spiked and recovered (listed, not flagged) when it is after",
            "runs": chk, "flagged": flagged,
            "spiked_recovered": sorted(f"{r}: training run {c['train_run']} " + ", ".join(
                f"block {z['block']} step {z['step']}" for z in c["spiked_recovered"])
                for r, c in chk.items() if c.get("spiked_recovered"))}


def extract(run, log, axis="PP4", allow_unchecked=False):
    """allow_unchecked: a read made before the in-job physics check (the control's own held-out reads, 339ec100 and
    380842ed) is kept with status 'unchecked'; used only where the arms read the same pools with the check ok (same
    recipes/catalog_v4_heldout_*.yaml, n_val equal on all 701 probes, var_logm2 within the arm-to-arm spread)."""
    tag = f"{axis} ZERO_SHOT "
    good, bad = {}, {}
    for line in open(log, errors="replace"):
        if not line.startswith(tag):
            continue
        d = json.loads(line[len(tag):])
        p = d.get("probe") or "?"
        x = d.get("mse_logm2_real")
        st = (d.get("physics") or {}).get("status", "unchecked")
        if "error" in d:
            bad[p] = "error: " + str(d["error"])[:200]
        elif x is None or not math.isfinite(x):
            bad[p] = f"non-finite mse {x}"
        elif st not in ("ok", "allowed") and not (allow_unchecked and st == "unchecked"):
            bad[p] = f"physics {st}"
        else:
            good[p] = float(x)
    os.makedirs(READS, exist_ok=True)
    out = os.path.join(READS, f"{run}_{axis}.json")
    json.dump({"run": run, "axis": axis, "log_lines_from": os.path.basename(log), "mse": good, "excluded": bad},
              open(out, "w"), indent=0, sort_keys=True)
    print(json.dumps({"out": out, "n": len(good), "excluded": len(bad)}))


def load(run, axis="PP4"):
    return json.load(open(os.path.join(READS, f"{run}_{axis}.json")))["mse"]


def class_of():
    c = json.load(open(os.path.join(HERE, "census_classes.json")))["classes"]
    m = {}
    for rep, members in c.items():
        for x in members:
            m[x] = rep
    return lambda probe: m.get("v4_" + probe.split("_", 1)[1], "v4_" + probe.split("_", 1)[1])


def gmean(xs):
    return float(np.exp(np.mean(np.log(xs))))


def class_ratios(num, den, probes, cls):
    by = {}
    for p in probes:
        by.setdefault(cls(p), []).append(num[p] / den[p])
    return {k: gmean(v) for k, v in by.items()}


def r_with_interval(ratios, r_seed, n_boot=10000, seed=0):
    v = np.array(sorted(ratios.values()))
    r = float(np.median(v))
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    w = abs(math.log(r_seed))
    return {"R": round(r, 4), "n_classes": len(v), "boot95": [round(float(lo), 4), round(float(hi), 4)],
            "widened": [round(float(lo) * math.exp(-w), 4), round(float(hi) * math.exp(w), 4)],
            "frac_classes_below_1": round(float(np.mean(v < 1)), 4)}


def bound():
    a42, a43 = (load(r) for r in BOUND["arm"])
    c42, c43 = (load(r) for r in BOUND["control"])
    probes = sorted(set(a42) & set(a43) & set(c42) & set(c43))
    arm = {p: math.sqrt(a42[p] * a43[p]) for p in probes}
    ctl = {p: math.sqrt(c42[p] * c43[p]) for p in probes}
    train = {p["name"] for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_pp_mixed_train.yaml")))["processes"]}
    cls = class_of()
    trained_cls = {cls(p) for p in probes if p in train}
    out = {"design": "plan catalog_coverage, passed 2026-10-30T17:29", "runs": BOUND, "n_probes": len(probes),
           "probes_missing": sorted((set(a42) | set(a43) | set(c42) | set(c43)) - set(probes))[:50]}
    for part, sel in (("trained", lambda p: cls(p) in trained_cls), ("held_out", lambda p: cls(p) not in trained_cls)):
        ps = [p for p in probes if sel(p)]
        seed_r = class_ratios(c43, c42, ps, cls)
        r_seed = float(np.median(list(seed_r.values())))
        res = r_with_interval(class_ratios(arm, ctl, ps, cls), r_seed)
        res.update(n_probes=len(ps), R_seed=round(r_seed, 4),
                   arm_seed_R=round(float(np.median(list(class_ratios(a43, a42, ps, cls).values()))), 4))
        out[part] = res
    t = out["trained"]
    lo, hi = t["widened"]
    if t["R"] <= YES and hi < 1:
        v = "YES"
    elif t["R"] >= NO or lo <= 1 <= hi:
        v = "NO"
    else:
        v = "small gain"
    out["blowups"] = blowups(BOUND["arm"] + BOUND["control"])
    out["verdict"] = None if out["blowups"]["flagged"] else v
    out["verdict_on_reads"] = v                  # the rule's outcome; the verdict only with no run flagged
    out["rule"] = f"YES R <= {YES} with the widened interval below 1; NO R >= {NO} or the interval contains 1"
    path = os.path.join(DATA, "pp_bound.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out))


def read_of(run, read):
    """{probe: mse} of one run on one read (the held-out axes pooled, catalog_v4 probes only: a control job of the
    shared-folder era also read catalog_v3's v3_* probes there), or None when a file is missing."""
    axes = HELD if read == "held" else (read,)
    out = {}
    for ax in axes:
        f = os.path.join(READS, f"{run}_{ax}.json")
        if not os.path.exists(f):
            return None
        out.update({(ax, p) if read == "held" else p: x for p, x in json.load(open(f))["mse"].items()
                    if read != "held" or p.startswith("v4_")})
    return out


def probe_name(k):
    return k[1] if isinstance(k, tuple) else k


def side(pair):
    """Geometric mean over the seeds read; the seeds present."""
    have = [s for s in pair if s is not None]
    keys = set.intersection(*(set(s) for s in have)) if have else set()
    return {k: gmean([s[k] for s in have]) for k in keys}


def compare(arm, ctl, c42, c43, keys, cls):
    cl = lambda k: cls(probe_name(k))
    seed_r = float(np.median(list(class_ratios(c43, c42, keys, cl).values())))
    res = r_with_interval(class_ratios(arm, ctl, keys, cl), seed_r)
    res.update(n_probes=len(keys), R_seed=round(seed_r, 4))
    return res


def _reads(table):
    """{arm: {read: (seed 42 reads, seed 43 reads)}} and the (arm, read, run) missing, as strings"""
    reads = {a: {r: tuple(read_of(run, r) for run in runs) for r, runs in v.items()} for a, v in table.items()}
    missing = sorted(f"{a} {r} {table[a][r][i]}" for a, v in reads.items() for r, pair in v.items()
                     for i, s in enumerate(pair) if s is None)
    return reads, missing


def _probes_missing(pair, ctl_pair):
    """Per seed of an arm, the control's probes that seed has no read of (a read cut by the job's limit): a count and
    the first names, so a partial read is reported as such, never as complete"""
    ref = set.union(*(set(x) for x in ctl_pair if x is not None)) if any(x is not None for x in ctl_pair) else set()
    out = {}
    for i, x in enumerate(pair):
        m = sorted((str(k) for k in ref - set(x or {})))
        if m:
            out[("seed42", "seed43")[i]] = {"n": len(m), "of": len(ref), "first": m[:10]}
    return out


def mixture(partial=False, het=False):
    cls = class_of()
    trained = {cls(p["name"]) for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")))["processes"]}
    table = MIX_HET if het else MIX
    reads, missing = _reads(table)
    if missing and not (partial or het):
        sys.exit("mixture: reads missing (extract them first; --partial for what is in hand): " + "; ".join(missing))
    c = reads["control"]
    if any(s is None for pair in c.values() for s in pair):
        sys.exit("mixture: a control read is missing: " + "; ".join(missing))
    design = "plan catalog_coverage step 15, passed 2026-10-31T04:56 (het loss)" if het else \
        "plan catalog_coverage, passed 2026-10-30T17:29"
    out = {"design": design, "runs": table, "missing_reads": missing, "probes_missing": {},
           "rule": f"GAIN R <= {GAIN} with the widened interval below 1 on PP4 (all classes) and EE4 (all); NO LOSS "
                   f"the pooled held-out widened upper end below {LOSS}; LOSS pooled R >= {LOSS} with the widened "
                   f"interval above 1; WIN = GAIN and NO LOSS; the held-out cost on default-sampled events (D227; "
                   f"held_default: the PP4/EE4 probes of the held-out classes, HD4 and HD5), the catalog-sampled "
                   f"read (held) beside",
           "shares": {}}
    for a in ("default50", "default25"):
        res = {}
        for r in ("PP4", "EE4", "held"):
            pm = _probes_missing(reads[a][r], c[r])
            if pm:
                out["probes_missing"][f"{a} {r}"] = pm
            arm = side(reads[a][r])
            ctl = side(c[r])
            keys = sorted(set(arm) & set(ctl), key=str)
            parts = {"all": keys,
                     "trained": [k for k in keys if cls(probe_name(k)) in trained],
                     "not_trained": [k for k in keys if cls(probe_name(k)) not in trained]}
            if r == "held":
                parts = {"pooled": keys, **{ax: [k for k in keys if k[0] == ax] for ax in HELD}}
            res[r] = {n: compare(arm, ctl, c[r][0], c[r][1], ks, cls) for n, ks in parts.items() if ks}
            res[r]["seeds_read"] = [run for run, s in zip(table[a][r], reads[a][r]) if s is not None]
            res[r]["probes_unmatched"] = sorted(str(k) for k in set(arm) ^ set(ctl))[:50]
            res[r]["n_probes_unmatched"] = len(set(arm) ^ set(ctl))
        if not all(k in res[r] for r, k in (("PP4", "all"), ("EE4", "all"), ("held", "pooled"))):
            res.update(gain=None, cost=None, wins=None)      # a read with no seed in hand
            out["shares"][a] = res
            continue
        pp, ee, h = res["PP4"]["all"], res["EE4"]["all"], res["held"]["pooled"]
        # D227 (Joaquin 2026-10-31: the check is against data generated the default way) and steps 17-20: the cost
        # is read on the default-sampled held-out events (H1-H3's PP4/EE4 probes, HD4, HD5: 700 classes), MSE and
        # het alike; the catalog-sampled held-out read (res["held"]) beside it as information
        hd, _ = _held_default(reads, a, cls, _held_axes(cls), DEFAULT_EXTRA[het])
        res["held_default"] = hd
        h = hd.get("pooled")
        if h is None or hd["default_missing"]:
            res.update(gain=None, cost=None, wins=None)
            out["shares"][a] = res
            continue
        gain = all(x["R"] <= GAIN and x["widened"][1] < 1 for x in (pp, ee))
        if h["widened"][1] < LOSS:
            cost = "NO LOSS"
        elif h["R"] >= LOSS and h["widened"][0] > 1:
            cost = "LOSS"
        else:
            cost = "undecided"
        res.update(gain=gain, cost=cost, wins=gain and cost == "NO LOSS")
        out["shares"][a] = res
    s50, s25 = out["shares"]["default50"], out["shares"]["default25"]
    if s50["wins"] is None or s25["wins"] is None:
        v = None
    elif s50["wins"] and s25["wins"]:
        below = all(s50[r][n]["widened"][1] < s25[r][n]["R"] for r, n in (("PP4", "all"), ("EE4", "all")))
        v = "default50" if below else "default25"
    elif s50["wins"] or s25["wins"]:
        v = "default50" if s50["wins"] else "default25"
    elif not (s50["gain"] or s25["gain"]):
        v = "keep the current mixture"
    else:
        v = "trade-off"
    out["blowups"] = blowups([run for v in table.values() for runs in v.values() for run in runs])
    complete = not missing and not out["probes_missing"] and not out["blowups"]["flagged"]
    # the default-sampled held-out pool is complete only with HD4 and HD5 read on every side
    ext = DEFAULT_EXTRA[het]
    hd_missing = [f"{a} {ax}" for ax, tab in ext.items() for a in table if not tab.get(a)] + \
        [f"{a} {ax} {run}" for ax, tab in ext.items() for a, runs in tab.items() for run in runs
         if read_of(run, ax) is None]
    out["held_default_missing"] = hd_missing
    out["blowups"] = blowups([run for v in table.values() for runs in v.values() for run in runs] +
                             [run for tab in ext.values() for runs in tab.values() for run in runs])
    complete = complete and not hd_missing and not out["blowups"]["flagged"]
    if partial and not het:
        out["verdict_if_complete"] = v if not missing else None
        out["would_be"] = v
        print(json.dumps(out, indent=1))
        return
    out["complete"] = complete
    out["verdict"] = v if complete else None
    if het:
        out["verdict_on_reads_in_hand"] = v                 # information only while reads are missing
        out["het_over_mse"] = het_over_mse(cls, trained)
        out["het_over_mse_note"] = "het control over MSE control: loss HETEROSC + sigma stage AND sign_head off together"
        path = os.path.join(DATA, "mixture_ab_het.json")
    else:
        out["bound"] = json.load(open(os.path.join(DATA, "pp_bound.json")))["trained"]
        path = os.path.join(DATA, "mixture_ab.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out))   # the whole result, the last JSON line (tools/verify_claims.py reads its claims from it)


def het_over_mse(cls, trained):
    """What the het setup costs or gains mu (information, design 04:56): the het control pair (MIX_HET) over the MSE
    control pair (MIX) on each read. Two things differ between them, the loss (HETEROSC plus its sigma stage) and
    sign_head (off in the het runs, on in the MSE runs): the ratio is the two together, not the loss alone, R as in mixture with R_seed from the MSE control's seeds"""
    h, _ = _reads({"control": MIX_HET["control"]})
    m, _ = _reads({"control": MIX["control"]})
    out = {}
    for r in ("PP4", "EE4", "held"):
        hp, mp = h["control"][r], m["control"][r]
        if any(x is None for x in hp + mp):
            out[r] = None
            continue
        arm, ctl = side(hp), side(mp)
        keys = sorted(set(arm) & set(ctl), key=str)
        parts = {"all": keys,
                 "trained": [k for k in keys if cls(probe_name(k)) in trained],
                 "not_trained": [k for k in keys if cls(probe_name(k)) not in trained]}
        if r == "held":
            parts = {"pooled": keys, **{ax: [k for k in keys if k[0] == ax] for ax in HELD}}
        out[r] = {n: compare(arm, ctl, mp[0], mp[1], ks, cls) for n, ks in parts.items() if ks}
        out[r]["het_seed_R"] = compare(hp[1], hp[0], mp[0], mp[1], keys, cls)["R"]   # het control seed 43 over 42
        out[r]["n_probes_unmatched"] = len(set(arm) ^ set(ctl))
    return out


def _held_axes(cls):
    """catalog_v4 class -> its held-out axis (recipes/catalog_v4_heldout_<axis>.yaml)"""
    ax_of = {}
    for ax in HELD:
        for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v4_heldout_{ax}.yaml")))["processes"]:
            if p["name"].startswith("v4_"):
                ax_of[cls(p["name"])] = ax
    return ax_of


def _held_default(reads, a, cls, ax_of, extra):
    """R of arm a over the control on the default-sampled events of the held-out classes: the PP4/EE4 probes whose
    class is held out, plus the default-sampled held-out pools of `extra` ({"HD4": {arm: (run 42, run 43)}, "HD5":
    ...}, DEFAULT_EXTRA) where all four seeds are read; per axis and pooled, with how many probes the control has and
    how many were read on every side"""
    tag = lambda r, d: {(r, k): x for k, x in (d or {}).items()}
    arm, ctl, c42, c43 = {}, {}, {}, {}
    srcs = [(r, reads[a][r], reads["control"][r]) for r in ("PP4", "EE4")]
    hd4_missing = []
    for ax, tab in extra.items():
        miss = [f"{x} {ax}" for x in (a, "control") if not tab.get(x)]
        if not miss:
            ap = tuple(read_of(x, ax) for x in tab[a])
            cp = tuple(read_of(x, ax) for x in tab["control"])
            miss = [f"{x} {ax} {run}" for x, runs, rd in ((a, tab[a], ap), ("control", tab["control"], cp))
                    for run, y in zip(runs, rd) if y is None]
            if not miss:                      # only with all four seeds read, never on one seed
                srcs.append((ax, ap, cp))
        hd4_missing += miss
    used = []
    for r, ap, cp in srcs:
        if any(x is None for x in cp):
            continue
        used.append(r)
        arm.update(tag(r, side(ap))); ctl.update(tag(r, side(cp)))
        c42.update(tag(r, cp[0])); c43.update(tag(r, cp[1]))
    held_ctl = [k for k in ctl if cls(k[1]) in ax_of]
    keys = sorted((k for k in set(arm) & set(ctl) if cls(k[1]) in ax_of), key=str)
    res = {"sources": used, "default_missing": hd4_missing, "n_probes_control": len(held_ctl), "n_probes_read": len(keys),
           "probes_missing": sorted(str(k) for k in set(held_ctl) - set(keys))[:20]}
    if keys:
        res["pooled"] = compare(arm, ctl, c42, c43, keys, cls)
    for ax in HELD:
        ks = [k for k in keys if ax_of[cls(k[1])] == ax]
        if ks:
            res[ax] = compare(arm, ctl, c42, c43, ks, cls)
    return res, keys


def heldout_default(het=False):
    """The held-out classes read on default-sampled events (Joaquin, 2026-10-31: the check is against data generated
    the default way): the PP4 and EE4 probes (MadEvent default, test split) whose class is in a held-out axis
    (recipes/catalog_v4_heldout_<axis>.yaml), per axis and pooled, R as in mixture (arm over control, R_seed of the
    same probes), beside the catalog-sampled held-out read on the same classes and on the classes with no
    default-sampled pool. -> data/mixture_heldout_default.json. With het, the het runs (MIX_HET, HD4 from HD4_HET
    once read): a missing read or probe is listed (missing_reads, probes_missing) and the classes read are the ones
    every side has -> data/mixture_heldout_default_het.json"""
    cls = class_of()
    ax_of = _held_axes(cls)
    table = MIX_HET if het else MIX
    reads, missing = _reads(table)
    # every seed of every read this uses, as mixture() without --partial: a missing seed would leave R on one seed
    if missing and not het:
        sys.exit("heldout_default: reads missing (extract them first): " + "; ".join(missing))
    if any(x is None for pair in reads["control"].values() for x in pair):
        sys.exit("heldout_default: a control read is missing: " + "; ".join(missing))
    out = {"design": "plan catalog_coverage step 15 (het), passed 2026-10-31T04:56" if het else
           "plan catalog_coverage step 14, REREAD on default-sampled held-out events",
           "runs": table, "missing_reads": missing, "shares": {}}
    for a in ("default50", "default25"):
        d, keys = _held_default(reads, a, cls, ax_of, DEFAULT_EXTRA[het])
        res = {"default": d}
        have = {cls(k[1]) for k in keys}
        harm, hctl = side(reads[a]["held"]), side(reads["control"]["held"])
        hk = sorted(set(harm) & set(hctl), key=str)
        hc = reads["control"]["held"]
        same = [k for k in hk if cls(k[1]) in have]
        # classes with a default-sampled pool: the control's PP4/EE4 held-out probes (its reads are complete), plus
        # those read (HD4); a pooled class an arm's cut read missed is in neither group (listed in probes_missing)
        pooled = have | {cls(p) for r in ("PP4", "EE4") for x in reads["control"][r] for p in x if cls(p) in ax_of}
        res["catalog_sampled"] = {
            "same_classes": compare(harm, hctl, hc[0], hc[1], same, cls) if same else None,
            "no_default_pool": compare(harm, hctl, hc[0], hc[1], [k for k in hk if cls(k[1]) not in pooled], cls)}
        res["axes_without_default_pool"] = sorted({ax for c, ax in ax_of.items() if c not in pooled})
        out["shares"][a] = res
    out["blowups"] = blowups([run for v in table.values() for runs in v.values() for run in runs] +
                             [run for tab in DEFAULT_EXTRA[het].values() for runs in tab.values() for run in runs])
    name = "mixture_heldout_default_het.json" if het else "mixture_heldout_default.json"
    json.dump(out, open(os.path.join(DATA, name), "w"), indent=1)
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("run"); e.add_argument("log"); e.add_argument("axis", nargs="?", default="PP4")
    e.add_argument("--allow-unchecked", action="store_true")
    v = sub.add_parser("val")
    v.add_argument("run"); v.add_argument("log")
    sub.add_parser("bound")
    hd = sub.add_parser("heldout_default")
    hd.add_argument("--het", action="store_true")
    m = sub.add_parser("mixture")
    m.add_argument("--partial", action="store_true")
    m.add_argument("--het", action="store_true")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.run, a.log, a.axis, a.allow_unchecked)
    elif a.cmd == "val":
        val(a.run, a.log)
    elif a.cmd == "mixture":
        mixture(a.partial, a.het)
    elif a.cmd == "heldout_default":
        heldout_default(a.het)
    else:
        bound()


if __name__ == "__main__":
    main()
