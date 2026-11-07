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
  bound_default                     the pure-default bound arm (step 11) on the same default-sampled reads as the
                                    shares (Joaquin 2026-10-31 21:3x): PP4 and EE4 (all, trained, not trained) and
                                    the 700 held-out classes (H1-H3's PP4/EE4 probes, HD4, HD5), pooled and per axis,
                                    against the same MSE control, the two shares' rows beside -> data/bound_default.json
  heldout_default [--het]           with --het the het runs, missing reads listed -> data/mixture_heldout_default_het.json
  width                             step 33 (the 1x width cells, D248): widths 16 and 32 over width 8 (num_heads, 4264
                                    steps, het loss, share 0.25), each run's best checkpoint (lowest aggregate
                                    val_loss_no_reg of the main block, each process at that step) and R per read as in
                                    mixture (seeds' geometric mean, class median, interval widened by width 8's seed
                                    pair), plus R per seed (seed 42 over seed 42, 43 over 43) -> data/width_x1.json
  arms                              the zero-shot reads per arm against the het control (share 0, 1e4 train
                                    events per process, 4264 steps): the share-0.25 het arm of step 15 and the 1x
                                    width cells (share 0.25, whole training set, widths 8, 16, 32), R as in mixture
                                    per read (PP4, EE4, held out default-sampled pooled and per axis, held out
                                    catalog-sampled pooled) -> data/arms_vs_het_control.json; beside R, under
                                    "seeds", each seed of every arm and of the control apart on the same probes: its
                                    absolute MSE (median over classes of the class's geometric mean over its probes of
                                    mse_logm2_real) and its R (median over classes of the class's geometric mean of
                                    seed / control, the control the geometric mean of its two seeds; the control's own
                                    seeds read the same way), and each arm's epochs (data/epochs.json)
  epochs <run id> <log>             the first 'Starting to train for N iterations ... on a dataset with B batches'
                                    line of a training log (the main block) -> data/epochs.json {run: iterations,
                                    batches, epochs = iterations / batches}; B is the training events over the batch
                                    size (16384), floored, so epochs = steps x batch / training events

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
(the classes catalog_v4 trains, recipes/catalog_v4_base_train.yaml, 797, and the others also reported apart: the design says
"on PP4", reporting the parts apart), EE4 on its 169 probes (split the same way), and the held-out H1..H7c pooled (catalog_v4 probes only, v4_*; each axis
reported as information), with R_seed taken per read:
  GAIN     R <= 0.90 with the widened interval below 1, on PP4 and on EE4
  NO LOSS  the pooled held-out widened interval's upper end below 1.05
  LOSS     the pooled held-out R >= 1.05 with the widened interval above 1;  otherwise the cost is undecided
A share WINS on GAIN and NO LOSS. Verdict: one winner -> it; both -> 0.25 unless 0.50's PP4 and EE4 widened intervals
both lie below 0.25's R on the same read (then 0.50); no winner -> 'keep the current mixture' when no share has GAIN,
else 'trade-off' (a GAIN with a LOSS or an undecided cost: the numbers go to Joaquin with the step-11 bound beside them).
  warm [<change>]                   plan warm_start (superset design passed 2026-11-04T21:51): the warm starts W_k
                                    and their matched controls C_k against the change from scratch B and the parent
                                    A, per k in 533/1066/2132, MATCH/SHORT/INCONCLUSIVE and g_k (warm()) ->
                                    data/warm_start_<change>.json
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
# the bound arm's EE4, HD4 and HD5: eval-only REREADs of afc96598 (seed 42) and d26f13aa (seed 43), plan step 23
BOUND_DEFAULT = {"PP4": BOUND["arm"], "EE4": ("FA-ccin2p3-f8ec724f", "FA-ccin2p3-049b863e"),
                 "HD": ("FA-ccin2p3-f8ec724f", "FA-ccin2p3-049b863e")}
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
              "FA-ccin2p3-f8ec724f": "FA-ccin2p3-afc96598", "FA-ccin2p3-049b863e": "FA-ccin2p3-d26f13aa",
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
    trained = {cls(p["name"]) for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_base_train.yaml")))["processes"]}
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


def bound_default():
    """The bound arm beside the shares, every read on default-sampled events, the same control pair and measure as
    mixture (MSE runs); the rule's verdict is not applied (the bound is a reference, not a candidate share)"""
    cls = class_of()
    trained = {cls(p["name"]) for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_base_train.yaml")))["processes"]}
    table = {"bound": {"PP4": BOUND_DEFAULT["PP4"], "EE4": BOUND_DEFAULT["EE4"]},
             **{a: {r: MIX[a][r] for r in ("PP4", "EE4")} for a in ("default50", "default25", "control")}}
    reads, missing = _reads(table)
    extra = {ax: dict(tab, bound=BOUND_DEFAULT["HD"]) for ax, tab in DEFAULT_EXTRA[False].items()}
    if missing:
        sys.exit("bound_default: reads missing: " + "; ".join(missing))
    c = reads["control"]
    out = {"design": "plan catalog_coverage step 11 (passed 2026-10-30T17:29), eval-only REREADs (step 23)",
           "runs": {**table, "HD4": {a: extra["HD4"][a] for a in ("bound", "default50", "default25", "control")},
                    "HD5": {a: extra["HD5"][a] for a in ("bound", "default50", "default25", "control")}},
           "arms": {}}
    ax_of = _held_axes(cls)
    for a in ("bound", "default50", "default25"):
        res = {}
        for r in ("PP4", "EE4"):
            arm, ctl = side(reads[a][r]), side(c[r])
            keys = sorted(set(arm) & set(ctl), key=str)
            parts = {"all": keys, "trained": [k for k in keys if cls(probe_name(k)) in trained],
                     "not_trained": [k for k in keys if cls(probe_name(k)) not in trained]}
            res[r] = {n: compare(arm, ctl, c[r][0], c[r][1], ks, cls) for n, ks in parts.items() if ks}
            res[r]["n_probes_unmatched"] = len(set(arm) ^ set(ctl))
        res["held_default"], hkeys = _held_default(reads, a, cls, ax_of, extra)
        # whether the table's 'PP4 not trained' and 'held out, default-sampled' rows read the same probes: the PP4
        # classes catalog_v4 does not train that are also in the held-out pool (through H1-H3's PP4 probes)
        nt = {cls(probe_name(k)) for k in set(side(reads[a]["PP4"])) & set(side(c["PP4"]))} - trained
        held = {cls(k[1]) for k in hkeys}
        res["overlap_PP4_not_trained_held_default"] = {
            "n_classes_PP4_not_trained": len(nt), "n_classes_held_default": len(held),
            "n_classes_shared": len(nt & held),
            "n_classes_held_default_via_PP4": len({cls(k[1]) for k in hkeys if k[0] == "PP4"})}
        out["arms"][a] = res
    runs = [run for v in table.values() for pair in v.values() for run in pair] + \
        [run for tab in extra.values() for pair in tab.values() for run in pair]
    out["blowups"] = blowups(runs)
    json.dump(out, open(os.path.join(DATA, "bound_default.json"), "w"), indent=1)
    print(json.dumps(out))


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


# step 33: the 1x width cells (scripts/job_catalog_v3_short.sh, LONG_FACTOR=1 LONG_MEMMAP=true, 4264 steps), every read
# from the run's own job; width 8 at 9439612d, widths 16/32 at deb290bc (only the cache lock differs) with
# LONG_CKPT=true (gradient checkpointing)
WIDTH = {"w8": ("FA-ccin2p3-804a65bc", "FA-ccin2p3-66e8c31f"),
         "w16": ("FA-ccin2p3-493c98db", "FA-ccin2p3-7288b3fe"),
         "w32": ("FA-ccin2p3-e26059bf", "FA-ccin2p3-fd7fc137")}
WIDTH_RUN_DIR = {"FA-ccin2p3-804a65bc": "v4_het_default25_w8_x1", "FA-ccin2p3-66e8c31f": "v4_het_default25_seed2_w8_x1",
                 "FA-ccin2p3-493c98db": "v4_het_default25_w16_x1", "FA-ccin2p3-7288b3fe": "v4_het_default25_seed2_w16_x1",
                 "FA-ccin2p3-e26059bf": "v4_het_default25_w32_x1", "FA-ccin2p3-fd7fc137": "v4_het_default25_seed2_w32_x1"}


def _best_of(run):
    """The run's best checkpoint from its fetched per_process_metrics.json (analysis/common best_checkpoint): index,
    aggregate val_loss_no_reg, each process at that step; None when not fetched"""
    sys.path.insert(0, ROOT)
    from analysis.common import best_checkpoint, run_metrics
    d = os.path.expanduser(f"~/.local/share/ccorch/artifacts/FA/{run}/sps/lpnhe/jiturrizaramirez01/"
                           f"Foundational_Amplitudes/runs/catalog_v4_long/{WIDTH_RUN_DIR[run]}")
    if not os.path.isdir(d):
        return None
    m = run_metrics(d)
    i, v, procs = best_checkpoint(m)
    return {"index": i, "n_validations": len(m["val_loss_no_reg"]), "val_loss_no_reg": v,
            "last": float(m["val_loss_no_reg"][-1]), "procs": procs}


def width():
    cls = class_of()
    ax_of = _held_axes(cls)
    reads = {w: {r: tuple(read_of(run, r) for run in runs) for r in ("PP4", "EE4", "held", "HD4", "HD5")}
             for w, runs in WIDTH.items()}
    missing = [f"{w} {r} {WIDTH[w][i]}" for w, v in reads.items() for r, pair in v.items()
               for i, x in enumerate(pair) if x is None]
    if missing:
        sys.exit("width: reads missing (extract them first): " + "; ".join(missing))
    out = {"design": "plan catalog_coverage step 33 (1x width cells, D248; the step-16 design's information read)",
           "runs": WIDTH, "best": {}, "arms": {}}
    best = {run: _best_of(run) for runs in WIDTH.values() for run in runs}
    for run, b in best.items():
        out["best"][run] = None if b is None else {k: b[k] for k in ("index", "n_validations", "val_loss_no_reg", "last")}
    c = reads["w8"]
    extra = {ax: {w: WIDTH[w] for w in WIDTH} for ax in ("HD4", "HD5")}
    tab = {w: {r: reads[w][r] for r in ("PP4", "EE4")} for w in WIDTH}
    tab["control"] = tab["w8"]
    ext = {ax: dict(t, control=WIDTH["w8"]) for ax, t in extra.items()}
    for w in ("w16", "w32"):
        res = {}
        for r in ("PP4", "EE4", "held", "HD4", "HD5"):
            arm, ctl = side(reads[w][r]), side(c[r])
            keys = sorted(set(arm) & set(ctl), key=str)
            parts = {"all": keys}
            if r == "held":
                parts = {"pooled": keys, **{ax: [k for k in keys if k[0] == ax] for ax in HELD}}
            res[r] = {n: compare(arm, ctl, c[r][0], c[r][1], ks, cls) for n, ks in parts.items() if ks}
            cl = lambda k: cls(probe_name(k))
            res[r]["per_seed_R"] = {s: round(float(np.median(list(class_ratios(reads[w][r][i], c[r][i], keys, cl)
                                                                   .values()))), 4)
                                    for i, s in enumerate(("seed42", "seed43"))}
            res[r]["n_probes_unmatched"] = len(set(arm) ^ set(ctl))
        res["held_default"], _ = _held_default({**tab, "arm": tab[w]}, "arm", cls, ax_of,
                                               {ax: {"arm": WIDTH[w], "control": WIDTH["w8"]} for ax in ext})
        # each process's val_loss_no_reg (het NLL) at the best checkpoint: width minus width 8, per seed
        dif = {}
        for i, s in enumerate(("seed42", "seed43")):
            a, b = best[WIDTH[w][i]], best[WIDTH["w8"][i]]
            if a and b:
                d = [a["procs"][p] - b["procs"][p] for p in a["procs"] if p in b["procs"]]
                dif[s] = {"n_processes": len(d), "median_delta_nll": round(float(np.median(d)), 4),
                          "frac_lower": round(float(np.mean(np.array(d) < 0)), 4),
                          "delta_aggregate": round(a["val_loss_no_reg"] - b["val_loss_no_reg"], 4)}
        res["val_per_process"] = dif
        out["arms"][w] = res
    out["blowups"] = blowups([run for runs in WIDTH.values() for run in runs])
    json.dump(out, open(os.path.join(DATA, "width_x1.json"), "w"), indent=1)
    print(json.dumps(out))


ARMS_VS_CONTROL = {"default25_1e4": {"PP4": MIX_HET["default25"]["PP4"], "EE4": MIX_HET["default25"]["EE4"],
                                     "held": MIX_HET["default25"]["held"]},
                   **{f"w{k}_full": {r: WIDTH[f"w{k}"] for r in ("PP4", "EE4", "held")} for k in (8, 16, 32)},
                   "control": MIX_HET["control"]}
ARMS_HD = {"default25_1e4": HD4_HET["default25"], **{f"w{k}_full": WIDTH[f"w{k}"] for k in (8, 16, 32)},
           "control": HD4_HET["control"]}
# one-seed arms: a long-run sweep's best trial (seed 42, its lr and HPs tuned by DyHPO), read by an eval job
# (scripts/job_catalog_v3_short.sh arm v4_het_long_best; design passed 2026-11-03T01:34): {arm: (eval run, the
# trial's key in data/size_curves.json, which holds its training curve, steps and batches)}
LONG_BEST = {"long4x_best": ("FA-ccin2p3-6f8a1eed", "17056_trial_1262"),
             "long16x_best": ("FA-ccin2p3-1a46f3e0", "68224_trial_1262")}
for _a, (_run, _key) in LONG_BEST.items():
    TRAINED_BY[_run] = f"trial_{_key}"


def trial_val(key):
    """A sweep trial's training curve from data/size_curves.json (size_figs.py pull) -> data/reads/trial_<key>_val.json,
    the format `val` writes, so blowups() checks the trial an eval run read"""
    t = json.load(open(os.path.join(DATA, "size_curves.json")))["trials"][key]
    os.makedirs(READS, exist_ok=True)
    out = os.path.join(READS, f"trial_{key}_val.json")
    json.dump({"run": f"trial_{key}", "loss": "HETEROSC", "log": f"size_curves.json trials/{key}",
               "blocks": [{"iterations": t["steps"], "every": t["every"], "vals": t["vals"]}]}, open(out, "w"), indent=0)
    return {"iterations": t["steps"], "batches": t["batches"], "epochs": round(t["steps"] / t["batches"], 4),
            "source": f"size_curves.json trials/{key}"}


def _tagged(pairs):
    """{(source, probe): mse} over [(run, source read), ...]; None when one is missing"""
    out = {}
    for run, r in pairs:
        d = read_of(run, r)
        if d is None:
            return None
        out.update({(r, k): x for k, x in d.items()})
    return out


def _one_point(x, ref, keys, cls):
    """One seed's absolute MSE (class median of the class's geometric mean over its probes) and R = x / ref (class
    median of the class's geometric mean of the ratios) on `keys`"""
    cl = lambda k: cls(probe_name(k))
    by = {}
    for k in keys:
        by.setdefault(cl(k), []).append(x[k])
    return {"mse": float(f"{np.median([gmean(v) for v in by.values()]):.5g}"),
            "R": round(float(np.median(list(class_ratios(x, ref, keys, cl).values()))), 4)}


def _long_best(a, cls, ax_of):
    """A one-seed arm: per read, its point against the het control's two-seed geometric mean (as `seeds`), and its
    ratio to each 1x width cell's seed 42 on the same probes (information: length and tuning together)"""
    run, key = LONG_BEST[a]
    ctl_runs = ARMS_VS_CONTROL["control"]
    hd_ctl = ARMS_HD["control"]
    reads = {r: (read_of(run, r), tuple(read_of(c, r) for c in ctl_runs[r]),
                 {w: read_of(WIDTH[w][0], r) for w in WIDTH}) for r in ("PP4", "EE4", "held")}
    dflt = lambda r42pp, r42ee, rhd: _tagged([(r42pp, "PP4"), (r42ee, "EE4"), (rhd, "HD4"), (rhd, "HD5")])
    reads["held_default"] = (dflt(run, run, run),
                             tuple(dflt(ctl_runs["PP4"][i], ctl_runs["EE4"][i], hd_ctl[i]) for i in (0, 1)),
                             {w: dflt(WIDTH[w][0], WIDTH[w][0], WIDTH[w][0]) for w in WIDTH})
    miss = [f"{a} {r}" for r, (x, cp, ws) in reads.items() if x is None or None in cp or None in ws.values()]
    if miss:
        sys.exit("arms: reads missing: " + "; ".join(miss))
    pts, vs = {}, {}
    for r, (x, cp, ws) in reads.items():
        keys = set(x) & set(cp[0]) & set(cp[1])
        if r == "held_default":
            keys = {k for k in keys if cls(k[1]) in ax_of}
        keys = sorted(keys, key=str)
        parts = {r: keys}
        if r == "held_default":
            parts.update({f"held_default/{ax}": [k for k in keys if ax_of[cls(k[1])] == ax] for ax in HELD})
        for name, ks in parts.items():
            if not ks:
                continue
            ctl = {k: gmean([cp[0][k], cp[1][k]]) for k in ks}
            pts[name] = {"n_probes": len(ks), "seed42": _one_point(x, ctl, ks, cls)}
            vs[name] = {}
            for w, cell in ws.items():
                kw = [k for k in ks if k in cell]
                p = _one_point(x, cell, kw, cls)
                vs[name][w] = {"n_probes": len(kw), "R_over_cell_seed42": p["R"], "mse_arm": p["mse"],
                               "mse_cell_seed42": _one_point(cell, cell, kw, cls)["mse"], "cell_run": WIDTH[w][0]}
    return pts, vs, trial_val(key)


def epochs(run, log):
    pat = re.compile(r"Starting to train for (\d+) iterations = [0-9.]+ epochs on a dataset with (\d+) batches")
    for line in open(log, errors="replace"):
        m = pat.search(line)
        if m:
            it, b = int(m.group(1)), int(m.group(2))
            break
    else:
        sys.exit(f"epochs: no 'Starting to train' line in {log}")
    path = os.path.join(DATA, "epochs.json")
    d = json.load(open(path)) if os.path.exists(path) else {}
    d[run] = {"iterations": it, "batches": b, "epochs": round(it / b, 4), "log": os.path.basename(log)}
    json.dump(d, open(path, "w"), indent=1, sort_keys=True)
    print(json.dumps({run: d[run]}))


def _default_pairs(reads, a, extra):
    """(arm (seed 42, seed 43), control (seed 42, seed 43)) reads tagged (source, probe) over the default-sampled
    sources _held_default uses (PP4, EE4, and each extra axis read on all four seeds)"""
    tag = lambda r, d: {(r, k): x for k, x in (d or {}).items()}
    arm, ctl = ({}, {}), ({}, {})
    srcs = [(r, reads[a][r], reads["control"][r]) for r in ("PP4", "EE4")]
    for ax, tab in extra.items():
        if tab.get(a) and tab.get("control"):
            ap = tuple(read_of(x, ax) for x in tab[a])
            cp = tuple(read_of(x, ax) for x in tab["control"])
            if all(x is not None for x in ap + cp):
                srcs.append((ax, ap, cp))
    for r, ap, cp in srcs:
        if any(x is None for x in cp):
            continue
        for i in (0, 1):
            arm[i].update(tag(r, ap[i])); ctl[i].update(tag(r, cp[i]))
    return arm, ctl


def _seed_points(pair, ctl_pair, keys, cls):
    """Each seed of `pair` apart on `keys`: absolute MSE (class median of the class's geometric mean over its probes)
    and R against the control's two-seed geometric mean (class median of the class's geometric mean of the ratios)"""
    cl = lambda k: cls(probe_name(k))
    ctl = {k: gmean([ctl_pair[0][k], ctl_pair[1][k]]) for k in keys}
    out = {"n_probes": len(keys)}
    for i, s in enumerate(("seed42", "seed43")):
        by = {}
        for k in keys:
            by.setdefault(cl(k), []).append(pair[i][k])
        out[s] = {"mse": float(f"{np.median([gmean(v) for v in by.values()]):.5g}"),
                  "R": round(float(np.median(list(class_ratios(pair[i], ctl, keys, cl).values()))), 4)}
    return out


def _points(arm_seeds, ctl_seeds, cls, ax_of):
    """The figure's one point per arm and read (Joaquin 2026-11-03): the arm's per-probe geometric mean over its
    seeds, R against the control's two-seed geometric mean and the absolute MSE (class median of the class's geometric
    mean); the bar the design's interval (r_with_interval "widened": the 95% class bootstrap of the class median,
    10000 resamples, widened on each side by |ln R_seed|, R_seed the control's seed 43 over seed 42 class median on
    the same classes; the same for every arm, a one-seed arm included, and for the absolute MSE's class median). Reads: PP4 and EE4 on the TRAINED classes
    only (their held-out classes are in held_default), the held-out pool catalog-sampled (held) and default-sampled
    (held_default, pooled and per axis). arm_seeds / ctl_seeds: {read: [seed dicts]}, read in PP4, EE4, held,
    held_default (keys (source, probe))"""
    cl = lambda k: cls(probe_name(k))
    out = {}
    for r in ("PP4", "EE4", "held", "held_default"):
        xs, cs = arm_seeds[r], ctl_seeds[r]
        keys = set.intersection(*(set(x) for x in xs + cs))
        if r in ("PP4", "EE4"):
            keys = {k for k in keys if cl(k) not in ax_of}
        elif r == "held_default":
            keys = {k for k in keys if cl(k) in ax_of}
        parts = {r + ("_trained" if r in ("PP4", "EE4") else ""): sorted(keys, key=str)}
        if r == "held_default":
            parts.update({f"held_default/{ax}": sorted((k for k in keys if ax_of[cl(k)] == ax), key=str)
                          for ax in HELD})
        for name, ks in parts.items():
            if not ks:
                continue
            ctl = {k: gmean([c[k] for c in cs]) for k in ks}
            arm = {k: gmean([x[k] for x in xs]) for k in ks}
            r_seed = float(np.median(list(class_ratios(cs[1], cs[0], ks, cl).values())))
            res = {"n_probes": len(ks), "n_classes": len({cl(k) for k in ks}), "n_seeds": len(xs),
                   "R_seed_control": round(r_seed, 4)}
            for q, vals in (("R", class_ratios(arm, ctl, ks, cl)),
                            ("mse", {c: gmean(v) for c, v in _by_class(arm, ks, cl).items()})):
                iv = r_with_interval(vals, r_seed)
                res[q] = float(f"{np.median(list(vals.values())):.5g}")
                res[q + "_bar"] = [float(f"{x:.5g}") for x in iv["widened"]]
            out[name] = res
    return out


def _by_class(d, keys, cl):
    by = {}
    for k in keys:
        by.setdefault(cl(k), []).append(d[k])
    return by


def arms():
    cls = class_of()
    ax_of = _held_axes(cls)
    reads, missing = _reads(ARMS_VS_CONTROL)
    extra = {ax: dict(ARMS_HD) for ax in ("HD4", "HD5")}
    miss_hd = [f"{a} {ax} {run}" for ax in extra for a, runs in ARMS_HD.items() for run in runs
               if read_of(run, ax) is None]
    if missing or miss_hd:
        sys.exit("arms: reads missing: " + "; ".join(missing + miss_hd))
    c = reads["control"]
    out = {"design": "plan catalog_coverage steps 15 and 33 (information: every arm against the het control)",
           "runs": ARMS_VS_CONTROL, "runs_HD4_HD5": ARMS_HD, "arms": {}}
    for a in ARMS_VS_CONTROL:
        if a == "control":
            continue
        res = {}
        for r in ("PP4", "EE4", "held"):
            arm, ctl = side(reads[a][r]), side(c[r])
            keys = sorted(set(arm) & set(ctl), key=str)
            parts = {"pooled": keys, **{ax: [k for k in keys if k[0] == ax] for ax in HELD}} if r == "held" \
                else {"all": keys}
            res[r] = {n: compare(arm, ctl, c[r][0], c[r][1], ks, cls) for n, ks in parts.items() if ks}
            res[r]["n_probes_unmatched"] = len(set(arm) ^ set(ctl))
        res["held_default"], dkeys = _held_default(reads, a, cls, ax_of, extra)
        out["arms"][a] = res
        # each seed apart, the arm's and (on the same probes) the control's
        dpair, dctl = _default_pairs(reads, a, extra)
        for who, sel in ((a, lambda p, cp: p), ("control", lambda p, cp: cp)):
            pts = out.setdefault("seeds", {}).setdefault(who, {})
            for r in ("PP4", "EE4", "held"):
                ks = sorted(set(reads[a][r][0]) & set(reads[a][r][1]) & set(c[r][0]) & set(c[r][1]), key=str)
                if r == "held":
                    ks = [k for k in ks if str(probe_name(k)).startswith("v4_")]
                pts.setdefault(r, _seed_points(sel(reads[a][r], c[r]), c[r], ks, cls))
            pts.setdefault("held_default", _seed_points(sel(dpair, dctl), dctl, dkeys, cls))
            for ax in HELD:
                ks = [k for k in dkeys if ax_of[cls(k[1])] == ax]
                if ks:
                    pts.setdefault(f"held_default/{ax}", _seed_points(sel(dpair, dctl), dctl, ks, cls))
    ep = json.load(open(os.path.join(DATA, "epochs.json")))
    trained = {a: [TRAINED_BY.get(run, run) for run in v["PP4"]] for a, v in ARMS_VS_CONTROL.items()}
    out["epochs"] = {a: {run: ep[run] for run in runs} if all(run in ep for run in runs) else None
                     for a, runs in trained.items()}
    # the figure's points (one per arm and read, bars from the per-seed bootstrap intervals)
    dflt = lambda pp, ee, hd: _tagged([(pp, "PP4"), (ee, "EE4"), (hd, "HD4"), (hd, "HD5")])
    ctl_runs = ARMS_VS_CONTROL["control"]
    ctl_seeds = {r: list(c[r]) for r in ("PP4", "EE4", "held")}
    ctl_seeds["held_default"] = [dflt(ctl_runs["PP4"][i], ctl_runs["EE4"][i], ARMS_HD["control"][i]) for i in (0, 1)]
    out["points"] = {}
    for a in ARMS_VS_CONTROL:
        av = ARMS_VS_CONTROL[a]
        seeds_a = {r: list(reads[a][r]) for r in ("PP4", "EE4", "held")}
        seeds_a["held_default"] = [dflt(av["PP4"][i], av["EE4"][i], ARMS_HD[a][i]) for i in (0, 1)]
        out["points"][a] = _points(seeds_a, ctl_seeds, cls, ax_of)
    for a, (run, _k) in LONG_BEST.items():
        seeds_a = {r: [read_of(run, r)] for r in ("PP4", "EE4", "held")}
        seeds_a["held_default"] = [dflt(run, run, run)]
        out["points"][a] = _points(seeds_a, ctl_seeds, cls, ax_of)
    out["one_seed_arms"] = {"runs": LONG_BEST, "caveat": "one seed (42), the best of an 8-trial DyHPO sweep (lr and "
                            "HPs tuned, best-of-8 selection), not converged (best lr at the window's top edge, best "
                            "checkpoint at the last validation); against the 1x width cells: length and tuning "
                            "together, information only", "vs_width_cells": {}}
    for a in LONG_BEST:
        pts, vs, ep_a = _long_best(a, cls, ax_of)
        out["seeds"][a] = pts
        out["one_seed_arms"]["vs_width_cells"][a] = vs
        out["epochs"][a] = {LONG_BEST[a][0]: ep_a}
    out["blowups"] = blowups([run for v in ARMS_VS_CONTROL.values() for runs in v.values() for run in runs] +
                             [run for runs in ARMS_HD.values() for run in runs] + [v[0] for v in LONG_BEST.values()])
    json.dump(out, open(os.path.join(DATA, "arms_vs_het_control.json"), "w"), indent=1)
    print(json.dumps(out))


def length():
    """Step 16's length verdict (design 2026-11-01T16:14): the 16x sweep's best trial (68224 steps) over the 4x
    sweep's best (17056), one seed (42) each, read as the other cells (sigma-stage parent, D276). Per read R = median
    over classes of the class's geometric mean of 16x/4x per probe; 95% class bootstrap widened on each side by the
    1x width-8 seed pair's |ln R_seed| on the same probes (the seed-noise stand-in: no seed twin at either length).
    Reads: PP4 (all), EE4 (all), the 700 default-sampled held-out classes (PP4/EE4 probes of held-out classes, HD4,
    HD5) pooled and per axis, the catalog-sampled held-out H1-H7c pooled and per axis (beside).
    YES: R < 1 with the widened interval below 1 on PP4, EE4 and the default-sampled held-out pool;
    NO: the widened interval contains 1 on all three; else MIXED (per read). -> data/length_x16_over_x4.json"""
    cls = class_of()
    ax_of = _held_axes(cls)
    x16, x4 = LONG_BEST["long16x_best"][0], LONG_BEST["long4x_best"][0]
    w42, w43 = WIDTH["w8"]
    dflt = lambda r: _tagged([(r, "PP4"), (r, "EE4"), (r, "HD4"), (r, "HD5")])
    reads = {r: [read_of(x, r) for x in (x16, x4, w42, w43)] for r in ("PP4", "EE4", "held")}
    reads["held_default"] = [dflt(x) for x in (x16, x4, w42, w43)]
    miss = [f"{r} {x}" for r, v in reads.items() for x, d in zip((x16, x4, w42, w43), v) if d is None]
    if miss:
        sys.exit("length: reads missing: " + "; ".join(miss))
    out = {"design": "plan catalog_coverage step 16 (2026-11-01T16:14), step 32 verdict", "x16": x16, "x4": x4,
           "seed_pair": [w42, w43], "reads": {}}
    for r, (a, b, s42, s43) in reads.items():
        keys = set(a) & set(b) & set(s42) & set(s43)
        if r == "held_default":
            keys = {k for k in keys if cls(k[1]) in ax_of}
        keys = sorted(keys, key=str)
        parts = {"all": keys}
        if r == "held":
            parts = {"pooled": keys, **{ax: [k for k in keys if k[0] == ax] for ax in HELD}}
        if r == "held_default":
            parts = {"pooled": keys, **{ax: [k for k in keys if ax_of[cls(k[1])] == ax] for ax in HELD}}
        res = {n: compare(a, b, s42, s43, ks, cls) for n, ks in parts.items() if ks}
        res["n_probes_unmatched"] = len(set(a) ^ set(b))
        res["mse_x16"] = _one_point(a, b, keys, cls)["mse"]; res["mse_x4"] = _one_point(b, b, keys, cls)["mse"]
        out["reads"][r] = res
    main3 = [out["reads"][r]["pooled" if r == "held_default" else "all"] for r in ("PP4", "EE4", "held_default")]
    if all(x["R"] < 1 and x["widened"][1] < 1 for x in main3):
        v = "YES"
    elif all(x["widened"][0] <= 1 <= x["widened"][1] for x in main3):
        v = "NO"
    else:
        v = "MIXED"
    out["blowups"] = blowups([x16, x4])
    out["verdict_on_reads"] = v
    out["verdict"] = None if out["blowups"]["flagged"] else v       # withheld while a run it reads blew up
    out["caveat"] = ("one seed per length, each the best of an 8-trial DyHPO sweep; both not converged (best lr at the "
                     "window's top, best checkpoint at the last validation); interval widened by the 1x width-8 seed "
                     "pair's |ln R_seed|")
    json.dump(out, open(os.path.join(DATA, "length_x16_over_x4.json"), "w"), indent=1)
    print(json.dumps({r: {n: (x["R"], x["widened"], x["n_classes"]) for n, x in v.items() if isinstance(x, dict)}
                      for r, v in out["reads"].items()}))
    print("verdict", v)


# plan warm_start, change C1 (superset design passed 2026-11-04T21:51): the warm starts W_k (the het control's final
# model, retrained k steps on the share-0.25 recipe, the parent's schedule restarted over k) and their matched controls
# C_k (the same warm start on the parent's own recipe), scripts/job_catalog_v3_short.sh @7f945910 (deployed cf87b124),
# every read from the run's own job at its sigma-stage model (D276); (seed 42, seed 43) per k. B = share 0.25 from
# scratch (MIX_HET default25 with its HD4/HD5), A = the parent (the het control, step 0)
WARM_K = (533, 1066, 2132)
WARM_STEPS = 4264
WARM = {"C1": {"W": {533: ("FA-ccin2p3-341bf564", "FA-ccin2p3-254abf4f"),
                     1066: ("FA-ccin2p3-ed946736", "FA-ccin2p3-8e41ac1e"),
                     2132: ("FA-ccin2p3-f699e170", "FA-ccin2p3-04fc5b3b")},
               "C": {533: ("FA-ccin2p3-343e3d58", "FA-ccin2p3-c2acc1fc"),
                     1066: ("FA-ccin2p3-aa64733e", "FA-ccin2p3-c477a52c"),
                     2132: ("FA-ccin2p3-0564ad5b", "FA-ccin2p3-a97e8083")},
               "B": dict(MIX_HET["default25"], HD4=HD4_HET["default25"], HD5=HD5_HET["default25"]),
               "A": dict(MIX_HET["control"], HD4=HD4_HET["control"], HD5=HD5_HET["control"]),
               "deciding": ("PP4", "EE4")},
        # step 3 (design passed 2026-11-05T02:09, D284): the tuned warm start at k=533, the best trial hp_1860 of the
        # DyHPO sweep ws_c1_k533_hpo8 rerun at seeds 42 and 43 with the full job (arm v4_het_warm25t_k533[_seed2],
        # scripts/job_catalog_v3_short.sh @dab9ff19); no matched control (D286); beside, R over the untuned W_533
        "C1t": {"W": {533: ("FA-ccin2p3-56cca07a", "FA-ccin2p3-75d28cb4")}, "C": {},
                "ref": {"untuned": ("FA-ccin2p3-341bf564", "FA-ccin2p3-254abf4f")},
                "deciding": ("PP4", "EE4")}}
WARM["C1t"].update(B=WARM["C1"]["B"], A=WARM["C1"]["A"])   # C1's from-scratch B and parent A
# step 5 (D289, design passed 2026-11-05T04:06): the k=533 warm start at step 3's tuned HPs (hp_1860) from the rerun het
# control's EMA model at step 2132, at step 3198 (pre-decay checkpoints, cosine lr about 0.65 and 0.20 of peak) and at
# its final model (training only, so a pre-sigma start); reruns FA-ccin2p3-d521160b (s42), b688a859 (s43), arms
# v4_het_control_ckpt[_seed2]; warm starts arms v4_het_warm25t_k533_from<x>[_seed2], scripts/job_catalog_v3_short.sh
# @d9e1af1f. Each checkpoint read against W_final (same rerun; the design's keep rule) and against step 3's tuned
# confirmation runs (post-sigma parent); B and A as C1
_CK = {"final": ("FA-ccin2p3-dd32a907", "FA-ccin2p3-70c28bd9")}
_TUNED = WARM["C1t"]["W"][533]
for _x, _runs in (("2132", ("FA-ccin2p3-0898a7b0", "FA-ccin2p3-a6834028")),
                  ("3198", ("FA-ccin2p3-2a27e8a2", "FA-ccin2p3-0259e865")), ("final", _CK["final"])):
    WARM[f"C1ck{_x}"] = {"W": {533: _runs}, "C": {}, "B": WARM["C1"]["B"], "A": WARM["C1"]["A"],
                         "ref": dict({} if _x == "final" else _CK, tuned_post_sigma=_TUNED),
                         "deciding": ("PP4", "EE4"), "keep_ref": None if _x == "final" else "final",
                         "design": "plan warm_start step 5 (D289), design passed 2026-11-05T04:06"}
# step 4 (D285, design passed round 2 2026-11-05T04:30): width growth w8 -> w16, the w8 cell's final model grown
# function-preservingly (models/grow_width.py) and retrained k steps at the w16 cell's setup (GROW_K on
# v4_het_default25[_seed2], LONG_HEADS=16 LONG_CKPT LONG_MEMMAP, scripts/job_catalog_v3_short.sh @13fe9f64, the second resubmission
# after the grow_width device fix 2d701b40 and the sigma-stage fix b36bb809), every read
# from the run's own job; B = the w16 cell from scratch, A = the w8 parent; MATCH bar (A/B)^0.3 = 1.19 (width_x1.json)
WARM["C10w"] = {"W": {533: ("FA-ccin2p3-e046ac67", "FA-ccin2p3-f35b6a78"),
                      1066: ("FA-ccin2p3-380c7b14", "FA-ccin2p3-af86de05"),
                      2132: ("FA-ccin2p3-58723352", "FA-ccin2p3-93479cde")}, "C": {},
                "B": WIDTH["w16"], "A": WIDTH["w8"], "deciding": ("PP4", "EE4"), "match_hi": 1.19,
                "design": "plan warm_start step 4 (D285), design passed round 2 2026-11-05T04:30"}
# step 6 (D303, design passed round 2 2026-11-06T05:40): the k=533 default restart with the optimizer's moments filled
# at lr 0 for 100 batches before the ramp (arm v4_het_warm25f_k533[_seed2], fine_tune.moment_fill_steps=100) against
# the same-commit default restart (v4_het_warm25_k533[_seed2], SHORT_TAG=d303), scripts/job_catalog_v3_short.sh
# @117f19ca; BETTER when the widened upper end of R(W/default) < 1 on PP4 or EE4 and above 1.05 on neither; B and A as C1
WARM["C1f"] = {"W": {533: ("FA-ccin2p3-b2c69c1a", "FA-ccin2p3-f5245570")}, "C": {},
               "B": WARM["C1"]["B"], "A": WARM["C1"]["A"],
               "ref": {"default": ("FA-ccin2p3-0202ba7f", "FA-ccin2p3-7b2d6c24"),
                       "default_older": ("FA-ccin2p3-341bf564", "FA-ccin2p3-254abf4f")},
               "deciding": ("PP4", "EE4"), "design": "plan warm_start step 6 (D303), design passed round 2 2026-11-06T05:40"}
# step 6 (D291, D298, design passed 2026-11-06T04:48): the k=533 warm start at the val-MSE-of-the-mean sweep's best
# trial hp_1333 (ws_c1_k533_mse_wsd_hpo8: lr_scale 1.85, WSD cooldown 0.164) rerun at seeds 42 and 43 with the full
# job (arm v4_het_warm25m_k533[_seed2], scripts/job_catalog_v3_short.sh @30ea35ac); beside, R over the untuned W_533
# and over the NLL-tuned one (C1t); B and A as C1
WARM["C1m"] = {"W": {533: ("FA-ccin2p3-4f9e0532", "FA-ccin2p3-8cb6c7b5")}, "C": {},
               "B": WARM["C1"]["B"], "A": WARM["C1"]["A"],
               "ref": {"untuned": ("FA-ccin2p3-341bf564", "FA-ccin2p3-254abf4f"), "nll_tuned": _TUNED},
               "deciding": ("PP4", "EE4"), "design": "plan warm_start step 6 (D291), design passed 2026-11-06T04:48"}
# the same tuned restart (GROW_HP=1333) on the width growth w8 -> w16 at k=533 (v4_het_default25[_seed2] with
# LONG_HEADS=16 LONG_CKPT LONG_MEMMAP GROW_K=533, scripts/job_catalog_v3_short.sh @4272209a); beside, R over the
# untuned grown k=533 pair; B, A and the MATCH bar as C10w
WARM["C10wm"] = {"W": {533: ("FA-ccin2p3-31359864", "FA-ccin2p3-bfed460f")}, "C": {},
                 "B": WIDTH["w16"], "A": WIDTH["w8"], "deciding": ("PP4", "EE4"), "match_hi": 1.19,
                 "ref": {"untuned": WARM["C10w"]["W"][533]},
                 "design": "plan warm_start step 6 (D291, D285), design passed 2026-11-06T04:48"}
# step 11 (D285, D291, design passed round 2 2026-11-06T10:54): the width growth w8 -> w16 at k=533 re-run at the best
# trial of its zero-shot-objective sweep (ws_grow_k533_zeroshot_hpo8_rerun trial 1860, zero_shot_score 0.476 against
# the default growth's 0.535), v4_het_default25[_seed2] with LONG_HEADS=16 LONG_CKPT LONG_MEMMAP GROW_K=533
# GROW_SWEEP=ws_grow_k533_zeroshot_hpo8_rerun GROW_HP=1860 (scripts/job_catalog_v3_short.sh @6454ecda); BETTER when the
# widened upper end of R(W/default growth) < 1 on PP4 or EE4 and above 1.05 on neither; B, A and the MATCH bar as C10wm
WARM["C10wz"] = {"W": {533: ("FA-ccin2p3-4442e786", "FA-ccin2p3-3fae7799")}, "C": {},
                 "B": WIDTH["w16"], "A": WIDTH["w8"], "deciding": ("PP4", "EE4"), "match_hi": 1.19,
                 "ref": {"default": WARM["C10w"]["W"][533], "mse_tuned": WARM["C10wm"]["W"][533]},
                 "design": "plan warm_start step 11 (D285, D291), design passed round 2 2026-11-06T10:54"}
# step 12 (D307, D308, design passed round 2 2026-11-06): the restart study, k=533 from the checkpoint rerun
# v4_het_control_ckpt[_seed2] (arms v4_het_ws_*, scripts/job_catalog_v3_short.sh @a1b0b899): each arm against the
# same parent's final model at the default restart (C1r_ctrl, its own R beside the original parent's default restart);
# BETTER when the widened upper end of R(arm/control) < 1 on PP4 or EE4 and above 1.05 on neither; B and A as C1
_C1R = {"final_default": ("FA-ccin2p3-0de6be56", "FA-ccin2p3-89ecbc1e"),
        "step3198_default": ("FA-ccin2p3-1ef38bb2", "FA-ccin2p3-c07e6ad5"),
        "step3198_continue": ("FA-ccin2p3-4cb686ac", "FA-ccin2p3-1f7ae4ea"),
        "step2132_continue": ("FA-ccin2p3-f3d6af3a", "FA-ccin2p3-32cb8805"),
        "final_lr3x": ("FA-ccin2p3-ba567f27", "FA-ccin2p3-166c6378")}
WARM["C1r_ctrl"] = {"W": {533: _C1R["final_default"]}, "C": {}, "B": WARM["C1"]["B"], "A": WARM["C1"]["A"],
                    "ref": {"default_original_parent": ("FA-ccin2p3-341bf564", "FA-ccin2p3-254abf4f")},
                    "deciding": ("PP4", "EE4"), "design": "plan warm_start step 12 (D307, D308), passed round 2"}
for _k in ("step3198_default", "step3198_continue", "step2132_continue", "final_lr3x"):
    WARM["C1r_" + _k] = {"W": {533: _C1R[_k]}, "C": {}, "B": WARM["C1"]["B"], "A": WARM["C1"]["A"],
                         "ref": {"control": _C1R["final_default"]},
                         "deciding": ("PP4", "EE4"), "design": "plan warm_start step 12 (D307, D308), passed round 2"}
# step 13 (D307, D308, design passed round 2 2026-11-06): step3198_continue with the parent's optimizer state loaded
# (fine_tune.load_optimizer_state, arm v4_het_ws_step3198_continue_optstate[_seed2], scripts/job_catalog_v3_short.sh
# @7b50861c): BETTER against the control as C1r_*; the mechanism read against the fresh-optimizer continued arm
WARM["C1r_step3198_continue_optstate"] = {
    # the reruns after b91cf7c6 (run dirs <arm>_tagfix1): the first pair (05df656d, 07bfc8d7) ended in a sigma stage that
    # reloaded the run's own optimizer moments, which moved the frozen mu row (its first validation 2.3e8): not read
    "W": {533: ("FA-ccin2p3-1c90ae73", "FA-ccin2p3-013708af")}, "C": {}, "B": WARM["C1"]["B"], "A": WARM["C1"]["A"],
    "ref": {"control": _C1R["final_default"], "fresh_optimizer": _C1R["step3198_continue"],
            "step3198_default": _C1R["step3198_default"]},
    "deciding": ("PP4", "EE4"), "design": "plan warm_start step 13 (D307, D308), passed round 2"}
# step 8 (D303, design passed round 2 2026-11-06): the k=533 default restart with only the Muon groups' lr damped at
# its start (x min(1, step/168), fine_tune.muon_warmup_steps=168, arm v4_het_warm25u_k533[_seed2],
# scripts/job_catalog_v3_short.sh @dbb8daf0) against the same-data default restart of the fill test; B and A as C1;
# BETTER when the widened upper end of R(W/default) < 1 on PP4 or EE4 and above 1.05 on neither
WARM["C1u"] = {"W": {533: ("FA-ccin2p3-35f9a8a5", "FA-ccin2p3-a0416422")}, "C": {},
               "B": WARM["C1"]["B"], "A": WARM["C1"]["A"],
               "ref": {"default": WARM["C1f"]["ref"]["default"], "default_older": WARM["C1f"]["ref"]["default_older"]},
               "deciding": ("PP4", "EE4"), "design": "plan warm_start step 8 (D303), design passed round 2 2026-11-06"}
# step 10 (D303, step 8's NOT MUON branch; design passed 2026-11-06 10:26): the k=533 restart with AdamW's lr damped
# over its first 168 steps, Muon untouched (C1a, v4_het_warm25a_k533[_seed2]), and with every group damped (C1e,
# v4_het_warm25e_k533[_seed2]), against the same default restart pair as C1f/C1u; W filled at submission
for _c, _what in (("C1a", "AdamW damped"), ("C1e", "every group damped")):
    WARM[_c] = {"W": {}, "C": {}, "B": WARM["C1"]["B"], "A": WARM["C1"]["A"],
                "ref": {"default": WARM["C1f"]["ref"]["default"], "default_older": WARM["C1f"]["ref"]["default_older"]},
                "deciding": ("PP4", "EE4"), "design": f"plan warm_start step 10 (D303), {_what}, design passed 2026-11-06"}
WARM["C1a"]["W"][533] = ("FA-ccin2p3-fbb16a33", "FA-ccin2p3-cb458c82")   # v4_het_warm25a_k533[_seed2], submitted 2026-11-06
WARM["C1e"]["W"][533] = ("FA-ccin2p3-21a18a76", "FA-ccin2p3-21674d1e")   # v4_het_warm25e_k533[_seed2] (the launch's; the hand-submitted 5862b62e/d15462b0 duplicates cancelled before start)
# step 7 (D283, D300; the superset design passed 2026-11-04T21:51): the other changes at k in 533/1066/2132, arms of
# scripts/job_catalog_v3_short.sh (ws-step7). W and C are filled with the run ids at submission (empty: not submitted).
# Het changes from the het control (A as C1, C_k C1's v4_het_cont_k*): C2 share 0.50 (B the het share-0.50 pair),
# C8 the signed one-loop pools on their magnitude (B v4_het_control_magnitude, catalog_coverage step 47; deciding HD5,
# H5 beside). MSE changes from v4_dimensionless_muon (A the MSE control, C_k v4_mse_cont_k*): C3 share 0.25, C4 share
# 0.50 (B the MSE share pairs of catalog_coverage step 14), C5 the pp-mixed set (B the pp bound arm, deciding PP4 on
# the classes it trains on pp events). C6 (the sign extension) is read by signext_compare.py warm
_HET_C = WARM["C1"]["C"]
_MSE_A = dict(MIX["control"], HD4=HD4_MSE["control"], HD5=HD5_MSE["control"])
_MSE_C = {533: ('FA-ccin2p3-ac7aa971', 'FA-ccin2p3-4ebff80f'), 1066: ('FA-ccin2p3-dd010506', 'FA-ccin2p3-39516188'), 2132: ('FA-ccin2p3-4c4a5069', 'FA-ccin2p3-9cab9b5b')}  # v4_mse_cont_k<K>[_seed2] (seeds 42, 43) per k, submitted 2026-11-06
_D7 = "plan warm_start step 7 (D283, D300), superset design passed 2026-11-04T21:51"
WARM["C2"] = {"W": {533: ('FA-ccin2p3-9a357af2', 'FA-ccin2p3-d26c43ba'), 1066: ('FA-ccin2p3-f968a51d', 'FA-ccin2p3-d88a5db8'), 2132: ('FA-ccin2p3-0fae36f3', 'FA-ccin2p3-74bb21ec')}, "C": _HET_C, "A": WARM["C1"]["A"], "deciding": ("PP4", "EE4"), "design": _D7,
              "B": dict(MIX_HET["default50"], HD4=HD4_HET["default50"], HD5=HD5_HET["default50"])}
# C8 k=2132 pair: held by the site gate's 60-min window at submission, added when submitted
WARM["C8"] = {"W": {533: ('FA-ccin2p3-0d2a042d', 'FA-ccin2p3-e4eac156'), 1066: ('FA-ccin2p3-1218b835', 'FA-ccin2p3-7c2c1c71'), 2132: ('FA-ccin2p3-9e4bcd8c', 'FA-ccin2p3-84059582')}, "C": _HET_C, "A": WARM["C1"]["A"], "deciding": ("HD5",), "design": _D7, "stats_check": True,
              "B": {r: ("FA-ccin2p3-5abe4a0e", "FA-ccin2p3-92207f6a") for r in ("PP4", "EE4", "held", "HD4", "HD5")}}
WARM["C3"] = {"W": {533: ('FA-ccin2p3-11938ca8', 'FA-ccin2p3-22afe10b'), 1066: ('FA-ccin2p3-a767f033', 'FA-ccin2p3-f9582bbb'), 2132: ('FA-ccin2p3-88551ced', 'FA-ccin2p3-ddf740e1')}, "C": _MSE_C, "A": _MSE_A, "deciding": ("PP4", "EE4"), "design": _D7,
              "B": dict(MIX["default25"], HD4=HD4_MSE["default25"], HD5=HD5_MSE["default25"])}
WARM["C4"] = {"W": {533: ('FA-ccin2p3-f1390aa8', 'FA-ccin2p3-21fb7bd7'), 1066: ('FA-ccin2p3-57172f65', 'FA-ccin2p3-f14bc159'), 2132: ('FA-ccin2p3-75f1d8c6', 'FA-ccin2p3-040c3448')}, "C": _MSE_C, "A": _MSE_A, "deciding": ("PP4", "EE4"), "design": _D7,
              "B": dict(MIX["default50"], HD4=HD4_MSE["default50"], HD5=HD5_MSE["default50"])}
WARM["C5"] = {"W": {533: ('FA-ccin2p3-455bbfbd', 'FA-ccin2p3-2e583c2b'), 1066: ('FA-ccin2p3-26f62d9c', 'FA-ccin2p3-6f6df1b8'), 2132: ('FA-ccin2p3-df7ec695', 'FA-ccin2p3-3794e155')}, "C": _MSE_C, "A": _MSE_A, "deciding": (("PP4", "trained"),), "trained_pp": True, "design": _D7,
              "stats_check": True,
              "B": {"PP4": BOUND["arm"], "EE4": BOUND_DEFAULT["EE4"], "HD4": BOUND_DEFAULT["HD"], "HD5": BOUND_DEFAULT["HD"],
                    "held": ("FA-ccin2p3-afc96598", "FA-ccin2p3-d26f13aa")}}
MATCH_HI, CARRY = 1.35, 0.90


def _side_table(runs):
    """{PP4, EE4, held, HD4, HD5: (run 42, run 43)}: a warm or control pair reads every axis from its own job"""
    return runs if isinstance(runs, dict) else {r: runs for r in ("PP4", "EE4", "held", "HD4", "HD5")}


def _warm_ratio(num, den, cls, ax_of, trained):
    """R(num/den) on each read as in mixture: per probe each side the geometric mean of its two seeds, class geometric
    mean of the probe ratios, median over classes, 95% class bootstrap widened by |ln R_seed| of the denominator pair
    (den seed 43 over seed 42). Reads: PP4 and EE4 (all classes; trained and not trained apart), the held-out classes
    on default-sampled events (PP4/EE4 probes of held-out classes, HD4, HD5; pooled and per axis) and catalog-sampled
    (H1-H7c pooled)"""
    num, den = _side_table(num), _side_table(den)
    table = {"num": {r: num[r] for r in ("PP4", "EE4", "held")}, "control": {r: den[r] for r in ("PP4", "EE4", "held")}}
    reads, missing = _reads(table)
    extra = {ax: {"num": num[ax], "control": den[ax]} for ax in ("HD4", "HD5")}
    missing += [f"{w} {ax} {run}" for ax, t in extra.items() for w, runs in t.items() for run in runs
                if read_of(run, ax) is None]
    if missing:
        sys.exit("warm: reads missing: " + "; ".join(missing))
    c = reads["control"]
    res = {}
    for r in ("PP4", "EE4", "held"):
        arm, ctl = side(reads["num"][r]), side(c[r])
        keys = sorted(set(arm) & set(ctl), key=str)
        parts = {"pooled": keys} if r == "held" else \
            {"all": keys, "trained": [k for k in keys if cls(probe_name(k)) in trained],
             "not_trained": [k for k in keys if cls(probe_name(k)) not in trained]}
        res[r] = {n: compare(arm, ctl, c[r][0], c[r][1], ks, cls) for n, ks in parts.items() if ks}
        res[r]["n_probes_unmatched"] = len(set(arm) ^ set(ctl))
    res["held_default"], _ = _held_default(reads, "num", cls, ax_of, extra)
    # each default-sampled held-out axis alone (HD4, HD5: C8's deciding HD5) and each catalog-sampled held-out axis
    # alone (res["held"][<axis>]: C8's H5)
    for ax, t in extra.items():
        a2, c2 = tuple(read_of(x, ax) for x in t["num"]), tuple(read_of(x, ax) for x in t["control"])
        arm, ctl = side(a2), side(c2)
        keys = sorted(set(arm) & set(ctl), key=str)
        res[ax] = {"all": compare(arm, ctl, c2[0], c2[1], keys, cls)} if keys else {}
    arm, ctl = side(reads["num"]["held"]), side(c["held"])
    for ax in HELD:
        ks = sorted((k for k in set(arm) & set(ctl) if k[0] == ax), key=str)
        if ks:
            res["held"][ax] = compare(arm, ctl, c["held"][0], c["held"][1], ks, cls)
    runs = [x for t in (num, den) for pair in t.values() for x in pair]
    return res, runs


def _main_read(res, r):
    """One read of a _warm_ratio result: PP4, EE4, HD4, HD5 (all classes); (read, part) such as ("PP4", "trained");
    held, held_default (pooled); a catalog-sampled held-out axis alone (H1 ... H7c)"""
    if isinstance(r, tuple):
        return res[r[0]][r[1]]
    if r in HELD:
        return res["held"][r]
    return res[r]["all"] if r in ("PP4", "EE4", "HD4", "HD5") else res[r]["pooled"]


def _read_label(r):
    return "_".join(r) if isinstance(r, tuple) else r


def warm(change="C1"):
    """plan warm_start's outcome rule for one change -> data/warm_start_<change>.json. Per k: R(W_k/B), R(W_k/C_k),
    R(W_k/A), R(C_k/A), and g_k = ln R(W_k/A) / ln R(B/A) per read (B/A the change's from-scratch gain over the parent;
    g_k the share of it the warm start reaches; R(C_k/A) what k more steps on the parent's own setup give). On the
    deciding reads (C1: PP4 and EE4, all classes): MATCH when the widened upper end of R(W_k/B) is <= 1.35 on each,
    SHORT when its widened lower end is above 1 on any, INCONCLUSIVE otherwise; the change carried when R(W_k/C_k) <=
    0.90 with the widened interval below 1 on each. YES: the smallest k with MATCH and the change carried, as
    k/4264; NO: no k matches (a NO at the restarted, untuned schedule)"""
    cls = class_of()
    ax_of = _held_axes(cls)
    d = WARM[change]
    # the "trained" part of PP4/EE4: the classes the parent trains (catalog_v4_base_train.yaml); C5 (the pp-mixed set):
    # the classes it trains on pp events (its pp13 entries), as the pp bound's own read (bound())
    if d.get("trained_pp"):
        trained = {cls(p["name"]) for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_pp_mixed_train.yaml")))["processes"]
                   if p["name"].startswith("pp13")}
    else:
        trained = {cls(p["name"]) for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_base_train.yaml")))["processes"]}
    hi_bar = d.get("match_hi", MATCH_HI)
    out = {"design": d.get("design", "plan warm_start, superset design passed 2026-11-04T21:51") + " (change %s)" % change,
           "runs": {"W": d["W"], "C": d["C"], "B": d["B"], "A": d["A"]}, "steps_from_scratch": WARM_STEPS,
           "rule": f"MATCH widened upper end of R(W_k/B) <= {hi_bar} on each deciding read {[_read_label(r) for r in d['deciding']]}; "
                   f"SHORT widened lower end > 1 on any; INCONCLUSIVE otherwise; carried R(W_k/C_k) <= {CARRY} with "
                   f"the widened interval below 1 on each; YES smallest k with MATCH and carried",
           "B_over_A": None, "k": {}}
    runs = []
    ba, rr = _warm_ratio(d["B"], d["A"], cls, ax_of, trained)
    out["B_over_A"] = ba
    runs += rr
    reads_named = ("PP4", "EE4", "held_default", "held") + tuple(r for r in d["deciding"]
                                                               if r not in ("PP4", "EE4", "held_default", "held"))
    for k in sorted(d["W"]):
        row = {"fraction": round(k / WARM_STEPS, 4)}
        pairs = {"W_over_B": (d["W"][k], d["B"]), "W_over_A": (d["W"][k], d["A"])}
        if k in d["C"]:
            pairs.update(W_over_C=(d["W"][k], d["C"][k]), C_over_A=(d["C"][k], d["A"]))
        pairs.update({f"W_over_{n}": (d["W"][k], r) for n, r in d.get("ref", {}).items()})
        for name, (num, den) in pairs.items():
            row[name], rr = _warm_ratio(num, den, cls, ax_of, trained)
            runs += rr
        row["g"] = {_read_label(r): round(math.log(_main_read(row["W_over_A"], r)["R"]) / math.log(_main_read(ba, r)["R"]), 4)
                    for r in reads_named}
        if "C_over_A" in row:
            row["g_control"] = {_read_label(r): round(math.log(_main_read(row["C_over_A"], r)["R"]) / math.log(_main_read(ba, r)["R"]), 4)
                                for r in reads_named}
        wb = [_main_read(row["W_over_B"], r)["widened"] for r in d["deciding"]]
        wc = [_main_read(row["W_over_C"], r) for r in d["deciding"]] if "W_over_C" in row else []
        row["outcome"] = "MATCH" if all(hi <= hi_bar for _, hi in wb) else \
            "SHORT" if any(lo > 1 for lo, _ in wb) else "INCONCLUSIVE"
        if d.get("keep_ref"):
            # D289's keep rule: the checkpoint start resolved better than the final-model start on each deciding read
            kr = [_main_read(row[f"W_over_{d['keep_ref']}"], r) for r in d["deciding"]]
            row["better_than_" + d["keep_ref"]] = all(x["R"] <= CARRY and x["widened"][1] < 1 for x in kr)
        row["carried"] = all(x["R"] <= CARRY and x["widened"][1] < 1 for x in wc) if wc else None   # None: no C_k (D286)
        # inside the equivalence bound and still resolved worse than B (widened lower end above 1): MATCH by the rule,
        # listed so a MATCH is never read as 'equal'
        row["W_resolved_worse_than_B"] = [_read_label(r) for r in reads_named if _main_read(row["W_over_B"], r)["widened"][0] > 1]
        out["k"][k] = row
    ks = sorted(d["W"])
    if not ks:          # not submitted yet: the from-scratch gain B/A only, nothing written
        print(json.dumps({"change": change, "W": "not submitted",
                          "B_over_A": {_read_label(r): (_main_read(ba, r)["R"], _main_read(ba, r)["widened"]) for r in reads_named}}))
        return
    yes = next((k for k in ks if out["k"][k]["outcome"] == "MATCH" and out["k"][k]["carried"] is not False), None)
    v = f"YES at k={yes} ({yes / WARM_STEPS:.3f} of the from-scratch steps)" if yes else \
        f"NO (no k up to {max(ks)} matches)" if \
        all(out["k"][k]["outcome"] == "SHORT" for k in ks) else "not decided (no MATCH; INCONCLUSIVE at some k)"
    out["blowups"] = blowups(runs)
    out["verdict_on_reads"] = v
    out["verdict"] = None if out["blowups"]["flagged"] else v
    if d.get("stats_check"):
        # the design's equivalence check on the parent_shared W runs (ws_stats_check.py): no verdict until it passes;
        # a failed W run's C_k is rerun with parent_shared before W and C are compared
        from ws_stats_check import status
        out["stats_check"] = status(change)
        if out["stats_check"] is not True:
            out["verdict"] = None
    json.dump(out, open(os.path.join(DATA, f"warm_start_{change}.json"), "w"), indent=1)
    print(json.dumps({"B_over_A": {_read_label(r): _main_read(ba, r)["R"] for r in reads_named},
                      "k": {k: {"outcome": x["outcome"], "carried": x["carried"], "g": x["g"],
                                "worse": x["W_resolved_worse_than_B"],
                                "keep": {n: v for n, v in x.items() if n.startswith("better_than_")},
                                **{n: {_read_label(r): (_main_read(x[n], r)["R"], _main_read(x[n], r)["widened"])
                                       for r in reads_named} for n in x if n.startswith(("W_over", "C_over"))}}
                            for k, x in out["k"].items()},
                      "flagged": out["blowups"]["flagged"], "spiked": out["blowups"]["spiked_recovered"],
                      "stats_check": out.get("stats_check"), "verdict": out["verdict"]}))


# step 43(1) loader control (design passed 2026-11-05T05:20): same arm v4_het_default25[_seed2], same commit, V100
LOADER = {"memmap": ("FA-ccin2p3-5b5215b6", "FA-ccin2p3-e184bbc5"), "rerun": ("FA-ccin2p3-ae39ee5a", "FA-ccin2p3-75ad7b20"),
          "arm_1031": {"PP4": MIX_HET["default25"]["PP4"], "EE4": MIX_HET["default25"]["EE4"],
                       "held": MIX_HET["default25"]["held"]}}
LOADER_GAP = math.log(1.39)          # the design's g: PP4 MSE w8_full over default25_1e4


def loader():
    """Step 43(1): per seed, R_L = memmap / in-memory rerun and R_0 = rerun / the 2026-10-31 in-memory arm, each the
    median over classes of the class's geometric mean of the probe ratios, on PP4 (trained classes), EE4 (trained),
    held-out default-sampled (the PP4/EE4 probes of held-out classes; these runs read no HD4/HD5) and, beside, the
    catalog-sampled held-out pool; best sigma-stage val_loss_no_reg per run; the design's verdict
    -> data/loader_control.json"""
    cls = class_of()
    ax_of = _held_axes(cls)
    cl = lambda k: cls(probe_name(k))
    def reads(run, ref_src):
        out = {}
        for r in ("PP4", "EE4"):
            d = read_of(ref_src(r), r) if ref_src else read_of(run, r)
            out[r + "_trained"] = {p: x for p, x in d.items() if cls(p) not in ax_of}
            out.setdefault("held_default", {}).update({(r, p): x for p, x in d.items() if cls(p) in ax_of})
        out["held_catalog"] = read_of(ref_src("held") if ref_src else run, "held")
        return out
    res = {"design": "plan catalog_coverage step 43(1), passed 2026-11-05T05:20", "runs": LOADER, "seeds": {}}
    lnL, ln0 = {}, {}
    for i, seed in enumerate((42, 43)):
        m, rr = reads(LOADER["memmap"][i], None), reads(LOADER["rerun"][i], None)
        a = reads(None, lambda r: LOADER["arm_1031"][r][i])
        sd = {}
        for q in m:
            ks = sorted(set(m[q]) & set(rr[q]) & set(a[q]), key=str)
            RL = float(np.median(list(class_ratios(m[q], rr[q], ks, cl).values())))
            R0 = float(np.median(list(class_ratios(rr[q], a[q], ks, cl).values())))
            mse = {w: float(np.median([gmean(v) for v in _by_class(d, ks, cl).values()]))
                   for w, d in (("memmap", m[q]), ("rerun", rr[q]), ("arm_1031", a[q]))}
            sd[q] = {"n_probes": len(ks), "n_classes": len({cl(k) for k in ks}), "R_L": round(RL, 4),
                     "R_0": round(R0, 4), "mse_class_median": {w: float(f"{x:.5g}") for w, x in mse.items()}}
            if q != "held_catalog":
                lnL[(seed, q)], ln0[(seed, q)] = math.log(RL), math.log(R0)
        best = {}
        for w in ("memmap", "rerun"):
            f = os.path.join(READS, f"{LOADER[w][i]}_val.json")
            b = json.load(open(f))["blocks"]
            best[w] = {"main_best": min(v for _, v in b[0]["vals"]), "sigma_best": min(v for _, v in b[-1]["vals"])}
        sd["val_loss_no_reg"] = best
        res["seeds"][seed] = sd
    s = max(abs(x) for x in ln0.values())
    g = LOADER_GAP
    reads3 = ("PP4_trained", "EE4_trained", "held_default")
    clear = all(abs(lnL[(sd, q)]) <= max(s, 0.05) and abs(lnL[(sd, q)]) < g / 3 for sd in (42, 43) for q in reads3)
    impl = all(lnL[(sd, "PP4_trained")] >= g / 2 and lnL[(sd, "PP4_trained")] > 2 * s and
               any(lnL[(sd, q)] >= g / 2 and lnL[(sd, q)] > 2 * s for q in reads3[1:]) for sd in (42, 43))
    v = "LOADING PATH CLEAR" if clear else "LOADING PATH IMPLICATED" if impl else "NOISE-LIMITED" if s >= g / 2 \
        else "UNRESOLVED"
    res["rule"] = {"s_max_abs_ln_R0": round(s, 4), "g": round(g, 4), "g_over_3": round(g / 3, 4), "g_over_2": round(g / 2, 4),
                   "max_abs_ln_RL": round(max(abs(x) for x in lnL.values()), 4)}
    res["blowups"] = blowups(list(LOADER["memmap"]) + list(LOADER["rerun"]))
    res["verdict"] = None if res["blowups"]["flagged"] else v
    res["verdict_on_reads"] = v
    json.dump(res, open(os.path.join(DATA, "loader_control.json"), "w"), indent=1)
    print(json.dumps(res))


# step 14's prior (D319): the share-0.50 warm starts (C2's W_k, seeds 42 and 43) read against the share-0.25 change
# from scratch (C1's B) and over C1's share-0.25 warm start at the same k: a richer retraining mixture toward the C1
# change; C_k as C1 (share 0)
WARM["C1s50"] = {"W": dict(WARM["C2"]["W"]), "C": WARM["C1"]["C"], "B": WARM["C1"]["B"], "A": WARM["C1"]["A"],
                 "ref": {"share25_k533": WARM["C1"]["W"][533]}, "deciding": ("PP4", "EE4"),
                 "design": "plan warm_start step 14 prior (D319): C2's share-0.50 warm starts against C1's B"}
# step 14 (D319; single seed, D315): warm starts toward the C1 change (share 0.25, het, k=533) at seed 42, each read
# against the same-commit default restart (v4_het_wsx_default_rerun) on the seed-42 class ratios; the study's runs are
# filled at submission ("" = not submitted). Arms of scripts/job_catalog_v3_short.sh v4_het_wsx_*; the share-0.50
# warm start at k=533 (C2's seed-42 run) is read beside from step 7.
S14 = {"control": "FA-ccin2p3-238e6f81",     # submitted 2026-11-07 @74aa4ce6 (limit 02:30); WSD parent FA-ccin2p3-dccb61e8
       "arms": {"shrinkperturb": "FA-ccin2p3-0f346464",  # rerun after b91cf7c6; 71852916 shrank again in its sigma stage
                 "share50_k267": "FA-ccin2p3-77e8bf2b",
                # resubmitted unchained (E3971; the chained 85dd4eff/a79d5b14/554f3ec5 never started)
                "wsdbranch25": "FA-ccin2p3-1354daee", "wsdbranch50": "FA-ccin2p3-4dc35a98",
                "wsdfinal_default": "FA-ccin2p3-9079718e", "share50_k533": "FA-ccin2p3-9a357af2"},
       "control_older": "FA-ccin2p3-341bf564",           # the step-1 default restart at seed 42 (older commit)
       "B": "FA-ccin2p3-9538e607", "B_HD": "FA-ccin2p3-4dca09e5"}
# each arm's widening over the control (design round 2): same parent and data -> the largest paired seed spread of
# the same-data pairs; a changed data share -> the share pair's; the WSD parent's arms -> the largest measured spread
# of any kind (same-data, share pair, parent swap)
S14_KIND = {"shrinkperturb": "same", "control_older": "same", "share50_k267": "data", "share50_k533": "data",
            "wsdbranch25": "parent", "wsdfinal_default": "parent", "wsdbranch50": "parent_data"}
# prior two-seed paired comparisons at k=533 (arm pair, control pair), for the paired seed spread |ln R43 - ln R42|
SEED_PAIRS = {"moment fill / default": ("C1f", "default"), "Muon damped / default": ("C1u", "default"),
              "AdamW damped / default": ("C1a", "default"), "every group damped / default": ("C1e", "default"),
              "MSE-tuned / untuned": ("C1m", "untuned"),
              "step-3198 default / control": ("C1r_step3198_default", "control"),
              "step-3198 continued / control": ("C1r_step3198_continue", "control"),
              "step-2132 continued / control": ("C1r_step2132_continue", "control"),
              "lr x3 / control": ("C1r_final_lr3x", "control")}
# pairs with a run whose validation loss left the scale (par:v4_warm_start_ramp: every group damped, seed 43): listed
# apart, not counted in the widening
SEED_PAIRS_BLOWUP = {"every group damped / default"}


def _one(num, den, read, cls):
    """Seed-level R(num/den) on one read: per probe the ratio of the two runs' MSE, class geometric mean, median over
    classes; with the class ratios (for a bootstrap)"""
    a, c = read_of(num, read), read_of(den, read)
    if a is None or c is None:
        return None, None
    keys = sorted(set(a) & set(c), key=str)
    cr = class_ratios(a, c, keys, lambda k: cls(probe_name(k)))
    return float(np.median(list(cr.values()))), cr


def seed_pairs():
    """The paired seed spread of the step-14 single-seed reads -> data/ws_seed_pairs.json: for each prior k=533
    arm/control pair at seeds 42 and 43 (SEED_PAIRS, plus share 0.50 over share 0.25, C2/C1 at 533), the seed-level
    R(arm/control) per seed on PP4 and EE4 and d = |ln R43 - ln R42|; the median d per read is the widening of the
    single-seed interval; beside, the seed-42 R against the two-seed R"""
    cls = class_of()
    rows = {}
    pairs = {n: (WARM[c]["W"][533], WARM[c]["ref"][r]) for n, (c, r) in SEED_PAIRS.items()}
    pairs["share 0.50 / share 0.25"] = (WARM["C2"]["W"][533], WARM["C1"]["W"][533])
    for n, (arm, ctl) in pairs.items():
        row = {}
        for read in ("PP4", "EE4"):
            r42, _ = _one(arm[0], ctl[0], read, cls)
            r43, _ = _one(arm[1], ctl[1], read, cls)
            if r42 is None or r43 is None:
                continue
            row[read] = {"R42": round(r42, 4), "R43": round(r43, 4), "d": round(abs(math.log(r43 / r42)), 4)}
        rows[n] = row
    clean = {n: v for n, v in rows.items() if n not in SEED_PAIRS_BLOWUP}
    out = {"what": "paired seed spread d = |ln R(arm/control) seed 43 - ln R(arm/control) seed 42|, prior k=533 pairs",
           "pairs": rows, "blowup_pairs": sorted(SEED_PAIRS_BLOWUP),
           "max_d_no_blowup": {r: round(float(np.max([v[r]["d"] for v in clean.values() if r in v])), 4)
                               for r in ("PP4", "EE4")},
           "median_d": {r: round(float(np.median([v[r]["d"] for v in rows.values() if r in v])), 4) for r in ("PP4", "EE4")},
           "max_d": {r: round(float(np.max([v[r]["d"] for v in rows.values() if r in v])), 4) for r in ("PP4", "EE4")}}
    same = {n: v for n, v in clean.items() if n != "share 0.50 / share 0.25"}
    out["max_d_same_data"] = {r: round(float(np.max([v[r]["d"] for v in same.values() if r in v])), 4)
                              for r in ("PP4", "EE4")}
    out["share_pair_d"] = {r: rows["share 0.50 / share 0.25"][r]["d"] for r in ("PP4", "EE4")}
    # a restart from another pretraining of the same setup: the checkpoint rerun parent's default restart over the
    # original parent's, per seed |ln R| (the spread a run from a different parent carries at one seed)
    ps = {}
    for read in ("PP4", "EE4"):
        v = [_one(a, b, read, cls)[0] for a, b in zip(WARM["C1r_ctrl"]["W"][533],
                                                      WARM["C1r_ctrl"]["ref"]["default_original_parent"])]
        ps[read] = {"R42": round(v[0], 4), "R43": round(v[1], 4), "max_abs_lnR": round(max(abs(math.log(x)) for x in v), 4)}
    out["parent_swap"] = ps
    json.dump(out, open(os.path.join(DATA, "ws_seed_pairs.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def warm_single():
    """Step 14's reads -> data/warm_start_S14.json. Per arm, seed 42: R(arm/control) on PP4 and EE4 (deciding), held
    (catalog-sampled held-out, pooled), HD4, HD5; median over classes of the class geometric mean of probe ratios; 95%
    class bootstrap widened on each side by the spread of the arm's kind (S14_KIND, data/ws_seed_pairs.json: same
    parent and data max_d_same_data; changed share share_pair_d; the WSD parent's arms the largest of these and of
    parent_swap; per read, held, HD4, HD5 the larger of PP4's and EE4's); the branch over the WSD final
    model takes the same-data width.
    Beside: R(arm/B) at seed 42 (B's seed-42 run; biased low, seed 42 is the favourable seed of the default restart),
    and the estimate R(arm/B) ~ R(arm/control) x R(W_533/B), the two-seed step-1 read (data/warm_start_C1.json).
    Rules fixed in the design: BETTER when the widened upper end is below 1 on PP4 or EE4 and above 1.05 on neither;
    WORSE when a lower end is above 1; else NOT BETTER; MATCH-estimate when the estimated R(arm/B) is <= 1.35 on both;
    a BETTER arm whose held widened lower end is above 1.05 is flagged (held_flag). Readings: branch = R(wsdbranch25 /
    wsdfinal_default) by the same rule (BRANCH works when it and wsdbranch25 over the control are both BETTER); parent =
    wsdfinal_default within 5% of the control on both reads (R in [1/1.05, 1.05]); exposure = share50_k267's widened
    interval contains 1 on PP4 and EE4 and share50_k533 is BETTER."""
    cls = class_of()
    sp = json.load(open(os.path.join(DATA, "ws_seed_pairs.json")))
    kinds = {"same": sp["max_d_same_data"], "data": sp["share_pair_d"],
             "parent_swap": {r: sp["parent_swap"][r]["max_abs_lnR"] for r in ("PP4", "EE4")}}
    # the WSD parent differs from the control's parent in its schedule too, which no prior pair measures: its arms
    # take the largest measured spread of any kind
    kinds["parent"] = {r: max(kinds[k][r] for k in ("same", "data", "parent_swap")) for r in ("PP4", "EE4")}
    kinds["parent_data"] = kinds["parent"]

    def widths(kind):
        k = kinds[kind]
        return dict(k, **{r: max(k.values()) for r in ("held", "HD4", "HD5")})
    c1 = json.load(open(os.path.join(DATA, "warm_start_C1.json")))["k"]["533"]["W_over_B"]
    est = {r: c1[r]["all"]["R"] for r in ("PP4", "EE4")}
    ctl = S14["control"]
    if not ctl:
        sys.exit("warm_single: S14 control not filled")

    def read_pair(run, den, kind, bref=True):
        w = widths(kind)
        row = {"run": run, "over": den, "widening_kind": kind, "widening": w}
        for read in ("PP4", "EE4", "held", "HD4", "HD5"):
            r, cr = _one(run, den, read, cls)
            if r is None:
                row[read] = None
                continue
            res = r_with_interval(cr, math.exp(w[read]))
            if bref:
                rb, _ = _one(run, S14["B_HD"] if read in ("HD4", "HD5") else S14["B"], read, cls)
                res["R_over_B_seed42"] = None if rb is None else round(rb, 4)
                if read in est:
                    res["R_over_B_estimate"] = round(r * est[read], 4)
            row[read] = res
        dec = [row[r] for r in ("PP4", "EE4")]
        if all(x is not None for x in dec):
            hi = [x["widened"][1] for x in dec]; lo = [x["widened"][0] for x in dec]
            row["outcome"] = ("BETTER" if min(hi) < 1 and max(hi) <= 1.05 else
                              "WORSE" if max(lo) > 1 else "NOT BETTER")
            if bref:
                row["match_estimate"] = all(x["R_over_B_estimate"] <= 1.35 for x in dec)
            row["held_flag"] = bool(row["outcome"] == "BETTER" and row["held"] is not None
                                    and row["held"]["widened"][0] > 1.05)
        return row

    out = {"design": "plan warm_start step 14 (D319, D315)", "control": ctl, "widening_kinds": kinds,
           "estimate_factor_W533_over_B": est, "arms": {}, "readings": {}}
    arms = dict(S14["arms"], control_older=S14["control_older"])
    for name, run in arms.items():
        if not run:
            continue
        out["arms"][name] = row = read_pair(run, ctl, S14_KIND[name])
        print(name, run, {r: (row[r] or {}).get("R") for r in ("PP4", "EE4", "held", "HD4", "HD5")}, row.get("outcome"),
              "match_estimate" if row.get("match_estimate") else "")
    A, rd = S14["arms"], out["readings"]
    if A["wsdbranch25"] and A["wsdfinal_default"]:
        b = read_pair(A["wsdbranch25"], A["wsdfinal_default"], "same", bref=False)
        rd["branch_over_wsdfinal"] = b
        rd["BRANCH_works"] = b.get("outcome") == "BETTER" and out["arms"]["wsdbranch25"].get("outcome") == "BETTER"
    if A["wsdfinal_default"] and out["arms"]["wsdfinal_default"].get("PP4"):
        f = out["arms"]["wsdfinal_default"]
        rd["WSD_parent_as_good"] = all(1 / 1.05 <= f[r]["R"] <= 1.05 for r in ("PP4", "EE4"))
    if A["share50_k267"] and out["arms"]["share50_k267"].get("PP4"):
        e = out["arms"]["share50_k267"]
        rd["EXPOSURE_supported"] = (all(e[r]["widened"][0] <= 1 <= e[r]["widened"][1] for r in ("PP4", "EE4"))
                                    and out["arms"]["share50_k533"].get("outcome") == "BETTER")
    print("readings", {k: v for k, v in rd.items() if not isinstance(v, dict)},
          (rd.get("branch_over_wsdfinal") or {}).get("outcome"))
    json.dump(out, open(os.path.join(DATA, "warm_start_S14.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("run"); e.add_argument("log"); e.add_argument("axis", nargs="?", default="PP4")
    e.add_argument("--allow-unchecked", action="store_true")
    v = sub.add_parser("val")
    v.add_argument("run"); v.add_argument("log")
    sub.add_parser("bound")
    sub.add_parser("bound_default")
    hd = sub.add_parser("heldout_default")
    hd.add_argument("--het", action="store_true")
    m = sub.add_parser("mixture")
    m.add_argument("--partial", action="store_true")
    m.add_argument("--het", action="store_true")
    sub.add_parser("width")
    sub.add_parser("arms")
    sub.add_parser("length")
    sub.add_parser("loader")
    w = sub.add_parser("warm")
    w.add_argument("change", nargs="?", default="C1")
    sub.add_parser("seed_pairs")
    sub.add_parser("warm_single")
    ep = sub.add_parser("epochs")
    ep.add_argument("run"); ep.add_argument("log")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.run, a.log, a.axis, a.allow_unchecked)
    elif a.cmd == "val":
        val(a.run, a.log)
    elif a.cmd == "mixture":
        mixture(a.partial, a.het)
    elif a.cmd == "bound_default":
        bound_default()
    elif a.cmd == "heldout_default":
        heldout_default(a.het)
    elif a.cmd == "width":
        width()
    elif a.cmd == "arms":
        arms()
    elif a.cmd == "length":
        length()
    elif a.cmd == "loader":
        loader()
    elif a.cmd == "warm":
        warm(a.change)
    elif a.cmd == "seed_pairs":
        seed_pairs()
    elif a.cmd == "warm_single":
        warm_single()
    elif a.cmd == "epochs":
        epochs(a.run, a.log)
    else:
        bound()


if __name__ == "__main__":
    main()
