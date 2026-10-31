#!/usr/bin/env python3
"""Where a training job's wall time went, read from its SLURM log (`site logs <run> -n 3000000 > <file>`).

Per log: the job start (the SLURM header's Date) to the first log line; per training stage (an experiment that logs
"Starting to train": the main run, a het arm's sigma stage) its data phase (the stage's first log line to "Starting to
train"; the default-share composition, "Recipe data source" to "composed at load", apart), its training ("Starting to
train" to "Finished training", with the logged average s/step, validations included; the step rate between two
"Finished iteration" lines with no validation between them; the validations counted and their time derived as
(training time - steps x that rate) / validations), its own evaluation ("Finished training" to the stage's last log
line, with the "Collected N predictions in X s" forward passes), and the gap to the next stage; then the zero-shot reads
of scripts/job_catalog_v3_short.sh (one tools/zero_shot_parent.py process per held-out set, one experiment built per
probe): per set the probes built, its wall (the last log line before the set's V3_RECIPE_CHECK to the set's last log
line, or to the scheduler's cancel), s per probe, the start-up before its first probe, and per probe the median
"Creating new experiment" to "Instantiated model" (the probe's data load and model build) and "Instantiated model" to
the next probe's "Creating new experiment" (weights, physics check, forward pass, scoring, saving, next start).
Timestamps are the logger's [YYYY-mm-dd HH:MM:SS I] lines (1 s resolution, the node's local time, as the header's Date
is); what has no marker is "not found". The job's end is the scheduler's CANCELLED line when there is one, else the
last log line.

    python tools/job_wall_split.py <log> [<log> ...] [--json out.json]

Prints one JSON line per log."""
import argparse, json, re, statistics
from datetime import datetime

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) I\] (.*)$")
DATE = re.compile(r"^\* Date:\s+(.+?)\s*$")
CANCEL = re.compile(r"CANCELLED AT (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) DUE TO (.+?) \*\*\*")
AXIS = re.compile(r"^V3_RECIPE_CHECK (\S+): (\d+) recipes")
ZS = re.compile(r"^(\S+) ZERO_SHOT \{")
START = re.compile(r"Starting to train for (\d+) iterations.*validating every (\d+) steps")
FIN = re.compile(r"Finished training: (\d+) iterations.* in ([\d.]+)min \(avg ([\d.]+)s/iter\)")
ITER = re.compile(r"Finished iteration (\d+) after ([\d.]+)s")
NF = "not found"


def _ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _header_date(s):
    # "Sat Oct 31 12:24:41 AM CET 2026": the zone is dropped (the log lines carry the same local time)
    p = s.split()
    return datetime.strptime(" ".join(p[:5] + p[6:7]), "%a %b %d %I:%M:%S %p %Y")


def _med(x):
    return round(statistics.median(x), 2) if x else NF


def _sec(a, b):
    return (b - a).total_seconds()


def _stage(blk, prev_end):
    s = {}
    t0 = blk[0][1]
    if prev_end is not None:
        s["gap_from_previous_stage_s"] = _sec(prev_end, t0)
    st = next(e for e in blk if START.search(e[2]))
    n_it, every = map(int, START.search(st[2]).groups())
    s["iterations"], s["validate_every"] = n_it, every
    s["data_phase_s"] = _sec(t0, st[1])
    rds = next((e for e in blk if e[2].startswith("Recipe data source")), None)
    comp = next((e for e in blk if e[2].startswith("composed at load")), None)
    s["composition_s"] = _sec(rds[1], comp[1]) if (comp and rds) else NF
    if comp:
        s["composition"] = comp[2][len("composed at load: "):].split(" (")[0]
    fin = next((e for e in blk if FIN.search(e[2])), None)
    if fin is None:
        s["training"] = NF
        return s
    _, mins, avg = FIN.search(fin[2]).groups()
    train_s = float(mins) * 60
    s["train_s"] = round(train_s, 1)
    s["avg_s_per_step_incl_val"] = float(avg)
    nval = sum(1 for e in blk if e[2].startswith("Val loss (combined)"))
    s["validations"] = nval
    its = [tuple(map(float, ITER.search(e[2]).groups())) for e in blk if ITER.search(e[2])]
    # two progress lines with no validation step between them: steps n1+1..n2 (1-based), a validation after step k when
    # k % every == 0
    pure = [(t2 - t1) / (n2 - n1) for (n1, t1), (n2, t2) in zip(its, its[1:])
            if n2 > n1 and int(n2) // every == int(n1) // every]
    if pure:
        rate = statistics.median(pure)
        s["s_per_step_no_val"] = round(rate, 4)
        s["s_per_step_no_val_windows"] = len(pure)
        s["steps_s"] = round(n_it * rate, 1)
        s["s_per_validation_derived"] = round((train_s - n_it * rate) / nval, 2) if nval else NF
    else:
        s["s_per_step_no_val"] = NF
        s["s_per_validation_derived"] = NF
    s["stage_eval_s"] = _sec(fin[1], blk[-1][1])
    # its parts: forward passes and metrics ("Finished training" to "Creating plots"), plots ("Creating plots" to "GPU
    # RAM information"), the rest (checkpoint compression, experiment end)
    cp = next((e for e in blk if e[2].startswith("Creating plots")), None)
    gr = next((e for e in blk if e[2].startswith("GPU RAM information")), None)
    s["stage_eval_metrics_s"] = _sec(fin[1], cp[1]) if cp else NF
    s["stage_eval_plots_s"] = _sec(cp[1], gr[1]) if (cp and gr) else NF
    s["stage_eval_rest_s"] = _sec(gr[1], blk[-1][1]) if gr else NF
    s["stage_eval_forward_s"] = [float(re.search(r"in ([\d.]+)s", e[2]).group(1))
                                 for e in blk if e[2].startswith("Collected ")]
    return s


def split(path):
    lines = open(path, errors="replace").read().splitlines()
    out = {"log": path}
    ev, axes, zs = [], [], []  # (line, datetime, message) of every logger line; (line, set, recipes) per read set;
    #                            (line, set) per ZERO_SHOT record
    start = end = gpu = params = None
    for i, l in enumerate(lines):
        m = TS.match(l)
        if m:
            ev.append((i, _ts(m.group(1)), m.group(2)))
            if params is None and m.group(2).startswith("Instantiated model"):
                params = int(re.search(r"with (\d+) learnable", m.group(2)).group(1))
            continue
        if start is None and (m := DATE.match(l)):
            start = _header_date(m.group(1))
        elif (m := ZS.match(l)):
            zs.append((i, m.group(1)))
        elif (m := AXIS.match(l)):
            axes.append((i, m.group(1), int(m.group(2))))
        elif (m := CANCEL.search(l)):
            end = (datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S"), m.group(2))
        elif gpu is None and l.startswith("RUN_SUMMARY") and '"gpu"' in l:
            gpu = re.search(r'"gpu": "([^"]*)"', l).group(1)
    out.update(gpu=gpu or NF, params=params or NF, v3_short_done=any(l.startswith("V3_SHORT_DONE") for l in lines))
    if not ev:
        out["error"] = "no timestamped log lines"
        return out
    t_first, t_last = ev[0][1], ev[-1][1]
    out["job_start"] = str(start) if start else NF
    out["start_to_first_log_s"] = _sec(start, t_first) if start else NF
    out["end"] = {"at": str(end[0]), "by": end[1]} if end else {"at": str(t_last), "by": "last log line"}
    t_end = end[0] if end else t_last
    out["total_s"] = _sec(start or t_first, t_end)

    creates = [k for k, e in enumerate(ev) if e[2].startswith("Creating new experiment")]
    if not axes and zs:
        # a job script without V3_RECIPE_CHECK lines: a set starts after the previous set's last ZERO_SHOT record (the
        # first set just before the first experiment after the last training stage); recipes not logged. The H5 sign
        # scoring's experiments (tools/signed_ab_score.py, no ZERO_SHOT record) then count in the next set
        order = list(dict.fromkeys(a for _, a in zs))
        last = {a: max(i for i, b in zs if b == a) for a in order}
        first = min(i for i, _ in zs)
        # the last training stage's start; an eval-only job (REREAD, no training stage): the log's start
        ls = max((e[0] for e in ev if e[0] < first and START.search(e[2])), default=-1)
        b0 = min(ev[k][0] for k in creates if ev[k][0] > ls) - 1                 # just before the first probe's experiment
        axes = [(b0 if n == 0 else last[order[n - 1]], a, NF) for n, a in enumerate(order)]
    first_axis_line = axes[0][0] if axes else float("inf")
    stages, probes = [], []
    for a, b in zip(creates, creates[1:] + [len(ev)]):
        blk = ev[a:b]
        if any(START.search(e[2]) for e in blk):
            stages.append(blk)
        elif blk[0][0] > first_axis_line:
            probes.append(blk)
    out["stages"], prev_end = [], None
    for blk in stages:
        out["stages"].append(_stage(blk, prev_end))
        prev_end = blk[-1][1]
    out["reads"] = []
    for j, (li, ax, nrec) in enumerate(axes):
        nxt = axes[j + 1][0] if j + 1 < len(axes) else float("inf")
        pb = [p for p in probes if li < p[0][0] < nxt]
        r = {"set": ax, "recipes": nrec, "probes_read": sum(1 for i, b in zs if b == ax and li < i < nxt),
             "experiments_built": len(pb)}
        before = [e for e in ev if e[0] < li]
        inside = [e for e in ev if li < e[0] < nxt]
        if not (pb and before and inside):
            r["wall_s"] = NF
            out["reads"].append(r)
            continue
        t_from = before[-1][1]
        t_to = end[0] if (nxt == float("inf") and end) else inside[-1][1]
        r["wall_s"] = _sec(t_from, t_to)
        r["s_per_experiment"] = round(r["wall_s"] / len(pb), 2)
        r["set_start_s"] = _sec(t_from, pb[0][0][1])
        starts = [p[0][1] for p in pb]
        r["median_probe_s"] = _med([_sec(a, b) for a, b in zip(starts, starts[1:])])
        built = [(p, next((e for e in p if e[2].startswith("Instantiated model")), None)) for p in pb]
        r["median_build_s"] = _med([_sec(p[0][1], im[1]) for p, im in built if im])
        r["median_after_build_s"] = _med([_sec(im[1], q[0][1]) for (p, im), q in zip(built, pb[1:]) if im])
        out["reads"].append(r)
    if axes:
        out["reads_total_s"] = sum(r["wall_s"] for r in out["reads"] if isinstance(r["wall_s"], float))
        out["probes_read_total"] = sum(r["probes_read"] for r in out["reads"])
        out["experiments_built_total"] = sum(r["experiments_built"] for r in out["reads"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--json", default=None, help="write the list of records here")
    a = ap.parse_args()
    recs = [split(p) for p in a.logs]
    for r in recs:
        print(json.dumps(r))
    if a.json:
        json.dump(recs, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
