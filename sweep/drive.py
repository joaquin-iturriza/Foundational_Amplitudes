#!/usr/bin/env python3
"""
drive.py  —  Run a DyHPO sweep across sites from one place.

The classic path (generate_sweep + sweep_manager) keeps the search state in a file that
every trial locks on ONE site's filesystem, so a sweep can never use more than one site,
and pool prebuilds only exist as SLURM jobs. Here the trials are stateless and the state
lives with this driver, in the local checkout:

    sweeps_local/<sweep_name>/dyhpo_state.pkl      the sampler (single writer: this process)
    sweeps_local/<sweep_name>/driver_state.json    every trial: site, run id, hp, outcome
    sweeps_local/<sweep_name>/results/*.json       what each trial reported
    sweeps_local/<sweep_name>/summary.txt

Loop:  suggest() -> place with `site pick` (fitting free GPUs, caps, waits, per site)
       -> `site submit <site> FA sweep/trial_job.sh -- --detached --hp-idx ... --hp k=v ...`
       -> `site poll` -> read the trial's RESULT_JSON line from its log -> observe().
Before the first trial on a site, the recipe's pools are checked there and, if missing,
built by a CPU job (`scripts/prebuild_recipes.sh` via `site submit`) — SLURM or HTCondor.

Usage (from the checkout, code committed and pushed; the driver syncs each site once):
    python sweep/drive.py --config sweep/sweep_config_X.yaml [--config ...]
        [--n-trials N] [--parallel P] [--mem 8G] [--cpus 4] [--hours 0.5]
        [--sites ccin2p3 lxplus] [--allow-jeanzay] [--poll 60] [--once] [--dry-run]
Resumable: run it again and it continues from driver_state.json.
Limits: detached trials always cold-start (no cross-fidelity warm start), which is how
nearly every sweep runs anyway (single fidelity).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The sampler needs torch + scikit-learn; the laptop's system python has neither. Run from
# the project's driver env (sweep/driver_env.sh builds it) whenever this interpreter lacks it.
try:
    import torch  # noqa: F401
except ImportError:
    _venv_py = os.path.join(ROOT, ".venv-driver", "bin", "python")
    if not os.path.exists(_venv_py) and not os.environ.get("DRIVE_NO_REEXEC"):
        import subprocess as _sp
        _sp.run(["bash", os.path.join(ROOT, "sweep", "driver_env.sh")], check=True)
    _venv = os.path.dirname(os.path.dirname(_venv_py))
    if os.path.exists(_venv_py) and os.path.abspath(sys.prefix) != os.path.abspath(_venv):
        os.execv(_venv_py, [_venv_py] + sys.argv)      # (the venv's python is a symlink: compare prefixes)
    raise SystemExit("sweep/drive.py: no torch here and no .venv-driver; run bash sweep/driver_env.sh")
import siteconf                                   # noqa: E402
from sweep.dyhpo_sampler import DyHPOSampler     # noqa: E402
from sweep.generate_sweep import init_sampler    # noqa: E402

SITE = os.path.expanduser("~/.local/bin/site")
PROJECT = "FA"
JOB = "sweep/trial_job.sh"
PREBUILD = "scripts/prebuild_recipes.sh"
FINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "REMOVED")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def site(*args, timeout=600, check=True, quiet=False):
    cmd = [SITE, "--timeout", str(int(min(timeout, 3600)))] + [str(a) for a in args]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       timeout=timeout + 60)
    if p.returncode and check:
        raise RuntimeError("site %s failed (%d):\n%s" % (" ".join(args[:3]), p.returncode, p.stdout.rstrip()))
    if not quiet and p.returncode:
        log("site %s -> rc %d: %s" % (args[0], p.returncode, p.stdout.strip()[-200:]))
    return p.stdout


def hhmmss(hours):
    h = float(hours)
    return "%02d:%02d:00" % (int(h), int(round((h - int(h)) * 60)))


def hours_of(t):
    parts = [int(x) for x in str(t).split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] + parts[1] / 60.0 + parts[2] / 3600.0


class Sweep:
    def __init__(self, config_path, a):
        self.config_rel = os.path.relpath(os.path.abspath(config_path), ROOT)
        with open(config_path) as f:
            self.cfg = siteconf.resolve(yaml.safe_load(f))
        self.cfg.setdefault("fixed_params", {}).setdefault("seed", 42)
        self.name = self.cfg["sweep_name"]
        self.dir = os.path.join(ROOT, "sweeps_local", self.name)
        os.makedirs(os.path.join(self.dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "dyhpo_surrogate"), exist_ok=True)
        self.state_path = os.path.join(self.dir, "dyhpo_state.pkl")
        self.out_path = os.path.join(self.dir, "dyhpo_surrogate")
        if not os.path.exists(self.state_path):
            init_sampler(self.cfg, self.dir, self.dir)
        self.sampler = DyHPOSampler.load(self.state_path, self.out_path, force_cpu=True)
        self.n_trials = int(a.n_trials or self.cfg.get("n_trials", 40))
        dy = self.cfg.get("dyhpo", {}) or {}
        self.n_startup = int(dy.get("n_startup", 3))
        cl = self.cfg.get("cluster", {}) or {}
        self.mem = a.mem or cl.get("mem") or cl.get("request_memory") or "8G"
        self.cpus = int(a.cpus or cl.get("cpus_per_task") or 4)
        self.hours = float(a.hours) if a.hours else hours_of(cl.get("time", "01:00:00"))
        self.time = hhmmss(self.hours * 1.5 + 0.1)           # wall limit: margin over the estimate
        fp = self.cfg.get("fixed_params", {})
        self.spec = None
        if str(fp.get("data.source", "files")) == "recipes" and fp.get("data.processes_file"):
            spec = str(fp["data.processes_file"])
            pd = self.cfg["paths"]["project_dir"].rstrip("/") + "/"
            self.spec = spec[len(pd):] if spec.startswith(pd) else os.path.relpath(spec, ROOT)
        self.seed = int(fp.get("data.seed", 42))
        self.sp = os.path.join(self.dir, "driver_state.json")
        self.st = json.load(open(self.sp)) if os.path.exists(self.sp) else {"trials": {}, "prebuild": {}}

    # ---- persistence
    def save(self):
        self.sampler.save(self.state_path)
        tmp = self.sp + ".tmp"
        json.dump(self.st, open(tmp, "w"), indent=1)
        os.replace(tmp, self.sp)

    # ---- bookkeeping
    def trials(self, *status):
        return [t for t in self.st["trials"].values() if not status or t["status"] in status]

    def issued(self):
        return len(self.st["trials"])

    def finished(self):
        return len(self.trials("observed", "failed", "lost"))

    def in_flight(self):
        return self.trials("submitted")

    def done(self):
        return self.finished() >= self.n_trials

    def may_issue(self, parallel):
        if self.issued() >= self.n_trials:
            return 0
        room = parallel - len(self.in_flight())
        if room <= 0:
            return 0
        # wave rule (as sweep_manager's waves / the Condor DAG): the startup trials run
        # first; guided suggestions wait until every startup trial has been observed
        if self.issued() >= self.n_startup and self.finished() < self.n_startup:
            return 0
        if self.issued() < self.n_startup:
            room = min(room, self.n_startup - self.issued())
        return min(room, self.n_trials - self.issued())

    # ---- data on a site
    def data_ready(self, s, a):
        """True when the recipe's pools exist on site s; submits the CPU prebuild once."""
        if not self.spec:
            return True
        pb = self.st["prebuild"].get(s)
        if pb == "ok":
            return True
        if isinstance(pb, dict):                       # a prebuild run is in progress
            out = site("poll", pb["rid"], timeout=120, check=False)
            m = re.search(r"->\s+(\S+)", out)
            state = m.group(1) if m else "UNKNOWN"
            if state == "COMPLETED":
                self.st["prebuild"][s] = "ok"; self.save()
                log("%s: pools built on %s" % (self.name, s))
                return True
            if state in FINAL:
                log("%s: prebuild %s on %s ended %s; will retry" % (self.name, pb["rid"], s, state))
                self.st["prebuild"][s] = None; self.save()
            return False
        chk = ("python -c \"import sys; sys.path.insert(0,'sweep'); from sweep_manager import _spec_fully_cached as c; "
               "print('CACHED' if c(%r, %d) else 'MISSING')\"" % (self.spec, self.seed))
        out = site("run", s, PROJECT, "--", chk, timeout=300, check=False)
        if "CACHED" in out:
            self.st["prebuild"][s] = "ok"; self.save()
            return True
        if "MISSING" not in out:
            log("%s: could not check pools on %s: %s" % (self.name, s, out.strip()[-160:]))
            return False
        if a.dry_run:
            log("[dry-run] would submit prebuild of %s on %s" % (self.spec, s)); return False
        cpus = 8 if s == "lxplus" else 32
        out = site("submit", s, PROJECT, PREBUILD, "--no-sync", "--item", a.item, "--note", "prebuild %s" % self.name,
                   "--hdr=--cpus-per-task=%d" % cpus, "--hdr=--time=04:00:00",
                   "--", self.spec, "--seed", str(self.seed), timeout=300)
        m = re.search(r"\brun\s+(\S+)", out)
        if not m:
            raise RuntimeError("prebuild submit gave no run id:\n" + out)
        self.st["prebuild"][s] = {"rid": m.group(1), "ts": int(time.time())}; self.save()
        log("%s: pools missing on %s -> prebuild %s submitted (CPU job)" % (self.name, s, m.group(1)))
        return False

    # ---- one trial out
    def submit(self, s, a):
        hp_idx, hp, t_steps = self.sampler.suggest()
        idx = self.issued()
        args = ["submit", s, PROJECT, JOB, "--no-sync", "--item", a.item,
                "--note", "%s hp%04d t%d" % (self.name, hp_idx, t_steps),
                "--hdr=--mem=%s" % self.mem, "--hdr=--time=%s" % self.time,
                "--hdr=--cpus-per-task=%d" % self.cpus,
                "--hdr=--job-name=%s_%04d" % (self.name[:20], idx)]
        if s == "jeanzay":
            args += ["--est-gpu-hours", "%.2f" % self.hours, "--allow-jeanzay"]
        args += ["--", "--sweep-config", self.config_rel, "--detached", "--trial-idx", str(idx),
                 "--hp-idx", str(hp_idx), "--t-steps", str(t_steps)]
        for k, v in hp.items():
            v = v.item() if hasattr(v, "item") else v            # numpy scalars -> python
            args += ["--hp", "%s=%s" % (k, repr(float(v)) if isinstance(v, float) else v)]
        if a.dry_run:
            log("[dry-run] site " + " ".join(map(str, args)))
            self.sampler.report_failure(hp_idx)        # give the candidate back
            return
        out = site(*args, timeout=300)
        m = re.search(r"\brun\s+(\S+)", out)
        if not m:
            self.sampler.report_failure(hp_idx)
            raise RuntimeError("submit gave no run id:\n" + out)
        self.st["trials"][str(idx)] = {"idx": idx, "rid": m.group(1), "site": s, "hp_idx": hp_idx,
                                       "hp": hp, "t_steps": t_steps, "status": "submitted",
                                       "submitted": int(time.time())}
        self.save()
        log("%s: trial %d -> %s  (hp%04d t%d, run %s)" % (self.name, idx, s, hp_idx, t_steps, m.group(1)))

    # ---- results back
    def collect(self, a):
        fl = self.in_flight()
        if not fl:
            return
        out = site("poll", *[t["rid"] for t in fl], timeout=120 + 20 * len(fl), check=False)
        states = dict(re.findall(r"^\s*(\S+)\s+\S+\s+->\s+(\S+)", out, re.M))
        for t in fl:
            stt = states.get(t["rid"], "")
            if stt not in FINAL:
                continue
            res = self.result_of(t)
            if res and not res.get("failed"):
                self.sampler.observe(t["hp_idx"], t["t_steps"], float(res["observe_loss"]),
                                     res.get("proc_val_losses"))
                t.update(status="observed", val_loss=res.get("val_loss"), observe_loss=res["observe_loss"],
                         finished=int(time.time()), sched_state=stt)
                json.dump(res, open(os.path.join(self.dir, "results", "hp%04d_t%d.json" % (t["hp_idx"], t["t_steps"])), "w"), indent=1)
                log("%s: trial %d on %s -> val_loss %.6g" % (self.name, t["idx"], t["site"], float(res.get("val_loss", res["observe_loss"]))))
            else:
                from sweep.run_trial import _failure_penalty
                pen = _failure_penalty(self.sampler, self.cfg)
                if pen is not None:
                    self.sampler.observe(t["hp_idx"], t["t_steps"], pen, None)
                self.sampler.report_failure(t["hp_idx"])
                t.update(status="failed" if res else "lost", finished=int(time.time()), sched_state=stt,
                         error=(res or {}).get("error", "no RESULT_JSON in the log (%s)" % stt))
                log("%s: trial %d on %s FAILED (%s): %s" % (self.name, t["idx"], t["site"], stt, t["error"][:120]))
            if a.fetch:
                site("fetch", t["rid"], "--dir", "runs/%s/trial_%04d" % (self.name, t["hp_idx"]), timeout=600, check=False, quiet=True)
            self.save()

    def result_of(self, t):
        out = site("logs", t["rid"], "-n", "200", timeout=120, check=False)
        hits = [l for l in out.splitlines() if l.strip().startswith("RESULT_JSON ")]
        if not hits:
            return None
        try:
            return json.loads(hits[-1].strip()[len("RESULT_JSON "):])
        except ValueError:
            return None

    def summary(self):
        from sweep.run_trial import _write_summary
        _write_summary(self.cfg, self.sampler, self.dir)


def plan(k, sweeps, a):
    """Sites for the next k trials, from `site pick`: each entry is one job's site."""
    s0 = sweeps[0]
    args = ["pick", PROJECT, "--jobs", str(k), "--mem", s0.mem, "--cpus", str(s0.cpus),
            "--hours", "%.2f" % s0.hours, "--json"]
    if a.sites:
        args += ["--only"] + list(a.sites)
    if a.allow_jeanzay:
        args.append("--allow-jeanzay")
    out = site(*args, timeout=400, check=False)
    try:
        d = json.loads(out[out.index("{"):])
    except ValueError:
        log("site pick gave no plan: %s" % out.strip()[-200:])
        return [], []
    order = []
    for s, n in sorted((d.get("plan") or {}).items(), key=lambda kv: -kv[1]):
        order += [s] * int(n)
    return order, d.get("rows", [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--parallel", type=int, help="trials in flight per sweep (default: n_startup)")
    ap.add_argument("--mem"); ap.add_argument("--cpus", type=int); ap.add_argument("--hours", type=float)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--item", required=True, help="the work item these trials belong to (`site items FA`)")
    ap.add_argument("--allow-jeanzay", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--no-fetch", dest="fetch", action="store_false", help="do not mirror each trial's tier-0 outputs locally")
    ap.add_argument("--once", action="store_true", help="one pass (collect, then submit what fits) and exit")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sweeps = [Sweep(c, a) for c in a.config]
    parallel = a.parallel or max(s.n_startup for s in sweeps)
    if not a.dry_run:
        for s in sorted(set(a.sites or ["ccin2p3", "lxplus"] + (["jeanzay"] if a.allow_jeanzay else []))):
            log("sync %s" % s)
            site("sync", s, PROJECT, timeout=600, check=False)
    while True:
        for sw in sweeps:
            sw.collect(a)
        want = {sw: sw.may_issue(parallel) for sw in sweeps if not sw.done()}
        k = sum(want.values())
        if k:
            order, rows = plan(k, sweeps, a)
            for r in rows:
                if r.get("state") in ("UP", "HELD"):
                    log("  %-9s %-5s %s" % (r["site"], r["state"], r.get("reason", "")[:150]))
            i = 0
            for sw, n in want.items():
                for _ in range(n):
                    if i >= len(order):
                        break
                    s = order[i]; i += 1
                    try:
                        if sw.data_ready(s, a):
                            sw.submit(s, a)
                        else:
                            log("%s: %s not ready (pools); trial deferred" % (sw.name, s))
                    except Exception as e:
                        log("%s: submit to %s failed: %s" % (sw.name, s, str(e)[-300:]))
        if all(sw.done() for sw in sweeps):
            for sw in sweeps:
                sw.summary()
                log("%s: done, %d trials (%d failed). %s" % (sw.name, sw.finished(), len(sw.trials("failed", "lost")), os.path.join(sw.dir, "summary.txt")))
            return
        if a.once:
            for sw in sweeps:
                log("%s: %d issued, %d in flight, %d finished of %d" % (sw.name, sw.issued(), len(sw.in_flight()), sw.finished(), sw.n_trials))
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
