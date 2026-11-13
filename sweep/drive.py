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
    python sweep/drive.py --config sweep/sweep_config_X.yaml [--config ...] --item <work item>
        [--n-trials N] [--parallel P | --wave-sizes 3,3,2] [--mem 8G] [--cpus 4] [--hours 0.5]
        [--sites ccin2p3 lxplus] [--allow-jeanzay] [--poll 60] [--once] [--dry-run]
Resumable: run it again and it continues from driver_state.json.

Issuing trials: the order of sweep/ordering.py, the same as sweep_manager.py's (D396).
  waves (default; --wave-sizes 3,3,2 or `dyhpo.waves` in the sweep config name them, the flag wins; else
      ordering.wave_sizes: the n_startup random trials, the rest in two guided waves): trials go
      out in whole waves, and wave k+1 is suggested and submitted only once EVERY trial of waves 1..k
      has finished (observed, or failed/lost and reported to the sampler), counted from this driver's
      own driver_state.json, never from a scheduler's view. The first wave should be the n_startup
      random trials; each later wave is the surrogate's picks given every earlier result. The sizes
      must add up to n_trials. This is the likelihoods project's rule: trials that run together are
      suggested blind of each other, so all-at-once is a random search. Across searches the trials
      wanting out go longest first, round-robin across searches of equal length (ordering.dispatch_order).
  rolling (only with --parallel P and no waves named): the n_startup random trials first; once all of
      them are observed, up to P guided trials in flight, a new one suggested as soon as any one finishes.
Pruning (sweep/prune.py, D19): the trials write their results under the site's sweep dir as any trial does;
      once a search is done the driver writes DRIVE_DONE there on every site it ran on, which prune.py
      requires before it treats a search without jobs/ (a driven one) as finished.
--dry-run: no state written (a temporary directory), no sync, no pool check, no submission; prints the
      `site submit` commands of the first wave that would go out now and exits.
Re-pick: every REPICK_S, each trial still queued (PENDING/IDLE) on its site is weighed against the others by `site
      pick`'s model (finish_h): when another allowed site (--sites, --allow-jeanzay) ends it more than MOVE_H hours
      sooner and in under 1/MOVE_X of the time, its job is cancelled there and it is resubmitted at the same hp, t_steps
      and index on that site, with no penalty (it never ran), logged 'moved <run> <from> -> <to>'.
Limits: detached trials always cold-start (no cross-fidelity warm start), which is how
nearly every sweep runs anyway (single fidelity).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
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
from sweep import ordering                        # noqa: E402

SITE = os.path.expanduser("~/.local/bin/site")
PROJECT = "FA"
JOB = "sweep/trial_job.sh"
PREBUILD = "scripts/prebuild_recipes.sh"
FINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "REMOVED")
QUEUED = ("PENDING", "IDLE")      # not started (SLURM, HTCondor): a trial in one may be re-picked
MOVE_H, MOVE_X = 1.0, 2.0         # a queued trial moves when another site ends it > 1 h sooner and in < half the time
REPICK_S = 600                    # one re-pick `site pick` per 10 min at most: each one probes every site over ssh
STAY_S = 12 * 3600                # a moved trial stays where it arrived this long (a move gives up its queue age)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def site(*args, timeout=600, check=True, quiet=False, timeout_ok=True):
    """A `site` command's output. A hang past its timeout never kills the driver (E6983: one slow `site pick` ended a
    search): with check=False and timeout_ok it reads as empty output (no plan, no states: retried next pass),
    otherwise it raises RuntimeError as a failed command does."""
    cmd = [SITE, "--timeout", str(int(min(timeout, 3600)))] + [str(a) for a in args]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           timeout=timeout + 60)
    except subprocess.TimeoutExpired:
        if check or not timeout_ok:
            raise RuntimeError("site %s timed out after %d s" % (" ".join(map(str, args[:3])), timeout + 60))
        log("site %s timed out after %d s: retried on the next pass" % (" ".join(map(str, args[:3])), timeout + 60))
        return ""
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
        self.dir = (tempfile.mkdtemp(prefix="zz_drive_dry_%s_" % self.name) if a.dry_run
                    else os.path.join(ROOT, "sweeps_local", self.name))
        os.makedirs(os.path.join(self.dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "dyhpo_surrogate"), exist_ok=True)
        self.state_path = os.path.join(self.dir, "dyhpo_state.pkl")
        self.out_path = os.path.join(self.dir, "dyhpo_surrogate")
        if not os.path.exists(self.state_path):
            if not a.dry_run:
                # a new sweep on a cell another sweep already measures is refused: extend that one (sweep/cell_guard.py)
                from sweep.cell_guard import check_new
                with open(config_path) as f:
                    check_new(yaml.safe_load(f), self.dir)
            init_sampler(self.cfg, self.dir, self.dir)
        self.sampler = DyHPOSampler.load(self.state_path, self.out_path, force_cpu=True)
        self.n_trials = int(a.n_trials or self.cfg.get("n_trials", 40))
        dy = self.cfg.get("dyhpo", {}) or {}
        self.n_startup = int(dy.get("n_startup", 3))
        waves = a.wave_sizes or dy.get("waves")
        if isinstance(waves, str):
            waves = [int(x) for x in waves.split(",") if x.strip()]
        self.waves = None
        if waves or not a.parallel:                   # rolling only when asked for (--parallel) and no waves named
            try:
                self.waves = ordering.wave_sizes(self.n_trials, self.n_startup, waves)
            except ValueError as e:
                raise SystemExit("%s: %s" % (self.name, e))
            if self.waves[0] != self.n_startup:
                log("%s: WARNING first wave %d != n_startup %d: the random start-up trials and the first wave differ"
                    % (self.name, self.waves[0], self.n_startup))
        cl = self.cfg.get("cluster", {}) or {}
        self.mem = a.mem or cl.get("mem") or cl.get("request_memory") or "8G"
        self.cpus = int(a.cpus or cl.get("cpus_per_task") or 4)
        self.gpu_min_mem = cl.get("gpu_min_mem")          # e.g. 24G: HTCondor keeps the trial off smaller GPUs
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
        if (getattr(a, "retry_lost", False) or getattr(a, "retry_failed", None)
                or getattr(a, "recollect", None)) and not a.dry_run:
            self.requeue_lost(getattr(a, "retry_lost", False), getattr(a, "retry_failed", None),
                              getattr(a, "recollect", None))

    def requeue_lost(self, lost=True, failed_re=None, recollect_re=None):
        """--retry-lost: every "lost" trial (no RESULT_JSON in its log: the job died without reporting, e.g. a full
        filesystem, 2026-10-23) is taken back: the failure penalty it was observed at is removed from the sampler and
        the surrogate, its config leaves the diverged set, and it is resubmitted at the same hp and t_steps under the
        same trial index. A "failed" trial (it reported a failure: diverged, NaN) keeps its penalty, unless
        --retry-failed REGEX matches its run id and error (a crash of our code, fixed since: not a property of its hp);
        that also takes "lost" trials. --recollect REGEX: a failed or lost trial whose job did finish but whose log
        could not be read then (site access lost) goes back to "submitted", penalty removed, so collect() reads its
        RESULT_JSON again instead of rerunning it."""
        alg = self.sampler.algorithm
        todo = list(self.trials("lost")) if lost else []
        back = []
        for rx, out in ((failed_re, todo), (recollect_re, back)):
            if rx:
                out += [t for t in self.trials("failed", "lost") if t not in todo + back
                        and re.search(rx, "%s %s" % (t.get("rid"), t.get("error") or ""))]
        for t in todo + back:
            hp, combo = t["hp_idx"], self.sampler._to_combo(t["t_steps"])
            h = self.sampler._val_loss_history.get(hp, {})
            h.pop(combo, None)
            if not h:
                self.sampler._val_loss_history.pop(hp, None)
            self.sampler._eval_order = [x for x in self.sampler._eval_order if tuple(x) != (hp, combo)]
            o = getattr(alg, "observations", {}).get(hp, {})
            o.pop(combo, None)
            if hp in getattr(alg, "observations", {}) and not o:
                alg.observations.pop(hp)
            alg.diverged_configs.discard(hp)
            t.setdefault("lost_runs", []).append(t["rid"])
            if t in back:
                t["lost_runs"].pop()
                t["status"] = "submitted"
                self.sampler._in_flight.add(hp)
                t.pop("finished", None); t.pop("error", None)
                log("%s: trial %d (hp%04d t%d, %s) back to submitted to recollect its result, its penalty removed"
                    % (self.name, t["idx"], hp, t["t_steps"], t["rid"]))
                continue
            was, t["status"] = t["status"], "retry"
            log("%s: trial %d (hp%04d t%d, %s in %s) requeued, its penalty removed"
                % (self.name, t["idx"], hp, t["t_steps"], was, t["rid"]))
        self.save()

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
        if self.trials("retry"):                   # requeued lost trials go out first, whatever the wave rule
            return len(self.trials("retry"))
        if self.issued() >= self.n_trials:
            return 0
        if self.waves:
            # whole waves: the wave holding the next trial goes out only when every trial of the
            # waves before it has finished (tracked trials in driver_state.json), and all at once
            end = 0
            for w in self.waves:
                start, end = end, end + w
                if self.issued() < end:
                    return end - self.issued() if self.finished() >= start else 0
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
        if a.dry_run:
            log("[dry-run] pools of %s on %s not checked" % (self.spec, s)); return True
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
        out = site("submit", s, PROJECT, PREBUILD, "--no-sync", "--allow-stale", "--item", a.item, "--note", "prebuild %s" % self.name,
                   "--hdr=--cpus-per-task=%d" % cpus, "--hdr=--time=04:00:00",
                   "--", self.spec, "--seed", str(self.seed), timeout=300)
        m = re.search(r"\brun\s+(\S+)", out)
        if not m:
            raise RuntimeError("prebuild submit gave no run id:\n" + out)
        self.st["prebuild"][s] = {"rid": m.group(1), "ts": int(time.time())}; self.save()
        log("%s: pools missing on %s -> prebuild %s submitted (CPU job)" % (self.name, s, m.group(1)))
        return False

    # ---- one trial out
    def submit(self, s, a, t0=None):
        retry = [t0] if t0 else self.trials("retry")
        if retry:
            t0 = retry[0]
            hp_idx, hp, t_steps, idx = t0["hp_idx"], t0["hp"], t0["t_steps"], t0["idx"]
            self.sampler._in_flight.add(hp_idx)
        else:
            hp_idx, hp, t_steps = self.sampler.suggest()
            idx = self.issued()
        args = ["submit", s, PROJECT, JOB, "--no-sync", "--allow-stale", "--item", a.item,
                "--note", "%s hp%04d t%d" % (self.name, hp_idx, t_steps),
                "--hdr=--mem=%s" % self.mem, "--hdr=--time=%s" % self.time,
                "--hdr=--cpus-per-task=%d" % self.cpus,
                "--hdr=--job-name=%s_%04d" % (self.name[:20], idx)]
        if self.gpu_min_mem:
            args.append("--hdr=--gpu-min-mem=%s" % self.gpu_min_mem)
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
            if retry:      # a requeued trial stays "retry" with its hp healthy: report_failure would mark it diverged
                self.sampler._in_flight.discard(hp_idx)
            else:
                self.sampler.report_failure(hp_idx)
            raise RuntimeError("submit gave no run id:\n" + out)
        prev = self.st["trials"].get(str(idx), {}).get("lost_runs") if retry else None
        self.st["trials"][str(idx)] = {"idx": idx, "rid": m.group(1), "site": s, "hp_idx": hp_idx,
                                       "hp": hp, "t_steps": t_steps, "status": "submitted",
                                       "submitted": int(time.time()), **({"lost_runs": prev} if prev else {}),
                                       **({"moved": t0["moved"]} if t0 and t0.get("moved") else {})}
        self.save()
        log("%s: trial %d -> %s  (hp%04d t%d, run %s)" % (self.name, idx, s, hp_idx, t_steps, m.group(1)))

    def move(self, t, dest, a):
        """Cancel trial t's job, still queued on its site, and submit it on dest at the same hp, t_steps and index: lost
        with no penalty (none was observed, it never ran), as --retry-lost takes a trial back. False: it stays."""
        src, rid = t["site"], t["rid"]
        if a.dry_run:
            log("[dry-run] would move %s %s -> %s" % (rid, src, dest)); return False
        if not self.data_ready(dest, a):
            return False                               # pools building there: moves on a later pass
        out = site("poll", rid, timeout=120, check=False)
        m = re.search(r"->\s+(\S+)", out)
        if not m or m.group(1) not in QUEUED:
            return False                               # started (or ended) since the last poll
        job = sched_id(rid)
        try:
            site("run", src, PROJECT, "--", cancel_cmd(src, job), timeout=300)
        except (RuntimeError, ValueError, TypeError) as e:
            log("%s: could not cancel %s on %s, it stays: %s" % (self.name, rid, src, str(e)[-200:]))
            return False
        t.setdefault("lost_runs", []).append(rid)
        t["status"], t["moved"] = "retry", int(time.time())
        self.save()
        self.submit(dest, a, t)
        log("moved %s %s -> %s" % (rid, src, dest))
        return True

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
                t["sched_state"] = stt
                continue
            try:
                res = self.result_of(t)
            except RuntimeError as e:                  # its log could not be read: stays in flight, read next pass
                log("%s: trial %d: %s" % (self.name, t["idx"], e)); continue
            if res and not res.get("failed"):
                self.sampler.observe(t["hp_idx"], t["t_steps"], float(res["observe_loss"]),
                                     res.get("proc_val_losses"))
                from sweep.run_trial import result_fields
                vnr, oname, oval = result_fields(res)
                t.update(status="observed", val_loss=vnr, objective=oname, objective_value=oval,
                         observe_loss=res["observe_loss"], finished=int(time.time()), sched_state=stt)
                json.dump(res, open(os.path.join(self.dir, "results", "hp%04d_t%d.json" % (t["hp_idx"], t["t_steps"])), "w"), indent=1)
                log("%s: trial %d on %s -> %s %.6g" % (self.name, t["idx"], t["site"], oname,
                                                       float(oval if oval is not None else res["observe_loss"])))
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
        out = site("logs", t["rid"], "-n", "200", timeout=120, check=False, timeout_ok=False)   # unread is not lost
        hits = [l for l in out.splitlines() if l.strip().startswith("RESULT_JSON ")]
        # the run's own exception, digits blanked, for failure_streak (run_trial's wrapper RuntimeError says nothing)
        errs = [re.sub(r"\d+", "N", l.strip())[:200] for l in out.splitlines()
                if re.match(r"\s*\w+(Error|Exception): ", l) and "run.py exited with code" not in l]
        t["cause"] = errs[-1] if errs else None
        if not hits:
            return None
        try:
            return json.loads(hits[-1].strip()[len("RESULT_JSON "):])
        except ValueError:
            return None

    def summary(self):
        from sweep.run_trial import _write_summary
        _write_summary(self.cfg, self.sampler, self.dir)


def failure_streak(trials, n):
    """The cause shared by the last n finished trials of all searches when every one of them failed with it (a crash
    of the setup, not of a trial's hp: 84 b b~ > Z Z V/B trials went out on one stale Born record, 2026-11-12), else
    None. Divergences are a property of the sampled hp and never count."""
    done = sorted((t for t in trials if t.get("finished")), key=lambda t: (t["finished"], t.get("idx", 0)))[-n:]
    causes = {t.get("cause") for t in done if t.get("status") in ("failed", "lost")}
    if n and len(done) == n and len(causes) == 1 and all(t.get("status") in ("failed", "lost") for t in done):
        c = causes.pop()
        return c if c and "diverg" not in c.lower() else None
    return None


def dispatch_order(want):
    """The trials all sweeps want now, one entry per trial, in ordering.dispatch_order (sw.hours: the config's
    estimate of T x the step rate)."""
    return ordering.dispatch_order(want, {sw: sw.hours for sw in want})


def mark_cmd(name):
    """The command marking a driven search over, run on the site: its sweep dir is the site's siteconf.SWEEP_DIR
    (where prune.finished looks), never the local machine's path in sw.cfg."""
    return ("python -c \"import os, siteconf; d = os.path.join(siteconf.SWEEP_DIR, '%s'); os.makedirs(d, exist_ok=True); "
            "open(os.path.join(d, 'DRIVE_DONE'), 'w').close(); print('MARKED')\"" % name)


def mark_done(sw, a):
    """DRIVE_DONE in the search's sweep dir on every site it ran on: prune.py's sign that a driven search is over."""
    if a.dry_run or sw.st.get("done_marked"):
        return
    for s in sorted({t["site"] for t in sw.trials()}):
        out = site("run", s, PROJECT, "--", mark_cmd(sw.name), timeout=300, check=False)
        if "MARKED" not in out:
            log("%s: DRIVE_DONE not written on %s: %s" % (sw.name, s, out.strip()[-160:]))
            return
    sw.st["done_marked"] = True
    sw.save()


def finish_h(r, hours, stays=False):
    """Hours until one trial of `hours` ends on the site of `site pick` row r, by pick's own model (plan_batch's finish
    of one job). A trial that stays counts none of my queued jobs ahead of it (it may be first in line), so a move must
    beat the stay's best case, and pick's skips (no GPU fits now, verify timed out) do not count against staying: they
    pass. None: a destination that takes no work (not UP, held, submit cap 0, a skip)."""
    if not stays and (r.get("state") != "UP" or r.get("cap") == 0 or r.get("skip")):
        return None
    if (r.get("slots_now") or 0) > 0:
        return float(hours)
    conc = max(1, r.get("mine_run") or 0)
    run_h = (r.get("med_run") or hours * 3600.0) / 3600.0
    ahead = 0 if stays else (r.get("mine_q") or 0)
    return (r.get("wait_now") or 0) / 3600.0 + ahead / float(conc) * run_h + hours


def repick_dest(cur, rows, hours):
    """The site a trial queued on `cur` should move to, or None to keep it: the other site of the pick's rows (already
    limited to --sites / --allow-jeanzay) that ends it soonest, when that beats staying by > MOVE_H hours and MOVE_X
    times. A current site that takes no work (cap 0, held, no GPU fits) loses to any that does; one the pick did not
    probe keeps its trial (its queue is unknown)."""
    by = {r["site"]: r for r in rows}
    if cur not in by:
        return None
    best = min(((f, s) for s, r in by.items() if s != cur for f in [finish_h(r, hours)] if f is not None), default=None)
    if best is None:
        return None
    home = by[cur]
    if home.get("cap") == 0 or home.get("fit_ever") == 0:
        return best[1]                         # never starts there (submit cap 0, no node holds the request)
    if home.get("state") != "UP":
        return None                            # down or held for now: its queue is unknown, the trial keeps its place
    stay = finish_h(home, hours, stays=True)
    if stay - best[0] > MOVE_H and stay > MOVE_X * best[0]:
        return best[1]
    return None


def sched_id(rid):
    """The scheduler's job id of a run, from the site registry."""
    import sqlite3
    db = os.path.join(os.environ.get("CCORCH_HOME") or os.path.expanduser("~/.local/share/ccorch"), "runs.db")
    c = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        r = c.execute("SELECT sched_id FROM runs WHERE id=?", (rid,)).fetchone()
    finally:
        c.close()
    return r[0] if r else None


def cancel_cmd(s, job):
    """The command cancelling scheduler job `job` on site s (its scheduler from sites/sites.yaml)."""
    with open(os.path.join(ROOT, "sites", "sites.yaml")) as f:
        sched = yaml.safe_load(f)[s]["cluster"]["scheduler"]
    return "%s %s" % ("condor_rm" if sched == "htcondor" else "scancel", int(job))


def took(rows, s):
    """Site s's pick row after one more job of mine went there: queued, one free slot fewer."""
    for r in rows:
        if r["site"] == s:
            r["mine_q"] = (r.get("mine_q") or 0) + 1
            r["slots_now"] = max(0, (r.get("slots_now") or 0) - 1)


def repick(sweeps, a, rows, now=None):
    """Move each trial still queued on its site to the site `site pick` (rows) says ends it clearly sooner; a trial
    moved in the last STAY_S stays."""
    now = now or time.time()
    for sw in sweeps:
        for t in sw.in_flight():
            if t.get("sched_state") not in QUEUED or now - t.get("moved", 0) < STAY_S:
                continue
            dest = repick_dest(t["site"], rows, sw.hours)
            if dest and sw.move(t, dest, a):
                took(rows, dest)


repick.last = 0.0


def plan(k, sweeps, a, record=True):
    """Sites for the next k trials, from `site pick`: each entry is one job's site. record=False: a re-check that
    leaves the placement quota other agents submit under untouched."""
    s0 = sweeps[0]
    args = ["pick", PROJECT, "--jobs", str(k), "--mem", s0.mem, "--cpus", str(s0.cpus),
            "--hours", "%.2f" % s0.hours, "--json"] + ([] if record else ["--no-record"])
    if s0.gpu_min_mem:
        args += ["--gpu-min-mem", str(s0.gpu_min_mem)]     # count only the GPUs the trials may run on
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
    ap.add_argument("--parallel", type=int, help="rolling mode (only when no waves are named): trials in flight per sweep")
    ap.add_argument("--wave-sizes", type=lambda v: [int(x) for x in v.split(",")],
                    help="whole waves, e.g. 3,3,2 (sum = n_trials): wave k+1 starts when waves 1..k are all observed; "
                         "overrides the config's dyhpo.waves")
    ap.add_argument("--mem"); ap.add_argument("--cpus", type=int); ap.add_argument("--hours", type=float)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--item", required=True, help="the work item these trials belong to (`site items FA`)")
    ap.add_argument("--allow-jeanzay", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--stop-after", type=int, default=6,
                    help="stop submitting when the last N finished trials all failed with the same error (0: never)")
    ap.add_argument("--no-fetch", dest="fetch", action="store_false", help="do not mirror each trial's tier-0 outputs locally")
    ap.add_argument("--once", action="store_true", help="one pass (collect, then submit what fits) and exit")
    ap.add_argument("--retry-lost", action="store_true",
                    help="resubmit every lost trial (no RESULT_JSON: the job died without reporting, which includes a "
                         "TIMEOUT or OUT_OF_MEM as well as a crash of the site) at its hp, its penalty taken back out of "
                         "the surrogate")
    ap.add_argument("--retry-failed", metavar="REGEX",
                    help="resubmit every failed trial whose '<run id> <error>' matches REGEX (a code crash fixed since), as "
                         "--retry-lost does, its penalty taken back out")
    ap.add_argument("--recollect", metavar="REGEX",
                    help="failed or lost trials whose '<run id> <error>' matches REGEX and whose job finished: back to "
                         "submitted, penalty removed, so the next collect reads their RESULT_JSON (no rerun)")
    ap.add_argument("--dry-run", action="store_true", help="print the first wave's submit commands; nothing written or sent")
    a = ap.parse_args()
    if a.dry_run:
        a.once = True

    sweeps = [Sweep(c, a) for c in a.config]
    parallel = a.parallel or max(s.n_startup for s in sweeps)
    for sw in sweeps:
        log("%s: %d trials, n_startup %d, surrogate %s, objective scale %s, %s" % (
            sw.name, sw.n_trials, sw.n_startup, sw.sampler.surrogate, sw.sampler.objective_scale,
            "waves %s" % sw.waves if sw.waves else "rolling, %d in flight" % parallel))
    if not a.dry_run:
        # the sync sends commits only, so other agents' uncommitted files in this shared checkout never reach a site and
        # do not stop the launch (they did three times on 2026-11-12); this launch's own configs must be committed
        mine = subprocess.run(["git", "status", "--porcelain", "--"] + list(a.config), cwd=ROOT, text=True,
                              stdout=subprocess.PIPE).stdout.strip()
        if mine:
            sys.exit("drive.py: commit the configs first, nothing submitted:\n" + mine)
        synced, failed = [], []
        for s in sorted(set(a.sites or ["ccin2p3", "lxplus"] + (["jeanzay"] if a.allow_jeanzay else []))):
            log("sync %s" % s)
            try:
                site("sync", s, PROJECT, "--allow-dirty", timeout=600)
                synced.append(s)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                # never submit on an unsynced site (stale checkout): a site down at start-up (Jean Zay's probe timing
                # out, 2026-11-12) is left out of this driver's placement; a restart syncs and includes it again
                failed.append("start-up sync of %s failed: %s" % (s, str(e)[-300:]))
                log(failed[-1] + "; site left out")
        if not synced:
            sys.exit("drive.py: %s; nothing submitted" % "; ".join(failed))
        a.sites = synced
    while True:
        for sw in sweeps:
            sw.collect(a)
        c = failure_streak([t for sw in sweeps for t in sw.trials("observed", "failed", "lost")], a.stop_after)
        if c:
            sys.exit("drive.py: the last %d trials all failed with the same error, nothing more submitted: %s\n"
                     "fix it, then rerun the same command with --retry-failed '.' (penalties taken back out)"
                     % (a.stop_after, c))
        want = {sw: sw.may_issue(parallel) for sw in sweeps if not sw.done()}
        k = sum(want.values())
        rows = None
        if k:
            order, rows = plan(k, sweeps, a)
            if a.dry_run and not order:
                order = [(a.sites or ["<site pick>"])[0]] * k
            for r in rows:
                if r.get("state") in ("UP", "HELD"):
                    log("  %-9s %-5s %s" % (r["site"], r["state"], r.get("reason", "")[:150]))
            for i, sw in enumerate(dispatch_order(want)):
                if i >= len(order):
                    break
                s = order[i]
                try:
                    if sw.data_ready(s, a):
                        sw.submit(s, a)
                        took(rows, s)
                    else:
                        log("%s: %s not ready (pools); trial deferred" % (sw.name, s))
                except Exception as e:
                    log("%s: submit to %s failed: %s" % (sw.name, s, str(e)[-300:]))
        if any(t.get("sched_state") in QUEUED for sw in sweeps for t in sw.in_flight()) \
                and time.time() - repick.last >= REPICK_S:
            repick.last = time.time()
            try:
                repick(sweeps, a, rows if rows is not None else plan(1, sweeps, a, record=False)[1])
            except Exception as e:
                log("re-pick failed: %s" % str(e)[-300:])
        for sw in sweeps:
            if sw.done():
                mark_done(sw, a)
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
