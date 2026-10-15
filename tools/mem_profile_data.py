#!/usr/bin/env python
"""Host-memory profile of the data pipeline of a sweep's runs, CPU only (no model, no GPU).

Builds the experiment a sweep trial would build (the sweep config's fixed_params, its recipe cut to the first
--n-procs processes), runs init_physics, init_data, _init_dataloader and --batches training batches through the
train loader with its workers, and records host memory all along:
  * a sampler thread every --dt s: this process's RSS / RssAnon / RssFile (/proc/self/status), the workers' summed
    Pss and Private_Dirty (/proc/<pid>/smaps_rollup), and the job cgroup's usage and anon/file split when readable;
  * every log line as a stage marker (the per-process load lines are folded into one stage per role);
  * every np.concatenate whose result exceeds 64 MB, with its caller's file:line, size and the RSS after it.
Writes <out>/memprof.json and prints a per-stage table (end and peak RSS of each stage).

    python tools/mem_profile_data.py --sweep-config sweep/sweep_config_dancat_het_pre.yaml --n-procs 100 \
        --out $SCRATCH/zz_memprof/n100 [--set data.memmap=true] [--batches 300] [--workers 2]
"""
import argparse
import inspect
import json
import logging
import os
import re
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import yaml

GB = 1e9


def _status():
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM"):
                out[k] = int(v.split()[0]) * 1024
    return out


def _rollup(pid):
    out = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("Rss", "Pss", "Pss_Anon", "Pss_File", "Private_Dirty", "Private_Clean", "Shared_Clean"):
                    out[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return out


def _children(pid):
    kids = []
    try:
        for t in os.listdir(f"/proc/{pid}/task"):
            with open(f"/proc/{pid}/task/{t}/children") as f:
                kids += [int(x) for x in f.read().split()]
    except OSError:
        pass
    return kids


def _cgroup_files():
    """The memory cgroup of this job (v1 or v2): (usage file, peak file, stat file), or None."""
    try:
        lines = open("/proc/self/cgroup").read().splitlines()
    except OSError:
        return None
    for line in lines:
        _, ctrl, path = line.split(":", 2)
        if ctrl == "" and os.path.exists("/sys/fs/cgroup" + path + "/memory.current"):
            d = "/sys/fs/cgroup" + path
            # walk up to the job level (the step's cgroup is a child of it)
            while "/step_" in d or "/task_" in d:
                d = os.path.dirname(d)
            peak = d + "/memory.peak" if os.path.exists(d + "/memory.peak") else None
            return d + "/memory.current", peak, d + "/memory.stat", "v2"
        if "memory" in ctrl.split(","):
            d = "/sys/fs/cgroup/memory" + path
            while ("/step_" in d or "/task_" in d) and os.path.exists(os.path.dirname(d) + "/memory.usage_in_bytes"):
                d = os.path.dirname(d)
            if os.path.exists(d + "/memory.usage_in_bytes"):
                return d + "/memory.usage_in_bytes", d + "/memory.max_usage_in_bytes", d + "/memory.stat", "v1"
    return None


def _cg_read(cg):
    if cg is None:
        return {}
    cur, peak, stat, ver = cg
    out = {}
    try:
        out["cg_usage"] = int(open(cur).read())
        if peak:
            out["cg_peak"] = int(open(peak).read())
        for line in open(stat):
            k, v = line.split()
            if (ver == "v2" and k in ("anon", "file", "file_mapped")) or \
               (ver == "v1" and k in ("total_rss", "total_cache", "total_mapped_file")):
                out["cg_" + k] = int(v)
    except (OSError, ValueError):
        pass
    return out


class Recorder:
    def __init__(self, dt):
        self.t0 = time.time()
        self.dt = dt
        self.samples, self.markers, self.concats = [], [], []
        self.cg = _cgroup_files()
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        s = _status()
        kids = _children(os.getpid())
        kr = [_rollup(k) for k in kids]
        rec = {"t": round(time.time() - self.t0, 3), "rss": s.get("VmRSS", 0), "anon": s.get("RssAnon", 0),
               "file": s.get("RssFile", 0), "hwm": s.get("VmHWM", 0), "n_workers": len(kids),
               "workers_pss": sum(r.get("Pss", 0) for r in kr),
               "workers_private_dirty": sum(r.get("Private_Dirty", 0) for r in kr)}
        rec.update(_cg_read(self.cg))
        return rec

    def _loop(self):
        while not self._stop:
            self.samples.append(self.sample())
            time.sleep(self.dt)

    def mark(self, msg):
        s = self.sample()
        s["msg"] = msg
        self.markers.append(s)
        self.samples.append({k: v for k, v in s.items() if k != "msg"})

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop = True
        self.thread.join(timeout=5)


class MarkHandler(logging.Handler):
    def __init__(self, rec):
        super().__init__(level=logging.INFO)
        self.rec = rec

    def emit(self, record):
        try:
            self.rec.mark(record.getMessage()[:200])
        except Exception:
            pass


def hook_concatenate(rec, min_bytes=64 * 2 ** 20):
    orig = np.concatenate

    def concatenate(*a, **k):
        out = orig(*a, **k)
        nb = getattr(out, "nbytes", 0)
        if nb >= min_bytes:
            fr = inspect.stack()[1]
            rec.concats.append({"t": round(time.time() - rec.t0, 3), "where": f"{os.path.basename(fr.filename)}:{fr.lineno}",
                                "gb": round(nb / GB, 3), "rss_after": _status().get("VmRSS", 0)})
        return out
    np.concatenate = concatenate


STAGE_RULES = [  # (regex on the log line, stage name); a run of lines with one name is one stage
    (r"^\s*\[(train|val|test)\] ", lambda m: f"load+boost [{m.group(1)}]"),
]


def stage_of(msg):
    for pat, name in STAGE_RULES:
        m = re.search(pat, msg)
        if m:
            return name(m)
    if msg.startswith("  Per-process val loader"):
        return "per-process loaders"
    return msg[:90]


def summarize(rec):
    """Stages = runs of markers with one stage name; peak = max over the samples since the previous stage's end."""
    samples = sorted(rec.samples, key=lambda s: s["t"])
    stages = []
    for m in rec.markers:
        name = stage_of(m["msg"])
        if stages and stages[-1]["name"] == name:
            stages[-1]["t_end"] = m["t"]; stages[-1]["end"] = m
        else:
            stages.append({"name": name, "t_end": m["t"], "end": m})
    t_prev = 0.0
    rows = []
    for st in stages:
        win = [s for s in samples if t_prev <= s["t"] <= st["t_end"]] or [st["end"]]
        tot = lambda s: s["rss"] + s.get("workers_private_dirty", 0)
        rows.append({"stage": st["name"], "t": st["t_end"],
                     "rss_end_gb": round(st["end"]["rss"] / GB, 3),
                     "anon_end_gb": round(st["end"]["anon"] / GB, 3),
                     "peak_rss_gb": round(max(s["rss"] for s in win) / GB, 3),
                     "peak_anon_gb": round(max(s["anon"] for s in win) / GB, 3),
                     "peak_rss_plus_workers_private_gb": round(max(tot(s) for s in win) / GB, 3),
                     "peak_cg_gb": round(max(s.get("cg_usage", 0) for s in win) / GB, 3)})
        t_prev = st["t_end"]
    return rows


def sweep_cfg(sweep_config, n_procs, out, extra=()):
    """The Hydra config a trial of `sweep_config` runs with (its fixed_params, an in-range lr, no MLflow), its recipe
    cut to the first n_procs processes (0: all), the run dir under `out`; `extra` overrides last."""
    sc = yaml.safe_load(open(sweep_config))
    fixed = dict(sc.get("fixed_params", {}))
    pdir = os.environ.get("PROJECT_DIR", ROOT)
    fixed = {k: (v.replace("${PROJECT_DIR}", pdir) if isinstance(v, str) else v) for k, v in fixed.items()}
    recipe = fixed.get("data.processes_file")
    os.makedirs(out, exist_ok=True)
    if recipe and n_procs:
        spec = yaml.safe_load(open(recipe))
        procs = spec["processes"] if isinstance(spec, dict) else spec
        cut = os.path.join(out, f"recipe_first{n_procs}.yaml")
        with open(cut, "w") as f:
            yaml.safe_dump({"processes": procs[:n_procs]}, f, sort_keys=False)
        fixed["data.processes_file"] = cut
    # the sweep's HPs do not touch the data; any in-range value builds the same pipeline
    ov = ["local=none"] + [f"{k}={v}" for k, v in fixed.items()]
    ov += ["training.lr=3e-3", "training.iterations=1", "use_mlflow=false", "warm_start_idx=null",
           f"base_dir={pdir}", "exp_name=zz_memprof", f"run_dir={os.path.join(out, 'run')}"]
    ov += list(extra)
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.join(ROOT, "config"), version_base=None):
        cfg = compose(config_name="amplitudes", overrides=ov)
    return cfg, ov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-config", required=True)
    ap.add_argument("--n-procs", type=int, default=0, help="first N processes of the recipe (0: all)")
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-pass", action="store_true", help="also one pass over the combined val loader")
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="extra override k=v (repeatable)")
    a = ap.parse_args()
    a.out = os.path.expandvars(a.out)           # a literal $SCRATCH survives the submission's quoting

    os.makedirs(a.out, exist_ok=True)
    rec = Recorder(a.dt)
    rec.start()
    rec.mark("start")
    hook_concatenate(rec)

    cfg, ov = sweep_cfg(a.sweep_config, a.n_procs, a.out, a.set + [f"training.num_workers={a.workers}"])
    import torch
    torch.set_default_dtype(torch.float32)
    from experiment import AmplitudeExperiment
    exp = AmplitudeExperiment(cfg)
    exp._init()
    # after _init: the run's logger is set up; mark every INFO line from here on
    from logger import LOGGER                     # the run's logger does not propagate to the root logger
    LOGGER.addHandler(MarkHandler(rec))
    t = time.time()
    exp.init_physics(); exp.init_geometric_algebra()
    rec.mark("== init_physics done")
    exp.init_data()
    rec.mark("== init_data done (%.0f s)" % (time.time() - t))
    exp._init_dataloader()
    rec.mark("== _init_dataloader done")
    it = iter(exp.train_loader)
    t = time.time()
    n_ev = 0
    for b in range(a.batches):
        batch = next(it)
        n_ev += int(batch[1].shape[0])
        if b in (0, 9) or (b + 1) % 50 == 0:
            rec.mark("== train batch %d (%.2f s/batch)" % (b + 1, (time.time() - t) / (b + 1)))
    if a.val_pass:
        t = time.time()
        for batch in exp.val_loader:
            pass
        rec.mark("== val pass done (%.0f s)" % (time.time() - t))
    rec.stop()

    rows = summarize(rec)
    peak = max(s["rss"] + s.get("workers_private_dirty", 0) for s in rec.samples)
    mdir = getattr(exp, "_mmap_dir", None)
    out = {"n_procs": len(exp.cfg.data.dataset), "mmap_dir": mdir,
           "mmap_gb": (sum(os.path.getsize(os.path.join(mdir, f)) for f in os.listdir(mdir)) / GB if mdir else None), "n_events": int(exp.N_events),
           "n_particles": int(exp.particles_flat.shape[0]), "role_counts": list(exp._role_counts),
           "overrides": ov, "peak_rss_gb": max(s["rss"] for s in rec.samples) / GB,
           "peak_anon_gb": max(s["anon"] for s in rec.samples) / GB,
           "peak_rss_plus_workers_private_gb": peak / GB,
           "peak_cg_gb": max(s.get("cg_peak", s.get("cg_usage", 0)) for s in rec.samples) / GB,
           "vm_hwm_gb": _status().get("VmHWM", 0) / GB,
           "stages": rows, "concats": rec.concats, "samples": rec.samples}
    with open(os.path.join(a.out, "memprof.json"), "w") as f:
        json.dump(out, f)
    print(f"\nprocesses {out['n_procs']}  events {out['n_events']}  particles {out['n_particles']}  "
          f"roles {out['role_counts']}")
    print(f"{'stage':92s} {'end':>7s} {'peak':>7s} {'pk+wk':>7s} {'cg':>7s}   (GB; RSS of the main process, "
          f"+ workers' private dirty pages, job cgroup)")
    for r in rows:
        print(f"{r['stage'][:92]:92s} {r['rss_end_gb']:7.2f} {r['peak_rss_gb']:7.2f} "
              f"{r['peak_rss_plus_workers_private_gb']:7.2f} {r['peak_cg_gb']:7.2f}")
    print("\nlarge concatenations (>= 64 MB):")
    for c in rec.concats:
        print(f"  t={c['t']:8.1f}s {c['where']:28s} {c['gb']:7.2f} GB  rss after {c['rss_after'] / GB:7.2f} GB")
    print(f"\nPEAK main RSS {out['peak_rss_gb']:.2f} GB (anon {out['peak_anon_gb']:.2f}), main + workers' private "
          f"{out['peak_rss_plus_workers_private_gb']:.2f} GB, cgroup peak {out['peak_cg_gb']:.2f} GB, "
          f"VmHWM {out['vm_hwm_gb']:.2f} GB")
    print("MEMPROF_JSON " + json.dumps({k: v for k, v in out.items() if k not in ("samples", "concats", "stages")}))


if __name__ == "__main__":
    main()
