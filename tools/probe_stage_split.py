"""Seconds per stage of a zero-shot read (tools/zero_shot_parent.py), from a job's full log: the reader logs one build per
probe with second-resolution timestamps, so each stage's mean over the probes of a job (all but the first build, which
carries the job's own setup) is its mean duration (a stage shorter than a second shows as the share of probes whose
timestamps tick inside it). Stages: data (recipe to dataloaders), model (instantiate, muP), listing_and_ft_load (the
module-name listing and the parent checkpoint's torch.load, logged after it returns), rest (the second weight load,
the physics check, the forward pass, scoring and the .npz; no timestamped line inside it).
    python tools/probe_stage_split.py LOG [LOG ...] [--out JSON]"""
import argparse, datetime as dt, json, os, re, statistics as st

MARKS = [("start", "Creating new experiment"), ("data_done", "Constructed dataloaders"),
         ("model_done", "Instantiated model"), ("ft_loaded", "Fine-tuning: loading pretrained")]


def split(path):
    rows, cur = [], None
    for line in open(path, errors="replace"):
        m = re.search(r"\[(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)", line)
        if not m:
            continue
        t = dt.datetime.fromisoformat(m.group(1).replace(" ", "T"))
        if MARKS[0][1] in line:
            if cur:
                rows.append(cur)
            cur = {"start": t}
            continue
        if cur is not None:
            for k, s in MARKS[1:]:
                if s in line and k not in cur:
                    cur[k] = t
    rows = [r for r in rows if all(k in r for k, _ in MARKS)][1:]
    if len(rows) < 3:
        return {"log": os.path.basename(path), "probes": len(rows)}
    mean = lambda xs: round(st.mean(xs), 3)
    nxt = list(zip(rows, rows[1:]))
    return {"log": os.path.basename(path), "probes": len(rows),
            "s_data": mean((r["data_done"] - r["start"]).total_seconds() for r in rows),
            "s_model": mean((r["model_done"] - r["data_done"]).total_seconds() for r in rows),
            "s_listing_and_ft_load": mean((r["ft_loaded"] - r["model_done"]).total_seconds() for r in rows),
            "s_rest": mean((b["start"] - a["ft_loaded"]).total_seconds() for a, b in nxt),
            "s_per_probe": mean((b["start"] - a["start"]).total_seconds() for a, b in nxt)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = [split(p) for p in a.logs]
    for r in res:
        print(json.dumps(r))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
