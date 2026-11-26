"""Plotting data of the held-out figures for Suprio: export_for_suprio/{per_process.csv, shape_dists_events.csv, README.txt}
and export_for_suprio.zip. Values come from the figures' own readers (heldout_per_process.PART/ARMS/GROUPS, suprio.llm,
shape_dists.main); per_process is asserted equal to what the figures plot.
    python analysis/llm_compare/export_for_suprio.py
"""
import csv, json, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import heldout_per_process as hp  # noqa: E402
import shape_dists as sd  # noqa: E402
import suprio  # noqa: E402

OUT = os.path.join(HERE, "export_for_suprio")
SHORT = {"a_trial5": "lloca_same_setup", "level_file": "lloca_same_setup_diagram_level", "c_best": "lloca_new_preprocessing"}
ARMS = list(SHORT)


def load(t, d):
    return json.load(open(os.path.join(HERE, d, f"{t}_heldout.json")))["processes"]


def main():
    os.makedirs(OUT, exist_ok=True)
    draft = json.load(open(os.path.join(hp.ROOT, "talks", "progress_oct2026b", "data", "draft_pid_by_process.json")))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(hp.DAN, "classes_default.json")))["processes"]}
    spl = {t: load(t, os.path.join("data", "split")) for t in ARMS}
    llm = {k: suprio.llm(k) for k in ("mse", "bias", "shape")}
    pid = {p["folder"]: i for i, p in enumerate(json.load(open(os.path.join(hp.DAN, "classes_default.json")))["processes"])}
    cols = ["row", "class", "process", "process_id", "is_2to2"]
    for t in ARMS:
        cols += [f"{SHORT[t]}_{k}" for k in ("mse_logm2", "bias_logm2", "shape_logm2")]
    cols += ["llm_mse_median", "llm_mse_min", "llm_mse_max", "llm_abs_bias_median", "llm_abs_bias_min", "llm_abs_bias_max",
             "llm_shape_median", "llm_shape_min", "llm_shape_max", "llm_bias_seed0", "llm_bias_seed1", "llm_bias_seed2"]
    rows = []
    for sp, rname in (("interp", "interpolation"), ("extrap", "extrapolation")):
        gx = {g: min(draft[sp][n][0] for n in draft[sp] if meta[n]["test_group"] == g) for g, _ in hp.GROUPS[sp]}
        short = dict(hp.GROUPS[sp])
        names = sorted(draft[sp], key=lambda n: (gx[meta[n]["test_group"]], np.median(llm["mse"][n])))
        for n in names:
            r = [rname, short[meta[n]["test_group"]].replace("$", ""), hp.proc_label(meta[n]["process"]).replace("$", ""),
                 meta[n]["folder"], int(meta[n]["n_out"] == 2)]
            for t in ARMS:
                q = spl[t][n]
                r += [q["mse_logm2"], q["bias_logm2"], q["mse_logm2"] - q["bias_logm2"] ** 2]   # one pass: data/split
            f = lambda a: [np.median(a), np.min(a), np.max(a)]
            r += f(llm["mse"][n]) + f(np.abs(llm["bias"][n])) + f(llm["shape"][n]) + list(llm["bias"][n])
            rows.append(r)
    assert len(rows) == 139 and sum(r[0] == "interpolation" for r in rows) == 96
    # reproduce the figures' plotted values by the figures' own PART readers
    for r in rows:
        n = "dc_" + r[3]
        for t in ARMS:
            i = cols.index(f"{SHORT[t]}_mse_logm2")
            s = json.load(open(os.path.join(HERE, hp.PART["mse"][0], f"{t}_heldout.json")))["processes"][n]
            assert hp.PART["mse"][0] == hp.PART["bias"][0] == hp.PART["shape"][0]   # one scoring pass for every part
            assert r[i] == s["mse_logm2"] and abs(r[i + 1]) == abs(s["bias_logm2"]) and r[i + 2] == s["mse_logm2"] - s["bias_logm2"] ** 2
    with open(os.path.join(OUT, "per_process.csv"), "w", newline="") as fh:
        fh.write("# arms: lloca_same_setup = LLoCa, same setup as LLM; lloca_same_setup_diagram_level = LLoCa, same setup as LLM + diagrams to remove bias; lloca_new_preprocessing = LLoCa + new preprocessing\n")
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    tags, pick, ev, smeta = sd.main()
    with open(os.path.join(OUT, "shape_dists_events.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["process_id", "event", "true_logm2"] + [f"{SHORT[t]}_{k}" for t in tags for k in ("pred_logm2", "pred_logm2_bias_removed")])
        for n in pick:
            y = ev[tags[0]][n][:, 0]
            q = {t: ev[t][n][:, 1] for t in tags}
            for i in range(len(y)):
                w.writerow([smeta[n]["folder"], i, y[i]] + [v for t in tags for v in (q[t][i], q[t][i] - np.mean(q[t] - y))])
    open(os.path.join(OUT, "README.txt"), "w").write(README.format(arms=", ".join(SHORT[t] for t in tags), absent="".join(
        f"\n  {SHORT[t]} ({hp.ARMS[t][0].replace(chr(10), ' ')}) is absent from this file and from the distribution figure: its "
        "per-event predictions are not scored yet (re-read queued); it is in per_process.csv." for t in ARMS if t not in tags), pids=", ".join(smeta[n]["folder"] for n in pick)))
    shutil.make_archive(OUT, "zip", HERE, "export_for_suprio")
    print("wrote", OUT + ".zip", os.path.getsize(OUT + ".zip"), "bytes")


README = """Plotting data of the held-out per-process comparison (LLoCa models and the LLM).

Held-out processes: 139 = 96 interpolation (relabels of training processes) + 43 extrapolation (new processes).
All quantities are of the natural logarithm ln|M|^2 of the squared amplitude, per process.

Models
  lloca_same_setup                 LLoCa, same setup as LLM
  lloca_same_setup_diagram_level   LLoCa, same setup as LLM + diagrams to remove bias
  lloca_new_preprocessing          LLoCa + new preprocessing
  llm                              LLM with pIDs (pID embedding only, seeds 0-2; median, min, max over seeds)

Definitions (per process)
  mse    = mean((pred - true)^2)
  bias   = mean(pred - true)            (signed)
  shape  = mse - bias^2                 (the error with the level removed)
  All three from one scoring pass, so shape = mse - bias^2 exactly; the figures plot these values.

per_process.csv: one row per held-out process, in the x order of the figures (row, then family block, then LLM MSE).
  row: interpolation or extrapolation; class: family label of the figure; process: tick label; process_id: folder name;
  is_2to2: 1 for 2->2 processes (shaded in the figures), 0 for 2->3.
  <model>_mse_logm2, _bias_logm2 (signed), _shape_logm2: the LLoCa columns, from the first 10000 test events of each process.
  llm_mse_*, llm_abs_bias_*, llm_shape_*: median, min, max over the three LLM seeds (the figures plot the absolute bias);
  llm_bias_seed0..2: the signed LLM bias per seed. LLM values are from the 16384 test events of each process.

shape_dists_events.csv: the six extrapolation processes drawn in the distribution figure (one per family): {pids}.
  Per event (first 5000 test events per process, same events for every model): process_id, event index, true_logm2,
  and per model (available: {arms}) pred_logm2 and pred_logm2_bias_removed = pred - mean(pred - true) over the process.
  The LLM has no per-event predictions here.{absent}
"""

if __name__ == "__main__":
    main()
