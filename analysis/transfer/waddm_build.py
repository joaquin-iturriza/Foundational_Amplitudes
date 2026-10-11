"""Build analysis/transfer/waddm_curves.json (read by waddm_tradeoff.py) from waddm_collect.py's ADDHB64 lines and the
finale's reference evaluation.
    for s in ccin2p3 lxplus; do site run $s FA -- python - < analysis/transfer/waddm_collect.py > /tmp/waddm_$s.txt; done
    site logs FA-ccin2p3-4e4e6153 > /tmp/finale_eval.txt          # its first EVAL line is the finale (trial 73)
    python analysis/transfer/waddm_build.py --eval /tmp/finale_eval.txt ccin2p3=/tmp/waddm_ccin2p3.txt lxplus=/tmp/waddm_lxplus.txt
Per run: W = the run's val_loss_no_reg curve (geometric mean over W+4g, W+5g, preprocessed), all30 = its
val_loss_no_reg_all, old28 = the 28 monitor processes' curves. A trial present on two sites (pending trials moved between
sites re-ran under the same HPs and gave the same curves) counts once, from the first site given; *_r<k> requeues of a
trial count once, the first run. Runs without per_process_metrics.json (not started or pruned) are left out."""
import argparse, base64, gzip, json, os, re

W_PROCS = ("udbar_Wgggg", "udbar_Wggggg")
ap = argparse.ArgumentParser()
ap.add_argument("--eval", required=True, help="log holding the finale's EVAL line (the first one)")
ap.add_argument("--ref-run", default="FA-ccin2p3-4e4e6153")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "waddm_curves.json"))
ap.add_argument("sites", nargs="+", help="SITE=collect_output, first site wins on duplicates")
a = ap.parse_args()

ev = next(json.loads(l[5:]) for l in open(a.eval) if l.startswith("EVAL "))
assert "tp3_finale/" in ev["weights"], ev["weights"]
fin = {k: v for k, v in ev["val_mse_prepd"].items() if k not in W_PROCS and k != "combined"}
assert len(fin) == 28, len(fin)

runs, seen = [], set()
for spec in a.sites:
    site, path = spec.split("=", 1)
    line = next(l for l in open(path) if l.startswith("ADDHB64 "))
    data = json.loads(gzip.decompress(base64.b64decode(line[len("ADDHB64 "):])))
    for sweep, x in sorted(data.items()):
        for trial, r in sorted(x["runs"].items()):
            base = re.sub(r"_r\d+$", "", trial)
            if r.get("pm") is None or (sweep, base) in seen:
                continue
            seen.add((sweep, base))
            old = {p: r["proc"][p] for p in fin}
            runs.append({"sweep": sweep, "trial": base, "site": site, "every": r["every"], "hp": r["hp"],
                         "W": r["sel"], "all30": r["all"], "old28": old})
json.dump({"finale_old28": fin, "ref_run": f"{a.ref_run} (finale trial_0073, val MSE preprocessed)", "runs": runs},
          open(a.out, "w"))
print(f"{len(runs)} runs from {len({r['sweep'] for r in runs})} sweeps -> {a.out}")
