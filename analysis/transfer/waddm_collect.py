"""Continual add-W along the run (addhoriz): per tp3_waddm_ trial on this site, the W and monitor validation curves (per_process_metrics.json) and the HPs, read-only. Run on each site, stdin: site run <site> FA -- python - < analysis/transfer/waddm_collect.py; the ADDHB64 line is gzip+base64 JSON, merged into waddm_curves.json."""
import siteconf, os, glob, json, math, yaml
out = {}
roots = {siteconf.SWEEP_DIR, getattr(siteconf, "RESULTS_DIR", siteconf.SWEEP_DIR)}
runroots = {os.path.join(siteconf.PROJECT_DIR, "runs")}
for d in sorted(glob.glob(os.path.join(siteconf.SWEEP_DIR, "tp3_waddm_*"))):
    name = os.path.basename(d)
    res = []
    for r in roots:
        res += glob.glob(os.path.join(r, name, "results", "hp*_t*.json"))
    rs = {}
    for f in sorted(set(res)):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        rs[os.path.basename(f)] = {"val_loss": j.get("val_loss"), "best_step": j.get("best_step"),
                                   "h": j.get("traintime_hours"), "mtime": os.path.getmtime(f)}
    runs = {}
    for rr in runroots:
        for td in sorted(glob.glob(os.path.join(rr, name, "trial_*"))):
            pm = glob.glob(os.path.join(td, "**", "per_process_metrics.json"), recursive=True)
            if not pm:
                runs[os.path.basename(td)] = {"pm": None}
                continue
            p = json.load(open(pm[0]))
            cfgf = os.path.join(td, "config.yaml")
            hp = {}
            if os.path.exists(cfgf):
                c = yaml.safe_load(open(cfgf))
                t, ft = c.get("training", {}), c.get("fine_tune", {})
                hp = {"lr": t.get("lr"), "lam": t.get("regularization_lambda"),
                      "ewc": (ft.get("ewc") or {}).get("lambda") if (ft.get("ewc") or {}).get("enabled") else None,
                      "frz": ft.get("freeze_blocks"), "ld": ft.get("layer_decay"),
                      "lora": (ft.get("lora") or {}).get("enabled"), "steps": t.get("iterations")}
            runs[os.path.basename(td)] = {
                "pm": os.path.relpath(pm[0], td), "every": p.get("validate_every_n_steps"),
                "mon": p.get("monitor_processes"), "sel": p.get("val_loss_no_reg"),
                "all": p.get("val_loss_no_reg_all"),
                "proc": p.get("proc_val_losses_no_reg"), "hp": hp, "mtime": os.path.getmtime(pm[0])}
    out[name] = {"results": rs, "runs": runs, "moved": os.path.exists(os.path.join(d, "MOVED_TO"))}
s = json.dumps(out)
print("ADDH", len(s))
import gzip, base64
print("ADDHB64 " + base64.b64encode(gzip.compress(s.encode())).decode())
