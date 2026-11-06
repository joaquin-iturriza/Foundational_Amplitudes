#!/usr/bin/env python3
"""plan warm_start step 6 lead (D303): what the restart's lr/schedule does to a warm start in its first steps.
Per run, from its train_steps.npz on CC (per-step lr, grad norm before clipping, training loss): the lr, grad norm and
loss at steps 0-2, the largest grad norm and training loss over the first 20 / 40 steps, the peak lr and its step,
and the final loss; for the ws_c1_k533_hpo8 trials beside each its lr_scale and warmup fraction; and what the parent's
checkpoint carries of optimizer state (number of AdamW and Muon state entries; the fine-tune path loads none of it).
Usage: python3 analysis/catalog_v4/ws_restart_steps.py [--cached]   -> data/ws_restart_steps.json (--cached: reread it)
The last line printed is the whole record as one JSON line (docs/results.tex claims)."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from analysis.common import site_python  # noqa: E402

SCRIPT = r'''
import glob, gzip, json, numpy as np, torch, yaml
RUNS = {"warm start, default lr (k=533, s42)": "runs/catalog_v4_short/v4_het_warm25_k533",
        "grown 8 to 16, 533 steps (s42)": "runs/catalog_v4_long/v4_het_default25_w16_grow_k533",
        "pretrained start (het control, s42)": "runs/catalog_v4_short/v4_het_control"}
RUNS.update({"hpo8 " + d.rsplit("/", 1)[1]: d for d in sorted(glob.glob("runs/ws_c1_k533_hpo8/trial_*"))})
out = {}
for lab, d in RUNS.items():
    z = np.load(d + "/train_steps.npz"); g, l, lr = z["grad_norm"], z["loss"], z["lr"]
    c = yaml.safe_load(open(d + "/config.yaml"))
    out[lab] = {"run_dir": d, "steps": len(g), "lr_scale": (c.get("fine_tune") or {}).get("lr_scale"),
                "warmup_frac": c["training"]["cosanneal_warmup_frac"], "peak_lr": float(lr.max()),
                "peak_at": int(lr.argmax()), "lr_0_2": lr[:3].tolist(), "grad_0_2": g[:3].tolist(),
                "loss_0_2": l[:3].tolist(), "grad_max20": float(g[:20].max()), "grad_max20_at": int(g[:20].argmax()),
                "loss_max40": float(l[:40].max()), "loss_end": float(l[-1])}
ck = torch.load(gzip.open("runs/catalog_v4_short/v4_het_control/models/model_run0.pt.gz"), map_location="cpu",
                weights_only=False)
o = ck["optimizer"]
out["_parent_optimizer_state"] = {"muon_state_entries": len(o["state"]), "adamw_state_entries": len(o["adamw"]["state"])}
print("RESTART " + json.dumps(out))
'''

if __name__ == "__main__":
    f = os.path.join(HERE, "data", "ws_restart_steps.json")
    if "--cached" in sys.argv:      # reread the committed JSON (docs/results.tex claims), no site call
        out = json.load(open(f))
    else:
        out = site_python(SCRIPT, [], "RESTART")
        json.dump(out, open(f, "w"), indent=1)
    for k, r in out.items():
        if k.startswith("_"):
            print(k, r); continue
        print(f"{k:42s} lr_scale {r['lr_scale']} lr@2 {r['lr_0_2'][2]:.2e} grad@0,1,2 "
              + " ".join(f"{x:.3g}" for x in r["grad_0_2"]) + f" gmax20 {r['grad_max20']:.3g}@{r['grad_max20_at']}"
              f" loss@0 {r['loss_0_2'][0]:.3f} max40 {r['loss_max40']:.3f} end {r['loss_end']:.3f}")
    # the last line, keyed without spaces for docs/results.tex claims: default, grown, parent, trial_NNNN
    short = {"warm start, default lr (k=533, s42)": "default", "grown 8 to 16, 533 steps (s42)": "grown",
             "pretrained start (het control, s42)": "parent"}
    print(json.dumps({short.get(k, k.replace("hpo8 ", "").lstrip("_")): v for k, v in out.items()}))
