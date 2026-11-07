#!/usr/bin/env python3
"""plan warm_start step 6 lead (D303): what the restart's lr/schedule does to a warm start in its first steps.
Per run, from its train_steps.npz on CC (per-step lr, grad norm before clipping, training loss): the lr, grad norm and
loss at steps 0-2, the largest grad norm and training loss over the first 20 / 40 steps, the peak lr and its step,
and the final loss; for the ws_c1_k533_hpo8 trials beside each its lr_scale and warmup fraction; and what the parent's
checkpoint carries of optimizer state (number of AdamW and Muon state entries; the fine-tune path loads none of it).
Usage: python3 analysis/catalog_v4/ws_restart_steps.py [--cached]   -> data/ws_restart_steps.json (--cached: reread it)
The last line printed is the whole record as one JSON line (docs/results.tex claims)."""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from analysis.common import site_python  # noqa: E402

SCRIPT = r'''
import glob, gzip, json, os, numpy as np, torch, yaml
RUNS = {"warm start, default lr (k=533, s42)": "runs/catalog_v4_short/v4_het_warm25_k533",
        "grown 8 to 16, 533 steps (s42)": "runs/catalog_v4_long/v4_het_default25_w16_grow_k533",
        "pretrained start (het control, s42)": "runs/catalog_v4_short/v4_het_control",
        "moment fill (k=533, s42)": "runs/catalog_v4_short/v4_het_warm25f_k533",
        "moment fill (k=533, s43)": "runs/catalog_v4_short/v4_het_warm25f_k533_seed2",
        "default restart, same commit (k=533, s42)": "runs/catalog_v4_short/v4_het_warm25_k533_tagd303",
        "default restart, same commit (k=533, s43)": "runs/catalog_v4_short/v4_het_warm25_k533_seed2_tagd303",
        "Muon lr damped (k=533, s42)": "runs/catalog_v4_short/v4_het_warm25u_k533",
        "Muon lr damped (k=533, s43)": "runs/catalog_v4_short/v4_het_warm25u_k533_seed2",
        "AdamW lr damped (k=533, s42)": "runs/catalog_v4_short/v4_het_warm25a_k533",
        "AdamW lr damped (k=533, s43)": "runs/catalog_v4_short/v4_het_warm25a_k533_seed2",
        "every group damped (k=533, s42)": "runs/catalog_v4_short/v4_het_warm25e_k533",
        "every group damped (k=533, s43)": "runs/catalog_v4_short/v4_het_warm25e_k533_seed2"}
RUNS.update({"hpo8 " + d.rsplit("/", 1)[1]: d for d in sorted(glob.glob("runs/ws_c1_k533_hpo8/trial_*"))})
# step 12 (D307, D308): the restart study from the checkpoint rerun v4_het_control_ckpt[_seed2], arms v4_het_ws_*
for _a in ("final_default", "step3198_default", "step3198_continue", "step2132_continue", "final_lr3x",
           "step3198_continue_optstate"):   # step 13 (D307, D308): continued with the parent's optimizer state
    for _s, _sfx in (("s42", ""), ("s43", "_seed2")):
        RUNS["restart study " + _a + " " + _s] = "runs/catalog_v4_short/v4_het_ws_" + _a + _sfx
out = {}
for lab, d in RUNS.items():
    if not os.path.exists(d + "/train_steps.npz"):
        out[lab] = {"run_dir": d, "missing": True}; continue
    z = np.load(d + "/train_steps.npz"); g, l, lr = z["grad_norm"], z["loss"], z["lr"]
    c = yaml.safe_load(open(d + "/config.yaml"))
    out[lab] = {"run_dir": d, "steps": len(g), "lr_scale": (c.get("fine_tune") or {}).get("lr_scale"),
                "warmup_frac": c["training"]["cosanneal_warmup_frac"], "peak_lr": float(lr.max()),
                "peak_at": int(lr.argmax()), "lr_0_2": lr[:3].tolist(), "grad_0_2": g[:3].tolist(),
                "loss_0_2": l[:3].tolist(), "grad_max20": float(g[:20].max()), "grad_max20_at": int(g[:20].argmax()),
                "loss_max40": float(l[:40].max()), "loss_end": float(l[-1]),
                "grad_max_2_10": float(g[2:11].max()), "grad_max_2_10_at": 2 + int(g[2:11].argmax()),
                "loss_max_2_10": float(l[2:11].max()), "grad_max_all": float(g.max()), "grad_max_all_at": int(g.argmax()),
                "loss_max_all": float(l.max()), "loss_max_all_at": int(l.argmax()), "loss_0_60": l[:60].tolist(),
                "grad_0_60": g[:60].tolist()}
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
        if k.startswith("_") or r.get("missing"):
            print(k, r); continue
        print(f"{k:42s} lr_scale {r['lr_scale']} lr@2 {r['lr_0_2'][2]:.2e} grad@0,1,2 "
              + " ".join(f"{x:.3g}" for x in r["grad_0_2"]) + f" gmax20 {r['grad_max20']:.3g}@{r['grad_max20_at']}"
              f" loss@0 {r['loss_0_2'][0]:.3f} max40 {r['loss_max40']:.3f} end {r['loss_end']:.3f}")
    # the last line, keyed without spaces for docs/results.tex claims: default, grown, parent, trial_NNNN
    short = {"warm start, default lr (k=533, s42)": "default", "grown 8 to 16, 533 steps (s42)": "grown",
             "pretrained start (het control, s42)": "parent", "moment fill (k=533, s42)": "fill_s42",
             "moment fill (k=533, s43)": "fill_s43", "default restart, same commit (k=533, s42)": "default_d303_s42",
             "default restart, same commit (k=533, s43)": "default_d303_s43",
             "Muon lr damped (k=533, s42)": "muon_damped_s42", "Muon lr damped (k=533, s43)": "muon_damped_s43",
             "AdamW lr damped (k=533, s42)": "adamw_damped_s42", "AdamW lr damped (k=533, s43)": "adamw_damped_s43",
             "every group damped (k=533, s42)": "all_damped_s42", "every group damped (k=533, s43)": "all_damped_s43"}
    res = {short.get(k, k.replace("hpo8 ", "").replace("restart study ", "ws_").replace(" ", "_").lstrip("_")): v
           for k, v in out.items()}
    # step 8 (D303, design passed round 2 2026-11-06): Muon's share of the jump over steps 2-10,
    # m = ln(G_ctrl / G_arm) / ln(G_ctrl / G_all), G_ctrl the same-seed default restart, G_all trial_1860's window max
    # (every group damped, seed 42), scaled to seed 43 by the controls' ratio (an assumption: hpo8 ran seed 42 only)
    g = lambda k: res.get(k, {}).get("grad_max_2_10")
    if g("muon_damped_s42") and g("trial_1860"):
        res["muon_share"] = {}
        for s in ("s42", "s43"):
            ctrl, arm = g("default_d303_" + s), g("muon_damped_" + s)
            if not arm:
                continue
            g_all = g("trial_1860") * ctrl / g("default_d303_s42")
            res["muon_share"][s] = {"G_ctrl": ctrl, "G_arm": arm, "G_all": g_all,
                                    "m": math.log(ctrl / arm) / math.log(ctrl / g_all)}
        print("muon_share", res["muon_share"])
    # step 9 (D303, step 8's NOT MUON branch): AdamW's share a = ln(G_ctrl / G_adamw) / ln(G_ctrl / G_both) and Muon's
    # against the same reference, G_both the same-seed every-group-damped arm (same data stream); arm e's ratio to the
    # control beside (the design's EVERY GROUP / NONE bar: below half the control's window max)
    if g("all_damped_s42") or g("all_damped_s43"):
        res["step9_share"] = {}
        for s in ("s42", "s43"):
            ctrl, ad, mu, both = (g("default_d303_" + s), g("adamw_damped_" + s), g("muon_damped_" + s),
                                  g("all_damped_" + s))
            if not (ctrl and both):
                continue
            den = math.log(ctrl / both)
            res["step9_share"][s] = {"G_ctrl": ctrl, "G_adamw": ad, "G_muon": mu, "G_both": both,
                                     "both_over_ctrl": both / ctrl,
                                     "a": math.log(ctrl / ad) / den if ad and den else None,
                                     "m_vs_both": math.log(ctrl / mu) / den if mu and den else None}
        print("step9_share", res["step9_share"])
    # step 12 (D307, D308, design passed round 2): each arm's jump against the same-seed control (final_default),
    # G = largest grad norm over restart steps 0-19; the jump moved when G_arm / G_ctrl < 0.5 or > 2 at both seeds
    if res.get("ws_final_default_s42", {}).get("grad_max20"):
        res["step12_jump"] = {}
        for a in ("step3198_default", "step3198_continue", "step2132_continue", "final_lr3x",
                  "step3198_continue_optstate"):
            row = {}
            for s in ("s42", "s43"):
                c, r = res.get(f"ws_final_default_{s}", {}), res.get(f"ws_{a}_{s}", {})
                if c.get("grad_max20") and r.get("grad_max20"):
                    row[s] = {"G_ctrl": c["grad_max20"], "G_arm": r["grad_max20"],
                              "G_ratio": r["grad_max20"] / c["grad_max20"],
                              "loss0_ctrl": c["loss_0_2"][0], "loss_max40_ctrl": c["loss_max40"],
                              "loss0_arm": r["loss_0_2"][0], "loss_max40_arm": r["loss_max40"]}
            gr = [v["G_ratio"] for v in row.values()]
            row["moved"] = len(gr) == 2 and (all(x < 0.5 for x in gr) or all(x > 2 for x in gr))
            res["step12_jump"][a] = row
        print("step12_jump", json.dumps(res["step12_jump"]))
    print(json.dumps(res))
