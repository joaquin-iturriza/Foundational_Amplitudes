"""Consistency check of the het retrains' test evaluations (Fig. het_delta_summary): per cell, the raw-space log ratio
ln(A_pred/A_true) from preds_test.npz against the standardized residual (mu - y) of sigma_test.npz and of preds_test.npz,
and the run's eval settings. A correct raw inversion gives ln r = (mu - y) * s per event for a per-cell scale s, so
events with |ln r| > 1 and a small standardized residual are an evaluation error, not the model.
Run on the site holding $SCRATCH/zz_delta_het (CC):  python analysis/transfer/het_eval_check.py <key> ...
Prints one line 'HETCHK <json>'."""
import glob, json, os, sys
import numpy as np

root = os.path.join(os.environ["SCRATCH"], "zz_delta_het")
out = {}
for key in sys.argv[1:]:
    ds = glob.glob(os.path.join(root, key + "__sigma_stage")) + glob.glob(os.path.join(root, "*", key.split("__")[-1] + "__sigma_stage"))
    if not ds:
        out[key] = "missing"; continue
    d = ds[0]; o = {"dir": d, "files": sorted(os.listdir(d))[:40]}
    P = np.load(os.path.join(d, "preds_test.npz")); S = np.load(os.path.join(d, "sigma_test.npz"))
    p, t = P["pred"].reshape(len(P["raw_pred"]), -1), P["truth"].reshape(-1)
    rp, rt = P["raw_pred"].reshape(-1).astype(float), P["raw_truth"].reshape(-1).astype(float)
    mu, y, sg = (S[k].reshape(-1).astype(float) for k in ("pred", "truth", "sigma"))
    lr = np.log(np.abs(rp) / np.abs(rt)); res = mu - y
    ok = np.abs(lr) < 0.5
    s = float(np.median(lr[ok] / np.where(res[ok] == 0, np.nan, res[ok]))) if ok.any() else None
    bad = np.abs(lr) > 1
    o.update(n=len(lr), pred_cols=int(p.shape[1]), preds_vs_sigma_mu_maxdiff=float(np.max(np.abs(p[:, 0] - mu))),
             truth_maxdiff=float(np.max(np.abs(t - y))), scale=s, n_bad=int(bad.sum()),
             neg_pred=int((rp < 0).sum()), neg_truth=int((rt < 0).sum()), zero_pred=int((rp == 0).sum()),
             bad_examples=[dict(lnr=float(lr[i]), res_std=float(res[i]), sigma=float(sg[i]), y=float(y[i]),
                                raw_true=float(rt[i]), raw_pred=float(rp[i])) for i in np.where(bad)[0][:6]],
             good_res_median=float(np.median(np.abs(res[~bad]))), bad_res_median=float(np.median(np.abs(res[bad]))) if bad.any() else None,
             sigma_median_good=float(np.median(sg[~bad])), sigma_median_bad=float(np.median(sg[bad])) if bad.any() else None)
    for f in ("eval_delta.json",):
        try: o[f] = json.load(open(os.path.join(d, f)))
        except Exception as e: o[f] = str(e)
    cf = os.path.join(d, "config.yaml")
    if os.path.exists(cf):
        o["cfg"] = [l.strip() for l in open(cf) if any(k in l for k in ("allow_tf32", "loss:", "sign_head", "preprocess_per_dataset", "amp_trafos", "target_physics", "dtype"))][:20]
    out[key] = o
print("HETCHK " + json.dumps(out, default=str))
