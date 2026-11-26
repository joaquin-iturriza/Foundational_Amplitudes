"""Wrong-sign fraction per sign-changing pool of a sign-head run, from its preds_val.npz (experiment.py writes
`sign` = (N, 2) [predicted > 0, true > 0] and `process_id`, `names` when training.sign_head is on). A pool counts as
sign-changing when its validation truth has both signs. No magnitude is read (no standardization involved).
    python tools/kit_sign_read.py <run dir> [--out <json>]      (--selfcheck: synthetic file)
"""
import glob, json, os, sys
import numpy as np


def read(npz):
    z = np.load(npz)
    assert "sign" in z.files, f"{npz}: no sign column (run without training.sign_head?)"
    s, pid, names = z["sign"].astype(bool), z["process_id"], [str(n) for n in z["names"]]
    out = {}
    for p, n in enumerate(names):
        sel = pid == p
        t = s[sel, 1]
        if sel.any() and t.any() and not t.all():
            out[n] = dict(n=int(sel.sum()), neg_share=float(1 - t.mean()), wrong_sign=float((s[sel, 0] != t).mean()))
    return out


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        import tempfile
        f = os.path.join(tempfile.mkdtemp(), "preds_val.npz")
        np.savez(f, sign=np.array([[1, 1], [0, 1], [0, 0], [1, 1], [1, 1]]), process_id=np.array([0, 0, 0, 1, 1]),
                 names=np.array(["signed", "positive"]))
        r = read(f)
        assert list(r) == ["signed"] and abs(r["signed"]["wrong_sign"] - 1 / 3) < 1e-12, r
        print("selfcheck ok"); sys.exit(0)
    run = sys.argv[1]
    npz = sorted(glob.glob(os.path.join(run, "**", "preds_val.npz"), recursive=True))
    assert npz, f"no preds_val.npz under {run}"
    r = read(npz[-1])
    for n, v in r.items():
        print(f"{n}: wrong sign {v['wrong_sign']:.4f}, negative share {v['neg_share']:.3f}, {v['n']} events")
    if "--out" in sys.argv:
        json.dump(dict(run=run, npz=npz[-1], pools=r), open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
