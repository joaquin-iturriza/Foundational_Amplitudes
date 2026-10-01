"""One cell of the transfer study (a probe at D = 10^(k/2)), scratch and fine-tuned from the factor-off ee->uu
pretraining, read from analysis/transfer/scratch_sweeps.json. Shared by transfer_all.py and transfer_ch1.py so the
table and every figure use one rule.

A cell is one search. Where its best sat at the top of its lr window (scratch lr, fine-tune lr_scale) the search was
extended in place (sweep/extend_sweep.py): the window-shifted runs (scratch tp3_scrh, tp3_scrh10; fine-tune
tp3_ftph, tp3_fth) were folded into its DyHPO state and trials added, so its trials are spread over those sweeps'
result files; its value is the best over all of them (MSE of log|M|^2 at the best checkpoint). The scratch arm is the
search on the fine-tune's target (equal prepd_std, asserted): tp3_scr for the probes the factors touch, tp2_scr where
only the t-channel factor reached the pool, tp_scr where neither did. ee->WW takes the sigma-steered pool from
D = 10^2.5 on (tp3s_scr, tp3s_ftp; the user's call, docs/results.tex sec:ladder), scored on its own validation split.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))
ARM = {"ee_ddbar": "tp3_scr", "ee_nnbar": "tp3_scr", "ee_dd_nlo": "tp3_scr", "ee_bb_nlo": "tp3_scr", "ee_WW": "tp3_scr",
       "ee_Za": "tp2_scr", "ud_ud": "tp2_scr", "uubar_gg": "tp2_scr", "uubar_Zg": "tp2_scr",
       "ee_ttbar": "tp_scr", "uubar_Zgg": "tp_scr", "uubar_Zggg": "tp_scr"}
STEERED_FROM = {"ee_WW": 5}


def best(name):
    """(loss, trial) of one sweep's best trial, (None, None) if it has none."""
    tr = [t for t in S.get(name, []) if t.get("val_loss") is not None and t.get("prepd_std")]
    if not tr:
        return None, None
    b = min(tr, key=lambda t: t["val_loss"] * t["prepd_std"] ** 2)
    return b["val_loss"] * b["prepd_std"] ** 2, b


def _pick(names):
    got = [g for g in (best(n) for n in names) if g[0] is not None]
    stds = {round(g[1]["prepd_std"], 5) for g in got}
    assert len(stds) <= 1, f"{names}: searches on different targets (prepd_std {stds})"
    return min(got, key=lambda g: g[0]) if got else (None, None)


def scratch(p, k):
    if k >= STEERED_FROM.get(p, 99):
        return _pick([f"tp3s_scr_{p}_d{k}"])
    return _pick([f"{ARM[p]}_{p}_d{k}", f"tp3_scrh_{p}_d{k}", f"tp3_scrh10_{p}_d{k}"])


def finetune(p, k):
    if k >= STEERED_FROM.get(p, 99):
        return _pick([f"tp3s_ftp_{p}_d{k}"])
    if f"tp3_ftp_{p}_d{k}" in S:          # internal Z: the pretraining's off-shellness scale
        return _pick([f"tp3_ftp_{p}_d{k}", f"tp3_ftph_{p}_d{k}"])
    return _pick([f"tp3_ft_{p}_d{k}", f"tp3_fth_{p}_d{k}"])
