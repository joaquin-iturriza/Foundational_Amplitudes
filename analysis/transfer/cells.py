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
import atexit, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))
ARM = {"ee_ddbar": "tp3_scr", "ee_nnbar": "tp3_scr", "ee_dd_nlo": "tp3_scr", "ee_bb_nlo": "tp3_scr", "ee_WW": "tp3_scr",
       "ee_Za": "tp2_scr", "ud_ud": "tp2_scr", "uubar_gg": "tp2_scr", "uubar_Zg": "tp2_scr",
       "ee_ttbar": "tp_scr", "uubar_Zgg": "tp_scr", "uubar_Zggg": "tp_scr",
       # the star arms' probes, factors off from the start
       "ee_ddbarg": "tp3_scr", "ee_ttbarg": "tp3_scr", "ee_ttbar_nlo_thr": "tp3_scr", "ee_dd_nlo_hi": "tp3_scr",
       # the redesigned arms' probes (2026-10-04)
       "udbar_enu": "tp3_scr", "ee_dd_ew_nlo": "tp3_scr"}
STEERED_FROM = {"ee_WW": 5}
# trials with a result but no prepd_std (their run dir's data_stats.json was not found): never dropped silently,
# every one a figure skipped is listed when the script ends (CLAUDE.md, Reported values)
UNCONVERTED = set()
atexit.register(lambda: UNCONVERTED and print(f"cells.py: {len(UNCONVERTED)} trial(s) with a result but no prepd_std, "
                                              f"not counted: " + ", ".join(f"{n} hp{h}" for n, h in sorted(UNCONVERTED)),
                                              file=sys.stderr))


def _note_unconverted(name, trials):
    for t in trials:
        if t.get("val_loss") is not None and not t.get("prepd_std"):
            UNCONVERTED.add((name, t["hp"]))


def best(name):
    """(loss, trial) of one sweep's best trial, (None, None) if it has none."""
    # a sweep finished on another site continues as <name>_002 (sweep/sweep_config_*_002.yaml): one search
    _note_unconverted(name, S.get(name, []) + S.get(name + "_002", []))
    tr = [t for t in S.get(name, []) + S.get(name + "_002", []) if t.get("val_loss") is not None and t.get("prepd_std")]
    if not tr:
        return None, None
    b = min(tr, key=lambda t: t["val_loss"] * t["prepd_std"] ** 2)
    return b["val_loss"] * b["prepd_std"] ** 2, b


def _pick(names):
    got = [g for g in (best(n) for n in names) if g[0] is not None]
    stds = {round(g[1]["prepd_std"], 5) for g in got}
    assert len(stds) <= 1, f"{names}: searches on different targets (prepd_std {stds})"
    return min(got, key=lambda g: g[0]) if got else (None, None)


def scratch(p, k, steered=True):
    """steered=False keeps ee->WW on the mixture pool at every D (a comparison across D on one measure)."""
    if steered and k >= STEERED_FROM.get(p, 99):
        return _pick([f"tp3s_scr_{p}_d{k}"])
    return _pick([f"{ARM[p]}_{p}_d{k}", f"tp3_scrh_{p}_d{k}", f"tp3_scrh10_{p}_d{k}"])


def finetune(p, k, steered=True):
    if steered and k >= STEERED_FROM.get(p, 99):
        return _pick([f"tp3s_ftp_{p}_d{k}"])
    if f"tp3_ftp_{p}_d{k}" in S:          # internal Z: the pretraining's off-shellness scale
        return _pick([f"tp3_ftp_{p}_d{k}", f"tp3_ftph_{p}_d{k}"])
    return _pick([f"tp3_ft_{p}_d{k}", f"tp3_fth_{p}_d{k}"])


# The study's final horizons: 8k steps up to D = 10^3, 32k at D = 10^3.5, 10^4 (tp3_scr32k, tp3_<parent>fte32k), and
# for ud -> ud also at 10^2.5, 10^3 (its 8k cells looked under-trained; the user's call, 2026-10-05). A cell whose 32k
# search is not in yet, or still needs work, keeps its 8k value and is flagged as having less compute for now.
LONG_K = {7, 8}
LONG_K_PROBE = {"ud_ud": {5, 6, 7, 8}}
# scratch ee -> nu_e nu_e: at 32k worse than 8k at D = 10^3.5 even after four extra random points (2.4e-3 against 7.4e-4),
# so that cell keeps 8k; at 10^4 the extra points beat 8k (2.1e-5 against 6.8e-5) and the cell is scored on all its
# trials, since they were run to explore it (the user's call, 2026-10-05)
HOLD_8K = {("scr", "ee_nnbar", 7)}
ALL_TRIALS32 = {"tp3_scr32k_ee_nnbar_d8"}
# On since 2026-10-06, when every pretraining and scratch had its 32k cells (all 420 chosen points: the twelve probes at
# D = 10^3.5, 10^4, ud -> ud also at 10^2.5, 10^3): those cells are read at 32k for every family alike. Off, every figure
# is the 8k grid (the first 32k round covered scratch, ee->uu and one rung per probe only; the user's call, 2026-10-05).
# TP_USE_32K=0 draws the 8k grid alone (the paper draft's figures, whose text quotes the 8k grid)
USE_32K = os.environ.get("TP_USE_32K", "1") == "1"


CHOSEN32 = json.load(open(os.path.join(ROOT, "analysis", "transfer", "horizon32k_chosen.json")))
# the 32k cells added after that file (the finale's and the synthetic pretraining's, ...) list their two chosen points in
# expect_tp3.json; a cell named in both keeps horizon32k_chosen.json's
for _n, _v in json.load(open(os.path.join(ROOT, "analysis", "transfer", "expect_tp3.json"))).items():
    if isinstance(_v, list) and "32k_" in _n:
        CHOSEN32.setdefault(_n, _v)


def best_chosen(name):
    """A 32k cell on its two chosen points only (horizon32k_chosen.json; sweep/pick_points.py), so every 32k cell is the
    best of the same two points: the first round's sweeps also hold 1-2 random start-up trials, the later ones none,
    and counting those would favour the pretrainings of the first round."""
    _note_unconverted(name, [t for t in S.get(name, []) + S.get(name + "_002", [])
                             if name in ALL_TRIALS32 or t["hp"] in CHOSEN32.get(name, [])])
    tr = [t for t in S.get(name, []) + S.get(name + "_002", [])
          if t.get("val_loss") is not None and t.get("prepd_std") and (name in ALL_TRIALS32 or t["hp"] in CHOSEN32.get(name, []))]
    return min(t["val_loss"] * t["prepd_std"] ** 2 for t in tr) if tr else None


def final(fam, p, k):
    """(value, at the final horizon?) of one cell. fam: "scr" for scratch, else a fine-tune family (tp3_r7fte, ...).
    Only the twelve ladder probes have long horizons; every other cell is at its grid horizon and counts as final."""
    short = scratch(p, k, steered=False)[0] if fam == "scr" else best(f"{fam}_{p}_d{k}")[0]
    if not USE_32K:
        return short, True
    if p not in ARM or k not in LONG_K_PROBE.get(p, LONG_K) or p in ("ee_ddbarg", "ee_ttbarg", "ee_ttbar_nlo_thr",
                                                                         "ee_dd_nlo_hi", "udbar_enu", "ee_dd_ew_nlo"):
        return short, True
    v32 = None if (fam, p, k) in HOLD_8K else best_chosen(f"tp3_scr32k_{p}_d{k}" if fam == "scr" else f"{fam}32k_{p}_d{k}")
    return (v32, True) if v32 is not None else (short, False)
