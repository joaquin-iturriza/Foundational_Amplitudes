#!/usr/bin/env python3
"""pretrain_data_saving step 7: the t t~ off-shellness diagnostic's readout (design passed 2026-11-11).

Per cell (t t~ LO and t t~ V, k=5..8) three fine-tunes at the same HPs and seed: offown (fine_tune.offshell_stats=own),
parscale (the parent's scale, an unselected rerun of the stage-2 best trial) and that original best trial. Val is the
run's RESULT_JSON val_loss (DyHPO's objective, mu-MSE in the target's standardized units, the same target stats in the
three); test is data/stage2_diag_test.json (stage2_readout.py --diag) and, for the originals, data/stage2_test.json:
MSE(ln|M|^2) for LO, MSE(V/B) for V. Verdict by the design's rule: YES if offown < parscale in >= 6 of 8 cells and
the geometric mean of offown/parscale <= 1/3; NO if the geometric mean >= 1/1.5 or fewer than 5 of 8 below; else
inconclusive. Writes data/stage2_diag_readout.json and figs/stage2_diag.{png,pdf}.
With the argument `tops` (D400 stage A, same rule): the fine-tunes from the r5 parents (catalog_v4's physical-mass
top classes added) in place of offown, against the same parscale reruns from r4 (the 'own' fields then hold the tops
arm); writes data/stage2_diag_tops_readout.json and figs/stage2_diag_tops.{png,pdf}.

    python analysis/data_saving/diag_readout.py [tops]
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

DATA = os.path.join(REPO, "analysis/data_saving/data")
RUNS = json.load(open(os.path.join(DATA, "stage2_diag_runs.json")))
_ok = [r for r in json.load(open(os.path.join(DATA, "stage2_diag_test.json"))) if "error" not in r]
assert len({r["sweep"] for r in _ok}) == len(_ok), "two good attempts of one diagnostic sweep: key the test by run dir"
TEST_DIAG = {r["sweep"]: r for r in _ok}
TEST_S2 = {(r["sweep"], r["hp_idx"]): r for r in json.load(open(os.path.join(DATA, "stage2_test.json")))}
ARM = sys.argv[1] if len(sys.argv) > 1 else "offown"
SUF = "" if ARM == "offown" else "_" + ARM
LABEL = {"offown": "own scale", "tops": "top parent"}[ARM]
TARGETS = [("epem_ttx", r"$e^+e^-\to t\bar t$", "mse_ln"), ("ee_ttbar_nlo", r"$e^+e^-\to t\bar t$, $V$", "mse_vb")]


def val(rid):
    """The run's RESULT_JSON val_loss, read from its log."""
    out = subprocess.run(["site", "logs", rid, "-n", "100000"], capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if "RESULT_JSON" in l][-1]
    return json.loads(line.split("RESULT_JSON", 1)[1].strip(" :="))["val_loss"]


def main():
    rows = []
    for tg, _, key in TARGETS:
        for k in (5, 6, 7, 8):
            own, par = (RUNS[f"dsave_diag_{a}_ft_{tg}_d{k}_t8500"] for a in (ARM, "parscale"))
            s2 = f"dsave_s2_ft_{tg}_d{k}_t8500"
            r = dict(target=tg, k=k, D=10 ** (k / 2), own=own["rid"], par=par["rid"], orig=own["control"],
                     val_own=val(own["rid"]), val_par=val(par["rid"]), val_orig=val(own["control"]),
                     test_own=TEST_DIAG[f"dsave_diag_{ARM}_ft_{tg}_d{k}_t8500"][key],
                     test_par=TEST_DIAG[f"dsave_diag_parscale_ft_{tg}_d{k}_t8500"][key],
                     test_orig=TEST_S2[(s2, own["hp_idx"])][key])
            r["own_over_par"] = r["val_own"] / r["val_par"]
            r["par_over_orig"] = r["val_par"] / r["val_orig"]
            rows.append(r)
    q = np.array([r["own_over_par"] for r in rows])
    below, gm = int((q < 1).sum()), float(np.exp(np.log(q).mean()))
    verdict = "YES" if below >= 6 and gm <= 1 / 3 else "NO" if gm >= 1 / 1.5 or below < 5 else "inconclusive"
    spread = [r["par_over_orig"] for r in rows]
    summ = dict(n_below=below, gmean_own_over_par=gm, verdict=verdict,
                par_over_orig_range=[min(spread), max(spread)],
                gmean_own_over_par_by_target={tg: float(np.exp(np.mean([np.log(r["own_over_par"]) for r in rows
                                                                          if r["target"] == tg]))) for tg, _, _ in TARGETS},
                # per target on both measures (the pre-set rule pools LO and V; read per target 2026-11-12)
                by_target={tg: {m: {"n_below": int(sum(q < 1 for q in qq)), "gmean": float(np.exp(np.mean(np.log(qq))))}
                                for m, qq in (("val", [r["own_over_par"] for r in rows if r["target"] == tg]),
                                              ("test", [r["test_own"] / r["test_par"] for r in rows if r["target"] == tg]))}
                           for tg, _, _ in TARGETS},
                test_par_over_orig_range=[min(r["test_par"] / r["test_orig"] for r in rows),
                                          max(r["test_par"] / r["test_orig"] for r in rows)])
    json.dump(dict(cells=rows, summary=summ), open(os.path.join(DATA, f"stage2_diag{SUF}_readout.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['target']:13s} D={r['D']:7.0f} val own {r['val_own']:.2e} par {r['val_par']:.2e} orig "
              f"{r['val_orig']:.2e}  own/par {r['own_over_par']:.2f}  par/orig {r['par_over_orig']:.2f}  "
              f"test own {r['test_own']:.2e} par {r['test_par']:.2e} orig {r['test_orig']:.2e}")
    print(json.dumps(summ))

    fig, axs = ps.figure(ncols=2)
    # words for the arms (Joaquin 2026-11-12): what each run is, no internal names; one seed and the same HPs in all three
    par = "parent without tops, rerun" if ARM == "tops" else "parent's off-shell scale, rerun"
    arms = [("val_own", "parent with tops" if ARM == "tops" else "own off-shell scale", ps.C.vermillion, "s"),
            ("val_par", par, ps.C.blue, "o"), ("val_orig", "parent without tops, search's pick", ps.C.grey, "^")]
    for n, (ax, (tg, lab, _)) in enumerate(zip(axs, TARGETS)):
        cc = [r for r in rows if r["target"] == tg]
        # val is in the target's standardized units (same stats in the three arms): re-based to ln units
        sd2 = [TEST_DIAG[f"dsave_diag_parscale_ft_{tg}_d{r['k']}_t8500"]["prepd_std"] ** 2 for r in cc]
        for f, name, col, m in arms:
            ax.plot([r["D"] for r in cc], [r[f] * s for r, s in zip(cc, sd2)], marker=m, linestyle="-", color=col,
                    label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$D$")
        ax.set_ylabel(r"val MSE$(\ln|\mathcal{M}|^2)$" if tg == "epem_ttx" else r"val MSE$(\ln(V/B))$")
        ps.process_label(ax, lab)
        if n == 0:
            ps.legend(ax, "lower left")
    ps.save(fig, os.path.join(REPO, "analysis/data_saving/figs/stage2_diag" + SUF))


if __name__ == "__main__":
    main()
