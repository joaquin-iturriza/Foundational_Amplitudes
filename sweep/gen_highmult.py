"""High-multiplicity test (site decisions D20, D21 and the user's 2026-10-09 call that every arm but the continual one gets
its full grid): the Z+4g and Z+5g probes on the study's grid, for scratch and the three pretrainings fine-tuned on them.

  D = 10^(k/2): k = 2..6 on the 8k grid (steps and limits as the ladder's), k = 7..10 at 32k
  scratch     tp3_hmscr_<p>_d<k> (8k), tp3_hmscr32k_<p>_d<k> (32k)        8-trial searches, as the ladder's scratch
  finale      tp3_finfte_<p>_d<k>, tp3_finfte32k_<p>_d<k>                 5-trial searches
  + W+4g/5g   tp3_fwngfte_<p>_d<k>, tp3_fwngfte32k_<p>_d<k>               from tp3_finale_wng hp73
  + syn 2->5/6 tp3_fsynfte_<p>_d<k>, tp3_fsynfte32k_<p>_d<k>               from tp3_finale_synhm hp73
Each config is the uubar_Zggg cell of the same family and D with the probe swapped (recipe, name, its own DyHPO seed);
the 32k cells are searches, not the ladder's two chosen points, since these probes have no 8k landscape to choose from;
d8 at 32k was written first by hand (c5fb2ac) and is kept as is. Time limits are the Zggg cell's x3 at 8k (a Z+5g 32k
trial ran 1.6 h against Zggg's 0.5 h), 10/14/16 h at 32k for k = 7, 9, 10.
    python sweep/gen_highmult.py [--dry-run]     writes sweep/sweep_config_<name>.yaml, prints the names
"""
import argparse, os, re, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES = ["uubar_Zgggg", "uubar_Zggggg"]
PARENT = {"fwngfte": "tp3_finale_wng", "fsynfte": "tp3_finale_synhm"}
T32 = {7: "'10:00:00'", 9: "'14:00:00'", 10: "'16:00:00'"}


def hms3(t):
    h, m, s = (int(x) for x in t.strip("'\"").split(":"))
    m = 3 * (60 * h + m)
    return f"'{m // 60:02d}:{m % 60:02d}:00'"


def cfg(path):
    return open(os.path.join(HERE, f"sweep_config_{path}.yaml")).read()


def make(src, name, probe, time, parent=None):
    """parent: None for the finale (the template's own), a run name, or False for scratch."""
    s = src.replace("uubar_Zggg", probe)
    s = re.sub(r"(?m)^sweep_name: .*$", f"sweep_name: {name}", s)
    s = re.sub(r"(?m)^(  time: ).*$", rf"\g<1>{time}", s, count=1)
    # fine-tunes draw their own DyHPO candidates per sweep (gen_transfer_pilot --eff-lr); scratch keeps the ladder's 42
    seed = zlib.crc32(name.encode()) & 0x7FFFFFFF if parent is not False else 42
    s = re.sub(r"(?m)^(dyhpo:\n(?:  .*\n)*?  seed: )\d+", rf"\g<1>{seed}", s)
    s = s.replace("data.target_propagators: 'true'", "data.target_propagators: 'false'")
    if parent is False:  # the scratch protocol's 8 trials (tab:ladder_scratch); the 32k d9/d10 templates carry 5
        s = re.sub(r"(?m)^n_trials: \d+$", "n_trials: 8", s)
    if parent:  # a run name
        s = s.replace("tp3_finale/trial_0073", f"{parent}/trial_0073")
    head = (f"# High-multiplicity test: {probe}, written by sweep/gen_highmult.py from the uubar_Zggg cell of the same "
            "family and D.\n")
    return head + "\n".join(l for l in s.splitlines() if not l.startswith("#")) + "\n"


ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
out = {}
for p in PROBES:
    for k in range(2, 7):
        src = cfg(f"tp2_scr_uubar_Zggg_d{k}")
        t = re.search(r"(?m)^  time: (.*)$", src).group(1)
        out[f"tp3_hmscr_{p}_d{k}"] = make(src, f"tp3_hmscr_{p}_d{k}", p, hms3(t), False)
        src = cfg(f"tp3_finfte_uubar_Zggg_d{k}")
        for fam in ("finfte", "fwngfte", "fsynfte"):
            out[f"tp3_{fam}_{p}_d{k}"] = make(src, f"tp3_{fam}_{p}_d{k}", p, hms3(t), PARENT.get(fam))
    for k in (7, 9, 10):
        out[f"tp3_hmscr32k_{p}_d{k}"] = make(cfg(f"tp3_scr32k_uubar_Zggg_d{k}"), f"tp3_hmscr32k_{p}_d{k}", p, T32[k], False)
        src = cfg(f"tp3_finfte32k_uubar_Zggg_d{k}")
        for fam in ("finfte", "fwngfte", "fsynfte"):
            out[f"tp3_{fam}32k_{p}_d{k}"] = make(src, f"tp3_{fam}32k_{p}_d{k}", p, T32[k], PARENT.get(fam))
for n, s in out.items():
    if not a.dry_run:
        open(os.path.join(HERE, f"sweep_config_{n}.yaml"), "w").write(s)
    print(n)
