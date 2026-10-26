"""Copy home the validation curves of the Muon test's catalog_v3 fine-tunes (tools/catalog_ft.py result files under
$SCRATCH/zz_catft*/ on CC): the runs before the EMA-start fix (zz_catft) and the reruns at ddfc0314 (zz_catft_emafix,
the parent-statistics kind; zz_catft_refit_emafix, the refit kind). Read-only on the site (site run, no job).
Per result file: val_loss, best_step, validate_every, val_curve, val_loss_step0 (only the reruns have it).
Prints how many of the 128 rerun files (2 kinds x 4 parent-and-optimizer pairs x 4 probes x 4 D cells) are present,
so a partial copy is never read as the whole rerun.
    python3 analysis/catalog_v3/collect_catft_curves.py   -> analysis/catalog_v3/data/catft_curves.json
"""
import base64, json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE = r'''
import json, glob, os
S = os.environ["SCRATCH"]; out = {}
for f in sorted(glob.glob(S + "/zz_catft*/*/*_result.json")):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    out[f[len(S) + 1:]] = {k: r.get(k) for k in ("val_loss", "best_step", "validate_every", "val_curve", "val_loss_step0")}
print("CATJSON" + json.dumps(out))
'''


def main():
    p = subprocess.run(["site", "run", "ccin2p3", "FA", "--", # the script travels base64-encoded: no quoting through ssh and the remote shell
                        "python3 -c \"import base64; exec(base64.b64decode('%s'))\"" % base64.b64encode(REMOTE.encode()).decode()],
                       capture_output=True, text=True, timeout=600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("CATJSON")), None)
    if line is None:
        raise SystemExit(f"no CATJSON line from the site (exit {p.returncode}): {p.stderr[-500:]}")
    d = json.loads(line[len("CATJSON"):])
    json.dump(d, open(os.path.join(HERE, "data", "catft_curves.json"), "w"))
    rer = [k for k in d if "emafix" in k]
    print(f"{len(d)} result files; reruns at the fix: {len(rer)} of 128 "
          f"(parent statistics {sum('zz_catft_emafix/' in k for k in rer)}, refit {sum('refit_emafix' in k for k in rer)})")


if __name__ == "__main__":
    main()
