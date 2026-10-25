"""Reads the level check of the hybrid reference design (prep_level.py, its json copied home into data/prep_level.json)
next to the shape proxy (data/prep_shape.json) and the short pretrainings' zero-shot reads (data/short_reads.json), and
prints one flat JSON line, the numbers the design paragraph in docs/results.tex cites:
  comp_<axis>_<target>, compls_<axis>_<target>   median squared error of a class mean (orders removed) predicted
                          by the notes' composition model, and by its variant with the mean log s_hat (prep_level.py)
  net_<axis>_<arm>        the arm's median zero-shot level part (raw: baseline, qperp: dimensionless, dless:
                          dimensionless_shat), the network's level on the same classes
  proxy_<axis>_<target>   the shape proxy's median MSE (the hybrid's shape is dless's by construction)
  cp_<axis>_{median, val_vs_test, n200_vs_2000}   the constant c_p: its median, the median |c_p(val) - c_p(test)| and
                          |c_p(200 events) - c_p(2000)|
  n_<axis>, skipped
    python3 analysis/catalog_v3/prep_level_read.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
AXES = ("train", "H1", "H2", "H2p5", "H3", "H6", "H7c", "H7E")
ARM = {"raw": "baseline", "qperp": "dimensionless", "dless": "dimensionless_shat"}


def main():
    lv = json.load(open(os.path.join(HERE, "data", "prep_level.json")))
    sh = json.load(open(os.path.join(HERE, "data", "prep_shape.json")))
    sr = json.load(open(os.path.join(HERE, "data", "short_reads.json")))
    out = {"skipped": len(lv["skipped"])}
    for ax in AXES:
        if ax not in lv["summary"]["composition"]:
            continue
        out[f"n_{ax}"] = len(lv["classes"].get(ax, {}))
        for model, key in (("composition", "comp"), ("composition+Ls", "compls")):
            for t, v in lv["summary"][model].get(ax, {}).items():
                out[f"{key}_{ax}_{t}"] = round(v, 4)
        for k, v in lv["c_p"][ax].items():
            out[f"cp_{ax}_{k.replace('200_vs_2000', 'n200_vs_2000')}"] = round(v, 4)
        for t in ("raw", "dless", "qperp"):
            if ax in sh and t in sh[ax]:
                out[f"proxy_{ax}_{t}"] = round(sh[ax][t]["mse"], 4)
            if ax != "train" and ax in sr.get(ARM[t], {}):
                out[f"net_{ax}_{ARM[t]}"] = round(sr[ARM[t]][ax]["level"], 4)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
