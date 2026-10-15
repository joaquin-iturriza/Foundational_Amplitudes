#!/usr/bin/env python
"""Wall time of the lhe_pp path (catalog_daniel) for one process at a few event counts: MG5 output +
standalone compile, MadEvent generate_events, LHE parsing, labelling. Everything goes to a fresh
temporary directory (pools, standalone, MadEvent dir) that is deleted at the end, so nothing lands
where a recipe would find it and no pool cache is touched.

    python tools/time_lhe_pp.py --process dc_gb_bbbx --process dc_gg_ggg --n 10000 --n 100000 [--mg-cores 1]
"""
import argparse, copy, os, shutil, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mg5_pipeline_final as mg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--process", action="append", required=True)
    ap.add_argument("--n", action="append", type=int, required=True, help="events (train block only)")
    ap.add_argument("--mg-cores", type=int, default=1)
    ap.add_argument("--tmp", default=os.environ.get("SCRATCH") or None, help="parent of the temp dir")
    a = ap.parse_args()
    root = tempfile.mkdtemp(prefix="zz_lhe_timing_", dir=a.tmp)
    rows = []
    try:
        mg.OUTPUT_DIR = root                       # write_recipe's manifest stays in the temp dir
        for p in a.process:
            for n in a.n:
                name = f"zz_time_{p}_{n}"
                cfg = copy.deepcopy(mg.PROCESSES[p])
                cfg["lhe"].update(n_val=0, n_test=0, n_train=n)
                mg.PROCESSES[name] = cfg
                mg.WORK_DIR = os.path.join(root, f"mg5work_{name}")     # compile counted every time
                os.makedirs(mg.WORK_DIR)
                rec = mg.variable_energy_recipe(name, 13000, 13000, n, role="train", seed=42)
                out = mg.recipe_output_path(rec, os.path.join(root, "pools"))
                t0 = time.time()
                res = mg.build_lhe_pp(name, [(rec, out)], work_root=os.path.join(root, "lhe"), mg_cores=a.mg_cores)
                wall = time.time() - t0
                run = res[0][1]["lhe_run"]
                du = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(mg.WORK_DIR) for f in fs)
                rows.append((p, n, wall, run["seconds"], run["xsec_pb"], os.path.getsize(out), du))
                print(f"[TIME] {p} n={n}: {wall:.0f} s wall  {run['seconds']}  xsec {run['xsec_pb']} pb  "
                      f"pool {os.path.getsize(out)/1e6:.1f} MB  standalone {du/1e6:.1f} MB", flush=True)
                del mg.PROCESSES[name]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\nprocess              n        wall_s  output  madevent  parse  label+rest  pool_MB  standalone_MB")
    for p, n, wall, s, _, sz, du in rows:
        print(f"{p:20s} {n:7d} {wall:9.0f} {s.get('output', 0):7.0f} {s.get('madevent', 0):9.0f} "
              f"{s.get('parse', 0):6.1f} {wall - sum(s.values()):12.1f} {sz/1e6:8.1f} {du/1e6:14.1f}")


if __name__ == "__main__":
    main()
