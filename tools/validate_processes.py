"""Validate the pretrain25 process definitions: compile each backend and sample
a few hundred events, checking the amplitudes are finite and positive. Writes a
clear PASS/FAIL line per process. Samples go to a throwaway temp dir."""
import os, sys, time, tempfile, traceback
import numpy as np, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datagen, mg5_pipeline_final as mg

procs = yaml.safe_load(open("recipes/pretrain25.yaml"))["processes"]
tmp = tempfile.mkdtemp(prefix="validate25_", dir=os.environ.get("SCRATCH", "/tmp"))
print(f"validating {len(procs)} processes -> {tmp}\n", flush=True)

results = []
for p in procs:
    name = p["name"]
    t0 = time.time()
    try:
        datagen.ensure_backend(name)
        path = datagen.ensure_dataset(
            name, float(p["sqrts"][0]), float(p["sqrts"][1]), 300,
            role="test", seed=7, dest_dir=tmp)
        a = np.load(path)
        amp = a[:, -1]
        nfinal = mg.PROCESSES[name]["nfinal"]
        ncols_ok = a.shape[1] == (nfinal + 2) * 5 + 1
        finite = np.isfinite(amp).all()
        positive = (amp > 0).all()
        varied = float(amp.std()) > 0
        ok = ncols_ok and finite and positive and varied
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name:12s} shape={a.shape} amp∈[{amp.min():.2e},{amp.max():.2e}] "
              f"finite={finite} pos={positive} ({time.time()-t0:.0f}s)", flush=True)
        results.append((name, ok))
    except Exception as e:
        print(f"[FAIL] {name:12s} ERROR: {e}", flush=True)
        traceback.print_exc()
        results.append((name, False))

bad = [n for n, ok in results if not ok]
print(f"\n=== {len(results)-len(bad)}/{len(results)} passed ===", flush=True)
if bad:
    print("FAILED:", bad, flush=True)
