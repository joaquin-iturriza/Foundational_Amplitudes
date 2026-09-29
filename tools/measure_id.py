"""Intrinsic dimension of a trained model's representation, measured after the fact on its best checkpoint.

What is measured: the representation the output reads, one vector per event. The LLoCa backbone reads a
scalar out of every particle (models/transformer_lloca_mup.py, linear_out) and the wrapper mean-pools it per
event (wrappers._pool_events); both are linear, so the event's input to the output is the event mean of the
per-particle features entering linear_out. That vector, on the first N test events, goes to the TwoNN
estimator (IntrinsicDimDeep/IDNN/intrinsic_dimension.estimate, the slope it returns), repeated on R random 90%
subsets: the estimator and defaults of IntrinsicDimDeep/get_dim.py (N = 1000, R = 3), which was written for the
MLP with one dataset per batch and cannot drive the LLoCa batches.
The model is rebuilt from the run's own config.yaml and its best checkpoint (models/model_run<idx>_best.pt.gz),
under the EMA weights when the run used EMA (as evaluation does), with save=False so the run dir is untouched (its data_stats.json and tokenizer are read, not rewritten).

    python tools/measure_id.py --sweep <sweep_dir> [--sweep ...] [--run <run_dir> ...] --out <file.jsonl>
A --sweep resolves to its best trial (results/hp<ID>_t<T>_*.json with the lowest val_loss -> checkpoint_index
run_dir). One JSON line per run: {"sweep", "run_dir", "val_loss", "id_mean", "id_std", "n_events"}.
"""
import argparse, glob, inspect, json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)


def best_run(sweep_dir):
    idx = json.load(open(os.path.join(sweep_dir, "checkpoint_index.json")))
    best = None
    for f in glob.glob(os.path.join(sweep_dir, "results", "*.json")):
        m = re.match(r"hp(\d+)_t\d+_", os.path.basename(f))
        v = json.load(open(f)).get("val_loss")
        if m and v is not None and (best is None or v < best[0]) and str(int(m.group(1))) in idx:
            best = (float(v), idx[str(int(m.group(1)))]["run_dir"])
    if best is None:
        raise SystemExit(f"{sweep_dir}: no result matched to a checkpoint")
    return best


def measure(run_dir, n_events, n_rep, seed=0):
    import torch
    from omegaconf import OmegaConf
    from scipy.spatial.distance import pdist, squareform
    from experiment import AmplitudeExperiment
    from base_experiment import _torch_load
    from wrappers import _pool_events
    from IntrinsicDimDeep.IDNN.intrinsic_dimension import estimate

    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    OmegaConf.set_struct(cfg, False)
    cfg.save = False; cfg.train = False; cfg.evaluate = False; cfg.plot = False
    # warm_start_idx = the run's own index: init_data then reads the run's data_stats.json and tokenizer, so the
    # inputs are normalized exactly as in training; the best checkpoint is loaded on top below.
    cfg.warm_start_idx = cfg.run_idx; cfg.use_mlflow = False; cfg.run_dir = run_dir
    torch.set_default_dtype({"float16": torch.float16, "float64": torch.float64}.get(cfg.training.dtype, torch.float32))
    exp = AmplitudeExperiment(cfg)
    exp._init()
    exp.init_physics(); exp.init_data(); exp.init_model()
    ck = _torch_load(os.path.join(run_dir, "models", f"model_run{cfg.run_idx}_best.pt.gz"), map_location=exp.device, weights_only=False)
    exp.model.load_state_dict(ck["model"])
    if exp.ema is not None and ck.get("ema") is not None:
        exp.ema.load_state_dict(ck["ema"])

    feats, ptrs = [], []
    readout = [m for n, m in exp.model.named_modules() if n.endswith("linear_out")]
    if len(readout) != 1:
        raise SystemExit(f"{run_dir}: expected one linear_out, found {len(readout)}")
    readout[0].register_forward_pre_hook(lambda mod, inp: feats.append(inp[0].detach().reshape(-1, inp[0].shape[-1])))
    sig = inspect.signature(type(exp.model).forward)
    def grab_ptr(mod, args, kwargs):
        ptrs.append(sig.bind(mod, *args, **kwargs).arguments["ptr"].detach())
    exp.model.register_forward_pre_hook(grab_ptr, with_kwargs=True)

    import contextlib
    ctx = exp.ema.average_parameters() if exp.ema is not None else contextlib.nullcontext()
    exp.model.eval()
    with torch.no_grad(), ctx:
        exp._collect_predictions(exp.test_loader)
    if len(feats) != len(ptrs):
        raise SystemExit(f"{run_dir}: {len(feats)} readout calls against {len(ptrs)} forward calls")
    ev = []
    for h, p in zip(feats, ptrs):
        if h.shape[0] != int(p[-1]):
            raise SystemExit(f"{run_dir}: {h.shape[0]} particle rows against ptr[-1] = {int(p[-1])}")
        ev.append(_pool_events(h, p).float().cpu().numpy())
    X = np.concatenate(ev)[:n_events].astype(np.float64)
    rng = np.random.default_rng(seed); ids = []
    for _ in range(n_rep):
        sel = rng.permutation(len(X))[: int(0.9 * len(X))]
        ids.append(float(estimate(squareform(pdist(X[sel])), verbose=False)[2]))
    return float(np.mean(ids)), float(np.std(ids)), len(X)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", default=[]); ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--rep", type=int, default=3)
    a = ap.parse_args()
    jobs = [(os.path.basename(s.rstrip("/")),) + best_run(s) for s in a.sweep] + [(os.path.basename(r.rstrip("/")), None, r) for r in a.run]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["run_dir"] for l in open(a.out) if l.strip()}
    for name, v, rd in jobs:
        if rd in done:
            continue
        try:
            m, s, n = measure(rd, a.n, a.rep)
            rec = {"sweep": name, "run_dir": rd, "val_loss": v, "id_mean": m, "id_std": s, "n_events": n}
        except BaseException as e:           # one bad run must not lose the others; recorded, not dropped
            rec = {"sweep": name, "run_dir": rd, "val_loss": v, "error": f"{type(e).__name__}: {e}"[:300]}
        print(json.dumps(rec), flush=True)
        open(a.out, "a").write(json.dumps(rec) + "\n")
