"""Extend a finished single-fidelity sweep in place: widen one HP's range, sample candidates in the added zone,
and fold in trials that ran outside the sweep (on the same target and fidelity) as observations, so the cell
stays one search whose surrogate keeps everything it saw. Then `generate_sweep.py --extend --n-trials N`
continues the same sweep. Runs on the site that holds the sweep (its dyhpo_state.pkl is locked while edited).

    python sweep/extend_sweep.py <sweep_dir> --param training.lr [--high 0.03] \
        [--fold <other sweep_dir or dyhpo_state.pkl> ...] [--n-new 50] [--dry-run]
Without --low/--high the range grows to cover the folded sweeps' own ranges of that HP.

The folded sweep's searched ranges must lie inside the extended one (asserted per point), and its fidelity
must equal this sweep's. Each fold is recorded in <sweep_dir>/folded.json (source, hp indices, losses).
"""
import argparse, json, os, pickle, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep.dyhpo_sampler import DyHPOSampler  # noqa: E402


def state_of(path):
    return path if path.endswith(".pkl") else os.path.join(path, "dyhpo_state.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--param", required=True)
    ap.add_argument("--low", type=float)
    ap.add_argument("--high", type=float)
    ap.add_argument("--fold", nargs="*", default=[])
    ap.add_argument("--n-new", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    state = state_of(a.sweep_dir)
    log = os.path.join(a.sweep_dir, "folded.json")
    done = json.load(open(log)) if os.path.exists(log) else []
    already = {d["source"] for d in done}
    srcs = [(src, os.path.abspath(state_of(src))) for src in a.fold]
    other = {st: DyHPOSampler.load(st, os.path.dirname(st)) for _, st in srcs}
    bounds = [next(e for e in o.hp_space if e["name"] == a.param) for o in other.values()]
    low = a.low if a.low is not None else min((b["low"] for b in bounds), default=None)
    high = a.high if a.high is not None else max((b["high"] for b in bounds), default=None)
    with DyHPOSampler.locked(state, a.sweep_dir) as s:
        (T,) = s.fidelity_grid["t_steps"]
        ext = s.extend_range(a.param, low=low, high=high, n_new=a.n_new)
        print(f"{a.sweep_dir}: {a.param} {ext or 'unchanged'}")
        for src, src_state in srcs:
            if src_state in already:
                print(f"  {src}: folded before, skipped"); continue
            o = other[src_state]
            assert o.fidelity_grid["t_steps"] == [T], (src, o.fidelity_grid, T)
            rec = []
            for hp, by_combo in o._val_loss_history.items():
                for combo, loss in by_combo.items():
                    proc = o._proc_val_loss_history.get(hp, {}).get(combo)
                    new = s.add_observed(o.candidates_raw[hp], T, loss, proc)
                    rec.append({"src_hp": hp, "hp": new, "val_loss": loss})
            print(f"  {src}: {len(rec)} observations folded")
            done.append({"source": src_state, "param": a.param, "range": ext.get("new"), "points": rec})
        if a.dry_run:
            raise SystemExit("dry run: state not saved")   # leaves the lock before save()
    json.dump(done, open(log, "w"), indent=1)


if __name__ == "__main__":
    main()
