"""A new sweep on a cell that already has one is refused: the existing sweep is extended instead (sweep/extend_sweep.py:
widen an HP's range in place, keep every observation and the surrogate, then continue it with more trials).

A cell is what a sweep measures: its fixed_params (data, model, target, horizon-independent settings, seed) and its
fidelity (t_steps). Two sweep configs with the same cell and different sweep_names are the same search run twice, which
throws away the first one's observations and pays for them again (2026-10-23: 79 cells in sweep/ already had two or
three sweeps, e.g. tp3_scr_/tp3_scrh_/tp3_scrh10_ee_ddbar_d6; the user: "Why would you do an entirely different sweep
if you already have points from a previous sweep and a train surrogate instead of extending the sweep?").

Called by sweep/drive.py and sweep/generate_sweep.py before a sweep's state is created; resuming an existing sweep is
never refused. The one way past it is FA_NEW_SWEEP_REASON="<D-number or the user's words>", written into the new sweep
directory as new_sweep_reason.txt so the reason stays with the sweep.
    python sweep/cell_guard.py sweep/sweep_config_X.yaml      # prints the sweeps sharing X's cell, exit 1 if any
"""
import glob, json, os, sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cell_key(cfg):
    fp = {k: str(v) for k, v in (cfg.get("fixed_params") or {}).items()}
    return json.dumps([fp, (cfg.get("fidelity_schedule") or {}).get("t_steps")], sort_keys=True)


def same_cell(cfg):
    """sweep_names of the other sweep configs in sweep/ with cfg's cell"""
    k, name, out = cell_key(cfg), cfg.get("sweep_name"), []
    for f in sorted(glob.glob(os.path.join(ROOT, "sweep", "sweep_config_*.yaml"))):
        try:
            c = yaml.safe_load(open(f))
        except Exception:
            continue
        if isinstance(c, dict) and c.get("fixed_params") and c.get("sweep_name") != name and cell_key(c) == k:
            out.append(c.get("sweep_name"))
    return out


def check_new(cfg, sweep_dir):
    """Refuse (SystemExit) creating sweep_dir's state when another sweep already measures cfg's cell."""
    others = same_cell(cfg)
    if not others:
        return
    reason = os.environ.get("FA_NEW_SWEEP_REASON", "").strip()
    if not reason:
        raise SystemExit(
            f"cell_guard: {cfg.get('sweep_name')} measures the same cell (fixed_params and t_steps) as {others}. "
            f"Extend that sweep instead: python sweep/extend_sweep.py <its dir> --param <hp> --low/--high ..., then "
            f"continue it with more trials (drive.py --n-trials N / generate_sweep.py --extend). A new sweep only with "
            f"FA_NEW_SWEEP_REASON='<D-number or the user's words>'.")
    os.makedirs(sweep_dir, exist_ok=True)
    with open(os.path.join(sweep_dir, "new_sweep_reason.txt"), "w") as f:
        f.write(f"{reason}\nsame cell as: {', '.join(others)}\n")


if __name__ == "__main__":
    o = same_cell(yaml.safe_load(open(sys.argv[1])))
    print("\n".join(o) if o else "no other sweep on this cell")
    sys.exit(1 if o else 0)
