#!/usr/bin/env python3
"""
aggregate_hpo_optima.py — Harvest optimal HPs from every DyHPO sweep on disk.

For each sweep with a dyhpo_state.pkl it records:
  * the full best HP config (min val_loss over all observations),
  * per-HP position within its search range (edge fraction),
  * the lr-convergence verdict (reusing analyze_lr_boundary.analyze_sweep on the
    effective learning-rate knob: training.lr, or fine_tune.lr_scale for FT sweeps),
  * context: dataset(s), num_heads (muP width), num_blocks, configured batchsize,
    t_steps (iterations), subsample/split,
  * derived n_train and effective batch size = min(bs, n_train//2).

The goal: see how the optimum (esp. lr) moves with dataset size, iterations and
effective batch size, so search ranges / #runs per sweep can be shrunk.

Usage:
    python sweep/aggregate_hpo_optima.py [--out-prefix sweep/hpo_optima]
Outputs <prefix>.json (full records) and <prefix>.csv (flat table).
"""
import argparse, json, math, os, pickle, re, sys, glob, csv
import numpy as np, yaml

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
from sweep.analyze_lr_boundary import analyze_sweep  # tested-range lr edge logic

LOG_TYPES = {"float_log", "int_log"}
EDGE_FRAC = 0.15          # < this fraction of the range from an end == "at edge"
MIN_OBS   = 6             # sweeps with fewer observations are too sparse to trust

# ---- .npy row-count cache (header only, cheap) ------------------------------
_NPY_ROWS = {}
def npy_rows(path):
    if path in _NPY_ROWS:
        return _NPY_ROWS[path]
    n = None
    try:
        with open(path, "rb") as f:
            ver = np.lib.format.read_magic(f)
            shp, _, _ = np.lib.format._read_array_header(f, ver)
            n = int(shp[0])
    except Exception:
        n = None
    _NPY_ROWS[path] = n
    return n

# ---- name parsing (scaling grids encode nh/D/t) -----------------------------
def parse_name(name):
    out = {}
    m = re.search(r'nh(\d+)', name);            out['nh_name']   = int(m.group(1)) if m else None
    m = re.search(r'_t0*(\d+)', name) or re.search(r'_t(\d+)', name)
    out['t_name'] = int(m.group(1)) if m else None
    m = re.search(r'D([0-9]+e[0-9]+|[0-9]+k|[0-9]+)', name)
    if m:
        s = m.group(1)
        try:
            out['D_name'] = int(float(s.replace('k','e3'))) if ('e' in s or 'k' in s) else int(s)
        except Exception:
            out['D_name'] = None
    else:
        out['D_name'] = None
    return out

def as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, str):
        s = x.strip().strip('[]')
        return [p.strip() for p in s.split(',') if p.strip()]
    return [x]

def to_int(x):
    try: return int(x)
    except Exception: return None

# ---- train-size / effective-batch derivation --------------------------------
def derive_sizes(cfg, fp):
    """Return (N_total, n_train, eff_bs, reliable, note)."""
    data_path = fp.get('data.data_path')
    datasets  = as_list(fp.get('data.dataset'))
    ttv       = fp.get('data.train_test_val')
    if isinstance(ttv, str):
        try: ttv = json.loads(ttv)
        except Exception: ttv = None
    sub       = fp.get('data.subsample')
    trsub     = fp.get('data.train_subsample')
    bs        = to_int(fp.get('training.batchsize'))

    # normalise subsample string 'null'/'none'
    if isinstance(sub, str) and sub.lower() in ('none', 'null'):
        sub = None

    N_total = None; note = ''
    if datasets and data_path:
        per = []
        for ds in datasets:
            rows = npy_rows(os.path.join(data_path, f"{ds}.npy"))
            if sub is not None and rows is not None:
                per.append(min(int(sub), rows))
            elif sub is not None:
                per.append(int(sub))
            elif rows is not None:
                per.append(rows)
            else:
                per.append(None)
        if all(p is not None for p in per):
            N_total = sum(per)
        else:
            note = 'missing_npy'

    n_train = None; reliable = False
    if N_total is not None and ttv and isinstance(ttv, (list, tuple)) and len(ttv) >= 1:
        n_train = int(N_total * float(ttv[0]))
        if n_train % 2 and n_train > 1: n_train -= 1
        reliable = True
    elif trsub is not None and datasets:
        # source/recipe path: train_subsample caps train events per process
        n_train = int(trsub) * max(len(datasets), 1)
        note = (note + ';train_subsample_est').strip(';')
        reliable = False
    elif N_total is not None:
        n_train = int(N_total * 0.7)  # assume default split
        note = (note + ';assumed_split_0.7').strip(';')
        reliable = False

    eff_bs = None
    if bs is not None and n_train is not None:
        eff_bs = int(min(bs, max(n_train // 2, 1)))
    elif bs is not None:
        eff_bs = bs
    return N_total, n_train, eff_bs, reliable, note

# ---- per-HP edge fraction within DECLARED range -----------------------------
def edge_frac_declared(entry, val):
    lo, hi = entry.get('low'), entry.get('high')
    if val is None or lo is None or hi is None: return None
    if entry.get('type') in LOG_TYPES:
        if lo <= 0 or hi <= 0 or val <= 0: return None
        lo, hi, v = math.log(lo), math.log(hi), math.log(val)
    else:
        v = val
    if hi <= lo: return None
    pos = (v - lo) / (hi - lo)
    return max(0.0, min(pos, 1.0 - pos))

# ---- best observation extraction --------------------------------------------
def best_observation(state, cfg):
    vh = state.get('val_loss_history', {})
    cands = state.get('candidates_raw', [])
    t_steps_list = cfg.get('fidelity_schedule', {}).get('t_steps', [None])
    best = None  # (val_loss, hp_idx, t_steps, config)
    for hp_idx, obs in vh.items():
        for combo, vl in obs.items():
            try: vlf = float(vl)
            except Exception: continue
            if not math.isfinite(vlf): continue
            fid_idx = combo[0] if isinstance(combo, (tuple, list)) and combo else 0
            ts = t_steps_list[fid_idx] if isinstance(t_steps_list, list) and fid_idx < len(t_steps_list) else t_steps_list[-1] if t_steps_list else None
            if best is None or vlf < best[0]:
                cfgd = cands[hp_idx] if hp_idx < len(cands) else {}
                best = (vlf, hp_idx, ts, dict(cfgd))
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweeps-root', default=os.path.join(_PROJ, 'sweeps'))
    ap.add_argument('--out-prefix',  default=os.path.join(_PROJ, 'sweep', 'hpo_optima'))
    args = ap.parse_args()

    leafs = sorted(os.path.dirname(p) for p in
                   glob.glob(os.path.join(args.sweeps_root, '**', 'dyhpo_state.pkl'), recursive=True))
    print(f"Found {len(leafs)} sweep dirs with state", file=sys.stderr)

    records = []
    for d in leafs:
        name = os.path.basename(d)
        try:
            state = pickle.load(open(os.path.join(d, 'dyhpo_state.pkl'), 'rb'))
        except Exception as e:
            print(f"skip {name}: {e}", file=sys.stderr); continue
        cf = os.path.join(d, 'sweep_config.yaml')
        cfg = yaml.safe_load(open(cf)) if os.path.exists(cf) else {}
        fp  = cfg.get('fixed_params', {}) or {}

        hp_space = state.get('hp_space', [])
        swept = {e['name']: e for e in hp_space}
        lr_key = ('training.lr' if 'training.lr' in swept else
                  'fine_tune.lr_scale' if 'fine_tune.lr_scale' in swept else None)

        best = best_observation(state, cfg)
        n_obs = sum(len(o) for o in state.get('val_loss_history', {}).values())

        # lr convergence verdict via the tested-range logic
        lr_metric = None
        if lr_key:
            try:
                lr_metric = analyze_sweep(state, lr_param=lr_key)
            except Exception as e:
                print(f"lr_metric fail {name}: {e}", file=sys.stderr)

        best_cfg = best[3] if best else {}
        # per-HP best value + declared-range edge fraction
        hp_best = {}
        for nm, e in swept.items():
            v = best_cfg.get(nm)
            hp_best[nm] = {'val': v, 'low': e.get('low'), 'high': e.get('high'),
                           'type': e.get('type'),
                           'edge_frac_declared': edge_frac_declared(e, v)}

        N_total, n_train, eff_bs, reliable, note = derive_sizes(cfg, fp)
        pn = parse_name(name)

        lr_at_edge = None; lr_edge_pct = None; lr_dir = None; lr_incon = None
        if lr_metric and lr_metric.get('n_obs', 0) > 0:
            lr_at_edge  = bool(lr_metric.get('is_flagged'))
            lr_edge_pct = lr_metric.get('edge_pct')
            lr_dir      = lr_metric.get('direction')
            lr_incon    = bool(lr_metric.get('inconclusive'))

        converged = (lr_key is not None and n_obs >= MIN_OBS and
                     lr_at_edge is False)

        rec = {
            'sweep': name, 'path': d,
            'family': re.sub(r'\d.*', '', name),
            'n_obs': n_obs,
            'n_candidates': len(state.get('candidates_raw', [])),
            'n_trials_cfg': cfg.get('n_trials'),
            'swept_hps': list(swept.keys()),
            'lr_key': lr_key,
            'best_val_loss': best[0] if best else None,
            'best_t_steps': best[2] if best else None,
            'best_config': best_cfg,
            'hp_best': hp_best,
            'lr_at_edge': lr_at_edge, 'lr_edge_pct': lr_edge_pct,
            'lr_direction': lr_dir, 'lr_inconclusive': lr_incon,
            'converged': converged,
            # context
            'datasets': as_list(fp.get('data.dataset')),
            'n_datasets': len(as_list(fp.get('data.dataset'))),
            'num_heads': to_int(fp.get('model.net.num_heads')),
            'num_blocks': to_int(fp.get('model.net.num_blocks')),
            'batchsize_cfg': to_int(fp.get('training.batchsize')),
            't_steps': best[2] if best else pn.get('t_name'),
            'subsample': fp.get('data.subsample'),
            'train_test_val': fp.get('data.train_test_val'),
            'train_subsample': fp.get('data.train_subsample'),
            # derived
            'N_total': N_total, 'n_train': n_train, 'eff_bs': eff_bs,
            'size_reliable': reliable, 'size_note': note,
            'nh_name': pn.get('nh_name'), 'D_name': pn.get('D_name'), 't_name': pn.get('t_name'),
        }
        records.append(rec)

    with open(args.out_prefix + '.json', 'w') as f:
        json.dump(records, f, indent=2, default=str)

    # flat CSV of the key columns (best HP values expanded)
    all_hp_names = sorted({h for r in records for h in r['swept_hps']})
    cols = ['sweep', 'family', 'lr_key', 'converged', 'lr_at_edge', 'lr_edge_pct',
            'lr_direction', 'lr_inconclusive', 'n_obs', 'n_candidates',
            'best_val_loss', 'n_datasets', 'num_heads', 'num_blocks',
            'batchsize_cfg', 't_steps', 'N_total', 'n_train', 'eff_bs',
            'size_reliable', 'size_note'] + ['best.' + h.split('.')[-1] for h in all_hp_names]
    with open(args.out_prefix + '.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for r in records:
            row = [r.get(c.replace('best.', '') if c.startswith('best.') else c) for c in cols[:21]]
            for h in all_hp_names:
                hb = r['hp_best'].get(h)
                row.append(hb['val'] if hb else '')
            w.writerow(row)

    n_conv = sum(r['converged'] for r in records)
    n_lr   = sum(r['lr_key'] is not None for r in records)
    print(f"Wrote {args.out_prefix}.json / .csv  —  {len(records)} sweeps, "
          f"{n_lr} with an lr knob, {n_conv} converged (lr interior, n_obs>={MIN_OBS}).",
          file=sys.stderr)

if __name__ == '__main__':
    main()
