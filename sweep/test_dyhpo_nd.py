#!/usr/bin/env python3
"""
test_dyhpo_nd.py  —  Unit tests for the independent-axes DyHPO + DeepSets encoder.

Runs entirely in-process (no HTCondor, no GPU, no actual training).
Tests the full chain: DyHPOSampler → DyHPOAlgorithmND → FeatureExtractor.

Usage (from lxplus login node, inside ~/Foundational_Amplitudes/):
    python sweep/test_dyhpo_nd.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label, condition, detail=""):
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}" + (f": {detail}" if detail else ""))
        sys.exit(1)


def main():
    from sweep.dyhpo_sampler import DyHPOSampler

    # ------------------------------------------------------------------
    # Grid with DIFFERENT numbers of levels per axis — key feature
    # ds1: 2 levels, ds2: 3 levels, t_steps: 2 levels
    # Cartesian product = 2 × 3 × 2 = 12 combos
    # ------------------------------------------------------------------
    grid = {
        'n_data': {
            'ds1': [100, 1000],
            'ds2': [200, 2000, 20000],
        },
        't_steps': [500, 5000],
    }
    hp_space = [
        {'name': 'training.lr',     'type': 'float_log',     'low': 1e-5, 'high': 1e-2},
        {'name': 'training.wd',     'type': 'float_log',     'low': 1e-9, 'high': 1e-4},
        {'name': 'model.depth',     'type': 'int_uniform',   'low': 2,    'high': 8   },
    ]
    N_HP = 3   # number of HP params

    s = DyHPOSampler(
        hp_space=hp_space,
        fidelity_grid=grid,
        n_candidates=10,
        n_startup=2,
        seed=42,
    )

    # ---- 1. Cartesian product size -----------------------------------
    check(
        "all_combos = 12 (2×3×2, independent axes with different lengths)",
        len(s.algorithm.all_combos) == 12,
        f"got {len(s.algorithm.all_combos)}",
    )

    # ---- 2. Cheapest combo is (0,0,0) --------------------------------
    check(
        "cheapest combo is (0,0,0)",
        s.algorithm.all_combos[0] == (0, 0, 0),
        f"got {s.algorithm.all_combos[0]}",
    )

    # ---- 3. Startup suggests cheapest fidelity -----------------------
    hp_idx, params, n_data_dict, t_steps = s.suggest()
    check(
        "startup suggests cheapest t_steps",
        t_steps == 500,
        f"got t_steps={t_steps}",
    )
    check(
        "startup suggests cheapest n_data for each dataset",
        list(n_data_dict.values()) == [100, 200],
        f"got {n_data_dict}",
    )
    check(
        "params dict has all HP names",
        set(params.keys()) == {'training.lr', 'training.wd', 'model.depth'},
        f"got {set(params.keys())}",
    )

    # ---- 4. Observe and check history format -------------------------
    s.observe(hp_idx, n_data_dict, t_steps, val_loss=0.5)
    check(
        "_val_loss_history is {hp_idx: {combo: val_loss}}",
        isinstance(s._val_loss_history[hp_idx], dict),
        f"got {type(s._val_loss_history[hp_idx])}",
    )
    combo_key = list(s._val_loss_history[hp_idx].keys())[0]
    check(
        "history key is a tuple of length 3 (N_datasets + t_steps)",
        isinstance(combo_key, tuple) and len(combo_key) == 3,
        f"got key={combo_key}",
    )

    # ---- 5. Second startup trial -------------------------------------
    hp_idx2, params2, nd2, ts2 = s.suggest()
    s.observe(hp_idx2, nd2, ts2, val_loss=0.4)

    # ---- 6. Surrogate activates after startup ------------------------
    # Third suggest() triggers surrogate predict (post-startup)
    hp_idx3, params3, nd3, ts3 = s.suggest()
    check(
        "surrogate model created after startup phase",
        s.algorithm.model is not None,
        "model is still None",
    )

    # ---- 7. FeatureExtractor architecture ----------------------------
    fe = s.algorithm.model.feature_extractor

    expected_fc1_in = N_HP + 3    # n_hp_features + budget_dim (N=2 datasets + t_steps)
    check(
        f"fc1.in_features = n_hp({N_HP}) + budget_dim(3) = {expected_fc1_in}",
        fe.fc1.in_features == expected_fc1_in,
        f"got {fe.fc1.in_features}",
    )

    expected_obs_in = 3 + 1   # budget_dim + val_loss
    check(
        f"obs_encoder input dim = budget_dim(3) + 1 = {expected_obs_in}",
        fe.obs_encoder[0].in_features == expected_obs_in,
        f"got {fe.obs_encoder[0].in_features}",
    )

    obs_embed_dim = fe.obs_embed_dim
    check(
        f"obs_encoder output dim = obs_embed_dim = {obs_embed_dim}",
        fe.obs_encoder[-1].out_features == obs_embed_dim,
        f"got {fe.obs_encoder[-1].out_features}",
    )

    expected_fc_last_in = 64 + obs_embed_dim   # layer1_units + obs_embed_dim
    check(
        f"fc_last.in_features = layer1_units(64) + obs_embed_dim({obs_embed_dim}) = {expected_fc_last_in}",
        fe.fc2.in_features == expected_fc_last_in,
        f"got {fe.fc2.in_features}",
    )

    # ---- 8. encode_contexts with empty and non-empty contexts --------
    import torch
    device = s.algorithm.dev
    ctx_empty    = [[]]
    ctx_one_obs  = [[ ((0.1, 0.2, 0.5), 0.42) ]]
    ctx_two_obs  = [[ ((0.1, 0.2, 0.5), 0.42), ((0.5, 0.8, 1.0), 0.31) ]]

    emb_empty   = fe.encode_contexts(ctx_empty,   device)
    emb_one     = fe.encode_contexts(ctx_one_obs,  device)
    emb_two     = fe.encode_contexts(ctx_two_obs,  device)

    check(
        "encode_contexts: empty context → zero embedding",
        emb_empty.shape == (1, obs_embed_dim) and emb_empty.abs().sum().item() == 0.0,
        f"got shape={emb_empty.shape}, sum={emb_empty.abs().sum().item()}",
    )
    check(
        "encode_contexts: single observation → non-zero embedding",
        emb_one.shape == (1, obs_embed_dim) and emb_one.abs().sum().item() > 0.0,
        f"got shape={emb_one.shape}",
    )
    check(
        "encode_contexts: two observations → embedding shape correct",
        emb_two.shape == (1, obs_embed_dim),
        f"got shape={emb_two.shape}",
    )

    # ---- 9. Combo round-trip -----------------------------------------
    test_combo = (1, 2, 1)    # ds1 level 1 → 1000, ds2 level 2 → 20000, t_steps level 1 → 5000
    nd_rt, ts_rt = s._from_combo(test_combo)
    combo_rt = s._to_combo(nd_rt, ts_rt)
    check(
        "combo round-trip: _from_combo → _to_combo is identity",
        combo_rt == test_combo,
        f"{test_combo} → {nd_rt},{ts_rt} → {combo_rt}",
    )
    check(
        "combo (1,2,1) decodes to correct dataset sizes",
        nd_rt == {'ds1': 1000, 'ds2': 20000} and ts_rt == 5000,
        f"got n_data={nd_rt}, t_steps={ts_rt}",
    )

    # ---- 10. all_results keys ----------------------------------------
    s.observe(hp_idx3, nd3, ts3, val_loss=0.3)
    results = s.all_results()
    check(
        "all_results() returns entries with required keys",
        all({'hp_idx','val_loss','n_data_dict','t_steps','params'}.issubset(r.keys())
            for r in results),
        f"got keys {results[0].keys() if results else 'empty'}",
    )
    check(
        "all_results() sorted ascending by val_loss",
        all(results[i]['val_loss'] <= results[i+1]['val_loss'] for i in range(len(results)-1)),
        f"losses: {[r['val_loss'] for r in results]}",
    )

    # ---- 11. best_result ---------------------------------------------
    best_params, best_loss = s.best_result()
    check(
        "best_result() returns minimum val_loss",
        best_loss == 0.3,
        f"got {best_loss}",
    )

    # ---- 12. Leave-one-out context for training ----------------------
    # For a training point (hp_idx, combo), context should exclude that combo
    if s.algorithm.observations:
        hp_i = next(iter(s.algorithm.observations))
        obs_dict = s.algorithm.observations[hp_i]
        if obs_dict:
            combo_j = next(iter(obs_dict))
            ctx = s.algorithm._build_context(hp_i, exclude_combo=combo_j)
            # Context should not contain combo_j
            ctx_combos = [
                s._to_combo({'ds1': int(bv[0]*1000), 'ds2': int(bv[1]*20000)},
                            int(bv[2]*5000))
                for bv, vl in ctx
            ]
            check(
                "leave-one-out context excludes the training combo",
                combo_j not in ctx_combos,
                f"combo_j={combo_j} found in context combos",
            )

    # ---- 13. Save / load round-trip ----------------------------------
    import tempfile, pickle
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp_path = f.name
    try:
        s.save(tmp_path)
        s2 = DyHPOSampler.load(tmp_path)

        check(
            "save/load: all_combos preserved",
            s2.algorithm.all_combos == s.algorithm.all_combos,
        )
        check(
            "save/load: observations preserved",
            s2.algorithm.observations == s.algorithm.observations,
        )
        check(
            "save/load: val_loss_history preserved",
            s2._val_loss_history == s._val_loss_history,
        )
    finally:
        os.unlink(tmp_path)

    print()
    print(f"All checks passed. ({len(s.algorithm.all_combos)} combos, "
          f"budget_dim={s.algorithm.budget_dim}, obs_embed_dim={obs_embed_dim})")


if __name__ == "__main__":
    main()
