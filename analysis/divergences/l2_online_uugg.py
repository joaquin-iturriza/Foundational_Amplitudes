#!/usr/bin/env python
"""On-the-fly, continuous-schedule, growing-pool online generation for ee->uugg (L2).

This is the CORRECTED online-generation loop (the previous l2_online.py had four flaws: a per-round
step budget instead of total/rounds; a whole-pool resample instead of a growing pool; a per-round
scheduler reset; and -- worst -- it never generated on the fly at all, it re-selected a frozen
pre-labeled batch). Here:

  * ONE long-lived process. Model + optimizer + a SINGLE cosine over the full horizon are built once
    (round 0 via the normal init_data, which also FREEZES the amp/mom preprocessing stats). The
    scheduler is never reset -- rounds are just boundaries inside the one train() loop, reached via the
    `_online_hook` added to base_experiment.train().
  * GENUINE on-the-fly generation. At each round boundary we PROPOSE fresh momenta from a
    process-agnostic base (gen_ir_democratic + flat RAMBO for the bulk), [score by the current sigma],
    KEEP n, LABEL the survivors with the exact standalone, preprocess with the FROZEN stats, and APPEND
    to a growing training pool. No frozen pre-labeled candidate set.
  * Matched budget. total_steps and N_total are the run inputs; per round = total_steps//R and
    N_total//R. A static baseline and the sigma arm see the SAME totals, differing only in WHERE the
    per-round increment is drawn.

Arms:
  base   : keep a uniform subset of the base proposal   -> trains on the fixed "best hand-coded
           sampler" (IR-democratic + RAMBO) every round. The baseline the sigma arm must beat.
  sigma  : keep proposals with prob ∝ sigma(x)^gamma     -> reallocates the increment toward where the
           model is CURRENTLY wrong. [needs the heteroscedratic sigma head -- ported separately.]

CPU generation/labeling; GPU training. Run under sbatch (xformers attention is CUDA-only).
"""
import argparse
import os
import sys

import numpy as np
import torch

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, WT)
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
import gen_ir_democratic as G          # noqa: E402
import mg5_pipeline_final as mp         # noqa: E402
from preprocessing import preprocess_amplitude  # noqa: E402
from particle_ids import global_encode  # noqa: E402
from lloca.utils.rand_transforms import rand_lorentz          # noqa: E402
from lloca.utils.polar_decomposition import restframe_boost   # noqa: E402

DATASET = "ee_uugg_91-1000GeV_amplitudes"
PDG = np.array([11, -11, 2, -2, 21, 21], dtype=int)
MASSES = np.zeros(4)                    # final-state masses (u ubar g g, all massless)
NP = 6                                  # particles per event
WORK = os.environ["WORK"]
STANDALONE = f"{WORK}/mg5amcnlo/ee_uugg_standalone"
BASE22 = f"{REPO}/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt"
# Lowered fiducial cuts (open the deep IR), matching gen_uug_sampling.LOW_CUTS.
LOW_CUTS = {"pt_min": 1.0, "cos_max": 0.9, "dr_min": 0.05, "m_min": 0.3}

# mu finetune HPs -- the validated ee->uugg finetune (finetune_addback_uugg.sh).
MU_OVERRIDES = [
    "model.use_diagrams=false", "model.particle_encoder_hidden=0",
    "model.net.num_blocks=8", "model.net.num_heads=8",
    "fine_tune.lr_scale=0.339", "fine_tune.layer_decay=0.999",
    "training.lr=0.004", "training.batchsize=16384", "evaluation.batchsize=16384",
    "training.regularization=L2", "training.regularization_lambda=2.47e-7",
    "training.scheduler=CosineAnnealingLR", "training.loss_aggregation=geometric_mean",
    "training.cosanneal_warmup_frac=0.191", "training.cosanneal_eta_min=1.6e-7",
    f"fine_tune.pretrained_path={BASE22}",
]
DATA_OVERRIDES = [
    "data.source=files", f"data.dataset=[{DATASET}]", "data.preprocess_per_dataset=true",
    "data.train_test_val=[0.9,0.05,0.05]", "data.subsample=null",
]
# sigma arm: train the whole loop as HETEROSC with a DETACHED sigma head at beta=1. Then the mu
# channel gets a pure-MSE gradient (beta=1 cancels the sigma^2 weighting) and mu trains EXACTLY as
# MSE, while sigma is fit continuously as a read-only calibration head off detached features -- no
# per-round sigma-fit, no grow_sigma_head, no wasted iters. sigma is always live to score proposals.
SIG_ARM_OVERRIDES = [
    "training.loss=HETEROSC", "training.heterosc_beta=1.0",
    "training.heterosc_sigma_only=false",
    "model.net.detach_sigma_backbone=true", "model.net.sigma_after_pool=true",
]


# ---------------------------------------------------------------- base proposal (process-agnostic)
def propose_momenta(n, y_lo, mix_ir, cuts, rng):
    """Propose n full events (n,6,4) from the base: mix_ir fraction from the IR-democratic generator
    (reaches soft/collinear corners of ANY massless process), the rest flat RAMBO (fills the O(1)
    bulk). Fiducial cuts applied by oversample-and-reject. No labels, no weights -- only WHERE."""
    n_ir = int(round(mix_ir * n))
    n_ram = n - n_ir

    def draw_ir(nb):
        sq = rng.uniform(91.0, 1000.0, nb)
        Pf = G.democratic_draw(nb, sq, MASSES, y_lo, rng)
        return G.build_full_event(Pf, sq), sq

    parts = []
    if n_ir > 0:
        P_ir, _ = mp._collect_with_cuts(draw_ir, n_ir, list(MASSES), PDG, cuts)
        parts.append(P_ir)
    if n_ram > 0:
        ev, _ = mp.sample_nbody_phase_space(n_ram, 91.0, 1000.0, list(MASSES), PDG, rng=rng, cuts=cuts)
        P_ram = np.stack([e[0] for e in ev], axis=0)
        parts.append(P_ram)
    P = np.concatenate(parts, axis=0)
    return P[rng.permutation(len(P))]


def label_events(P):
    """Exact tree |M|^2 via the compiled ee_uugg standalone. P: (n,6,4)."""
    events = [(P[i], PDG) for i in range(len(P))]
    with mp.CppDriverPipe(f"{STANDALONE}/driver", STANDALONE) as pipe:
        me2 = np.asarray(pipe.compute(events), dtype=np.float64)
    return me2


# ---------------------------------------------------------------- frozen-stats preprocessing
def preprocess_momenta(exp, P):
    """Momentum path only (no amplitude): mirror experiment.init_data's LLoCa branch with the FROZEN
    round-0 mom scale. Returns per-event particle/token/order/pid lists. Used for BOTH the training
    increment (label added separately) and sigma scoring (no label needed)."""
    n = len(P)
    particles_t = torch.tensor(P, dtype=torch.float64).reshape(-1, NP, 4)
    # enforce the mass shell from the 3-momenta (identical to init_data)
    m2 = particles_t[..., 0] ** 2 - (particles_t[..., 1:] ** 2).sum(dim=-1)
    particles_t[..., 0] = torch.sqrt((particles_t[..., 1:] ** 2).sum(dim=-1) + m2.clamp(min=0))
    lab = particles_t[..., :2, :].sum(dim=-2)
    to_com = restframe_boost(lab)
    trafo = rand_lorentz(particles_t.shape[:-2], generator=None, dtype=particles_t.dtype)
    trafo = torch.einsum("...ij,...jk->...ik", trafo, to_com)
    particles_t = torch.einsum("...ij,...kj->...ki", trafo, particles_t)
    particles_prepd = (particles_t / exp.mom_div).numpy()          # (n,6,4), FROZEN mom scale

    toks = global_encode(np.tile(PDG, (n, 1)))                     # (n,6) per-particle tokens
    order0 = exp.train_loader.dataset.order_labels[0].cpu().numpy()  # uugg LO row (same every event)
    parts = [particles_prepd[j] for j in range(n)]
    toks_l = [toks[j] for j in range(n)]
    orders = np.tile(order0, (n, 1))
    pids = np.zeros(n, dtype=np.int32)                             # single process
    return parts, toks_l, orders, pids


def preprocess_increment(exp, P, me2):
    """A labeled training increment: momentum path (frozen mom scale) + amplitude (frozen amp stats)."""
    parts, toks_l, orders, pids = preprocess_momenta(exp, P)
    amp_prepd, _, _ = preprocess_amplitude(
        me2.reshape(-1, 1), trafos=exp.cfg.data.amp_trafos,
        mean=exp.prepd_mean[0], std=exp.prepd_std[0])              # FROZEN amp stats
    return parts, toks_l, amp_prepd, orders, pids


def score_sigma(exp, P):
    """Forward the LIVE (mu,sigma) model over proposed momenta -> per-proposal sigma (the model's own
    uncertainty). No labels needed. Reuses exp._collect_predictions (handles the HETEROSC split)."""
    from dataset import AmplitudeDataset, build_flat_arrays, collate_variable_length
    parts, toks_l, orders, pids = preprocess_momenta(exp, P)
    pf, tf, off = build_flat_arrays(parts, toks_l)
    ds = AmplitudeDataset(
        particles_flat=pf, offsets=off,
        amplitudes=np.zeros((len(parts), 1), dtype=np.float64),     # dummy (unused for sigma)
        tokens_flat=tf, order_labels=np.asarray(orders),
        process_ids=pids.astype(np.int64), dtype=exp.dtype)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=int(exp.cfg.evaluation.batchsize), shuffle=False, drop_last=False,
        collate_fn=collate_variable_length, num_workers=0)
    was_training = exp.model.training
    exp.model.eval()
    with torch.no_grad():
        _, _, sig = exp._collect_predictions(loader)
    if was_training:
        exp.model.train()
    return np.asarray(sig, dtype=np.float64).reshape(-1)            # (N,)


def _dataset_to_lists(ds):
    """Explode an AmplitudeDataset (round-0 train pool) back into per-event lists to seed the pool."""
    pf = ds.particles_flat.cpu().numpy()
    tf = ds.tokens_flat.cpu().numpy()
    off = ds.offsets
    parts = [pf[int(off[i, 0]):int(off[i, 1])] for i in range(len(off))]
    toks = [tf[int(off[i, 0]):int(off[i, 1])] for i in range(len(off))]
    amps = ds.amplitudes.cpu().numpy()
    orders = ds.order_labels.cpu().numpy()
    pids = ds.process_ids.cpu().numpy() if ds.process_ids is not None else np.zeros(len(off), np.int32)
    return parts, toks, amps, orders, pids


def _make_train_loader(exp, pool):
    """Rebuild exp.train_loader from the growing per-event pool (all events are 6-particle)."""
    from dataset import AmplitudeDataset, build_flat_arrays, collate_variable_length
    pf, tf, off = build_flat_arrays(pool["parts"], pool["toks"])
    ds = AmplitudeDataset(
        particles_flat=pf, offsets=off,
        amplitudes=np.asarray(pool["amps"]).reshape(-1, 1),
        tokens_flat=tf,
        order_labels=np.asarray(pool["orders"]),
        process_ids=np.asarray(pool["pids"]).astype(np.int64),
        dtype=exp.dtype,
    )
    bs = int(min(exp.cfg.training.batchsize, len(ds) // 2))
    exp.train_loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True, drop_last=True,
        collate_fn=collate_variable_length, pin_memory=False,
        num_workers=exp.cfg.training.num_workers,
        persistent_workers=exp.cfg.training.num_workers > 0,
    )


# ---------------------------------------------------------------- cfg build
def build_cfg(total_steps, round0_dir, exp_name, run_name, seed, arm, pretrained=None):
    from hydra import compose, initialize_config_dir
    arm_ov = SIG_ARM_OVERRIDES if arm == "sigma" else []
    # later overrides win: a grown 2-ch checkpoint (sigma arm) supersedes the 1-ch BASE22 in MU_OVERRIDES.
    pre_ov = [f"fine_tune.pretrained_path={pretrained}"] if pretrained else []
    overrides = DATA_OVERRIDES + MU_OVERRIDES + arm_ov + pre_ov + [
        f"exp_name={exp_name}", f"run_name={run_name}", f"seed={seed}",
        f"data.data_path={round0_dir}/",
        f"training.iterations={total_steps}",
        "plot=false", "save=true", "training.save_intermediate=true",
    ]
    # Compose from the MAIN-repo config, NOT the worktree's: the runtime imports the core modules
    # (experiment/base_experiment/models) from the main repo, so the config tree must match them --
    # e.g. the sigma-head fields (heterosc_beta, detach_sigma_backbone) live in the main-repo config.
    with initialize_config_dir(config_dir=os.path.join(REPO, "config"), version_base=None):
        cfg = compose(config_name="amplitudes", overrides=overrides)
    return cfg


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "sigma"])
    ap.add_argument("--tag", default="uugg")
    ap.add_argument("--total_steps", type=int, default=4000)
    ap.add_argument("--n_total", type=int, default=300000)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=1.0, help="sigma arm: p(x) ∝ sigma^gamma")
    ap.add_argument("--sigma0", type=float, default=0.1, help="sigma arm: initial sigma for grow_sigma_head")
    ap.add_argument("--oversample", type=float, default=4.0, help="propose oversample*inc_n, keep inc_n")
    ap.add_argument("--y_lo", type=float, default=1e-6)
    ap.add_argument("--mix_ir", type=float, default=0.5, help="base = mix_ir IR-democratic + rest RAMBO")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate_prep", action="store_true",
                    help="CPU check: increment preprocessing reproduces init_data on the same rows")
    args = ap.parse_args()

    R = args.rounds
    inc_n = args.n_total // R
    steps_per_round = args.total_steps // R
    exp_name = f"eeuu_l2uugg"
    run_name = f"{args.tag}_{args.arm}_s{args.seed}"
    rng_gen = np.random.default_rng(1000 + args.seed)

    # --- round 0 increment -> disk, so init_data builds it + FREEZES the stats ---
    round0_dir = os.path.join(REPO, f"data_l2uugg/{run_name}_r0")
    os.makedirs(round0_dir, exist_ok=True)
    r0_npy = os.path.join(round0_dir, DATASET + ".npy")
    if not os.path.exists(r0_npy):
        print(f"[r0] proposing+labeling {inc_n} base events", flush=True)
        P0 = propose_momenta(inc_n, args.y_lo, args.mix_ir, LOW_CUTS, rng_gen)
        me0 = label_events(P0)
        rows0 = np.concatenate([P0.reshape(len(P0), -1), np.tile(PDG.astype(np.float64), (len(P0), 1)),
                                me0.reshape(-1, 1)], axis=1)
        np.save(r0_npy, rows0.astype(np.float64))
        print(f"[r0] saved {r0_npy} N={len(rows0)}  |M|^2 [{me0.min():.2e},{me0.max():.2e}]", flush=True)

    # --- sigma arm: GROW base22 (1-ch MSE) -> 2-ch (mu row verbatim, fresh sigma row) so warm-start
    #     into the HETEROSC net matches shapes AND keeps base22's converged mu head. ---
    pretrained = None
    if args.arm == "sigma":
        grown = os.path.join(REPO, f"data_l2uugg/{run_name}_base_grown.pt")
        if not os.path.exists(grown):
            import subprocess
            subprocess.run([sys.executable, os.path.join(WT, "analysis/divergences/grow_sigma_head.py"),
                            BASE22, grown, str(args.sigma0)], check=True)
        pretrained = grown

    # --- build experiment (warm-start base22[grown], freeze stats, ONE cosine over total_steps) ---
    from experiment import AmplitudeExperiment
    torch.set_default_dtype(torch.float32)
    cfg = build_cfg(args.total_steps, round0_dir, exp_name, run_name, 42 + args.seed, args.arm, pretrained)
    exp = AmplitudeExperiment(cfg)
    exp._init()                     # run_dir, logger, backend (device/dtype/tf32) -- normally via __call__

    # We need init_data to have run (stats frozen) before defining the hook, but train() is called
    # inside full_run(). So we run the init prefix here, install the hook, then the loop.
    exp.init_physics(); exp.init_geometric_algebra(); exp.init_data(); exp._init_dataloader()

    if args.validate_prep:
        _validate_prep(exp, args, rng_gen); return

    exp.init_model(); exp._init_loss(); exp._init_regularization()
    exp._init_ewc(); exp._init_optimizer(); exp._init_scheduler()

    # seed the growing pool from the round-0 TRAIN split
    pool_lists = _dataset_to_lists(exp.train_loader.dataset)
    pool = {"parts": list(pool_lists[0]), "toks": list(pool_lists[1]),
            "amps": list(pool_lists[2]), "orders": list(pool_lists[3]), "pids": list(pool_lists[4])}
    print(f"[pool] round-0 train pool seeded: {len(pool['parts'])} events", flush=True)

    def online_hook(step):
        r = step // steps_per_round
        rng = np.random.default_rng(2000 + args.seed * 97 + r)
        M = int(round(args.oversample * inc_n))
        P_prop = propose_momenta(M, args.y_lo, args.mix_ir, LOW_CUTS, rng)
        keep_n = min(inc_n, len(P_prop))
        if args.arm == "sigma":
            # score fresh proposals by the LIVE model's uncertainty; keep WITHOUT replacement ∝ σ^γ.
            sig = score_sigma(exp, P_prop)
            w = np.clip(sig, 1e-12, None) ** args.gamma
            p = w / w.sum()
            keep = rng.choice(len(P_prop), size=keep_n, replace=False, p=p)
            print(f"[round {r}] σ p50/90/99={np.percentile(sig,[50,90,99])} "
                  f"kept σ mean={sig[keep].mean():.3g} vs all {sig.mean():.3g}", flush=True)
        else:
            keep = rng.choice(len(P_prop), size=keep_n, replace=False)          # base arm: uniform
        P_keep = P_prop[keep]
        me = label_events(P_keep)
        parts, toks, amp_p, orders, pids = preprocess_increment(exp, P_keep, me)
        pool["parts"] += parts; pool["toks"] += toks
        pool["amps"] += [amp_p[i] for i in range(len(amp_p))]
        pool["orders"] += [orders[i] for i in range(len(orders))]
        pool["pids"] += [pids[i] for i in range(len(pids))]
        _make_train_loader(exp, pool)
        print(f"[round {r}] +{len(P_keep)} events -> pool={len(pool['parts'])} "
              f"(step {step}/{args.total_steps})", flush=True)

    # --- drive the round loop by MONKEYPATCHING exp._cycle (no core-module edit) ---
    # train() builds its training iterator once via `iter(self._cycle(self.train_loader))` and pulls
    # one batch/step from it. We replace _cycle with a generator that, every steps_per_round yields,
    # calls online_hook (regenerate+extend the pool, rebuild exp.train_loader) and re-iterates the new
    # loader. The optimizer/scheduler/EMA are train()'s own and are never touched -> ONE continuous
    # cosine across rounds. (Editing base_experiment.train() itself is futile here: the runtime imports
    # the core modules from the MAIN repo, not this worktree -- gen_ir_democratic inserts REPO on
    # sys.path -- so a worktree edit to train() is dead code. The monkeypatch works regardless.)
    def online_cycle(_iterable_ignored=None):
        step = 0
        it = iter(exp.train_loader)
        while True:
            if step > 0 and step % steps_per_round == 0:
                online_hook(step)
                it = iter(exp.train_loader)          # loader rebuilt by the hook
            try:
                batch = next(it)
            except StopIteration:
                it = iter(exp.train_loader)
                batch = next(it)
            yield batch
            step += 1

    exp._cycle = online_cycle
    print(f'[driver] online _cycle installed: round_steps={steps_per_round} inc_n={inc_n} R={R}', flush=True)

    # run training (regeneration fires inside _cycle at each round boundary) + full_run's tail
    exp.train(); exp._save_model()
    if exp.is_main_process():
        exp.evaluate()
        exp.compress_models()
    print(f"[done] arm={args.arm} run={run_name}", flush=True)


def _validate_prep(exp, args, rng):
    """Generate a small labeled batch, preprocess it two ways -- via a fresh init_data on disk vs via
    preprocess_increment with the frozen stats -- and assert they match."""
    P = propose_momenta(2000, args.y_lo, args.mix_ir, LOW_CUTS, rng)
    me2 = label_events(P)
    parts, toks, amp_p, orders, pids = preprocess_increment(exp, P, me2)
    # reference: amp through the exact frozen path already; the mom path is deterministic up to the
    # RANDOM Lorentz aug, so we can only check amp exactly + mom scale/COM invariants statistically.
    amp_ref, _, _ = preprocess_amplitude(me2.reshape(-1, 1), trafos=exp.cfg.data.amp_trafos,
                                         mean=exp.prepd_mean[0], std=exp.prepd_std[0])
    amp_err = np.abs(np.asarray(amp_p) - amp_ref).max()
    momstd = np.std([p for p in parts])
    print(f"[validate_prep] amp max|err|={amp_err:.2e} (frozen stats)  "
          f"mom_div={exp.mom_div:.4g}  increment mom std≈{momstd:.3f} (~1 expected)", flush=True)
    print(f"[validate_prep] amp_prepd mean={float(np.mean(amp_p)):.3f} std={float(np.std(amp_p)):.3f} "
          f"(round-0 frozen mean={float(np.asarray(exp.prepd_mean[0])):.3f} "
          f"std={float(np.asarray(exp.prepd_std[0])):.3f})", flush=True)
    assert amp_err < 1e-8, "amp increment preprocessing does not match frozen stats"
    print("[validate_prep] OK", flush=True)


if __name__ == "__main__":
    main()
