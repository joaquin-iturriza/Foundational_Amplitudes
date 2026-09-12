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
           model is CURRENTLY wrong (het-head ALEATORIC sigma). The point-uncertainty baseline.
  bbb    : keep proposals with prob ∝ sigma_epi(x)^gamma  -> sigma_epi is the EPISTEMIC predictive std
           of a full-network Bayes-by-backprop posterior (K weight samples). High where training data
           is sparse -- the arguably-right signal for what to generate. See bbb.py.

CPU generation/labeling; GPU training. Run under sbatch (xformers attention is CUDA-only).
"""
import argparse
import os
import sys

import numpy as np
import torch

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, WT)
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
import gen_ir_democratic as G          # noqa: E402
import mg5_pipeline_final as mp         # noqa: E402
from preprocessing import preprocess_amplitude  # noqa: E402
from particle_ids import global_encode  # noqa: E402
from lloca.utils.rand_transforms import rand_lorentz          # noqa: E402
from lloca.utils.polar_decomposition import restframe_boost   # noqa: E402

WORK = os.environ["WORK"]
BASE22 = f"{REPO}/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt"
# Lowered fiducial cuts (open the deep IR), matching gen_uug_sampling.LOW_CUTS.
LOW_CUTS = {"pt_min": 1.0, "cos_max": 0.9, "dr_min": 0.05, "m_min": 0.3}

# ---------------------------------------------------------------- process registry
# The L2 loop is coordinate-free: the ONLY process-specific facts are the particle content
# (PDG/masses/multiplicity), the compiled tree standalone that labels proposals, and the round-0
# frozen-stats dataset. gen_ir_democratic reaches the soft/collinear corners of any process, so
# extending to a new multiplicity, a resonance+IR process, or a MASSIVE final state is just a new
# row here -- no new sampler, no per-process tuning (exactly the transfer claim being stress-tested).
#
# Rows carry the final-state PDG and MASSES explicitly rather than deriving them from a gluon count,
# so massive final states are expressible. `dataset` is only a FILENAME for the round-0 pool the
# driver generates itself (nothing needs to pre-exist in data/).
#
# SLOT-ORDER WARNING: each compiled standalone has its own final-state slot convention, and getting
# it wrong mirrors the forward-backward asymmetry -- a model then trains perfectly on its own set yet
# ANTI-correlates on production data (this silently inverted an earlier ee->uu result). The
# convention is not shared across builds: ee_ttbar takes production order as-is while ee_uu needs its
# final pair swapped. Every row below has been checked -- massless ones against production data,
# ee_bbbarg via the sign of A_FB at the Z pole against the validated ee_uug (both negative under the
# same construction, with |A_FB(b)|>|A_FB(u)| as the SM requires).
MB = 4.7                                                     # m_b in the standalone's param_card
PROCESSES = {
    "uug":    dict(pdg=[11, -11, 2, -2, 21],      masses=[0.0, 0.0, 0.0],
                   standalone="ee_uug_standalone",    dataset="ee_uug_91-1000GeV_amplitudes"),
    "uugg":   dict(pdg=[11, -11, 2, -2, 21, 21],  masses=[0.0, 0.0, 0.0, 0.0],
                   standalone="ee_uugg_standalone",   dataset="ee_uugg_91-1000GeV_amplitudes"),
    "uuggg":  dict(pdg=[11, -11, 2, -2, 21, 21, 21], masses=[0.0] * 5,
                   standalone="ee_uuggg_standalone",  dataset="ee_uuggg_91-1000GeV_amplitudes"),
    # MASSIVE: ee -> b bbar g. The direct massive analogue of uug -- same Z resonance, same soft
    # gluon singularity, but the COLLINEAR limit is regulated by the b mass (dead cone). With
    # sqrt(s) in [91,1000] the dead-cone scale m_b^2/s ~ 2e-5..3e-3 sits inside the y_min window we
    # bin over, so the mass genuinely reshapes the divergence rather than just relabelling it.
    # (ee->ttbar was the original candidate for the "massive threshold" test and is NOT usable: at
    # tree level |M|^2 varies only ~8.7x in total and is FLATTEST at threshold -- the 2m_t cusp is a
    # phase-space/cross-section effect, not an amplitude feature. See results.tex.)
    "bbbarg": dict(pdg=[11, -11, 5, -5, 21],      masses=[MB, MB, 0.0],
                   standalone="ee_bbbarg_standalone", dataset="ee_bbbarg_91-1000GeV_amplitudes"),
}

# module globals set by set_process() (default uugg -> unchanged behaviour for existing importers).
PROCESS = None; DATASET = None; PDG = None; MASSES = None; NP = None; STANDALONE = None
GLUONS = None; COLORED = None


def set_process(name):
    """Point the module globals at process `name`. Importers (make_heldout_uugg, eval_heldout_uugg)
    read L.PDG/NP/... so this must set module globals, not just locals."""
    spec = PROCESSES[name]
    pdg = np.asarray(spec["pdg"], dtype=int)
    masses = np.asarray(spec["masses"], dtype=float)
    assert len(pdg) == 2 + len(masses), f"{name}: pdg/masses length mismatch"
    g = globals()
    g["PROCESS"] = name
    g["PDG"] = pdg
    g["NP"] = len(pdg)                                       # 2 beams + final state
    g["MASSES"] = masses                                     # FINAL-state masses only
    g["STANDALONE"] = f"{WORK}/mg5amcnlo/{spec['standalone']}"
    g["DATASET"] = spec["dataset"]
    fin = np.arange(2, len(pdg))
    g["GLUONS"] = [int(i) for i in fin if pdg[i] == 21]
    g["COLORED"] = [int(i) for i in fin if (abs(pdg[i]) <= 6 or pdg[i] == 21)]
    return spec


set_process("uugg")     # default: identical to the original hard-coded constants

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
def data_overrides():
    """Built from the CURRENT DATASET (set by set_process) -- NOT frozen at import, else a
    --process switch would leave the config pointing at the default (uugg) dataset name."""
    return [
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
def propose_momenta(n, y_lo, mix_ir, cuts, rng, sqrt_s_lo=91.0, sqrt_s_hi=1000.0):
    """Propose n full events (n,6,4) from the base: mix_ir fraction from the IR-democratic generator
    (reaches soft/collinear corners of ANY massless process), the rest flat RAMBO (fills the O(1)
    bulk). Fiducial cuts applied by oversample-and-reject. No labels, no weights -- only WHERE."""
    n_ir = int(round(mix_ir * n))
    n_ram = n - n_ir

    def draw_ir(nb):
        sq = rng.uniform(sqrt_s_lo, sqrt_s_hi, nb)
        Pf = G.democratic_draw(nb, sq, MASSES, y_lo, rng)
        return G.build_full_event(Pf, sq), sq

    parts = []
    if n_ir > 0:
        P_ir, _ = mp._collect_with_cuts(draw_ir, n_ir, list(MASSES), PDG, cuts)
        parts.append(P_ir)
    if n_ram > 0:
        ev, _ = mp.sample_nbody_phase_space(n_ram, sqrt_s_lo, sqrt_s_hi, list(MASSES), PDG, rng=rng, cuts=cuts)
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


def _score_loader(exp, P):
    """Build a shuffle-free eval loader over proposed momenta (dummy amps) for scoring."""
    from dataset import AmplitudeDataset, build_flat_arrays, collate_variable_length
    parts, toks_l, orders, pids = preprocess_momenta(exp, P)
    pf, tf, off = build_flat_arrays(parts, toks_l)
    ds = AmplitudeDataset(
        particles_flat=pf, offsets=off,
        amplitudes=np.zeros((len(parts), 1), dtype=np.float64),     # dummy (unused for sigma)
        tokens_flat=tf, order_labels=np.asarray(orders),
        process_ids=pids.astype(np.int64), dtype=exp.dtype)
    return torch.utils.data.DataLoader(
        ds, batch_size=int(exp.cfg.evaluation.batchsize), shuffle=False, drop_last=False,
        collate_fn=collate_variable_length, num_workers=0)


def score_sigma(exp, P):
    """Forward the LIVE (mu,sigma) model over proposed momenta -> per-proposal sigma (the model's own
    HETEROSC uncertainty). No labels needed. Reuses exp._collect_predictions (handles the split)."""
    loader = _score_loader(exp, P)
    was_training = exp.model.training
    exp.model.eval()
    with torch.no_grad():
        _, _, sig = exp._collect_predictions(loader)
    if was_training:
        exp.model.train()
    return np.asarray(sig, dtype=np.float64).reshape(-1)            # (N,)


def score_epistemic(exp, P, k_samples):
    """EPISTEMIC sigma for the bbb arm: draw k_samples posterior weight samples, forward each over the
    proposals, and return the per-proposal STD of the predicted mu (log|M|^2). High where the posterior
    disagrees = where training data is sparse -- exactly what L2 wants to fill. No labels needed."""
    import bbb as BBB
    loader = _score_loader(exp, P)
    was_training = exp.model.training
    exp.model.eval()                                    # disables dropout...
    BBB.set_sample_in_eval(exp.model, True)             # ...but force posterior weight sampling
    preds = []
    with torch.no_grad():
        for _ in range(k_samples):
            mu, _, _ = exp._collect_predictions(loader)   # fresh weight sample each pass
            preds.append(np.asarray(mu, dtype=np.float64).reshape(-1))
    BBB.set_sample_in_eval(exp.model, False)
    if was_training:
        exp.model.train()
    return np.std(np.stack(preds, axis=0), axis=0)      # (N,) epistemic predictive std


def keep_logweight(sig, args):
    """Log of the un-normalised keep weight for a batch of proposal sigmas.

    The deployed rule has been p ∝ sigma^gamma -- one scalar "temperature" gamma. This generalises it
    to a POLYNOMIAL in u = log(sigma) - median(log sigma):

        log w = c1*u + c2*u^2 + c3*u^3        (c1 = gamma, c2 = c3 = 0  =>  exactly sigma^gamma)

    so the power law is the DEGREE-1 MEMBER and the comparison is properly nested -- a polynomial
    that cannot beat c2=c3=0 has genuinely found nothing. Two design points:

    * Centering on the median (not the max/mean) makes the constant term irrelevant -- it cancels in
      the normalisation -- so c1 reproduces gamma EXACTLY rather than approximately, which is what
      makes the nesting exact. Centering is also what keeps the basis conditioned as sigma drifts
      downward over training: u stays O(1) while log(sigma) does not.
    * We do NOT rescale u by its spread. That would make c1 = gamma*std(log sigma), and since the
      spread changes round to round a fixed c1 would no longer correspond to a fixed gamma -- the
      nesting would silently break.

    Motivation for going beyond degree 1: E[err^2 | sigma] measured on held-out sets is NOT a power
    law in sigma -- fitting log E[err^2|sigma] against log sigma gives R^2 0.89 -> 0.99 (uugg) and
    0.68 -> 0.93 (uuggg) going from degree 1 to 2, with POSITIVE curvature. Positive c2 concentrates
    extra budget only in the high-sigma tail while staying mild at moderate sigma, which is exactly
    the trade-off a single large gamma cannot express (it over-concentrates everywhere at once, which
    is the measured bulk penalty).
    """
    ls = np.log(np.clip(sig, 1e-12, None))
    u = ls - np.median(ls)
    return args.keep_c1 * u + args.keep_c2 * u ** 2 + args.keep_c3 * u ** 3


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
# Capacity axis (--num_heads / --no_pretrain), set once from argv in main().
# num_heads is the muP width axis, so a width change makes every backbone tensor a different
# shape and BASE22 (num_heads=8) is no longer loadable: strict load_state_dict raises, and the
# reset_output_head path would silently shape-filter almost the whole checkpoint away and train
# from scratch while still logging "loading pretrained weights". So a width sweep MUST also pass
# --no_pretrain and is a fresh-init regime, not comparable in absolute level to the warm-started
# saturation runs; only the trend across widths is.
NUM_HEADS = None
NO_PRETRAIN = False


def build_cfg(total_steps, round0_dir, exp_name, run_name, seed, arm, pretrained=None):
    from hydra import compose, initialize_config_dir
    # BOTH arms train the identical HETEROSC(detach, beta=1) model (same mu training, same sigma head);
    # they differ ONLY in the per-round keep rule (uniform vs prop sigma^gamma), so the base arm is a
    # clean control that isolates the sigma-sampling effect with no loss-function confound.
    # bbb arm is a pure-MSE net (variationalized post-init); base/sigma are HETEROSC(detach,beta=1).
    arm_ov = [] if arm == "bbb" else SIG_ARM_OVERRIDES
    # later overrides win: a grown 2-ch checkpoint (sigma arm) supersedes the 1-ch BASE22 in MU_OVERRIDES.
    # Rounds >= 1 always chain from THIS run's own previous checkpoint, which is already at the
    # right width, so --no_pretrain only has to suppress the round-0 BASE22 warm start.
    if pretrained:
        pre_ov = [f"fine_tune.pretrained_path={pretrained}"]
    elif NO_PRETRAIN:
        pre_ov = ["fine_tune.pretrained_path=null"]
    else:
        pre_ov = []
    width_ov = [f"model.net.num_heads={NUM_HEADS}"] if NUM_HEADS else []
    overrides = data_overrides() + MU_OVERRIDES + width_ov + arm_ov + pre_ov + [
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
    ap.add_argument("--arm", required=True, choices=["base", "sigma", "bbb"],
                    help="base=uniform keep; sigma=keep prop het-head sigma^gamma; "
                         "bbb=keep prop EPISTEMIC sigma^gamma from a full-network Bayes-by-backprop net")
    ap.add_argument("--bbb_beta", type=float, default=1e-2, help="bbb: ELBO KL weight (loss += beta*KL/N)")
    ap.add_argument("--bbb_sigma_rel", type=float, default=0.05,
                    help="bbb: initial posterior std as a fraction of each layer's muP weight scale")
    ap.add_argument("--bbb_ksamples", type=int, default=16,
                    help="bbb: posterior forward samples for the epistemic predictive std")
    ap.add_argument("--process", default="uugg", choices=list(PROCESSES),
                    help="ee -> u ubar + n_g gluons: uug (multi-scale Z-res+IR), uugg, uuggg (more legs)")
    ap.add_argument("--tag", default=None, help="default: the process name")
    ap.add_argument("--total_steps", type=int, default=4000)
    ap.add_argument("--n_total", type=int, default=300000)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=1.0, help="sigma arm: p(x) ∝ sigma^gamma")
    # Polynomial generalisation of the keep rule: log w = c1*u + c2*u^2 + c3*u^3, u = log sigma -
    # median. c1 defaults to --gamma and c2=c3=0, so omitting these reproduces p ∝ sigma^gamma
    # EXACTLY and every existing gamma result stands unchanged. See keep_logweight().
    ap.add_argument("--keep_c1", type=float, default=None,
                    help="keep rule linear coeff (default: --gamma, i.e. the plain power law)")
    ap.add_argument("--keep_c2", type=float, default=0.0, help="keep rule quadratic coeff in log-sigma")
    ap.add_argument("--keep_c3", type=float, default=0.0, help="keep rule cubic coeff in log-sigma")
    # Saturation: how much data does a region actually need? Logs, every round, the sigma distribution
    # over the FRESH proposal batch (drawn from the same base every round, so it is a like-for-like
    # measure of the model's remaining uncertainty over the whole space) alongside the pool size.
    # A high-sigma tail that stops falling as the pool grows is the region saturating.
    ap.add_argument("--sat_log", default=None,
                    help="write per-round saturation diagnostics (pool size, sigma percentiles) to this json")
    ap.add_argument("--stop_on_saturation", action="store_true",
                    help="stop GROWING the pool (training continues) once the sigma tail plateaus")
    # Capacity axis: does the sigma tail floor fall when the model gets wider at fixed data?
    ap.add_argument("--num_heads", type=int, default=None,
                    help="muP width axis. Overrides the default 8. Requires --no_pretrain: BASE22 is "
                         "num_heads=8 and cannot be loaded into another width.")
    ap.add_argument("--no_pretrain", action="store_true",
                    help="fresh init instead of the round-0 BASE22 warm start (needed for a width sweep)")
    ap.add_argument("--sat_tol", type=float, default=0.02,
                    help="relative fall in the sigma p99 below which a round counts as saturated")
    ap.add_argument("--sat_patience", type=int, default=2,
                    help="consecutive saturated rounds required before data addition stops")
    ap.add_argument("--sigma0", type=float, default=0.1, help="sigma arm: initial sigma for grow_sigma_head")
    ap.add_argument("--oversample", type=float, default=4.0, help="propose oversample*inc_n, keep inc_n")
    ap.add_argument("--y_lo", type=float, default=1e-6)
    ap.add_argument("--mix_ir", type=float, default=0.5, help="base = mix_ir IR-democratic + rest RAMBO")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout_eval", action="store_true",
                    help="fold the fixed held-out deep-IR eval into the run tail (reloads the best ckpt, "
                         "scores it in-process, saves heldout_eval_<label>.npz) -- no separate eval job")
    ap.add_argument("--heldout_label", default=None,
                    help="label for the held-out npz (default: <base|sigma|g{γ}>_s{seed}, the plot tags)")
    ap.add_argument("--heldout_path", default=None, help="held-out npz (default: heldout_uugg_deepIR.npz)")
    ap.add_argument("--objective", default="deep", choices=["deep", "overall", "logflat"],
                    help="which held-out metric becomes val_loss in --result_path (sweep objective). "
                         "'deep' is the historical default; 'logflat' is correct for concentration "
                         "trade-off questions -- 'deep' rewards starving the bulk.")
    ap.add_argument("--result_path", default=None,
                    help="if set, write {'val_loss': deep-IR MSE, ...} here after the held-out eval "
                         "(the objective a DyHPO sweep reads).")
    ap.add_argument("--validate_prep", action="store_true",
                    help="CPU check: increment preprocessing reproduces init_data on the same rows")
    args = ap.parse_args()

    global NUM_HEADS, NO_PRETRAIN
    if args.num_heads is not None and not args.no_pretrain:
        ap.error("--num_heads requires --no_pretrain: BASE22 is num_heads=8, so at any other width "
                 "the warm start either raises or silently drops the whole backbone.")
    NUM_HEADS, NO_PRETRAIN = args.num_heads, args.no_pretrain

    if args.keep_c1 is None:
        args.keep_c1 = args.gamma             # degree-1 default == the plain sigma^gamma power law
    set_process(args.process)                 # point PDG/NP/MASSES/STANDALONE/DATASET at this process
    tag = args.tag or args.process
    R = args.rounds
    inc_n = args.n_total // R
    steps_per_round = args.total_steps // R
    exp_name = f"eeuu_l2{args.process}"
    # encode gamma for the sigma/bbb arms (else two gamma runs of the same arm share a run dir and
    # collide); gamma=1 stays the canonical unsuffixed name for backward compatibility.
    gsuf = f"_g{int(args.gamma)}" if (args.arm in ("sigma", "bbb") and args.gamma != 1.0) else ""
    run_name = f"{tag}_{args.arm}{gsuf}_s{args.seed}"
    rng_gen = np.random.default_rng(1000 + args.seed)
    print(f"[driver] process={args.process} PDG={PDG.tolist()} NP={NP} standalone={STANDALONE}", flush=True)

    # --- round 0 increment -> disk, so init_data builds it + FREEZES the stats ---
    # NB: encode inc_n in the dir name -- else a small smoke run (small n_total) and a full run that
    # share a run_name would share this dir, and the full run would silently REUSE the smoke's tiny
    # round-0 pool (10x less data -> wrong, and misleadingly faster). inc_n makes them disjoint.
    round0_dir = os.path.join(REPO, f"data_l2{args.process}/{run_name}_r0n{inc_n}")
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

    # --- warm-start checkpoint ---
    # base/sigma arms are HETEROSC: GROW base22 (1-ch MSE) -> 2-ch (mu row verbatim, fresh sigma row)
    # so the warm start matches the 2-ch net AND keeps base22's converged mu head.
    # bbb arm is a pure-MSE (1-ch) net that gets variationalized post-init, so it warm-starts from the
    # ORIGINAL 1-ch base22 directly (its weights become the posterior means) -- no grow needed.
    if NO_PRETRAIN:
        # Width sweep: BASE22 is num_heads=8, so there is nothing loadable to warm-start from at
        # another width. Skip the grow step too -- it only exists to turn the 1-ch BASE22 checkpoint
        # into a 2-ch one, and a fresh HETEROSC init already emits both channels.
        pretrained = None
    elif args.arm == "bbb":
        pretrained = BASE22
    else:
        grown = os.path.join(REPO, f"data_l2{args.process}/{run_name}_base_grown.pt")
        if not os.path.exists(grown):
            import subprocess
            subprocess.run([sys.executable, os.path.join(WT, "analysis/divergences/grow_sigma_head.py"),
                            BASE22, grown, str(args.sigma0)], check=True)
        pretrained = grown

    # --- build experiment (warm-start base22[grown], freeze stats, ONE cosine over total_steps) ---
    from experiment import AmplitudeExperiment
    torch.set_default_dtype(torch.float32)
    cfg = build_cfg(args.total_steps, round0_dir, exp_name, run_name, 42 + args.seed, args.arm, pretrained)
    # Assert rather than trust the override chain: at num_heads=8 a leftover warm start LOADS
    # CLEANLY (shapes happen to match) and the run silently becomes warm-started while the sweep
    # reports it as fresh-init. Only the other widths would have failed loudly.
    _pp = cfg.fine_tune.get("pretrained_path", None)
    if NO_PRETRAIN and _pp not in (None, "", "null"):
        raise RuntimeError(f"--no_pretrain requested but fine_tune.pretrained_path resolved to {_pp!r}")
    print(f"[cfg] num_heads={cfg.model.net.num_heads} pretrained_path={_pp}", flush=True)
    # idempotent rerun: base_experiment aborts on an existing run dir, so clear THIS run's dir first
    # (disposable L2 run; the round-0 data dir + grown ckpt above are reused, not cleared).
    import shutil
    shutil.rmtree(os.path.join("runs", exp_name, run_name), ignore_errors=True)
    exp = AmplitudeExperiment(cfg)
    exp._init()                     # run_dir, logger, backend (device/dtype/tf32) -- normally via __call__
    exp._save_config("config.yaml")  # full_run normally does this; we bypass full_run, so save it here
                                     # (the held-out eval reloads config.yaml to rebuild the model).

    # We need init_data to have run (stats frozen) before defining the hook, but train() is called
    # inside full_run(). So we run the init prefix here, install the hook, then the loop.
    exp.init_physics(); exp.init_geometric_algebra(); exp.init_data(); exp._init_dataloader()

    if args.validate_prep:
        _validate_prep(exp, args, rng_gen); return

    exp.init_model()
    bbb_layers = None
    if args.arm == "bbb":
        # Turn the warm-started (base22 means) MSE net into a full-network Bayes-by-backprop posterior:
        # every Linear (except the equivariant framesnet) gets a Gaussian weight posterior. Done AFTER
        # init_model (means = base22, muP finalised) and BEFORE _init_optimizer so the fresh rho params
        # are picked up by build_ft_param_groups (matched to their layer's depth by name prefix).
        import bbb as BBB
        bbb_layers = BBB.variationalize(exp.model, sigma_rel=args.bbb_sigma_rel, prior_rel=1.0)
        n_rho = sum(p.numel() for m in bbb_layers for p in (m.weight_rho,
                    *( (m.bias_rho,) if m.has_bias else ())))
        print(f"[bbb] variationalized {len(bbb_layers)} Linear layers ({n_rho} posterior-std params); "
              f"beta={args.bbb_beta} sigma_rel={args.bbb_sigma_rel} K={args.bbb_ksamples}", flush=True)
    exp._init_loss(); exp._init_regularization()
    exp._init_ewc(); exp._init_optimizer(); exp._init_scheduler()

    if args.arm == "bbb":
        # ELBO: add the KL(q||prior) term to the MINIMISED loss (loss += beta*KL/N), leaving
        # loss_no_reg the PURE data MSE so best-checkpoint selection and the comparison metric stay
        # like-for-like with the het-head arms. N = current pool size (KL is a per-example prior).
        import bbb as BBB
        _orig_batch_loss = exp._batch_loss
        def _bbb_batch_loss(data):
            loss, lnr, mse = _orig_batch_loss(data)
            n = max(1, len(exp.train_loader.dataset))
            loss = loss + (args.bbb_beta / n) * BBB.total_kl(exp.model)
            return loss, lnr, mse
        exp._batch_loss = _bbb_batch_loss

        # Checkpoint RANKING is Bayesian too: score validation with the PREDICTIVE MEAN over K_val
        # posterior samples (matches how the model is evaluated + reported), not a single mean-weights
        # forward. Cheaper K for the frequent validation than the eval/scoring K.
        _orig_validate = exp._validate
        K_val = min(args.bbb_ksamples, 8)
        def _bbb_validate(step):
            with BBB.predictive_forward(exp.model, K_val):
                return _orig_validate(step)
        exp._validate = _bbb_validate

    # seed the growing pool from the round-0 TRAIN split
    pool_lists = _dataset_to_lists(exp.train_loader.dataset)
    pool = {"parts": list(pool_lists[0]), "toks": list(pool_lists[1]),
            "amps": list(pool_lists[2]), "orders": list(pool_lists[3]), "pids": list(pool_lists[4])}
    print(f"[pool] round-0 train pool seeded: {len(pool['parts'])} events", flush=True)

    # --- saturation state: per-round record + the "stop growing" latch ---
    sat = {"rounds": [], "stopped_at": None, "n_saturated": 0, "prev_p99": None}

    def saturation_probe(r, sig, pool_n):
        """Record how much uncertainty is left, and decide whether the pool has saturated.

        The probe is the sigma distribution over the FRESH proposal batch. That batch is drawn from
        the same fixed base proposal every round, so its sigma percentiles are directly comparable
        across rounds and measure the model's remaining uncertainty over the whole space -- unlike a
        pool statistic, which drifts because the pool itself is being reshaped by the keep rule.
        The p99 (the hard tail) is the stopping statistic: the bulk saturates long before the
        singular region does, so a mean would call saturation far too early."""
        p50, p90, p99 = (float(x) for x in np.percentile(sig, [50, 90, 99]))
        rec = {"round": int(r), "pool": int(pool_n), "sigma_p50": p50,
               "sigma_p90": p90, "sigma_p99": p99, "sigma_mean": float(sig.mean())}
        prev = sat["prev_p99"]
        rec["rel_drop_p99"] = (float((prev - p99) / prev) if prev else None)
        if prev is not None and (prev - p99) / prev < args.sat_tol:
            sat["n_saturated"] += 1
        else:
            sat["n_saturated"] = 0
        sat["prev_p99"] = p99
        rec["n_saturated"] = sat["n_saturated"]
        sat["rounds"].append(rec)
        if args.sat_log:
            import json
            with open(args.sat_log, "w") as f:
                json.dump({"process": args.process, "arm": args.arm, "seed": args.seed,
                           "gamma": args.gamma, "n_total": args.n_total, "rounds": sat["rounds"],
                           "stopped_at": sat["stopped_at"]}, f, indent=2)
        return sat["n_saturated"] >= args.sat_patience

    def online_hook(step):
        r = step // steps_per_round
        rng = np.random.default_rng(2000 + args.seed * 97 + r)
        if sat["stopped_at"] is not None:
            print(f"[round {r}] pool SATURATED at round {sat['stopped_at']} "
                  f"(pool={len(pool['parts'])}); training on, not growing", flush=True)
            return
        M = int(round(args.oversample * inc_n))
        P_prop = propose_momenta(M, args.y_lo, args.mix_ir, LOW_CUTS, rng)
        keep_n = min(inc_n, len(P_prop))
        # The base arm needs no sigma to CHOOSE, but the saturation probe needs one to MEASURE. Both
        # arms carry the same detached sigma head, so scoring the base arm costs one extra forward and
        # keeps the saturation curves comparable between arms.
        want_sat = bool(args.sat_log or args.stop_on_saturation)
        sig = None
        if args.arm in ("sigma", "bbb") or want_sat:
            # score fresh proposals by the LIVE model's uncertainty; keep WITHOUT replacement ∝ σ^γ.
            # sigma arm = het-head aleatoric σ; bbb arm = epistemic predictive std (K posterior samples).
            if args.arm == "bbb":
                sig = score_epistemic(exp, P_prop, args.bbb_ksamples)
            else:
                sig = score_sigma(exp, P_prop)
        if args.arm in ("sigma", "bbb"):
            # Keep-rule weight, always built in LOG space and max-shifted. Mathematically identical
            # to the direct power, but safe at large gamma: a direct sigma**gamma underflows to exact
            # 0 for the small-sigma tail once gamma is big (sigma~1e-2, gamma~30 -> 1e-60, worse for
            # wider spreads), and if fewer than keep_n entries stay non-zero
            # np.random.choice(replace=False) raises "Fewer non-zero entries in p than size".
            logw = keep_logweight(sig, args)
            w = np.exp(logw - logw.max())
            p = w / w.sum()
            keep = rng.choice(len(P_prop), size=keep_n, replace=False, p=p)
            # Spearman(logw, sigma) instruments MONOTONICITY of the keep rule. For the pure power law
            # it is identically +1; a polynomial with negative curvature can turn over and start
            # up-weighting the LOW-sigma tail, which is a pathology worth seeing rather than assuming
            # away, so it is logged rather than forbidden.
            def _rank(a): return np.argsort(np.argsort(a))
            mono = float(np.corrcoef(_rank(logw), _rank(sig))[0, 1])
            print(f"[round {r}] {args.arm} σ p50/90/99={np.percentile(sig,[50,90,99])} "
                  f"kept σ mean={sig[keep].mean():.3g} vs all {sig.mean():.3g} "
                  f"keep-rule monotonicity ρ={mono:+.3f}", flush=True)
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
        # Probe AFTER the pool has grown, so `pool` is the size that produced the next round's sigma.
        if want_sat:
            saturated = saturation_probe(r, sig, len(pool["parts"]))
            rec = sat["rounds"][-1]
            drop = rec["rel_drop_p99"]
            drop_s = "n/a" if drop is None else f"{drop:.2%}"
            print(f"[round {r}] saturation: σ p99={rec['sigma_p99']:.4g} (drop {drop_s}) "
                  f"consec_saturated={rec['n_saturated']}", flush=True)
            if saturated and args.stop_on_saturation and sat["stopped_at"] is None:
                sat["stopped_at"] = int(r)
                print(f"[round {r}] SATURATION REACHED -> pool frozen at {len(pool['parts'])} events "
                      f"({args.sat_patience} rounds with <{args.sat_tol:.1%} σ-p99 fall)", flush=True)

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
        # Fold the held-out deep-IR eval into the run tail: reload the BEST checkpoint (the same
        # model_run0_best.pt the standalone eval scores, so the number matches) and score it in-process
        # BEFORE compress_models() gzips the checkpoints. Eliminates the separate eval job + queue wait.
        if args.heldout_eval:
            import eval_heldout_uugg as EV
            # This driver runs as __main__, but EV does `import l2_online_uugg as L`, which is a SECOND
            # module object still on the default set_process("uugg"). EV.score_and_save reads L.NP/PDG,
            # so point THAT copy at our process too, else uug/uuggg momenta get reshaped with NP=6.
            EV.L.set_process(args.process)
            if args.arm == "base":
                htag = "base"
            elif args.arm == "bbb":
                htag = "bbb" if args.gamma == 1.0 else f"bbb_g{int(args.gamma)}"
            else:
                htag = "sigma" if args.gamma == 1.0 else f"g{int(args.gamma)}"
            label = args.heldout_label or f"{args.process}_{htag}_s{args.seed}"
            heldout_path = args.heldout_path or os.path.join(
                REPO, f"analysis/divergences/heldout_{args.process}_deepIR.npz")
            state = EV._load_ckpt_state(exp.cfg.run_dir, "model_run0_best.pt")
            exp.model.load_state_dict(state)
            exp.model.to(exp.device, dtype=exp.dtype).eval()
            # bbb: evaluate THROUGH the posterior (predictive mean over K samples + calibration).
            metrics = EV.score_and_save(exp, label, heldout_path,
                                        mc_samples=(args.bbb_ksamples if args.arm == "bbb" else 1))
            # a sweep reads the objective from here (val_loss = the deep-IR MSE we minimise).
            if args.result_path and isinstance(metrics, dict):
                import json
                # Which metric the sweep minimises. 'deep' is the historical default (the BBB sweep
                # used it); 'logflat' is the right choice for any question about the concentration
                # trade-off, since 'deep' rewards starving the bulk and 'overall' is bulk-dominated.
                key = {"deep": "deep_mse", "overall": "overall_mse",
                       "logflat": "logflat_mse"}[args.objective]
                with open(args.result_path, "w") as f:
                    json.dump({"val_loss": metrics[key], "objective": args.objective,
                               "overall_mse": metrics["overall_mse"], "deep_mse": metrics["deep_mse"],
                               "logflat_mse": metrics.get("logflat_mse"),
                               "arm": args.arm, "gamma": args.gamma,
                               "keep_c1": args.keep_c1, "keep_c2": args.keep_c2,
                               "keep_c3": args.keep_c3,
                               "bbb_beta": args.bbb_beta, "bbb_sigma_rel": args.bbb_sigma_rel}, f)
                print(f"[result] wrote {args.result_path} val_loss({key})={metrics[key]:.6e}",
                      flush=True)
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
