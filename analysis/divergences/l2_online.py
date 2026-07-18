#!/usr/bin/env python
"""L2 -- online sigma-informed GENERATION during training (ee->uu Z resonance).

The aspirational third layer of the divergence-sampling study (docs/results.tex Thread D):
L0 = coverage by GENERATION (flat-log|M|^2 admixture), L1 = sigma-reweighting a FIXED pool
(scoped to starved/un-regenerable pools), L2 = let sigma DRIVE fresh generation DURING training
so it can track the migrating worst-region that no static density stays optimal against.

Design (crispest testable form). One large FROZEN candidate pool (pole+bulk covered, labeled once
with the exact ee_uu standalone). A shared fixed coverage base p_cov = the L0 optimum mixture
(f flat-log|M|^2 + (1-f) uniform-sqrt(s)), expressed as a per-candidate importance weight so both
arms draw from the SAME coverage. Then per round r a training pool of N events is drawn WITHOUT
replacement:
  static : pi = p_cov                          (fixed coverage, re-used every round -- the L0 baseline)
  l2     : pi = p_cov * sigma_r^alpha          (coverage x the CURRENT model's uncertainty)
  oracle : pi = p_cov * |pred_r-true|^alpha     (coverage x the perfect error signal -- ceiling)
Round 0's pool is coverage-only for every arm (no sigma yet) -> identical start; arms diverge from
round 1. mu is a warm CHAIN across rounds (round r finetunes from round r-1's mu), so the only thing
that differs between static and l2 is the round>=1 training-pool composition -- isolating the
sigma-tracking effect. sigma_r comes from the resolved two-stage recipe (freeze trunk+mu, fit only
the sigma readout row at beta=0) on round r-1's mu.

All GPU work (mu finetune, sigma-fit, sigma extraction) runs as a FRESH run.py/py subprocess -- the
muP base-shape globals leak across AmplitudeExperiment instances in one process (documented eval bug),
so each stage must be its own process. Generation + coverage-weighting + thinning are pure numpy here.

Resumable: a stage is skipped if its output (ckpt / pool / sigma npz) already exists.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
import gen_uu_sampling as g  # noqa: E402  (candidate_sqrts / build_momenta / label)

DATASET = "ee_uu_91-1000GeV_amplitudes"
# run.py subprocesses run with cwd=WT, so its default output dir is WT/runs/<exp>/<run>.
RUNS = os.path.join(WT, "runs/eeuu_l2")

# mu finetune HPs -- the validated ee->uu L1 finetune (finetune_l1raw.sh).
MU_HPS = [
    "model.use_diagrams=false", "model.particle_encoder_hidden=0",
    "model.net.num_blocks=8", "model.net.num_heads=8",
    "fine_tune.lr_scale=0.339", "fine_tune.layer_decay=0.999",
    "training.lr=0.004", "training.batchsize=16384",
    "training.regularization=L2", "training.regularization_lambda=2.47e-7",
    "training.scheduler=CosineAnnealingLR", "training.loss_aggregation=geometric_mean",
    "training.cosanneal_warmup_frac=0.191", "training.cosanneal_eta_min=1.6e-7",
]
# sigma-fit HPs -- the resolved two-stage recipe (sigma_fit_raw.sh): freeze mu, fit sigma row at beta=0.
SIG_HPS = [
    "model.use_diagrams=false", "model.particle_encoder_hidden=0",
    "model.net.num_blocks=8", "model.net.num_heads=8",
    "model.net.detach_sigma_backbone=false", "model.net.sigma_after_pool=true",
    "fine_tune.reset_output_head=false", "fine_tune.lr_scale=1.0", "fine_tune.layer_decay=1.0",
    "training.loss=HETEROSC", "training.heterosc_beta=0.0", "training.heterosc_sigma_only=true",
    "training.lr=0.0202319", "training.batchsize=16384", "evaluation.batchsize=16384",
    "training.clip_grad_norm=5", "training.regularization=L2",
    "training.regularization_lambda=9.892346e-07", "training.scheduler=CosineAnnealingLR",
    "training.cosanneal_warmup_frac=0.1537129", "training.cosanneal_eta_min=1.0e-8",
]
DATA_HPS = [
    "data.source=files", f"'data.dataset=[{DATASET}]'", "data.preprocess_per_dataset=true",
    "'data.train_test_val=[0.9, 0.05, 0.05]'", "data.subsample=null",
]


def sh(cmd, cwd=WT):
    print(f"\n$ {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


def ckpt_exists(run_name):
    d = os.path.join(RUNS, run_name, "models")
    return os.path.exists(os.path.join(d, "model_run0_best.pt")) or \
        os.path.exists(os.path.join(d, "model_run0_best.pt.gz"))


def ckpt_path(run_name):
    # run.py gzips the checkpoint to .pt.gz after training; return whichever exists so downstream
    # consumers (grow_sigma_head, which only decompresses when the path ends in .gz) get a real file.
    base = os.path.join(RUNS, run_name, "models", "model_run0_best.pt")
    return base if os.path.exists(base) else base + ".gz"


# ----------------------------------------------------------------------------- generation
def gen_candidate_pool(n_cand, frac_pole, floor, smin, smax, seed, out_npy):
    """Generate + label ONE broad pole+bulk-covering candidate pool; save (N,21) rows."""
    if os.path.exists(out_npy):
        print(f"[cand] reuse {out_npy}", flush=True)
        return
    rng = np.random.default_rng(seed)
    print(f"[cand] generating {n_cand} candidates (frac_pole={frac_pole}) -> label", flush=True)
    s = g.candidate_sqrts(n_cand, smin, smax, frac_pole, floor, rng)
    P = g.build_momenta(s, rng)
    me2 = g.label(P)
    rows = np.concatenate([P.reshape(len(P), -1),
                           np.tile(g.PDG.astype(np.float64), (len(P), 1)),
                           me2.reshape(-1, 1)], axis=1)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, rows.astype(np.float64))
    g._decade_report(me2, s, "cand")
    print(f"[cand] saved {out_npy}  N={len(rows)}", flush=True)


# ----------------------------------------------------------------------------- coverage base
def coverage_weight(rows, f, bins):
    """p_cov = f * flat-in-log|M|^2 + (1-f) * uniform-in-sqrt(s), as a per-candidate weight."""
    me2 = rows[:, -1]
    sqrt_s = 2.0 * rows[:, 0]
    u = np.log(me2)
    cu, eu = np.histogram(u, bins=bins)
    wi = np.clip(np.digitize(u, eu[1:-1]), 0, len(cu) - 1)
    w_flat = np.where(cu[wi] > 0, 1.0 / cu[wi], 0.0)
    cs, es = np.histogram(sqrt_s, bins=bins)
    si = np.clip(np.digitize(sqrt_s, es[1:-1]), 0, len(cs) - 1)
    w_uni = np.where(cs[si] > 0, 1.0 / cs[si], 0.0)
    w_flat = w_flat / w_flat.sum()
    w_uni = w_uni / w_uni.sum()
    return f * w_flat + (1.0 - f) * w_uni


def build_pool(rows, p_cov, score, alpha, n, seed, out_npy, bins=40, q_cov=None):
    """Draw n WITHOUT replacement, pi ∝ p_cov * clip(score)^alpha. score=None -> coverage only.
    Also writes is_weights.json: per-log|M|^2-bin unbiased-IS factor c_b = Q_b/pi_b, where Q_b is the
    OBJECTIVE marginal (q_cov aggregated into |M|^2 bins, defaulting to the sampling base p_cov) and
    pi_b the realized pool density -- so a sigma-emphasized draw is trained UNBIASEDLY against the
    objective. Decoupling q_cov from p_cov lets a STARVED sampling base (e.g. uniform-sqrt(s)) be
    trained toward a GOOD coverage objective (e.g. the f=0.25 mixture) that sigma must fill."""
    pi = p_cov.astype(np.float64).copy()
    if score is not None and alpha != 0:
        pi = pi * np.clip(score, 1e-12, None) ** alpha
    pi = pi / pi.sum()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(rows), size=n, replace=False, p=pi)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, rows[idx])

    # IS correction table over fixed log|M|^2 bins (aligned to the loss-side bucketize).
    q_src = p_cov if q_cov is None else q_cov
    u_all = np.log(rows[:, -1])
    edges = np.linspace(u_all.min(), u_all.max(), bins + 1)
    which_all = np.clip(np.digitize(u_all, edges[1:-1]), 0, bins - 1)
    Qb = np.array([q_src[which_all == b].sum() for b in range(bins)], float)      # objective marginal
    Qb = Qb / Qb.sum()
    cnt_pool = np.bincount(which_all[idx], minlength=bins).astype(float)          # realized pool
    pib = cnt_pool / cnt_pool.sum()
    cb = np.where(pib > 0, Qb / np.clip(pib, 1e-12, None), 0.0)                   # c_b = Q_b/pi_b
    import json as _json
    with open(os.path.join(os.path.dirname(out_npy), "is_weights.json"), "w") as f:
        _json.dump({"logm_edges": edges.tolist(), "c": cb.tolist()}, f)

    sqrt_s = 2.0 * rows[idx, 0]
    reg = [(88, 95), (95, 150), (150, 400), (400, 1000)]
    frac = "  ".join(f"[{lo},{hi}):{100*np.mean((sqrt_s>=lo)&(sqrt_s<hi)):.1f}%" for lo, hi in reg)
    mean_w = float((pib * cb).sum())     # = sum_b Q_b over covered bins ≈ 1 (unbiasedness check)
    print(f"[pool] {os.path.basename(os.path.dirname(out_npy))}: N={n} uniq={len(np.unique(idx))} | "
          f"sqrt(s) {frac} | IS c in [{cb[cb>0].min():.2f},{cb.max():.2f}] mean_w={mean_w:.3f}", flush=True)


# ----------------------------------------------------------------------------- coordinate-free adaptive generation
def adapt_pool(rows, score, n, seed, out_npy, gamma=1.0, state_path=None, damp=0.5):
    """COORDINATE-FREE, self-driven adaptive sampling: the per-event training density is p(x) ∝
    sigma(x)^gamma -- shaped ONLY by the model's own uncertainty score(x) (learned sigma, or |residual|
    for the oracle), with NO reference density, NO physics coordinate (sqrt(s)/y_min/|M|^2), NO bins,
    and NO loss correction. This is the whole point of sigma: it flags where the model is wrong in ANY
    process without naming the divergence variable. Iterated (each round re-scores the SAME candidate
    pool with the current model and re-draws), the density flows toward wherever sigma is large and,
    as those regions get trained down, sigma there falls and the density rebalances -> a fixed point of
    EQUALIZED sigma (uniform error) over phase space. DAMPED via a per-candidate sigma EMA (state_path,
    aligned to the fixed candidate pool) so a huge sigma ratio doesn't collapse the whole budget onto a
    handful of events and oscillate. Draws n WITHOUT replacement by pi ∝ sigma_ema^gamma."""
    sc = np.clip(np.asarray(score, float), 1e-12, None)
    if state_path and os.path.exists(state_path):
        prev = np.load(state_path)["s"]
        s_ema = (1.0 - damp) * prev + damp * sc               # per-candidate EMA (pool is fixed across rounds)
    else:
        s_ema = sc
    if state_path:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        np.savez(state_path, s=s_ema)
    pi = s_ema ** gamma
    pi = pi / pi.sum()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(rows), size=n, replace=False, p=pi)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, rows[idx])
    # diagnostics only (NOT used by the algorithm): sigma spread of the draw + a coarse sqrt(s) view.
    ss = 2.0 * rows[idx, 0]
    q = np.percentile(s_ema[idx], [50, 90, 99])
    reg = [(88, 95), (95, 150), (150, 400), (400, 1000)]
    frac = " ".join(f"[{lo},{hi}):{100*np.mean((ss>=lo)&(ss<hi)):.0f}%" for lo, hi in reg)
    print(f"[adapt] {os.path.basename(os.path.dirname(out_npy))}: N={n} gamma={gamma} "
          f"| sigma(sel) p50/90/99={q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g} | diag sqrt(s) {frac}", flush=True)


# ----------------------------------------------------------------------------- GPU stages (subprocess)
def run_finetune(prev_ckpt, pool_dir, iters, run_name, is_weight_path=None, train_seed=42):
    if ckpt_exists(run_name):
        print(f"[mu] reuse {run_name}", flush=True)
        return ckpt_path(run_name)
    ov = " ".join(DATA_HPS + MU_HPS)
    isw = f"training.is_weight_path={is_weight_path} " if is_weight_path else ""
    sh(f"python run.py exp_name=eeuu_l2 run_name={run_name} seed={train_seed} "
       f"data.data_path={pool_dir}/ {ov} {isw}"
       f"fine_tune.pretrained_path={prev_ckpt} "
       f"training.iterations={iters} plot=false save=true")
    return ckpt_path(run_name)


def run_sigma_fit(mu_ckpt, pool_dir, iters, run_name, sigma0=1e-2):
    sig_dir = os.path.join(RUNS, run_name)
    if os.path.exists(os.path.join(sig_dir, "models", "model_run0_best.pt")) or \
       os.path.exists(os.path.join(sig_dir, "models", "model_run0_best.pt.gz")):
        print(f"[sig] reuse {run_name}", flush=True)
        return sig_dir
    grown = os.path.join(RUNS, run_name + "_grown.pt")
    sh(f"python analysis/divergences/grow_sigma_head.py {mu_ckpt} {grown} {sigma0}")
    ov = " ".join(DATA_HPS + SIG_HPS)
    sh(f"python run.py exp_name=eeuu_l2 run_name={run_name} "
       f"data.data_path={pool_dir}/ {ov} "
       f"fine_tune.pretrained_path={grown} "
       f"training.iterations={iters} plot=false save=true")
    return sig_dir


def extract_score(sig_run_dir, cand_npy, out_npz):
    """Forward the (mu,sigma) model over the candidate pool -> npz with sigma_ln, pred/true, sqrt_s."""
    if os.path.exists(out_npz):
        print(f"[score] reuse {out_npz}", flush=True)
    else:
        sh(f"python analysis/divergences/extract_sigma_eeuu.py "
           f"--pool {cand_npy} --run_dir {sig_run_dir} --ckpt model_run0_best.pt "
           f"--out {out_npz} --batch_events 16384")
    d = np.load(out_npz)
    return np.asarray(d["sigma_ln"], float), np.abs(np.asarray(d["pred_logamp"], float) - np.asarray(d["true_logamp"], float))


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["static", "l2", "oracle", "reweight", "adapt", "adapt_oracle"])
    ap.add_argument("--tag", default="run", help="namespace for cand pool / training pools / run dirs")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=1500, help="mu iters per round")
    ap.add_argument("--sig_iters", type=int, default=800, help="sigma-fit iters per round")
    ap.add_argument("--n", type=int, default=400000, help="training pool size per round")
    ap.add_argument("--n_cand", type=int, default=3000000, help="frozen candidate pool size")
    ap.add_argument("--frac_pole", type=float, default=0.5)
    ap.add_argument("--floor", type=float, default=0.02)
    ap.add_argument("--smin", type=float, default=91.0)
    ap.add_argument("--smax", type=float, default=1000.0)
    ap.add_argument("--f", type=float, default=0.25, help="SAMPLING-base flat-log|M|^2 fraction (0=starved uniform-sqrt(s))")
    ap.add_argument("--q_f", type=float, default=None, help="OBJECTIVE (IS-correction Q) flat-log|M|^2 fraction; default=f. Set >f for a starved base trained toward good coverage.")
    ap.add_argument("--alpha", type=float, default=1.0, help="sigma/err emphasis exponent (l2/oracle arms)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="coordinate-free adaptive exponent p(x) ∝ sigma(x)^gamma (adapt arms). gamma=1 = "
                         "sample ∝ uncertainty; larger = more aggressive concentration on high-sigma events.")
    ap.add_argument("--correct", action="store_true",
                    help="apply the unbiased IS loss correction (c_b=Q_b/pi_b) on emphasized arms")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cand_seed", type=int, default=1234)
    ap.add_argument("--base_ckpt", default=os.path.join(REPO, "runs/pretrain22_heldout_uug/base/models/model_run0_best.pt"))
    args = ap.parse_args()

    cand_npy = os.path.join(REPO, f"data_l2/{args.tag}_cand", DATASET + ".npy")
    gen_candidate_pool(args.n_cand, args.frac_pole, args.floor, args.smin, args.smax, args.cand_seed, cand_npy)
    rows = np.load(cand_npy)
    p_cov = coverage_weight(rows, args.f, args.bins)
    q_f = args.f if args.q_f is None else args.q_f
    q_cov = coverage_weight(rows, q_f, args.bins) if q_f != args.f else None
    print(f"[cov] sampling f={args.f}  objective q_f={q_f} over {len(rows)} candidates", flush=True)

    prev_mu = args.base_ckpt
    prev_sig = None
    for r in range(args.rounds):
        pool_dir = os.path.join(REPO, f"data_l2/{args.tag}_{args.arm}_r{r}")
        pool_npy = os.path.join(pool_dir, DATASET + ".npy")

        # --- build this round's training pool ---
        # reweight = mechanism CONTROL: same uniform sampling as static, but the loss is IS-corrected
        # to the q_f objective (upweight rare pole events) WITHOUT sigma-resampling -> isolates
        # "objective reweighting" from "sigma adds pole samples (generation)".
        uses_sigma = args.arm in ("l2", "oracle", "adapt", "adapt_oracle")
        corrected = (r >= 1 and args.arm in ("l2", "oracle", "reweight"))
        if os.path.exists(pool_npy):
            print(f"[pool] reuse {pool_npy}", flush=True)
        elif args.arm in ("adapt", "adapt_oracle"):
            if r == 0:                                    # coord-free uniform start (generator's natural density)
                adapt_pool(rows, np.ones(len(rows)), args.n, args.seed, pool_npy, gamma=1.0, state_path=None)
            else:
                sig, err = extract_score(prev_sig, cand_npy,
                                         os.path.join(REPO, "analysis/divergences", f"l2_{args.tag}_{args.arm}_score_r{r-1}.npz"))
                score = sig if args.arm == "adapt" else err
                adapt_pool(rows, score, args.n, args.seed + r, pool_npy, gamma=args.gamma,   # p ∝ σ^γ, NO correction
                           state_path=os.path.join(RUNS, f"{args.tag}_{args.arm}_semastate.npz"), damp=0.5)
        elif r == 0:
            build_pool(rows, p_cov, None, 0, args.n, args.seed + r, pool_npy, args.bins)   # coverage-only start
        elif args.arm in ("static", "reweight"):
            qc = q_cov if args.arm == "reweight" else None
            # FRESH draw every round (args.seed + r) so a static arm sees the SAME total unique data as
            # an adaptive one -- matching the training budget on the data axis, not just the iter axis.
            build_pool(rows, p_cov, None, 0, args.n, args.seed + r, pool_npy, args.bins, q_cov=qc)
        else:
            sig, err = extract_score(prev_sig, cand_npy,
                                     os.path.join(REPO, "analysis/divergences", f"l2_{args.tag}_{args.arm}_score_r{r-1}.npz"))
            score = sig if args.arm == "l2" else err
            build_pool(rows, p_cov, score, args.alpha, args.n, args.seed + r, pool_npy, args.bins, q_cov=q_cov)

        # --- warm-chain mu finetune (IS-correct the emphasized/reweight arms when --correct) ---
        isw = os.path.join(pool_dir, "is_weights.json") if (args.correct and corrected) else None
        prev_mu = run_finetune(prev_mu, pool_dir, args.iters, f"{args.tag}_{args.arm}_mu_r{r}",
                               is_weight_path=isw, train_seed=42 + args.seed)

        # --- sigma-fit for the NEXT round (l2/oracle only) ---
        if uses_sigma and r < args.rounds - 1:
            prev_sig = run_sigma_fit(prev_mu, pool_dir, args.sig_iters, f"{args.tag}_{args.arm}_sig_r{r}")

    print(f"\n[done] arm={args.arm} rounds={args.rounds}  final mu = {prev_mu}", flush=True)


if __name__ == "__main__":
    main()
