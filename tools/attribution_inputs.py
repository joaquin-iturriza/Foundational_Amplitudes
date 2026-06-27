"""Input-feature importance for a trained LLoCa-μP amplitude model.

Loads a finished run, reconstructs the *exact* model + preprocessing it was
trained with, and attributes the model's predicted (preprocessed) log-amplitude
to its physical inputs. Two complementary, on-manifold methods:

  * PARTICLE IDENTITY -> exact discrete Shapley over particle slots.
      Players = the P particles of an event. A coalition S keeps those particles'
      true identity; every particle not in S is swapped to a fixed REFERENCE real
      particle (default: photon). Momenta are untouched. So every model evaluation
      uses only REAL particle property vectors — no fractional/unphysical particles
      (which is what broke path-integral / Integrated-Gradients attribution: this
      model is a sharp interpolator that extrapolates nonsensically off the
      discrete physical manifold). Exact via all 2^P coalitions (P<=6 -> <=64
      evals/event). Completeness: sum_p phi_p = F(event) - F(all-reference event).

  * MOMENTUM -> local gradient sensitivity at the real inputs, |dF/dp|*sigma_p.
      Continuous input, so a local on-manifold gradient is the robust measure
      (no off-manifold baseline traversed).

The 8 property numbers are normally a table lookup (property_matrix[token]); we
feed the looked-up vector in as a continuous input upstream of the model's learned
particle_encoder so the momentum gradient and the Shapley swaps both work.

Run on a GPU (xformers attention is CUDA-only):
    python attribution_inputs.py --run-dir runs/<run> --frame aug --n-per-process 512
"""

import argparse
import os
import sys
from math import factorial

# This script lives in tools/ but imports project-root modules (wrappers,
# experiment, base_experiment, particle_ids). Make the project root importable
# regardless of the cwd it's launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from wrappers import _pool_events

# PDG -> short name, for labelling slots/particles in the output
PDG_NAMES = {
    11: "e-", -11: "e+", 12: "nu", -12: "nu~", 13: "mu-", -13: "mu+",
    22: "gamma", 23: "Z", 24: "W+", -24: "W-", 25: "H", 21: "g",
    1: "d", -1: "d~", 2: "u", -2: "u~", 3: "s", -3: "s~",
    4: "c", -4: "c~", 5: "b", -5: "b~", 6: "t", -6: "t~",
}


# ---------------------------------------------------------------------------
# Model + data loading (reconstruct the run exactly, then load the checkpoint)
# ---------------------------------------------------------------------------
def load_experiment(run_dir, ckpt_name, frame):
    """Build an AmplitudeExperiment from a run's saved config, run the init
    pipeline (physics/data/model), and load the trained weights.

    frame="com": neutralise the random-Lorentz augmentation so each event is in
    its physical COM frame (momentum components interpretable). frame="aug": keep
    the training-time random augmentation (predictions identical by invariance)."""
    import experiment as experiment_mod
    from experiment import AmplitudeExperiment
    from base_experiment import _torch_load

    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    with open_dict(cfg):
        cfg.save = False
        cfg.use_mlflow = False
        cfg.train = False
        cfg.evaluate = False
        cfg.plot = False
        cfg.count_flops = False
        cfg.warm_start_idx = None
        cfg.run_dir = run_dir

    if frame == "com":
        def _identity_lorentz(shape, generator=None, dtype=torch.float64):
            return torch.eye(4, dtype=dtype).expand(*shape, 4, 4).clone()
        experiment_mod.rand_lorentz = _identity_lorentz

    exp = AmplitudeExperiment(cfg)
    exp.warm_start = False
    exp._init_backend()
    exp.init_physics()
    exp.init_geometric_algebra()
    exp.init_data()
    exp.init_model()

    ckpt_path = os.path.join(run_dir, "models", ckpt_name)
    state = _torch_load(ckpt_path, map_location=exp.device, weights_only=False)
    exp.model.load_state_dict(state["model"])
    exp.model.eval()
    exp.model.to(exp.device)
    print(f"Loaded weights from {ckpt_path}")
    return exp


# ---------------------------------------------------------------------------
# Functional forward, taking the continuous property vector directly
# ---------------------------------------------------------------------------
class ForwardFn:
    """Mirror of AmplitudeLLoCaWrapper.forward (use_PIDs=False) but taking the
    continuous property vector `props` (N,8) instead of an integer token."""

    def __init__(self, exp):
        self.encoder = exp.model.particle_encoder
        self.net = exp.model.net
        self.mean = float(exp.mom_mean[0])
        self.std = float(exp.mom_std[0])

    def __call__(self, fm, props, order, ptr, seq_lens):
        particle_type = self.encoder(props)
        counts = ptr[1:] - ptr[:-1]
        order_pp = order.repeat_interleave(counts, dim=0)
        particle_type = torch.cat(
            [particle_type, order_pp.to(particle_type.dtype)], dim=-1)
        out = self.net(fm, particle_type, self.mean, self.std,
                       ptr=ptr, seq_lens=seq_lens)
        return _pool_events(out, ptr)


# ---------------------------------------------------------------------------
# Per-process dense input assembly
# ---------------------------------------------------------------------------
def gather_process_events(exp, proc_idx, n_events, rng):
    """Dense tensors for up to n_events events of one process:
    fm (B,P,4), props (B,P,8), order (B,n_order), y (B,1), tokens (P,) slot ids."""
    idx = np.where(exp.all_process_ids == proc_idx)[0]
    if idx.size == 0:
        return None
    if idx.size > n_events:
        idx = rng.choice(idx, size=n_events, replace=False)
    offsets = exp.offsets
    fm_list, tok_list = [], []
    for e in idx:
        s, t = int(offsets[e, 0]), int(offsets[e, 1])
        fm_list.append(exp.particles_flat[s:t])
        tok_list.append(exp.tokens_flat[s:t])
    P = fm_list[0].shape[0]
    assert all(f.shape[0] == P for f in fm_list), "ragged process block"
    fm = np.stack(fm_list).astype(np.float32)
    toks = np.stack(tok_list).astype(np.int64)
    props = exp.property_matrix[toks].astype(np.float32)
    order = exp.all_order_labels[idx].astype(np.float32)
    y = exp.all_amplitudes[idx].astype(np.float32)
    return dict(fm=fm, props=props, order=order, y=y, P=P, n=len(idx),
                tokens=toks[0])


# ---------------------------------------------------------------------------
# Exact discrete Shapley over particle identities (reference-particle baseline)
# ---------------------------------------------------------------------------
def _coalition_values(fn, fm_t, pr_t, or_t, bits_t, ref, P, M, device, budget):
    """v(S) for all 2^P coalitions, for a single reference particle `ref` (8,).
    Particles in S keep real identity, others -> ref. Returns v (B,M)."""
    B = fm_t.shape[0]
    v = torch.zeros((B, M), device=device)
    bsel = bits_t[None, :, :, None]                       # (1,M,P,1)
    ev_chunk = max(1, budget // M)
    for b0 in range(0, B, ev_chunk):
        b1 = min(b0 + ev_chunk, B); bs = b1 - b0
        xfm, xpr, xor = fm_t[b0:b1], pr_t[b0:b1], or_t[b0:b1]
        coal = bsel * xpr[:, None] + (1 - bsel) * ref.view(1, 1, 1, 8)  # (bs,M,P,8)
        E = bs * M
        fm_flat = xfm[:, None].expand(bs, M, P, 4).reshape(E * P, 4)
        pr_flat = coal.reshape(E * P, 8)
        or_flat = xor[:, None].expand(bs, M, xor.shape[-1]).reshape(E, -1)
        ptr = torch.arange(0, (E + 1) * P, P, device=device, dtype=torch.long)
        with torch.no_grad():
            out = fn(fm_flat, pr_flat, or_flat, ptr, tuple([P] * E))
        v[b0:b1] = out.reshape(bs, M)
    return v


def exact_particle_shapley(fn, fm, props, order, ref_set, device, budget):
    """Interventional Shapley value of each particle slot's identity.

    When a particle leaves a coalition, its identity is replaced by a particle
    drawn (uniformly) from `ref_set` (R real particles); the value function is the
    average over that background, and the Shapley value is the average of the
    single-reference Shapley values (Shapley is linear in v). With a 1-particle
    ref_set this reduces to a fixed-reference baseline. Exact over all 2^P subsets.
    Returns phi (B,P), v_full (B,), v_empty (B,) (the last two averaged over refs)."""
    B, P, _ = fm.shape
    M = 1 << P
    bits = ((np.arange(M)[:, None] >> np.arange(P)[None, :]) & 1).astype(np.float32)
    bits_t = torch.from_numpy(bits).to(device)
    fm_t = torch.from_numpy(fm).to(device)
    pr_t = torch.from_numpy(props).to(device)
    or_t = torch.from_numpy(order).to(device)
    refs = torch.as_tensor(np.asarray(ref_set), dtype=torch.float32, device=device)
    R = refs.shape[0]

    # Shapley combination weights (independent of reference)
    combos = []
    for p in range(P):
        wo = np.array([m for m in range(M) if not ((m >> p) & 1)])
        wp = wo | (1 << p)
        k = np.array([bin(int(m)).count("1") for m in wo])
        w = np.array([factorial(int(kk)) * factorial(P - int(kk) - 1) / factorial(P)
                      for kk in k], dtype=np.float32)
        combos.append((torch.from_numpy(wp).to(device),
                       torch.from_numpy(wo).to(device),
                       torch.from_numpy(w).to(device)))

    phi = torch.zeros((B, P), device=device)
    vfull = torch.zeros(B, device=device)
    vempty = torch.zeros(B, device=device)
    for r in range(R):
        v = _coalition_values(fn, fm_t, pr_t, or_t, bits_t, refs[r], P, M, device, budget)
        for p, (wp, wo, w) in enumerate(combos):
            phi[:, p] += ((v[:, wp] - v[:, wo]) * w).sum(dim=1)
        vfull += v[:, M - 1]
        vempty += v[:, 0]
    return (phi.cpu().numpy() / R, vfull.cpu().numpy() / R, vempty.cpu().numpy() / R)


# ---------------------------------------------------------------------------
# Momentum + physical-property: local gradient sensitivity at the real inputs
# ---------------------------------------------------------------------------
# For the continuous inputs (the 4-momenta, and the 8 property numbers) the robust
# on-manifold importance is the LOCAL gradient at the real event (one backward, no
# off-manifold path): mean_events |dF/dx| * sigma_x, with sigma_x the input's
# natural spread in the data. (Path-integral / IG over these blew up off-manifold.)
def local_sensitivity(fn, fm, props, order, scale_fm, scale_pr, device, mb_events):
    """Returns sfm (B,P,4), spr (B,P,8) sigma-scaled |grad|, and preds (B,1)."""
    B, P, _ = fm.shape
    fm_t = torch.from_numpy(fm).to(device)
    pr_t = torch.from_numpy(props).to(device)
    or_t = torch.from_numpy(order).to(device)
    g_fm = torch.zeros_like(fm_t)
    g_pr = torch.zeros_like(pr_t)
    preds = torch.zeros((B, 1), device=device)
    for b0 in range(0, B, mb_events):
        b1 = min(b0 + mb_events, B); bs = b1 - b0
        fm_flat = fm_t[b0:b1].reshape(bs * P, 4).clone().requires_grad_(True)
        pr_flat = pr_t[b0:b1].reshape(bs * P, 8).clone().requires_grad_(True)
        ptr = torch.arange(0, (bs + 1) * P, P, device=device, dtype=torch.long)
        out = fn(fm_flat, pr_flat, or_t[b0:b1], ptr, tuple([P] * bs))
        gfm, gpr = torch.autograd.grad(out.sum(), [fm_flat, pr_flat])
        g_fm[b0:b1] = gfm.reshape(bs, P, 4)
        g_pr[b0:b1] = gpr.reshape(bs, P, 8)
        preds[b0:b1] = out.detach()
    sfm = np.abs(g_fm.cpu().numpy()) * scale_fm
    spr = np.abs(g_pr.cpu().numpy()) * scale_pr
    return sfm, spr, preds.cpu().numpy()


# ---------------------------------------------------------------------------
# Property contrasts: on-manifold isolation of one property via real-particle swaps
# ---------------------------------------------------------------------------
def property_contrasts(fn, fm, props, order, contrasts, device, mb_events):
    """For each contrast (name, propsA, propsB) — a pair of REAL particles that
    differ in (essentially) one property — measure how much the amplitude moves
    when a particle is that A vs that B, with kinematics fixed. Both endpoints are
    real particle states (on-manifold, same footing as the Shapley). Score =
    mean over events and particle slots of |F(slot=A) - F(slot=B)|."""
    B, P, _ = fm.shape
    fm_t = torch.from_numpy(fm).to(device)
    pr_t = torch.from_numpy(props).to(device)
    or_t = torch.from_numpy(order).to(device)
    sums = {name: 0.0 for name, _, _ in contrasts}
    cnt = 0
    for b0 in range(0, B, mb_events):
        b1 = min(b0 + mb_events, B); bs = b1 - b0
        xfm = fm_t[b0:b1].reshape(bs * P, 4)
        xor = or_t[b0:b1]
        ptr = torch.arange(0, (bs + 1) * P, P, device=device, dtype=torch.long)
        seq = tuple([P] * bs)
        for name, pa, pb in contrasts:
            tA = torch.as_tensor(pa, dtype=torch.float32, device=device)
            tB = torch.as_tensor(pb, dtype=torch.float32, device=device)
            dsum = 0.0
            for s in range(P):                 # swap each slot in turn
                prA = pr_t[b0:b1].clone(); prA[:, s, :] = tA
                prB = pr_t[b0:b1].clone(); prB[:, s, :] = tB
                with torch.no_grad():
                    fA = fn(xfm, prA.reshape(bs * P, 8), xor, ptr, seq)
                    fB = fn(xfm, prB.reshape(bs * P, 8), xor, ptr, seq)
                dsum += (fA - fB).abs().sum().item()
            sums[name] += dsum
        cnt += bs * P
    return {name: sums[name] / cnt for name in sums}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(global_pdg_imp, per_proc, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Global particle-identity importance (Shapley) by particle type
    names = list(global_pdg_imp.keys())
    vals = np.array([global_pdg_imp[n] for n in names])
    o = np.argsort(vals)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(len(names)), vals[o], color="#3b6ea5")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([names[i] for i in o], rotation=40, ha="right")
    ax.set_ylabel("mean |Shapley value|")
    ax.set_title("Particle-identity importance (exact Shapley, by particle type)")
    fig.tight_layout(); fig.savefig(f"{out_prefix}_identity_importance.pdf"); plt.close(fig)

    # 2) Per-process: Shapley per slot (labelled) + momentum per slot
    nproc = len(per_proc)
    fig, axes = plt.subplots(2, nproc, figsize=(3.0 * nproc, 6.0), squeeze=False)
    for k, (name, d) in enumerate(per_proc.items()):
        labels = [PDG_NAMES.get(p, str(p)) for p in d["slot_pdg"]]
        x = np.arange(d["P"])
        ax = axes[0][k]
        ax.bar(x, d["shap_abs"], color="#3b6ea5")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_title(f"{name.split('_')[1]} (MSE={d['mse']:.0e})", fontsize=8)
        if k == 0:
            ax.set_ylabel("mean |Shapley| (identity)")
        ax = axes[1][k]
        ax.bar(x, d["mom_per_slot"], color="#c0504d")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, fontsize=7)
        if k == 0:
            ax.set_ylabel("mean |dF/dp|·σ (momentum)")
    fig.suptitle("Per-process: identity Shapley (top) and momentum sensitivity (bottom)")
    fig.tight_layout(); fig.savefig(f"{out_prefix}_per_process.pdf"); plt.close(fig)
    print(f"Saved plots to {out_prefix}_*.pdf")


def make_property_plot(global_prop_imp, prop_names, per_proc, out_prefix):
    """Global physical-property importance (top) + per-process panels (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nproc = len(per_proc)
    fig = plt.figure(figsize=(max(8, 2.4 * nproc), 7))
    gs = fig.add_gridspec(2, nproc)

    ax0 = fig.add_subplot(gs[0, :])
    o = np.argsort(global_prop_imp)[::-1]
    ax0.bar(np.arange(len(prop_names)), global_prop_imp[o], color="#3b6ea5")
    ax0.set_xticks(np.arange(len(prop_names)))
    ax0.set_xticklabels([prop_names[i] for i in o], rotation=30, ha="right")
    ax0.set_ylabel("mean |dF/dprop|·σ")
    ax0.set_title("Physical-property importance (pooled over all particles/processes)")

    for k, (name, d) in enumerate(per_proc.items()):
        ax = fig.add_subplot(gs[1, k])
        v = d["prop_per_dim"]; oo = np.argsort(v)[::-1]
        ax.bar(np.arange(len(prop_names)), v[oo], color="#3b6ea5")
        ax.set_xticks(np.arange(len(prop_names)))
        ax.set_xticklabels([prop_names[i] for i in oo], rotation=80, ha="right", fontsize=5)
        ax.set_title(name.split("_")[1], fontsize=8)
        if k == 0:
            ax.set_ylabel("mean |dF/dprop|·σ")
    fig.tight_layout(); fig.savefig(f"{out_prefix}_property_importance.pdf"); plt.close(fig)
    print(f"Saved property plot to {out_prefix}_property_importance.pdf")


def make_contrast_plot(global_contrast, out_prefix):
    """Property importance via on-manifold real-particle contrasts (mean |ΔF|)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(global_contrast.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    vals = np.array([v for _, v in items])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(labels)), vals, color="#3b6ea5")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean |ΔF|  (log-amplitude units)")
    ax.set_title("Property importance via real-particle contrasts (on-manifold)")
    fig.tight_layout(); fig.savefig(f"{out_prefix}_property_contrasts.pdf"); plt.close(fig)
    print(f"Saved contrast plot to {out_prefix}_property_contrasts.pdf")


def make_kin_vs_id_plot(per_proc, out_prefix):
    """Grouped bars per process comparing kinematic vs identity importance, both
    in predicted-log-amplitude units (mean |ΔF|)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n.split("_")[1] for n in per_proc]
    kin = np.array([d["kin_imp"] for d in per_proc.values()])
    ide = np.array([d["ident_imp"] for d in per_proc.values()])
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, kin, w, label="kinematics (momenta)", color="#c0504d")
    ax.bar(x + w / 2, ide, w, label="identity (particle type)", color="#3b6ea5")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("mean |ΔF|  (log-amplitude units)")
    ax.set_title("Kinematic vs identity importance, per process")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_prefix}_kin_vs_identity.pdf"); plt.close(fig)
    print(f"Saved kin-vs-identity plot to {out_prefix}_kin_vs_identity.pdf")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="model_run0_best.pt.gz")
    ap.add_argument("--frame", choices=["com", "aug"], default="aug")
    ap.add_argument("--n-per-process", type=int, default=512)
    ap.add_argument("--ref-mode", choices=["fixed", "background"], default="background",
                    help="fixed: single reference particle; background: average over "
                         "all real particle types present (interventional Shapley)")
    ap.add_argument("--ref-pdg", type=int, default=22,
                    help="reference particle for --ref-mode fixed (22=photon)")
    ap.add_argument("--budget", type=int, default=4096,
                    help="(events*coalitions) per forward for the Shapley pass")
    ap.add_argument("--mb-events", type=int, default=64, help="momentum chunk size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    out_prefix = args.out_prefix or os.path.join(args.run_dir, "attribution", "shap")
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    from particle_ids import GLOBAL_PDG_IDX, PARTICLE_FEATURE_NAMES
    idx2pdg = {v: k for k, v in GLOBAL_PDG_IDX.items()}
    prop_names = PARTICLE_FEATURE_NAMES

    exp = load_experiment(args.run_dir, args.ckpt, args.frame)
    fn = ForwardFn(exp)
    rng = np.random.default_rng(args.seed)
    dataset_names = list(exp.cfg.data.dataset)

    scale_fm = exp.particles_flat.std(axis=0).astype(np.float32)   # (4,)
    all_props = exp.property_matrix[exp.tokens_flat].astype(np.float32)
    scale_pr = all_props.std(axis=0).astype(np.float32)            # (8,)
    scale_pr[scale_pr == 0] = 1.0
    if args.ref_mode == "background":
        present = np.unique(exp.tokens_flat)
        present = present[present > 0]
        ref_set = exp.property_matrix[present].astype(np.float32)   # (R,8)
        ref_pdgs = [int(idx2pdg.get(int(i), 0)) for i in present]
        print("Shapley background (uniform over present types): "
              + ", ".join(PDG_NAMES.get(q, str(q)) for q in ref_pdgs))
    else:
        ref_set = exp.property_matrix[[GLOBAL_PDG_IDX[args.ref_pdg]]].astype(np.float32)
        print(f"Shapley reference particle: "
              f"{PDG_NAMES.get(args.ref_pdg, str(args.ref_pdg))} (PDG {args.ref_pdg})")

    # On-manifold property contrasts: real-particle pairs differing in ~one property.
    def _pp(pdg):
        return exp.property_matrix[GLOBAL_PDG_IDX[pdg]].astype(np.float32)
    contrast_spec = [
        ("gamma-Z: mass(boson)", 22, 23), ("u-t: mass(quark)", 2, 6),
        ("gamma-g: color", 22, 21), ("Z-W: EW charge", 23, 24),
        ("e--e+: charge sign", 11, -11), ("u-ubar: charge sign", 2, -2),
        ("W+-W-: charge sign", 24, -24),
    ]
    contrasts = [(nm, _pp(a), _pp(b)) for nm, a, b in contrast_spec]

    n_proc = int(exp.all_process_ids.max()) + 1
    per_proc = {}
    pdg_imp = {}     # pdg -> list of per-slot mean|phi| across processes
    prop_accum = []  # list of (n*P,8) sigma-scaled |dF/dprop| for global pooling

    for p in range(n_proc):
        d = gather_process_events(exp, p, args.n_per_process, rng)
        if d is None:
            continue
        name = dataset_names[p] if p < len(dataset_names) else f"proc{p}"
        phi, vfull, vempty = exact_particle_shapley(
            fn, d["fm"], d["props"], d["order"], ref_set, exp.device, args.budget)
        sfm, spr, preds = local_sensitivity(
            fn, d["fm"], d["props"], d["order"], scale_fm, scale_pr,
            exp.device, args.mb_events)
        contrast_vals = property_contrasts(
            fn, d["fm"], d["props"], d["order"], contrasts, exp.device, args.mb_events)

        mse = float(np.mean((preds - d["y"]) ** 2))
        comp_err = float(np.mean(np.abs(phi.sum(1) - (vfull - vempty))))
        shap_abs = np.abs(phi).mean(axis=0)             # (P,) magnitude
        shap_signed = phi.mean(axis=0)                  # (P,) direction
        mom_per_slot = sfm.sum(axis=2).mean(axis=0)     # (P,)
        prop_per_dim = spr.mean(axis=(0, 1))            # (8,) physical-property importance
        slot_pdg = [int(idx2pdg.get(int(t), 0)) for t in d["tokens"]]

        # group-level summary, both in predicted-log-amplitude units:
        #  identity = mean|F(real) - F(all-reference)| = mean|sum_p phi_p|
        #  kinematic = mean|F - mean_e F| (within a process only momenta vary)
        ident_imp = float(np.mean(np.abs(vfull - vempty)))
        kin_imp = float(np.mean(np.abs(vfull - vfull.mean())))

        per_proc[name] = dict(
            P=d["P"], n=d["n"], mse=mse, comp_err=comp_err,
            shap_abs=shap_abs, shap_signed=shap_signed,
            mom_per_slot=mom_per_slot, prop_per_dim=prop_per_dim, slot_pdg=slot_pdg,
            ident_imp=ident_imp, kin_imp=kin_imp, contrasts=contrast_vals,
        )
        for pdg, val in zip(slot_pdg, shap_abs):
            pdg_imp.setdefault(pdg, []).append(float(val))
        prop_accum.append(spr.reshape(-1, 8))

        labels = ", ".join(f"{PDG_NAMES.get(q, str(q))}={shap_abs[i]:.2e}"
                           for i, q in enumerate(slot_pdg))
        print(f"[{name}] P={d['P']} n={d['n']} pred-MSE={mse:.2e} "
              f"shapley-completeness={comp_err:.2e}")
        print(f"    identity Shapley |phi| per slot: {labels}")

    # global importance by particle type (mean over all slots/processes of that pdg)
    global_pdg_imp = {PDG_NAMES.get(k, str(k)): float(np.mean(v))
                      for k, v in pdg_imp.items()}

    global_prop_imp = np.concatenate(prop_accum, axis=0).mean(axis=0)   # (8,)

    print("\n=== GLOBAL particle-identity importance (Shapley, by particle type) ===")
    for nm in sorted(global_pdg_imp, key=global_pdg_imp.get, reverse=True):
        print(f"  {nm:>6s} : {global_pdg_imp[nm]:.4e}")

    print("\n=== GLOBAL physical-property importance (|dF/dprop|*sigma, local proxy) ===")
    for i in np.argsort(global_prop_imp)[::-1]:
        print(f"  {prop_names[i]:>16s} : {global_prop_imp[i]:.4e}")

    # property contrasts: weighted mean over processes (by event count)
    cnames = [nm for nm, _, _ in contrasts]
    wts = np.array([per_proc[n]["n"] for n in per_proc])
    global_contrast = {c: float(np.average([per_proc[n]["contrasts"][c]
                                            for n in per_proc], weights=wts))
                       for c in cnames}
    print("\n=== PROPERTY CONTRASTS (on-manifold real-particle swaps, mean|ΔF|) ===")
    for c in sorted(global_contrast, key=global_contrast.get, reverse=True):
        print(f"  {c:<24s} : {global_contrast[c]:.4e}")

    print("\n=== kinematics vs identity (mean |ΔF|, log-amplitude units) ===")
    print(f"  {'process':<8}{'kinematic':>11}{'identity':>10}{'kin/id':>8}")
    ks, ids = [], []
    for nm, dd in per_proc.items():
        k, i = dd["kin_imp"], dd["ident_imp"]
        ks.append(k); ids.append(i)
        print(f"  {nm.split('_')[1]:<8}{k:>11.3f}{i:>10.3f}{k / max(i, 1e-9):>8.2f}")
    print(f"  {'MEAN':<8}{np.mean(ks):>11.3f}{np.mean(ids):>10.3f}"
          f"{np.mean(ks) / max(np.mean(ids), 1e-9):>8.2f}")

    np.savez(
        f"{out_prefix}_results.npz",
        ref_pdg=args.ref_pdg,
        global_pdg=np.array(list(global_pdg_imp.keys())),
        global_imp=np.array(list(global_pdg_imp.values())),
        prop_names=np.array(prop_names),
        global_prop_imp=global_prop_imp,
        **{f"{n}__shap_abs": d["shap_abs"] for n, d in per_proc.items()},
        **{f"{n}__shap_signed": d["shap_signed"] for n, d in per_proc.items()},
        **{f"{n}__mom_slot": d["mom_per_slot"] for n, d in per_proc.items()},
        **{f"{n}__prop_per_dim": d["prop_per_dim"] for n, d in per_proc.items()},
        **{f"{n}__slot_pdg": np.array(d["slot_pdg"]) for n, d in per_proc.items()},
        **{f"{n}__mse": d["mse"] for n, d in per_proc.items()},
        **{f"{n}__kin_imp": d["kin_imp"] for n, d in per_proc.items()},
        **{f"{n}__ident_imp": d["ident_imp"] for n, d in per_proc.items()},
        contrast_names=np.array(cnames),
        contrast_global=np.array([global_contrast[c] for c in cnames]),
    )
    make_plots(global_pdg_imp, per_proc, out_prefix)
    make_property_plot(global_prop_imp, prop_names, per_proc, out_prefix)
    make_contrast_plot(global_contrast, out_prefix)
    make_kin_vs_id_plot(per_proc, out_prefix)

    print("\n=== OUTPUT FILES (fixed set) ===")
    for suffix, desc in [
        ("identity_importance.pdf", "Shapley importance by particle type"),
        ("property_contrasts.pdf",  "property importance via real-particle contrasts (primary)"),
        ("property_importance.pdf", "gradient sensitivity by property (crude local proxy)"),
        ("kin_vs_identity.pdf",     "kinematics vs identity, per process"),
        ("per_process.pdf",         "per-process identity Shapley + momentum"),
        ("results.npz",             "all numbers"),
    ]:
        print(f"  {out_prefix}_{suffix}  —  {desc}")


if __name__ == "__main__":
    main()
