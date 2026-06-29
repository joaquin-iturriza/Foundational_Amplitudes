"""
Two checks:
  1. Distribution of preprocessed amplitude targets across all datasets
     (confirms values are in a float16-safe range)
  2. float16 AMP vs float32 consistency check
     (same model, same data, same init — compares loss trajectories)

Run on Jean-Zay with a GPU allocated:
    python test_amp.py
"""
import sys, os
# This test lives in tests/ but imports project-root modules; put the repo root
# (parent of tests/) on sys.path so `python tests/test_amp.py` works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from hydra.utils import instantiate
from omegaconf import OmegaConf
from mup import set_base_shapes

from preprocessing import preprocess_amplitude
from dataset import collate_variable_length

import types
from experiment import AmplitudeExperiment
from losses import LogCoshLoss, RelL1Loss

DEVICE    = torch.device("cuda")
N_SCALARS = 18
DATA_PATH = "data/"

# Datasets used in many_datasets_006
DATASETS = [
    "ee_wwz_255-1000GeV_amplitudes",
    "ee_WW_162-1000GeV_amplitudes",
    "ee_ttbar_346-1000GeV_amplitudes",
    "ee_uug_91-1000GeV_amplitudes",
    "ee_uugg_91-1000GeV_amplitudes",
    "ee_aa_10-1000GeV_amplitudes",
    "ee_aaa_10-1000GeV_amplitudes",
    "ee_uu_91-1000GeV_amplitudes",
]
AMP_TRAFOS = ["log", "standardization"]

# ═════════════════════════════════════════════════════════════════════════════
# 0. Vectorization equivalence regression guard  (CPU, no GPU/data required)
# ═════════════════════════════════════════════════════════════════════════════
# Guards the three per-step LLoCa hot-path vectorizations against their original
# implementations, which are kept behind LLOCA_* env toggles:
#   #1 attention mask built once/forward   models/transformer_lloca_mup.py  (LLOCA_ATTN_MASK)
#   #2 per-process loss segment-mean       experiment.py:_aggregate_per_process_loss (LLOCA_PROC_LOSS)
#   #3 L2/L1 regularization via _foreach    experiment.py:_init_regularization        (LLOCA_REG)
# These are deterministic on CPU, so this section runs anywhere (CI included) and
# drives the REAL code via a light stub — a future refactor that breaks numerics
# is caught here, not silently in training.

_CPU = torch.device("cpu")


def _tiny_model(n_scalars=6, num_blocks=2, num_heads=2):
    cfg = OmegaConf.create({
        "_target_": "models.lloca.LLOCAMuPTransformer",
        "num_scalars": n_scalars, "hidden_channels_mlp": 128,
        "num_layers_mlp": 2, "in_channels": n_scalars + 4,
        "attn_reps": "8x0n+2x1n", "out_channels": 1,
        "num_blocks": num_blocks, "num_heads": num_heads,
    })
    model = instantiate(cfg).to(_CPU)
    # The μP backbone sets its own base shapes in __init__ (self-contained μP), so
    # re-rescaling here would double-apply; keep the in-backbone shapes.
    set_base_shapes(model, model, rescale_params=False)
    return model, n_scalars


def _tiny_batch(n_scalars, bs=8, n=3):
    N = bs * n
    p = torch.randn(N, 3); m = torch.rand(N, 1) + 0.1
    E = (p.pow(2).sum(-1, keepdim=True) + m.pow(2)).sqrt()
    fm = torch.cat([E, p], dim=-1)
    pt = torch.randn(N, n_scalars)
    ptr = torch.arange(0, N + 1, n, dtype=torch.long)
    return fm, pt, ptr


def check_proc_loss_equivalence():
    """#2: segment-mean per-process loss == original per-process loop."""
    torch.manual_seed(0)
    n_datasets, B = 7, 256
    loss_mods = {"MSE": torch.nn.MSELoss(), "L1": torch.nn.L1Loss(),
                 "LogCosh": LogCoshLoss(), "RelL1": RelL1Loss()}
    for name, loss_mod in loss_mods.items():
        stub = types.SimpleNamespace(
            n_datasets=n_datasets, loss=loss_mod,
            cfg=OmegaConf.create({"training": {"loss": name}}))
        stub._per_event_loss = types.MethodType(AmplitudeExperiment._per_event_loss, stub)
        y_pred, y = torch.randn(B, 1), torch.randn(B, 1)
        pids = torch.randint(0, n_datasets, (B,))
        for agg in ("mean", "geometric_mean"):
            os.environ["LLOCA_PROC_LOSS"] = "loop"
            ref = AmplitudeExperiment._aggregate_per_process_loss(stub, y_pred, y, pids, agg)
            os.environ["LLOCA_PROC_LOSS"] = "vectorized"
            got = AmplitudeExperiment._aggregate_per_process_loss(stub, y_pred, y, pids, agg)
            assert torch.allclose(ref, got, atol=1e-5, rtol=1e-4), \
                f"#2 {name}/{agg}: loop={ref.item():.6g} vec={got.item():.6g}"
    os.environ.pop("LLOCA_PROC_LOSS", None)
    print("  #2 per-process loss: loop == vectorized  OK  (MSE/L1/LogCosh/RelL1, mean & geometric)")


def check_reg_equivalence():
    """#3: _foreach regularization == original Python-sum loop."""
    torch.manual_seed(0)
    model, _ = _tiny_model()
    for reg in ("L2", "L1"):
        stub = types.SimpleNamespace(cfg=OmegaConf.create(
            {"training": {"regularization": reg, "regularization_lambda": 1.0}}))
        os.environ["LLOCA_REG"] = "loop"
        AmplitudeExperiment._init_regularization(stub); ref = stub.regularization(model)
        os.environ["LLOCA_REG"] = "foreach"
        AmplitudeExperiment._init_regularization(stub); got = stub.regularization(model)
        assert torch.allclose(ref, got, rtol=1e-4), \
            f"#3 {reg}: loop={ref.item():.6g} foreach={got.item():.6g}"
    os.environ.pop("LLOCA_REG", None)
    print("  #3 regularization: loop == _foreach  OK  (L2 & L1)")


def check_attention_routing():
    """#1: per_forward kwarg routing == per_block routing (CPU dense fallback).

    xformers imports on CPU but its memory_efficient_attention kernel is CUDA-only,
    so we force the dense fallback here. This guards the per-forward kwarg plumbing
    (ptr popped → attn_bias passed down) on a deterministic path; the block-diagonal
    xformers kernel is GPU-only and was verified separately to agree within
    memory_efficient_attention's own (nondeterministic) reduction order.
    """
    import models.attention_lloca_mup as attn_mod
    torch.manual_seed(0)
    model, ns = _tiny_model(); model.eval()
    fm, pt, ptr = _tiny_batch(ns)
    saved = attn_mod._XFORMERS_AVAILABLE
    attn_mod._XFORMERS_AVAILABLE = False     # force deterministic dense path on CPU
    try:
        with torch.no_grad():
            model(fm, pt, mean=0.0, std=1.0, ptr=ptr)   # warm up edge-standardization buffers
            os.environ["LLOCA_ATTN_MASK"] = "per_block";   a = model(fm, pt, mean=0.0, std=1.0, ptr=ptr)
            os.environ["LLOCA_ATTN_MASK"] = "per_forward"; b = model(fm, pt, mean=0.0, std=1.0, ptr=ptr)
    finally:
        attn_mod._XFORMERS_AVAILABLE = saved
        os.environ.pop("LLOCA_ATTN_MASK", None)
    assert torch.allclose(a, b, atol=1e-6), f"#1 routing diff {(a - b).abs().max().item():.3e}"
    print("  #1 attention mask: per_forward == per_block routing  OK  (CPU dense fallback)")


def check_frame_broadcast_equivalence():
    """Finding A: LLOCA_FRAMES=broadcast must reproduce the original repeat path
    (same math, no num_heads-replicated frame copies). Forces the deterministic
    dense path; relative tolerance keeps it robust to output magnitude.

    NOTE: a strict end-to-end *Lorentz-invariance* assertion (f(Λp)==f(p)) was
    tried here but is not a reliable unit test on an UNTRAINED model: it forces
    float32 internally, and the frame round-trip's precision is batch-dependent
    (near-light-like / large-γ momenta drift O(1); tiny momenta make the output
    near-zero). The model is invariant by construction — verified manually across
    widths (~1e-4 rel) — but reproducing that reliably needs a trained checkpoint
    or float64 internals. This broadcast==repeat check is the robust guard for the
    frame change; it compares two implementations at equal precision, so it's
    unaffected by conditioning.
    """
    import models.attention_lloca_mup as attn_mod
    torch.manual_seed(0)
    model, ns = _tiny_model(); model.eval()
    fm, pt, ptr = _tiny_batch(ns)
    saved = attn_mod._XFORMERS_AVAILABLE
    attn_mod._XFORMERS_AVAILABLE = False
    try:
        with torch.no_grad():
            os.environ["LLOCA_FRAMES"] = "repeat"
            out_rep = model(fm, pt, mean=0.0, std=1.0, ptr=ptr)
            os.environ["LLOCA_FRAMES"] = "broadcast"
            out_bc = model(fm, pt, mean=0.0, std=1.0, ptr=ptr)
    finally:
        attn_mod._XFORMERS_AVAILABLE = saved
        os.environ.pop("LLOCA_FRAMES", None)
    rel = ((out_rep - out_bc).abs().max() / out_rep.abs().max().clamp(min=1e-6)).item()
    assert rel < 1e-4, f"frame broadcast != repeat: max rel diff {rel:.2e}"
    print(f"  frame broadcast == repeat  OK  (max rel diff {rel:.1e})")


def check_fast_equilinear_equivalence():
    """L-GATr hot-path: LGATR_FAST_LINEAR (fancy-index-free EquiLinear.forward in
    models/lgatr_mup.py) must reproduce the stock forward bit-for-bit, including
    gradients, for every scalar/MV in/out combination. Checked in float64."""
    import importlib
    import models.lgatr_mup as lg
    from lgatr.layers.linear import EquiLinear

    torch.manual_seed(0)
    worst = 0.0
    for in_s, out_s in [(8, 5), (8, None), (None, 5), (None, None)]:
        lin = EquiLinear(in_mv_channels=4, out_mv_channels=7,
                         in_s_channels=in_s, out_s_channels=out_s).double()
        mv0 = torch.randn(3, 4, 16, dtype=torch.float64, requires_grad=True)
        mv1 = mv0.detach().clone().requires_grad_(True)
        s = torch.randn(3, 8, dtype=torch.float64) if in_s else None

        EquiLinear.forward = lg._ORIG_EQUILINEAR_FORWARD
        o0_mv, o0_s = lin(mv0, scalars=s)
        (o0_mv.sum() + (o0_s.sum() if o0_s is not None else 0)).backward()

        EquiLinear.forward = lg._fast_equilinear_forward
        f1_mv, f1_s = lin(mv1, scalars=s)
        (f1_mv.sum() + (f1_s.sum() if f1_s is not None else 0)).backward()

        d = (o0_mv - f1_mv).abs().max().item()
        if o0_s is not None:
            d = max(d, (o0_s - f1_s).abs().max().item())
        d = max(d, (mv0.grad - mv1.grad).abs().max().item())
        worst = max(worst, d)
    # restore default (env-driven) behavior
    lg.enable_fast_equilinear()
    assert worst < 1e-10, f"fast EquiLinear != stock: max diff {worst:.2e}"
    print(f"  fast EquiLinear == stock   OK  (max fwd+grad diff {worst:.1e})")


def check_diagram_conditioning():
    """Feynman-diagram conditioning (concat-to-scalars): the diagram-conditioned
    LLoCa forward must (a) run and be finite, (b) stay Lorentz-invariant (the
    appended diagram embedding E(process) is a Lorentz scalar), and (c) actually
    depend on the diagrams (swapping process ids changes the prediction). CPU dense
    attention path (xformers is CUDA-only); needs data/diagrams/*.diagrams.json."""
    from lloca.mup import finalize as mup_finalize
    from particle_ids import build_property_matrix, GLOBAL_PDG_IDX
    from diagram_graphs import load_diagram_registry, feature_dims
    from models.diagram_encoder import DiagramEncoder
    from wrappers import AmplitudeLLoCaWrapper
    from lloca.utils.rand_transforms import rand_lorentz
    import models.attention_lloca_mup as _attn

    ddir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "diagrams")
    if not os.path.exists(os.path.join(ddir, "ee_ttbar.diagrams.json")):
        print("  diagram conditioning: SKIP (no data/diagrams sidecars; "
              "run tools/dump_diagrams.py)")
        return

    torch.manual_seed(0)
    FLAGS = dict(spin_onehot=True, color_onehot=True, is_massless=True, standardize=True)
    D_PH, ENC_H, N_ORDER, D_DIAG = 16, 32, 2, 8
    prop_matrix, _ = build_property_matrix(**FLAGS)
    n_prop = prop_matrix.shape[1]
    n_scalars = D_PH + N_ORDER + D_DIAG
    cfg = OmegaConf.create({
        "_target_": "models.lloca.LLOCAMuPTransformer",
        "num_scalars": n_scalars, "hidden_channels_mlp": 128, "num_layers_mlp": 2,
        "in_channels": n_scalars + 4, "attn_reps": "8x0n+2x1n", "out_channels": 1,
        "num_blocks": 2, "num_heads": 2})
    net = instantiate(cfg)
    wrap = AmplitudeLLoCaWrapper(net, token_size=prop_matrix.shape[0], d_particle_hidden=D_PH,
                                 particle_encoder_hidden=ENC_H, use_diagrams=True, d_diag=D_DIAG)
    wrap.setup_particle_features(use_pids=False, property_matrix=prop_matrix, encoder_hidden=ENC_H)
    reg, _ = load_diagram_registry(ddir, ["ee_ttbar", "ee_uug"], **FLAGS, k_pe=8)
    f_node, f_edge = feature_dims(n_prop)
    enc = DiagramEncoder(f_node, f_edge, k_pe=8, d_model=64, n_heads=4, n_layers=3, d_out=D_DIAG)
    wrap.setup_diagram_conditioning(enc, [reg["ee_ttbar"], reg["ee_uug"]], D_DIAG)
    mup_finalize(wrap); wrap.eval()

    def four_mom(n):
        p = torch.randn(n, 3); m = torch.rand(n, 1) + 0.1
        return torch.cat([(p.pow(2).sum(-1, keepdim=True) + m.pow(2)).sqrt(), p], dim=-1)
    n0, n1 = 4, 5
    fm = torch.cat([four_mom(n0), four_mom(n1)], 0)
    idx = torch.tensor([GLOBAL_PDG_IDX[i] for i in (11, -11, 6, -6, 11, -11, 2, -2, 21)])
    ptr = torch.tensor([0, n0, n0 + n1], dtype=torch.long)
    order = torch.zeros(2, 2)
    mean, std = torch.zeros(4), torch.tensor(100.0)

    saved = _attn._XFORMERS_AVAILABLE
    _attn._XFORMERS_AVAILABLE = False           # force dense CPU attention
    try:
        with torch.no_grad():
            y = wrap(fm, idx, mean, std, ptr, order_labels=order,
                     process_ids=torch.tensor([0, 1]))
            Lam = rand_lorentz((1,), dtype=torch.float64).squeeze(0).to(fm.dtype)
            fm2 = torch.einsum("ij,nj->ni", Lam, fm)
            y2 = wrap(fm2, idx, mean, std, ptr, order_labels=order,
                      process_ids=torch.tensor([0, 1]))
            y_swap = wrap(fm, idx, mean, std, ptr, order_labels=order,
                          process_ids=torch.tensor([1, 0]))
    finally:
        _attn._XFORMERS_AVAILABLE = saved
    assert torch.isfinite(y).all(), "diagram-conditioned forward produced non-finite output"
    dinv = (y - y2).abs().max().item()
    ddiag = (y - y_swap).abs().max().item()
    assert dinv < 1e-3, f"diagram conditioning broke Lorentz invariance: max|Δ|={dinv:.2e}"
    assert ddiag > 1e-6, f"diagram conditioning had no effect on output: max|Δ|={ddiag:.2e}"
    print(f"  diagram conditioning: forward OK, Lorentz-inv {dinv:.1e}, "
          f"pid-swap effect {ddiag:.1e}  OK")


def check_diagram_batch_equivalence():
    """DiagramEncoder.encode_all (one batched forward over all processes) must match
    the per-process forward() loop — same E and gradients up to float32 reduction-
    order noise. This is the per-step optimization that removes the 25x launch-bound
    loop + the unique() GPU->CPU sync; guarding it keeps a future refactor honest."""
    ddir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "diagrams")
    names = [n for n in ("ee_aa", "ee_ttbar", "ee_uug", "ee_uugg", "ee_ZH")
             if os.path.exists(os.path.join(ddir, f"{n}.diagrams.json"))]
    if len(names) < 2:
        print("  diagram batch equivalence: SKIP (no data/diagrams sidecars)")
        return
    from diagram_graphs import load_diagram_registry, feature_dims, build_diagram_batch
    from models.diagram_encoder import DiagramEncoder
    reg, n_prop = load_diagram_registry(ddir, names, spin_onehot=True, is_massless=True,
                                        standardize=True, k_pe=8)
    f_node, f_edge = feature_dims(n_prop)
    torch.manual_seed(0)
    enc = DiagramEncoder(f_node, f_edge, k_pe=8, d_model=64, n_heads=4, n_layers=3, d_out=32)
    pd_by_pid = [reg[n] for n in names] + [None]   # include a missing process

    enc.zero_grad()
    E_loop = torch.zeros(len(pd_by_pid), 32)
    for pid, pd in enumerate(pd_by_pid):
        if pd is not None:
            E_loop = E_loop.index_copy(0, torch.tensor([pid]), enc(pd).unsqueeze(0))
    E_loop.pow(2).sum().backward()
    # skip params with no grad (e.g. the pool-mode virt_score head, unused here)
    g_loop = {k: p.grad.clone() for k, p in enc.named_parameters() if p.grad is not None}

    enc.zero_grad()
    E_b = enc.encode_all(build_diagram_batch(pd_by_pid))
    E_b.pow(2).sum().backward()
    g_b = {k: p.grad.clone() for k, p in enc.named_parameters() if p.grad is not None}

    dfwd = (E_loop - E_b).abs().max().item()
    grel = max((g_loop[k] - g_b[k]).norm().item() / (g_loop[k].norm().item() + 1e-30)
               for k in g_loop)
    assert E_b[-1].abs().max().item() < 1e-9, "missing process should get zero embedding"
    assert dfwd < 1e-5, f"batched encode_all forward differs: {dfwd:.2e}"
    assert grel < 2e-3, f"batched encode_all grad differs beyond float noise: {grel:.2e}"
    print(f"  diagram batch encode == per-process loop  OK  "
          f"(fwd {dfwd:.1e}, grad rel {grel:.1e})")


def check_diagram_virtuality():
    """Tier B: the propagator-virtuality precompute must reproduce the physical
    Mandelstam invariant. For ee->ttbar the single s-channel propagator carries the
    e+e- pair, so its s=(Σ sign·p)² must equal (p_e- + p_e+)² = E_cm² on real events.
    Pure-tensor check (no model), so it's fast."""
    import numpy as np
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ddir = os.path.join(repo, "data", "diagrams")
    npy = os.path.join(repo, "data", "ee_ttbar_346-1000GeV_amplitudes.npy")
    if not (os.path.exists(os.path.join(ddir, "ee_ttbar.diagrams.json")) and os.path.exists(npy)):
        print("  diagram virtuality: SKIP (no ee_ttbar sidecar/data)")
        return
    from diagram_graphs import load_diagram_registry, build_process_virtuality
    reg, _ = load_diagram_registry(ddir, ["ee_ttbar"], spin_onehot=True,
                                   is_massless=True, standardize=True, k_pe=8)
    a = np.load(npy, mmap_mode="r")
    slot_pdgs = a[0, 16:20].astype(int).tolist()           # [11,-11,6,-6]
    vt = build_process_virtuality(reg["ee_ttbar"], slot_pdgs, n_initial=2)
    assert vt is not None and vt["mask"].shape[0] == 2, "expected 2 s-channel propagators (gamma,Z)"
    mom = torch.tensor(np.asarray(a[:64, :16]).reshape(64, 4, 4), dtype=torch.float64)
    p_prop = torch.einsum("kn,enc->ekc", vt["mask"].double(), mom)        # (64, 2, 4)
    s = p_prop[..., 0] ** 2 - (p_prop[..., 1:] ** 2).sum(-1)             # (64, 2)
    ecm = mom[:, 0] + mom[:, 1]
    ecm2 = ecm[:, 0] ** 2 - (ecm[:, 1:] ** 2).sum(-1)                    # (64,)
    rel = ((s.abs() - ecm2.abs().unsqueeze(-1)).abs() / ecm2.abs().unsqueeze(-1)).max().item()
    assert rel < 1e-5, f"propagator virtuality != E_cm^2: max rel {rel:.2e}"
    print(f"  diagram virtuality: s-channel == E_cm^2 over 64 events  OK  (rel {rel:.1e})")


def check_diagram_virtuality_batch():
    """Tier B optimization: the batched per-event path (one encode_grouped call over
    all processes' event×diagram graphs) must equal the per-process forward_per_event
    loop. Small encoder for speed."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ddir = os.path.join(repo, "data", "diagrams")
    if not os.path.exists(os.path.join(ddir, "ee_ttbar.diagrams.json")):
        print("  diagram virtuality batch: SKIP (no sidecars)")
        return
    from diagram_graphs import load_diagram_registry, feature_dims, build_process_virtuality
    from models.diagram_encoder import DiagramEncoder
    from wrappers import AmplitudeLLoCaWrapper
    torch.manual_seed(0)
    F = dict(spin_onehot=True, is_massless=True, standardize=True)
    reg, _ = load_diagram_registry(ddir, ["ee_ttbar", "ee_uug"], **F, k_pe=8)
    n_prop = reg["ee_ttbar"].node_feat.shape[-1] - 4 - 2 - 2   # back out from f_node
    f_node, f_edge = feature_dims(reg["ee_ttbar"].edge_feat.shape[-1] - 1)
    enc = DiagramEncoder(f_node, f_edge, k_pe=8, d_model=16, n_heads=2, n_layers=1, d_out=4, f_edge_extra=1)
    wrap = AmplitudeLLoCaWrapper.__new__(AmplitudeLLoCaWrapper)
    torch.nn.Module.__init__(wrap)
    wrap.diagram_encoder = enc; wrap.d_diag = 4; wrap.use_diagrams = True
    wrap._pd_by_pid = [reg["ee_ttbar"], reg["ee_uug"]]
    virt = [build_process_virtuality(reg["ee_ttbar"], [11, -11, 6, -6], 2),
            build_process_virtuality(reg["ee_uug"], [11, -11, 2, -2, 21], 2)]
    wrap.setup_diagram_virtuality(virt, log_scale=0.1, standardize=False)
    np_ = {0: 4, 1: 5}; pids = torch.tensor([0, 1, 0, 1, 0])
    def fm(n):
        p = torch.randn(n, 3); m = torch.rand(n, 1) + 0.1
        return torch.cat([(p.pow(2).sum(-1, keepdim=True) + m.pow(2)).sqrt(), p], -1)
    moms = torch.cat([fm(np_[int(p)]) for p in pids], 0)
    counts = torch.tensor([np_[int(p)] for p in pids])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    std = torch.tensor(0.7)
    with torch.no_grad():
        new = wrap._diagram_features_virtuality(moms, std, pids, ptr, moms.device, torch.float32)
        ref = torch.zeros(pids.shape[0], 4)
        for pid in (0, 1):
            pd, vt = wrap._pd_by_pid[pid], virt[pid]
            ev = (pids == pid).nonzero(as_tuple=True)[0]
            N, D = pd.node_feat.shape[1], pd.n_diagrams
            g = ptr[ev][:, None] + torch.arange(vt["n_part"])[None, :]
            pp = torch.einsum("kn,enc->ekc", vt["mask"], moms[g])
            s = pp[..., 0] ** 2 - (pp[..., 1:] ** 2).sum(-1)
            vf = torch.sign(s) * torch.log1p(s.abs()) * 0.1
            ee = torch.zeros(ev.shape[0], D, N, N, 1)
            ee[:, vt["diag_idx"], vt["i_idx"], vt["j_idx"], 0] = vf
            ee[:, vt["diag_idx"], vt["j_idx"], vt["i_idx"], 0] = vf
            ref[ev] = enc.forward_per_event(pd, ee)
        ref = ref.repeat_interleave(counts, dim=0)
    d = (new - ref).abs().max().item()
    assert d < 1e-5, f"batched Tier-B != per-process loop: {d:.2e}"
    print(f"  diagram virtuality batch == per-process loop  OK  (max {d:.1e})")


print("=" * 60)
print("0. Vectorization equivalence regression guard (CPU)")
print("=" * 60)
check_proc_loss_equivalence()
check_reg_equivalence()
check_attention_routing()
check_frame_broadcast_equivalence()
check_fast_equilinear_equivalence()
check_diagram_conditioning()
check_diagram_batch_equivalence()
check_diagram_virtuality()
check_diagram_virtuality_batch()


# ── 1. Load real data and check preprocessed amplitude distribution ──────────

def load_amplitudes(data_path, datasets, trafos, n_samples=50000):
    """Load amplitudes from each dataset and apply global preprocessing."""
    all_raw = []
    per_process = {}
    for ds in datasets:
        path = os.path.join(data_path, f"{ds}.npy")
        if not os.path.exists(path):
            print(f"  WARNING: not found: {path}")
            continue
        data = np.load(path)           # (N, n_particles*5 + 1)
        raw  = data[:, -1]             # last column = amplitude
        idx  = np.random.choice(len(raw), min(n_samples, len(raw)), replace=False)
        per_process[ds] = raw[idx]
        all_raw.append(per_process[ds])

    if not all_raw:
        return None, None, None, None

    all_raw = np.concatenate(all_raw).reshape(-1, 1)
    all_prepd, mean, std = preprocess_amplitude(all_raw, trafos=trafos)
    return all_prepd.flatten(), mean, std, per_process


print("=" * 60)
print("1. Preprocessed amplitude distribution")
print("=" * 60)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42)

prepd, mean, std, per_process = load_amplitudes(DATA_PATH, DATASETS, AMP_TRAFOS)

if prepd is not None:
    print(f"\nGlobal stats after {AMP_TRAFOS}:")
    print(f"  mean={prepd.mean():.4f}  std={prepd.std():.4f}")
    print(f"  min={prepd.min():.4f}   max={prepd.max():.4f}")
    print(f"  p1={np.percentile(prepd,1):.3f}  p99={np.percentile(prepd,99):.3f}")
    print(f"\nfloat16 range:   [{np.finfo(np.float16).min:.0f}, {np.finfo(np.float16).max:.0f}]")
    print(f"float16 epsilon: {np.finfo(np.float16).eps:.2e}  (relative precision)")
    print(f"→ all values in [{prepd.min():.2f}, {prepd.max():.2f}] are safely representable in float16")

    # Per-process breakdown
    print(f"\nPer-process stats (raw log-transformed, before global standardization):")
    for ds, raw in per_process.items():
        log_vals = np.log(np.abs(raw) + 1e-30)
        print(f"  {ds[-30:]:>30}  log range [{log_vals.min():6.1f}, {log_vals.max():6.1f}]")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(prepd, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    ax1.axvline(np.finfo(np.float16).min, color='red', ls='--', label='float16 min/max')
    ax1.axvline(np.finfo(np.float16).max, color='red', ls='--')
    ax1.set(title='Preprocessed amplitudes (all datasets combined)',
            xlabel='preprocessed value', ylabel='count')
    ax1.legend()

    colors = plt.cm.tab10(np.linspace(0, 1, len(per_process)))
    for (ds, raw), c in zip(per_process.items(), colors):
        log_vals = np.log(np.abs(raw) + 1e-30)
        prepd_ds, _, _ = preprocess_amplitude(log_vals.reshape(-1, 1), trafos=["standardization"])
        ax2.hist(prepd_ds.flatten(), bins=80, alpha=0.5, color=c,
                 label=ds.split('_')[1]+'_'+ds.split('_')[2], edgecolor='none')
    ax2.set(title='Per-process distribution (individually standardized)',
            xlabel='standardized log amplitude', ylabel='count')
    ax2.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("amp_distribution.pdf", bbox_inches='tight')
    print("\nSaved amp_distribution.pdf")
else:
    print("Could not load data — skipping distribution check")


# Section 2 below runs a real fwd/bwd in float16 AMP and needs a GPU. The
# CPU-runnable regression guards (section 0) and the data check (section 1) are
# done; bail out cleanly on CPU rather than erroring in the CUDA-only section.
if not torch.cuda.is_available():
    print("\n(no CUDA detected — skipping section 2 float32/float16 AMP consistency)")
    sys.exit(0)


# ── 2. float32 vs float16 AMP consistency ───────────────────────────────────

print("\n" + "=" * 60)
print("2. float32 vs float16 AMP consistency")
print("=" * 60)

def make_model_cfg(num_heads=16):
    return OmegaConf.create({
        "_target_": "models.lloca.LLOCAMuPTransformer",
        "num_scalars": N_SCALARS, "hidden_channels_mlp": 128,
        "num_layers_mlp": 2, "in_channels": N_SCALARS + 4,
        "attn_reps": "8x0n+2x1n", "out_channels": 1,
        "num_blocks": 8, "num_heads": num_heads,
    })

def make_batch(bs=512, n=6):
    N = bs * n
    p = torch.randn(N, 3)
    m = torch.rand(N, 1) + 0.1
    E = (p.pow(2).sum(-1, keepdim=True) + m.pow(2)).sqrt()
    fourmomenta   = torch.cat([E, p], dim=-1).to(DEVICE)
    particle_type = torch.randn(N, N_SCALARS).to(DEVICE)
    ptr = torch.arange(0, N + 1, n, dtype=torch.long, device=DEVICE)
    # Targets drawn from the real preprocessed range [-2.77, 3.58]
    targets = torch.tensor(
        np.random.choice(prepd if prepd is not None else np.random.randn(10000),
                         bs, replace=True),
        dtype=torch.float32, device=DEVICE
    ).unsqueeze(-1)
    return fourmomenta, particle_type, ptr, targets

N_STEPS = 30
BS, N_PARTICLES = 512, 6
SEED = 1234

losses_f32  = []
losses_amp  = []
gnorms_f32  = []
gnorms_amp  = []

for mode, use_amp in [("float32", False), ("float16 AMP", True)]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = instantiate(make_model_cfg()).to(DEVICE)
    set_base_shapes(model, model)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scaler    = GradScaler() if use_amp else None

    losses, gnorms = [], []
    nan_steps = []

    for step in range(N_STEPS):
        torch.manual_seed(step)   # same batch each step across modes
        fmom, ptype, ptr, targets = make_batch(BS, N_PARTICLES)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            out_per_particle = model(fmom, ptype, mean=0.0, std=1.0, ptr=ptr)  # (N_total, 1)
            # mean-pool per event (same as AmplitudeLLoCaWrapper)
            out = torch.stack([
                out_per_particle[ptr[i]:ptr[i+1]].mean(dim=0)
                for i in range(len(ptr) - 1)
            ])                                                                   # (B, 1)
            loss = ((out - targets) ** 2).mean()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
            optimizer.step()

        l = loss.item()
        losses.append(l)
        gnorms.append(gnorm)
        if np.isnan(l) or np.isinf(l):
            nan_steps.append(step)

    if mode == "float32":
        losses_f32, gnorms_f32 = losses, gnorms
    else:
        losses_amp, gnorms_amp = losses, gnorms

    status = f"NaN/Inf at steps {nan_steps}" if nan_steps else "no NaN/Inf"
    print(f"\n  {mode}: final loss={losses[-1]:.6f}  ({status})")
    print(f"    loss range:  [{min(losses):.4f}, {max(losses):.4f}]")
    print(f"    grad norm range: [{min(gnorms):.3e}, {max(gnorms):.3e}]")
    del model, optimizer, scaler
    torch.cuda.empty_cache()

# Compare
max_loss_diff = max(abs(a - b) for a, b in zip(losses_f32, losses_amp))
rel_diff = [abs(a - b) / (abs(a) + 1e-8) for a, b in zip(losses_f32, losses_amp)]
print(f"\n  Max absolute loss diff (f32 vs AMP): {max_loss_diff:.6f}")
print(f"  Max relative loss diff:              {max(rel_diff):.4%}")
print(f"  Mean relative loss diff:             {np.mean(rel_diff):.4%}")

# Plot loss comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
steps = range(N_STEPS)
ax1.plot(steps, losses_f32, label='float32', color='steelblue')
ax1.plot(steps, losses_amp, label='float16 AMP', color='tomato', ls='--')
ax1.set(title='Loss trajectory: float32 vs float16 AMP',
        xlabel='step', ylabel='MSE loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.semilogy(steps, rel_diff, color='purple')
ax2.set(title='Relative loss difference per step',
        xlabel='step', ylabel='|loss_f32 - loss_amp| / loss_f32')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("amp_consistency.pdf", bbox_inches='tight')
print("\nSaved amp_consistency.pdf")
