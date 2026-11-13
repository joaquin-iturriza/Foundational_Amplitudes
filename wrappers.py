import os

import math
import torch
from torch import nn
from lgatr.interface import embed_vector, extract_scalar

from models.attention_lloca_mup import build_block_diagonal_bias


def _property_matrix_prefix_hook(module):
    """load_state_dict pre-hook for the ``property_matrix`` buffer: a checkpoint written before non-SM
    rows were appended to particle_ids' table (D49) holds the first rows of the current matrix. If its
    rows equal the current first rows bit for bit, the current (longer) matrix is loaded in its place;
    anything else is left alone, so a genuinely different encoding still fails the strict load."""
    def hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        k = prefix + "property_matrix"
        own = getattr(module, "property_matrix", None)
        if k not in state_dict or own is None:
            return
        old = state_dict[k]
        if (old.dim() == 2 and old.shape[1] == own.shape[1] and old.shape[0] < own.shape[0]
                and torch.equal(old.to(device=own.device, dtype=own.dtype), own[: old.shape[0]])):
            state_dict[k] = own.detach().clone()
    return hook


def _build_particle_encoder(n_features: int, d_hidden: int, encoder_hidden: int = 0):
    """Project the physical property vector (n_features) → d_hidden.

    encoder_hidden == 0 → a single Linear (legacy: the embedding is a *linear*
    function of the quantum numbers, so only one-hot columns get independent
    directions). encoder_hidden > 0 → a 2-layer MLP (Linear→GELU→Linear), which
    lets the encoder form nonlinear *products* of quantum numbers (charged-AND-
    colored, fermion-AND-massive, …) a single Linear cannot. Negligible params
    vs the transformer. Marked standard-parametrization by mup_finalize either way.
    """
    if encoder_hidden and encoder_hidden > 0:
        enc = nn.Sequential(
            nn.Linear(n_features, encoder_hidden, bias=True),
            nn.GELU(),
            nn.Linear(encoder_hidden, d_hidden, bias=True),
        )
        for lin in (enc[0], enc[2]):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        return enc
    enc = nn.Linear(n_features, d_hidden, bias=True)
    nn.init.xavier_uniform_(enc.weight)
    nn.init.zeros_(enc.bias)
    return enc


def _block_diagonal_attn_mask(ptr, device):
    """Boolean (N_total, N_total) mask: True where two particles share an event.

    Built from the CPU ``ptr`` (event boundaries) so each event's particles attend
    only within that event. Passed as ``attn_mask`` to the (geometric) attention.
    """
    counts = (ptr[1:] - ptr[:-1]).tolist()
    blocks = [torch.ones(c, c, dtype=torch.bool, device=device) for c in counts]
    return torch.block_diag(*blocks)


def _segment_mean(values, ptr):
    """Mean of `values` over each event segment defined by `ptr`.

    Vectorised equivalent of
        torch.stack([values[ptr[i]:ptr[i+1]].mean(dim=0) for i in range(len(ptr)-1)])
    but using a single scatter (index_add_) instead of one slice+mean kernel per
    event — so its cost is independent of the number of events in the batch.

    values : (N_total, C)   per-particle outputs
    ptr    : (B+1,)         cumulative event boundaries (ptr[-1] == N_total)
    returns: (B, C)         per-event means
    """
    counts = ptr[1:] - ptr[:-1]                                  # (B,) particles per event
    B = counts.shape[0]
    seg = torch.repeat_interleave(
        torch.arange(B, device=values.device), counts
    )                                                            # (N_total,) event id per particle
    summed = values.new_zeros((B,) + tuple(values.shape[1:]))
    summed.index_add_(0, seg, values)
    denom = counts.clamp(min=1).view((B,) + (1,) * (values.dim() - 1)).to(values.dtype)
    return summed / denom


_ETA = (1.0, -1.0, -1.0, -1.0)


def _pair_log_inputs(p, ptr, n, threshold=False):
    """(N_total, n(n-1)/2): log(2 p_i.p_j / s)/5 of every pair i < j of each event, s = (p_0 + p_1)^2 of the two beams,
    or with `threshold` log(((p_i + p_j)^2 - (m_i + m_j)^2) / s)/5 = log(2 (p_i.p_j - m_i m_j) / s)/5, m_i = sqrt|p_i^2|:
    the same for a massless pair, log beta^2 of a massive pair at its threshold (model.input_pair_threshold),
    the event's row broadcast to each of its particles; pairs beyond an event's multiplicity are 0. Lorentz scalars,
    in float64 (a pair at 1e-7 of s is float32 rounding), floored at 1e-16 (model.input_pair_logs)."""
    cnt = ptr[1:] - ptr[:-1]
    assert int(cnt.max()) <= n, f"model.input_pair_logs={n} is below an event's {int(cnt.max())} particles"
    ar = torch.arange(n, device=p.device)
    valid = ar[None, :] < cnt[:, None]                                           # (B, n)
    P = p[(ptr[:-1, None] + ar[None, :]).clamp(max=p.shape[0] - 1)].to(torch.float64)
    dots = torch.einsum("bic,bjc,c->bij", P, P, torch.tensor(_ETA, device=p.device, dtype=P.dtype))
    s = (dots[:, 0, 0] + dots[:, 1, 1] + 2 * dots[:, 0, 1]).abs().clamp(min=1e-30)
    iu = torch.triu_indices(n, n, 1, device=p.device)
    pij = dots[:, iu[0], iu[1]]
    if threshold:
        m = dots.diagonal(dim1=1, dim2=2).abs().sqrt()
        pij = pij - m[:, iu[0]] * m[:, iu[1]]
    lp = torch.log((2 * pij / s[:, None]).abs().clamp(min=1e-16)) / 5
    return (lp * (valid[:, iu[0]] & valid[:, iu[1]])).repeat_interleave(cnt, dim=0)


def _div_form_width(n):
    """Columns of _div_form_inputs for an n-particle event: eikonal per (i<k, j), splitting and 1/beta per pair, per leg."""
    return n * (n - 1) * (n - 2) // 2 + n * (n - 1) + n


_CF, _CA, _TR = 4.0 / 3.0, 3.0, 0.5


def _div_form_inputs(p, ptr, n, cas, heavy):
    """(N_total, _div_form_width(n)): the functional forms of the universal singular limits as Lorentz-scalar inputs
    (model.input_div_form, kit_benchmark step 14), from the momenta and each particle's colour Casimir `cas` (0, C_F,
    C_A) and `heavy` flag (table mass >= 1 GeV), s = (p_0 + p_1)^2, s_ij = 2 p_i.p_j, the event's row on its particles:
      soft      per (i < k, j not i, k), j a gluon and i, k coloured: log(s s_ik / (s_ij s_jk))/5, the eikonal factor
      collinear per final pair i < j (legs >= 2) of light coloured partons: log P(z)/5, z = p_i.Q/(p_i + p_j).Q, Q = p_0 + p_1,
                P = C_F (1 + z_q^2)/(1 - z_q) (q g, z_q the quark's), C_A (z/(1-z) + (1-z)/z + z(1-z)) (g g),
                T_R (z^2 + (1-z)^2) (q q); the 1/s_ij that multiplies it is the pair log already (input_pair_logs)
      Coulomb   per pair i < j of heavy legs: log(1/beta)/5, beta = sqrt(lambda(M^2, m_i^2, m_j^2))/(M^2 - (m_i - m_j)^2),
                M^2 = (p_i + p_j)^2 (= sqrt(1 - 4m^2/M^2) for equal masses)
      Sudakov   per heavy leg: log^2(s/m_i^2)/25
    Every other entry, and every column beyond an event's multiplicity, is 0. Logs because the inputs must stay O(1)
    where the factors run over decades; the combinations (eikonal, P(z), 1/beta, the squared log) are the functional
    form, which the pair logs alone give only after the network learns them.
    shortcut: the splitting z is taken w.r.t. Q for final-state pairs only; an initial-state coloured leg (pp) needs the
    crossed form, add it when a pp process enters the KIT models."""
    cnt = ptr[1:] - ptr[:-1]
    assert int(cnt.max()) <= n, f"model.input_pair_logs={n} is below an event's {int(cnt.max())} particles"
    ar = torch.arange(n, device=p.device)
    valid = ar[None, :] < cnt[:, None]                                           # (B, n)
    idx = (ptr[:-1, None] + ar[None, :]).clamp(max=p.shape[0] - 1)
    P = p[idx].to(torch.float64)
    c, hv = cas[idx].to(torch.float64) * valid, heavy[idx] & valid
    dots = torch.einsum("bic,bjc,c->bij", P, P, torch.tensor(_ETA, device=p.device, dtype=P.dtype))
    s = (dots[:, 0, 0] + dots[:, 1, 1] + 2 * dots[:, 0, 1]).abs().clamp(min=1e-30)
    sij = (2 * dots).abs().clamp(min=1e-30)
    col, glu, qrk = c > 0, (c - _CA).abs() < 1e-3, (c - _CF).abs() < 1e-3
    lg = lambda x: torch.log(x.clamp(min=1e-300)) / 5
    # soft: ordered triples (i < k, j)
    tri = torch.tensor([(i, j, k) for i in range(n) for k in range(i + 1, n) for j in range(n) if j not in (i, k)],
                       device=p.device, dtype=torch.long).reshape(-1, 3)
    i, j, k = tri[:, 0], tri[:, 1], tri[:, 2]
    eik = lg(s[:, None] * sij[:, i, k] / (sij[:, i, j] * sij[:, j, k])) * (col[:, i] & col[:, k] & glu[:, j])
    # collinear and Coulomb: pairs i < j
    iu = torch.triu_indices(n, n, 1, device=p.device)
    a, b = iu[0], iu[1]
    Q = P[:, 0] + P[:, 1]
    xq = torch.einsum("bic,bc,c->bi", P, Q, torch.tensor(_ETA, device=p.device, dtype=P.dtype))   # p_i.Q
    z = (xq[:, a] / (xq[:, a] + xq[:, b])).clamp(1e-300, 1 - 1e-16)
    zq = torch.where(qrk[:, a], z, 1 - z)                                        # the quark's fraction in a q g pair
    Pqg = _CF * (1 + zq ** 2) / (1 - zq).clamp(min=1e-300)
    Pgg = _CA * (z / (1 - z) + (1 - z) / z + z * (1 - z))
    Pqq = _TR * (z ** 2 + (1 - z) ** 2)
    fin = (a >= 2) & (b >= 2)
    qg = (qrk[:, a] & glu[:, b]) | (glu[:, a] & qrk[:, b])
    spl = torch.where(qg, Pqg, torch.where(glu[:, a] & glu[:, b], Pgg, Pqq))
    spl = lg(spl) * (col[:, a] & col[:, b] & fin & ~hv[:, a] & ~hv[:, b])   # a heavy leg has no collinear limit
    m2 = dots.diagonal(dim1=1, dim2=2).abs()
    m = m2.sqrt()
    M2 = m2[:, a] + m2[:, b] + 2 * dots[:, a, b]
    lam = ((M2 - (m[:, a] + m[:, b]) ** 2) * (M2 - (m[:, a] - m[:, b]) ** 2)).abs()
    beta = lam.sqrt() / (M2 - (m[:, a] - m[:, b]) ** 2).abs().clamp(min=1e-300)
    coul = -lg(beta) * (hv[:, a] & hv[:, b])
    sud = (torch.log(s[:, None] / m2.clamp(min=1e-300)) ** 2 / 25) * hv
    return torch.cat([eik, spl, coul, sud], dim=1).repeat_interleave(cnt, dim=0)


# model.input_asym_factor: every learned coefficient within +-10 target std of the bulk mean. Needed (tools/kit_asym_bound.py,
# kit_asym_bound.json): the common coefficient (log|M|^2 - log sum D_t - mean)/sd over bulk_train and every hard_test
# stratum reaches |c| 3.05 (uug), 6.07 (uugg soft), 6.84 (tt Sudakov), the regular readout |z| 5.24; 10 leaves 3 std
_ASYM_BOUND = 10.0


def _asym_terms(p, ptr, n, cas, heavy, loops, sud_single=None):
    """The singular factors D_t of model.input_asym_factor (kit_benchmark step 18, D435), built from each particle's
    colour Casimir `cas` (0, C_F, C_A), `heavy` flag (table mass >= 1 GeV) and the event's QCD loop count `loops`,
    s = (p_0 + p_1)^2, s_ij = 2 p_i.p_j, final legs only (index >= 2):
      dipole   per ordered (i, j, k) of light coloured final legs, j a gluon, i a quark (q -> q g) or a gluon with i < j
               (g -> g g, once per pair), k the spectator: Catani-Seymour's spin-averaged massless final-final dipole
               D = (s/s_ij) V, y = s_ij/(s_ij + s_ik + s_jk), z = s_ik/(s_ik + s_jk),
               V_qg = C_F (2/(1 - z(1-y)) - (1+z)), V_gg = C_A (1/(1 - z(1-y)) + 1/(1 - (1-z)(1-y)) - 2 + z(1-z))
      Coulomb  per pair i < j of heavy coloured legs of a loop target: D = 1/beta, beta = sqrt(lambda(M^2, m_i^2, m_j^2))/(M^2 - (m_i - m_j)^2)
      Sudakov  per heavy coloured leg of a loop target: D = log^2(s/m_i^2)
      single   with `sud_single` (a learned scalar kappa, model.input_asym_sudakov_single): per heavy coloured leg of a
               loop target also D = exp(kappa) log(s/m_i^2), on the leg's coefficient: exp(c)(L^2 + exp(kappa) L), the
               subleading single log with one global ratio (step 21: log^2 alone overshoots the Sudakov strata)
    (a colourless heavy leg, W, Z, H, tau, has no QCD Coulomb or Sudakov factor: the target's loop count is L_QCD)
    Returns log D (B, T) in float64, on (B, T) (the term exists in that event) and members (T, 3): the legs whose
    coefficients set the term's coefficient, -1 padded. Every D is positive and Lorentz invariant.
    shortcut: no heavy emitter (CDST's massive Q -> Q g dipole) and no initial-state coloured leg; no KIT target has
    either (uug, uugg, tt virtual), add them when a t t~ g or a pp target is fit with the factor."""
    cnt = ptr[1:] - ptr[:-1]
    ar = torch.arange(n, device=p.device)
    valid = ar[None, :] < cnt[:, None]
    idx = (ptr[:-1, None] + ar[None, :]).clamp(max=p.shape[0] - 1)
    P = p[idx].to(torch.float64)
    c, hv = cas[idx].to(torch.float64) * valid, heavy[idx] & valid
    dots = torch.einsum("bic,bjc,c->bij", P, P, torch.tensor(_ETA, device=p.device, dtype=P.dtype))
    s = (dots[:, 0, 0] + dots[:, 1, 1] + 2 * dots[:, 0, 1]).abs().clamp(min=1e-300)
    sij = (2 * dots).abs().clamp(min=1e-300)
    light = (c > 0) & ~hv
    glu, qrk = light & ((c - _CA).abs() < 1e-3), light & ((c - _CF).abs() < 1e-3)
    loop = (loops > 0.5)[:, None] if loops is not None else torch.zeros_like(s, dtype=torch.bool)[:, None]
    lg = lambda x: torch.log(x.clamp(min=1e-300))
    dev, L = p.device, torch.long
    tri = torch.tensor([(i, j, k) for i in range(2, n) for j in range(2, n) for k in range(2, n) if len({i, j, k}) == 3],
                       device=dev, dtype=L).reshape(-1, 3)
    i, j, k = tri[:, 0], tri[:, 1], tri[:, 2]
    a, b, d = sij[:, i, j], sij[:, i, k], sij[:, j, k]
    y, z = a / (a + b + d), b / (b + d)
    wi, wj = d / (b + d) + z * y, b / (b + d) + (1 - z) * y          # 1 - z(1-y), 1 - (1-z)(1-y), stable at z -> 1
    qg, gg = qrk[:, i] & glu[:, j], glu[:, i] & glu[:, j] & (i < j)
    V = torch.where(qg, _CF * (2 / wi - (1 + z)), _CA * (1 / wi + 1 / wj - 2 + z * (1 - z)))
    dip = (lg(V) + lg(s[:, None] / a), (qg | gg) & light[:, k], tri)
    pr = torch.triu_indices(n, n, 1, device=dev)
    pr = pr[:, pr[0] >= 2]
    a, b = pr[0], pr[1]
    m2 = dots.diagonal(dim1=1, dim2=2).abs()
    m = m2.sqrt()
    M2 = m2[:, a] + m2[:, b] + 2 * dots[:, a, b]
    beta = (((M2 - (m[:, a] + m[:, b]) ** 2) * (M2 - (m[:, a] - m[:, b]) ** 2)).abs().sqrt()
            / (M2 - (m[:, a] - m[:, b]) ** 2).abs().clamp(min=1e-300))
    neg = torch.full_like(a, -1)
    hq = hv & (c > 0)                                                  # heavy and coloured
    cou = (-lg(beta), hq[:, a] & hq[:, b] & loop, torch.stack([a, b, neg], 1))
    leg = torch.arange(2, n, device=dev)
    sud = (2 * lg(lg(s[:, None] / m2[:, leg].clamp(min=1e-300))), hq[:, leg] & loop,
           torch.stack([leg, torch.full_like(leg, -1), torch.full_like(leg, -1)], 1))
    parts = (dip, cou, sud)
    if sud_single is not None:
        parts += ((lg(lg(s[:, None] / m2[:, leg].clamp(min=1e-300))) + sud_single.to(P.dtype).reshape(()),) + sud[1:],)
    return tuple(torch.cat(x, dim=d) for x, d in zip(zip(*parts), (1, 1, 0)))


def _asym_mu(mu_raw, alpha, ptr, n, logD, on, members, sd):
    """(B,) the standardized mu of model.input_asym_factor: log|M|^2 = log A(x) + g(x) with
    A = sum_t exp(c_t) D_t, c_t = sd * mean over the term's legs of alpha_i (per-particle readout), and
    g = log(1 + exp(r)/A), r = sd * mu_raw (the pooled readout), both readouts bounded by +-_ASYM_BOUND (tanh), so g is
    bounded and goes to 0 wherever a D_t diverges: there log|M|^2 - log(exp(c_t) D_t) -> 0, the limit's form is exact
    and only the coefficient (the reduced matrix element's role, smooth in the limit) is learned (Maitre-Truong,
    arXiv:2107.06625). In target units: mu = (log|M|^2 - mean)/sd; the bulk level `mean` cancels into the readouts.
    mu_raw (B,), alpha (N_total,) the raw readouts; logD, on, members from _asym_terms; sd the target's std (nats)."""
    B_ = _ASYM_BOUND
    cnt = ptr[1:] - ptr[:-1]
    ar = torch.arange(n, device=ptr.device)
    valid = (ar[None, :] < cnt[:, None]).to(torch.float64)
    al = (B_ * torch.tanh(alpha.to(torch.float64) / B_))[(ptr[:-1, None] + ar[None, :]).clamp(max=alpha.shape[0] - 1)]
    al = torch.cat([al * valid, torch.zeros_like(al[:, :1])], 1)               # column n: the -1 padding
    mem = torch.where(members < 0, torch.full_like(members, n), members)
    cbar = al[:, mem].sum(-1) / (members >= 0).sum(-1).to(torch.float64)        # (B, T)
    sd = sd.to(torch.float64)
    terms = torch.where(on, sd * cbar + logD, torch.full_like(logD, -torch.inf))
    reg = sd * B_ * torch.tanh(mu_raw.to(torch.float64) / B_)
    return torch.logsumexp(torch.cat([reg[:, None], terms], 1), 1) / sd


def _soft_rescale(p, ptr, eps):
    """p_i -> p_i log1p(x_i/eps) / (x_i log1p(1/eps)), x_i = 2 p_i.Q/s, Q = p_0 + p_1, s = Q^2: an invariant factor per
    momentum (equivariant), 1 at x_i = 1, so a soft momentum's scale reads logarithmically (model.input_soft_eps)."""
    cnt = ptr[1:] - ptr[:-1]
    eta = torch.tensor(_ETA, device=p.device, dtype=torch.float64)
    p64 = p.to(torch.float64)
    Q = (p64[ptr[:-1]] + p64[ptr[:-1] + 1]).repeat_interleave(cnt, dim=0)
    x = (2 * (p64 * Q * eta).sum(-1) / (Q * Q * eta).sum(-1)).abs().clamp(min=1e-30)
    return p * (torch.log1p(x / eps) / (x * math.log1p(1 / eps))).to(p.dtype)[:, None]


def _log_skip(p, ptr, types, n, head, threshold=False, quadratic=True):
    """(B,) a linear readout of the event's singular logarithms, added to mu (model.input_log_skip, D411/step 13):
    sum_{i<j} [c1_ij L_ij + c2_ij L_ij^2] + sum_i [b1_i M_i + b2_i M_i^2], L_ij = log(2 p_i.p_j / s)/5 (with `threshold`
    log(2 (p_i.p_j - m_i m_j)/s)/5: log beta^2 of a massive pair), M_i = log(m_i^2/s)/5 of a massive particle (0 if
    massless), s = (p_0 + p_1)^2. The coefficients come from the particles' types only, c_ij = <a_i, a_j> and b_i from
    `head` on each particle's encoded properties, so the same readout applies to any process: a power law or a double log
    in an invariant is linear or quadratic here and continues past the trained range, where a saturating backbone does not.
    quadratic=False keeps only the linear terms (c1, b1; model.input_log_skip_linear, step 13)."""
    cnt = ptr[1:] - ptr[:-1]
    ar = torch.arange(n, device=p.device)
    valid = (ar[None, :] < cnt[:, None]).to(torch.float64)                      # (B, n)
    idx = (ptr[:-1, None] + ar[None, :]).clamp(max=p.shape[0] - 1)
    P = p[idx].to(torch.float64)
    dots = torch.einsum("bic,bjc,c->bij", P, P, torch.tensor(_ETA, device=p.device, dtype=P.dtype))
    s = (dots[:, 0, 0] + dots[:, 1, 1] + 2 * dots[:, 0, 1]).abs().clamp(min=1e-30)
    m2 = dots.diagonal(dim1=1, dim2=2).abs()
    pij = dots - (m2[:, :, None] * m2[:, None, :]).sqrt() if threshold else dots
    L = torch.log((2 * pij / s[:, None, None]).abs().clamp(min=1e-16)) / 5
    massive = (m2 / s[:, None] > 1e-12).to(torch.float64)
    Mm = torch.log((m2 / s[:, None]).clamp(min=1e-16)) / 5 * massive
    h = head(types)[idx].to(torch.float64) * valid[..., None]                 # (B, n, 2k + 2)
    k = (h.shape[-1] - 2) // 2
    c1 = torch.einsum("bik,bjk->bij", h[..., :k], h[..., :k])
    c2 = torch.einsum("bik,bjk->bij", h[..., k:2 * k], h[..., k:2 * k])
    up = torch.triu(torch.ones(n, n, device=p.device, dtype=torch.float64), 1)
    q = float(quadratic)
    pair = ((c1 * L + q * c2 * L ** 2) * up).sum((1, 2))
    return pair + (h[..., -2] * Mm + q * h[..., -1] * Mm ** 2).sum(1)


def _pool_events(values, ptr):
    """Per-event mean pooling.

    Defaults to the vectorised path. Set env var LLOCA_POOL=loop to fall back to
    the original Python-loop implementation (kept only for A/B timing/equivalence
    checks); any other value uses the vectorised path.
    """
    if os.environ.get("LLOCA_POOL", "vectorized") == "loop":
        B = ptr.shape[0] - 1
        return torch.stack([values[ptr[i]:ptr[i + 1]].mean(dim=0) for i in range(B)])
    return _segment_mean(values, ptr)


def _build_block_diagonal_mask(ptr, device):
    """Boolean mask (N_total, N_total): True where two particles share an event."""
    N = int(ptr[-1])
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(len(ptr) - 1):
        s, e = int(ptr[i]), int(ptr[i + 1])
        mask[s:e, s:e] = True
    return mask


class AmplitudeMLPWrapper(nn.Module):
    """LEGACY (non-LLoCa). Restored verbatim from the reference implementation
    (heidelberg-hepml/amplitude_DSI, wrappers.py) — it was removed in the LLoCa refactor, which
    left `model=mup_mlp` un-instantiable.

    Restored ONLY to run the diagnostic control: the reference wires the HETEROSC head exclusively
    into the MLP (models/mup_mlp.py: softplus applied to an ALREADY PER-EVENT readout, no pooling),
    and never into LLoCa. So the MLP is the one architecture on which the het loss is actually
    validated, and it is the last untested axis behind het's failure on LLoCa (14-20x worse
    accuracy than MSE, reliability slope 1.3-1.5, on BOTH datasets, unfixed by the sigma-pool
    change and by three Bayesian sweeps).

    The net is not permutation invariant, so type_token / global_token / attn_mask are ignored —
    the flattened per-event feature vector is fed straight in.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token=None, global_token=None, attn_mask=None, **kwargs):
        return self.net(inputs)


class AmplitudeGATrWrapper(nn.Module):
    def __init__(self, net, token_size):
        super().__init__()
        self.net = net
        self.token_size = token_size

    def forward(self, particles, particle_type_indices, ptr, **kwargs):
        """
        particles             : (N_total, 4)   flat sparse (same format as LLoCa)
        particle_type_indices : (N_total,)     long
        ptr                   : (B+1,)         event boundaries
        """
        # one-hot encode particle types as scalars
        scalars = torch.nn.functional.one_hot(
            particle_type_indices, num_classes=self.token_size
        ).to(dtype=particles.dtype)                          # (N_total, token_size)

        # embed 4-momenta into multivectors: (1, N_total, 1, 16)
        mv = embed_vector(particles.unsqueeze(0).unsqueeze(-2))
        s  = scalars.unsqueeze(0)                            # (1, N_total, token_size)

        # block-diagonal attention mask: particles attend only within their event
        attn_mask = _build_block_diagonal_mask(ptr, particles.device)  # (N_total, N_total)

        mv_out, _ = self.net(mv, scalars=s, attention_mask=attn_mask)

        # extract scalar output per particle, then mean-pool per event
        amp_per_particle = extract_scalar(mv_out)[0, :, 0, 0]  # (N_total,)

        B = ptr.shape[0] - 1
        pooled = torch.stack([
            amp_per_particle[ptr[i]:ptr[i + 1]].mean()
            for i in range(B)
        ]).unsqueeze(-1)                                     # (B, 1)
        return pooled


class AmplitudeLLoCaWrapper(nn.Module):
    def __init__(self, net, token_size, d_particle_hidden: int = 16,
                 particle_encoder_hidden: int = 0,
                 use_diagrams: bool = False, d_diag: int = 32,
                 diagrams_dir: str = "data/diagrams", diagram_encoder=None,
                 use_diagram_virtuality: bool = False, virt_log_scale: float = 0.1,
                 virt_standardize: bool = True, virt_clamp: float = 4.0,
                 virt_mode: str = "edge",
                 diagram_scanned_mass: bool = False,   # read in experiment.init_model; accepted here so Hydra can pass it
                 use_pair_bias: bool = False, pair_bias_hidden: int = 16,
                 pair_bias_clamp: float = 4.0,
                 pair_bias_max_props: int = 48,
                 pair_bias_logit_cap: float = 8.0,    # ditto: consumed by the experiment (spec build + setup_pair_bias)
                 diagram_encoder_impl: str = "ours",  # "ours" | "llm4lhc" (models/diagram_encoder_llm4lhc.py)
                 diagram_encoder_llm4lhc=None,        # its hyperparameters; consumed by the experiment
                 context_dim: int = 0, context_dropout: float = 0.5,   # read in experiment._post_instantiate_model
                 context_names_file=None,
                 sign_pair_dim: int = 0, sign_pair_hidden: int = 32, sign_pair_bound: bool = False,
                 input_pair_logs: int = 0, input_soft_eps: float = 0.0, input_local_log_eps: float = 0.0,
                 input_pair_threshold: bool = False, input_scalars_only: bool = False,
                 input_route_off: str = "", input_log_skip: int = 0, input_log_skip_linear: bool = False,
                 input_div_form: bool = False, input_asym_factor: bool = False,
                 input_asym_sudakov_single: bool = False):
        super().__init__()
        self.net = net
        # input representations for singular limits (D374, kit_benchmark input arms; all Lorentz-exact, 0 = off):
        # input_pair_logs n: log(2 p_i.p_j/s)/5 of every pair of an n-particle event as extra scalars (_pair_log_inputs);
        # input_soft_eps: p_i -> p_i log1p(x_i/eps) / (x_i log1p(1/eps)), x_i = 2 p_i.Q/s (_soft_rescale);
        # input_local_log_eps: asinh(z/eps)/asinh(1/eps) of the standardized local-frame components (models/lloca.py)
        self.input_pair_logs = int(input_pair_logs or 0)
        self.input_pair_threshold = bool(input_pair_threshold)   # pair logs of (p_i+p_j)^2 - (m_i+m_j)^2: log beta^2
        self.input_soft_eps = float(input_soft_eps or 0.0)
        net.local_log_eps = float(input_local_log_eps or 0.0)
        # input_scalars_only: the backbone reads only the per-particle scalars (properties, orders, pair logs): local
        # momenta zeroed, identity frames (models/lloca.py), so no momentum reaches it but through the pair logs
        net.scalars_only = bool(input_scalars_only)
        # input_route_off: cut one route of the momenta into the backbone (D411 isolation): 'local' zeroes the local-frame
        # momenta (learned frames kept in the attention), 'frames' passes identity frames to the attention (local momenta kept)
        assert input_route_off in ("", "local", "frames"), input_route_off
        net.route_off = input_route_off
        # input_log_skip k: a type-dependent linear + quadratic readout of the pair logs and mass logs added to mu
        # (_log_skip; rank-k pair coefficients); needs input_pair_logs (its multiplicity) and the property encoding
        self.input_log_skip = int(input_log_skip or 0)
        self.input_log_skip_linear = bool(input_log_skip_linear)   # drop the quadratic terms (step 13)
        # input_div_form: the singular limits' functional forms per triple / pair / leg (_div_form_inputs, step 14); needs
        # input_pair_logs (its multiplicity) and the property encoding (colour Casimir and table mass of each particle)
        self.input_div_form = bool(input_div_form)
        assert not self.input_div_form or self.input_pair_logs, "model.input_div_form needs model.input_pair_logs"
        # input_asym_factor: log|M|^2 = log A + g, A the universal singular factors with learned per-leg coefficients,
        # g bounded and vanishing in every limit (_asym_terms, _asym_mu; step 18); the net carries one more per-particle
        # row (models/lloca.py extra_main) and asym_sd, the target's std, is set by the experiment from its data stats
        self.input_asym_factor = bool(input_asym_factor)
        if self.input_asym_factor:
            self.register_buffer("asym_sd", torch.ones((), dtype=torch.float64))
        # input_asym_sudakov_single: the Sudakov single log beside the double log, kappa = log of their ratio (_asym_terms)
        assert not input_asym_sudakov_single or self.input_asym_factor, "input_asym_sudakov_single needs input_asym_factor"
        self.asym_kappa = nn.Parameter(torch.zeros(1)) if input_asym_sudakov_single else None
        if self.input_log_skip:
            assert self.input_pair_logs, "model.input_log_skip needs model.input_pair_logs (the event multiplicity)"
            self.log_skip_head = nn.Linear(d_particle_hidden, 2 * self.input_log_skip + 2)
            nn.init.normal_(self.log_skip_head.weight, std=0.1)     # c_ij = <a_i, a_j>: zero init would never move
            nn.init.zeros_(self.log_skip_head.bias)
        # Pair sign head (model.sign_pair_dim, training.sign_head): the net's last sign_pair_dim per-particle channels
        # are pair features; the sign logit gets the mean over the event's particle pairs of an MLP of them and of the
        # pair's invariants (see _sign_pair_logit). Output layer zeroed: at init the head is the pooled channel alone.
        self.sign_pair_dim = int(sign_pair_dim or 0)
        # sign_pair_bound: the pair channels go through tanh before the sums and products, so h_i + h_j and h_i h_j stay
        # in [-2, 2] and [-1, 1] (the unbounded head's run blew up, par:v3_prep_signed); off keeps the first arm's head
        self.sign_pair_bound = bool(sign_pair_bound)
        if self.sign_pair_dim > 0:
            K = self.sign_pair_dim
            self.sign_pair_mlp = nn.Sequential(nn.Linear(2 * K + 4, int(sign_pair_hidden)), nn.SiLU(),
                                               nn.Linear(int(sign_pair_hidden), 1))
            nn.init.zeros_(self.sign_pair_mlp[-1].weight)
            nn.init.zeros_(self.sign_pair_mlp[-1].bias)
        # Per-process context vector (model.context_dim, D105): off until the experiment calls setup_context().
        self.context = None
        self.context_override = None
        self.network_dtype = torch.float32
        self.token_size = token_size
        self.d_particle_hidden = d_particle_hidden
        # 0 → single Linear encoder; >0 → 2-layer MLP (see _build_particle_encoder).
        # Hydra passes this from config/model/lloca.yaml; setup_particle_features
        # receives the same value explicitly from experiment._post_instantiate_model.
        self.particle_encoder_hidden = particle_encoder_hidden
        self.use_pids = True   # overridden by setup_particle_features()
        # particle_encoder and property_matrix are set by setup_particle_features()

        # Data-derived per-particle mass (data.mass_from_momenta); off until the
        # experiment calls setup_mass_from_momenta(). When on, the forward replaces
        # the gathered property matrix's log10_mass_gev column with the on-shell mass
        # m=sqrt(E^2-|p|^2) of each particle (see setup_mass_from_momenta).
        self.mass_from_momenta = False
        self._mom_div = 1.0
        self._mass_spec = None

        # Feynman-diagram conditioning config (Hydra passes these from lloca.yaml).
        # The encoder module + per-process graphs are attached later by the
        # experiment via setup_diagram_conditioning() (they need the loaded
        # diagram registry, which Hydra cannot build). use_diagrams stays False
        # until then, so an un-set-up wrapper behaves exactly like before.
        self.use_diagrams = False
        self.use_diagram_virtuality = False
        # Pairwise attention bias (model.use_pair_bias); attached later by the
        # experiment via setup_pair_bias() (needs the diagram sidecars).
        self.use_pair_bias = False
        self._cfg_use_diagrams = bool(use_diagrams)
        # "llm4lhc": the arXiv:2606.23791 encoder, its vector ADDED to every particle's
        # token embedding inside the backbone (setup_diagram_conditioning_llm4lhc), not
        # appended to the scalar features. Set for real by that setup call.
        self.diagram_encoder_impl = "ours"
        self._cfg_diagram_encoder_impl = str(diagram_encoder_impl)
        self.d_diag = int(d_diag)
        self.diagrams_dir = diagrams_dir
        self._diag_enc_cfg = diagram_encoder
        self._pd_by_pid = None
        self._pd_device = None
        self._virt_by_pid = None
        self._virt_log_scale = 0.1   # signed-log virtuality scale; see _diagram_features_virtuality

    def setup_particle_features(self, use_pids: bool, property_matrix=None,
                                encoder_hidden: int = 0):
        """
        Configure particle encoding after construction (called from experiment.init_model).

        use_pids=False (default):
            Looks up a fixed property matrix (global_pdg_idx → quantum numbers) then
            projects through a small learned encoder (n_features → d_particle_hidden).
            d_particle_hidden is fixed regardless of n_features, so adding a new quantum
            number only requires extending the projection — all transformer weights
            stay identical and can be reloaded without retraining. ``encoder_hidden>0``
            makes that encoder a 2-layer MLP instead of a single Linear.

        use_pids=True (legacy):
            One-hot encodes particle type indices; no projection layer.
        """
        self.use_pids = use_pids
        if not use_pids:
            assert property_matrix is not None
            t = torch.tensor(property_matrix, dtype=torch.float32)
            self.register_buffer("property_matrix", t)
            self._register_load_state_dict_pre_hook(_property_matrix_prefix_hook(self))
            n_features = property_matrix.shape[1]
            self.particle_encoder = _build_particle_encoder(
                n_features, self.d_particle_hidden, encoder_hidden
            )

    def setup_context(self, names, dim: int, dropout: float, own_names):
        """Per-process context vector c_p (model.context_dim, D105; CAVIA, arXiv:1909.09157): a learned row per
        process of ``names`` (the pretraining's processes, in its recipe order), ADDED to every particle's encoded
        properties of that process's events. In training each event's c_p is replaced by the zero vector with
        probability ``dropout``, so the zero vector is the zero-shot model: a process the table has no row for gets
        zero. ``own_names``: this run's processes in process-id order, mapped to table rows by name (a probe's run
        reads a parent's table). The table starts at zero, i.e. at the zero-shot model."""
        assert not self.use_pids, "the context vector is added to the property encoding (use_PIDs=false)"
        assert int(dim) == self.d_particle_hidden, \
            f"model.context_dim {dim} must equal d_particle_hidden {self.d_particle_hidden} (it is added to it)"
        self.context_names = [str(n) for n in names]
        self.context = nn.Parameter(torch.zeros(len(self.context_names), int(dim)))
        self.context_dropout = float(dropout)
        self.set_context_rows(own_names)

    def set_context_rows(self, own_names):
        """process id -> row of the context table (-1: no row, the zero vector)"""
        idx = {n: i for i, n in enumerate(self.context_names)}
        rows = torch.tensor([idx.get(str(n), -1) for n in own_names], dtype=torch.long)
        if "_ctx_row" in self._buffers:
            rows = rows.to(self._buffers["_ctx_row"].device)
        self.register_buffer("_ctx_row", rows, persistent=False)

    def _context_per_particle(self, process_ids, ptr, dtype):
        """(N_total, dim) context added to the particle encoding, or None. ``context_override`` (a (dim,) tensor,
        set by tools/kshot_context.py) replaces every event's vector: the K-shot fit of a held-out process."""
        counts = ptr[1:] - ptr[:-1]
        if self.context_override is not None:
            c = self.context_override.to(dtype).unsqueeze(0).expand(counts.shape[0], -1)
        elif process_ids is None:
            return None
        else:
            row = self._ctx_row[process_ids]                                  # (B,)
            c = self.context[row.clamp(min=0)].to(dtype) * (row >= 0).unsqueeze(-1).to(dtype)
            if self.training and self.context_dropout > 0:
                keep = (torch.rand(c.shape[0], device=c.device) >= self.context_dropout).to(dtype)
                c = c * keep.unsqueeze(-1)
        return c.repeat_interleave(counts, dim=0)

    def setup_mass_from_momenta(self, mom_div: float, mass_spec: dict):
        """Enable data-derived per-particle mass (``data.mass_from_momenta``).

        ``mom_div`` is the run's global momentum scale (the model sees momenta
        already divided by it), so physical GeV are recovered as ``p*mom_div``.
        ``mass_spec`` is :func:`particle_ids.mass_feature_spec` — the column index
        and transform constants that put the on-shell mass on the same scale the
        frozen ``log10_mass_gev`` column would have under the active encoding flags.
        Off (default) ⇒ the forward is unchanged."""
        self.mass_from_momenta = True
        self._mom_div = float(mom_div)
        self._mass_spec = dict(mass_spec)

    def setup_dimless_mass(self, mass_col: int, raw_log10m, massive, mean: float, scale: float, massless_value: float):
        """Enable data.dimensionless_inputs on the property side: the mass column becomes, per particle of each event,
        (log10(m / sqrt(s_hat)) - mean) / scale for a massive species and ``massless_value`` for a massless one, so no
        absolute scale reaches the network. The event's log10 sqrt(s_hat) arrives as the last column of order_labels,
        which this takes off before the order features are used. ``raw_log10m``/``massive``: (n_types,) per row of
        property_matrix, the generator's mass (not recomputed from the float32 momenta, where a massless leg reads as
        m ~ 1e-4 sqrt(s_hat)); ``mean``/``scale``: the column's statistics over the massive train particles
        (experiment: dimless_mass_stats, frozen with the data stats; scale 1 when the table is not standardized)."""
        self.dimless_mass = True
        self._dm_col = int(mass_col)
        self._dm_mean, self._dm_scale, self._dm_massless = float(mean), float(scale), float(massless_value)
        self.register_buffer("_dm_log10m", torch.as_tensor(raw_log10m, dtype=torch.float32), persistent=False)
        self.register_buffer("_dm_massive", torch.as_tensor(massive, dtype=torch.bool), persistent=False)

    def setup_dimless_pids(self):
        """data.dimensionless_inputs under use_PIDs: no mass column to put per sqrt(s_hat); the event's log10
        sqrt(s_hat), the last column of order_labels, is only taken off (and kept for the diagram encoder's s_hat)."""
        self.dimless_pids = True

    def _apply_dimless_mass(self, raw_props, particle_type_indices, ptr, order_labels):
        """(raw_props with the mass column per event, order_labels without the log10 sqrt(s_hat) column)"""
        lrs = order_labels[:, -1].repeat_interleave(ptr[1:] - ptr[:-1], dim=0)          # (N_total,)
        col = (self._dm_log10m[particle_type_indices] - lrs - self._dm_mean) / self._dm_scale
        col = torch.where(self._dm_massive[particle_type_indices], col, torch.full_like(col, self._dm_massless))
        raw_props = raw_props.clone()
        raw_props[:, self._dm_col] = col.to(raw_props.dtype)
        return raw_props, order_labels[:, :-1]

    def _apply_mass_from_momenta(self, raw_props, fourmomenta):
        """Replace the log10_mass_gev column of ``raw_props`` (N, n_features) with
        each particle's on-shell mass derived from ``fourmomenta`` (N, 4), matching
        the table column's massless-neutralise → standardize pipeline. Returns the
        (cloned) feature tensor. Shared by the LLoCa + GATr wrappers."""
        spec = self._mass_spec
        raw_props = raw_props.clone()
        p = fourmomenta * self._mom_div                       # scaled → physical GeV
        m2 = p[:, 0] ** 2 - (p[:, 1:] ** 2).sum(dim=-1)
        m = torch.sqrt(m2.clamp(min=0.0))
        floor = spec["floor_log10"]
        log10m = torch.log10(m.clamp(min=10.0 ** floor))      # massless → _MASSLESS floor
        massless = log10m < spec["threshold_log10"]
        if spec["neutralize_log10"] is not None:              # is_massless flag active
            log10m = torch.where(
                massless,
                torch.full_like(log10m, spec["neutralize_log10"]),
                log10m,
            )
        if spec["std_mu"] is not None:
            log10m = (log10m - spec["std_mu"]) / spec["std_sd"]
        raw_props[:, spec["mass_col"]] = log10m.to(raw_props.dtype)
        if spec["is_massless_col"] is not None:
            raw_props[:, spec["is_massless_col"]] = massless.to(raw_props.dtype)
        return raw_props

    def setup_diagram_conditioning(self, encoder, pd_by_pid, d_diag):
        """Attach the diagram graph encoder + per-process graphs (from experiment).

        encoder   : models.diagram_encoder.DiagramEncoder (a real submodule → trained,
                    checkpointed, and marked standard-parametrisation by mup_finalize).
        pd_by_pid : list indexed by process_id (0..n_processes-1) of
                    diagram_graphs.ProcessDiagrams | None (None → that process has no
                    sidecar; it gets a zero diagram embedding so the run still works).
        d_diag    : width of the appended embedding (must match in_channels sizing).
        """
        from diagram_graphs import build_diagram_batch
        self.diagram_encoder = encoder
        self._pd_by_pid = list(pd_by_pid)
        # Pre-batch every process's diagrams into one padded tensor bundle so each
        # step is ONE encoder forward (vs one launch-bound forward per process).
        self._diag_batch = build_diagram_batch(pd_by_pid)
        self._pd_device = None
        self.d_diag = int(d_diag)
        self.use_diagrams = True

    def setup_diagram_conditioning_llm4lhc(self, encoder):
        """Attach the arXiv:2606.23791 diagram encoder (models/diagram_encoder_llm4lhc).

        encoder : LLM4LHCDiagramEncoder holding every process's diagrams as buffers; its
                  ``out_dim`` must equal the backbone's token width. A real submodule:
                  trained, checkpointed, marked standard-parametrisation by mup_finalize.
        Its per-process vector is added to the token embedding of every particle of the
        event (``token_add``); the scalar features are left as they are.
        """
        if encoder.mode not in ("pooled", "pooled_external"):
            raise ValueError(
                f"diagram_encoder_llm4lhc.mode={encoder.mode!r}: only 'pooled' and "
                "'pooled_external' have an injection site here (per_diagram needs one "
                "prefix token per diagram; LLoCa tokens are particles with momenta)")
        width = int(self.net.net.hidden_channels)
        if encoder.out_dim != width:
            raise ValueError(f"encoder out_dim {encoder.out_dim} != backbone width {width}")
        self.diagram_encoder = encoder
        self.diagram_encoder_impl = "llm4lhc"
        self.use_diagrams = True

    def setup_pair_bias(self, specs_by_pid, num_heads, hidden=16, clamp=4.0,
                        mom_div=1.0, logit_cap=8.0):
        """Attach the diagram-derived PAIRWISE attention bias (``model.use_pair_bias``).

        Independent of the diagram encoder (use_diagrams): the diagram sidecars are
        used only to enumerate each process's internal propagators. Every propagator
        cuts the process into two leg clusters; per event its off-shellness
        s−m² = (Σ sign·p)² − m² (a Lorentz scalar) is signed-log compressed,
        standardized (first-batch calibration, persisted buffers), passed WITH the
        propagator's identity features (pfeat: log-mass + massless flag) through a
        small zero-init MLP → one score per head, soft-capped to ±logit_cap
        (cap·tanh(x/cap): linear for small scores, saturating instead of the
        unbounded −17-logit hard-mask regime the v1 arm learned), and added to the
        attention logits of every particle pair inside the resonating cluster(s).
        Zero-init makes the model EXACTLY the baseline at step 0 — the bias is
        opt-in by gradient.

        specs_by_pid : list indexed by process_id of None | dict with numpy arrays
                       mask (K,n_part) signed momentum-gather rows, m2 (K,) physical
                       GeV², pfeat (K,F) per-propagator identity features, trips
                       (T,3) int64 rows (k, i, j) — the (propagator, ordered pair)
                       index triples the score is scattered into.
        num_heads    : per-head bias width (the backbone's H).
        """
        P = len(specs_by_pid)
        K = max((s["mask"].shape[0] for s in specs_by_pid if s is not None), default=0)
        T = max((s["trips"].shape[0] for s in specs_by_pid if s is not None), default=0)
        n = max((s["mask"].shape[1] for s in specs_by_pid if s is not None), default=0)
        F = max((s["pfeat"].shape[1] for s in specs_by_pid
                 if s is not None and "pfeat" in s), default=0)
        if K == 0 or T == 0:
            return   # nothing usable; feature stays off
        mask = torch.zeros(P, K, n)
        m2 = torch.zeros(P, K)
        kvalid = torch.zeros(P, K)
        pfeat = torch.zeros(P, K, F) if F else None
        tk = torch.zeros(P, T, dtype=torch.long)
        ti = torch.zeros(P, T, dtype=torch.long)
        tj = torch.zeros(P, T, dtype=torch.long)
        tvalid = torch.zeros(P, T)
        for pid, s in enumerate(specs_by_pid):
            if s is None:
                continue
            k, t, m = s["mask"].shape[0], s["trips"].shape[0], s["mask"].shape[1]
            mask[pid, :k, :m] = torch.as_tensor(s["mask"], dtype=torch.float32)
            m2[pid, :k] = torch.as_tensor(s["m2"], dtype=torch.float32)
            kvalid[pid, :k] = 1.0
            if F and "pfeat" in s:
                pfeat[pid, :k] = torch.as_tensor(s["pfeat"], dtype=torch.float32)
            trips = torch.as_tensor(s["trips"], dtype=torch.long)
            tk[pid, :t], ti[pid, :t], tj[pid, :t] = trips[:, 0], trips[:, 1], trips[:, 2]
            tvalid[pid, :t] = 1.0
        # regenerable statics (not nn buffers) + device-migration flag, like _diag_batch
        self._pb_static = {"mask": mask, "m2": m2, "kvalid": kvalid,
                           "tk": tk, "ti": ti, "tj": tj, "tvalid": tvalid}
        if pfeat is not None:
            self._pb_static["pfeat"] = pfeat
        self._pb_device = None
        self._pb_num_heads = int(num_heads)
        self._pb_clamp = float(clamp)
        self._pb_mom_div = float(mom_div)
        self._pb_logit_cap = float(logit_cap)
        self._pb_in_dim = 1 + F
        self.pair_bias_head = nn.Sequential(
            nn.Linear(1 + F, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, num_heads, bias=True),
        )
        nn.init.xavier_uniform_(self.pair_bias_head[0].weight)
        nn.init.zeros_(self.pair_bias_head[0].bias)
        nn.init.zeros_(self.pair_bias_head[2].weight)   # zero-init: exact baseline at t=0
        nn.init.zeros_(self.pair_bias_head[2].bias)
        # calibration buffers (persisted in the checkpoint, same pattern as _virt_*)
        self.register_buffer("_pb_mu", torch.zeros(()))
        self.register_buffer("_pb_sigma", torch.ones(()))
        self.register_buffer("_pb_calibrated", torch.zeros((), dtype=torch.bool))
        self._pb_cal_done = False
        self.use_pair_bias = True

    def _pair_bias(self, fourmomenta, process_ids, ctx):
        """Per-head padded attention bias (B, H, n_max, n_max) for this batch.

        Fully batched over processes: static per-process tensors are gathered by
        process_ids (no per-pid Python loop), the per-propagator off-shellness is one
        einsum + Minkowski norm, and the per-head scores are scatter-added into the
        padded pair grid via precomputed (k, i, j) triples. Processes without a
        sidecar have all-zero/invalid static rows and contribute exactly zero bias.
        """
        st = self._pb_static
        device, dtype = fourmomenta.device, fourmomenta.dtype
        if self._pb_device != device:
            self._pb_static = st = {k: v.to(device) for k, v in st.items()}
            self._pb_device = device
        B, n_max, H = ctx["B"], ctx["n_max"], self._pb_num_heads

        # padded per-event momenta via the ctx index maps (dummy zero row at index N)
        momp = torch.cat([fourmomenta, fourmomenta.new_zeros(1, 4)], dim=0)
        momp = momp[ctx["pad2flat"]].reshape(B, n_max, 4)

        mask = st["mask"][process_ids]                        # (B, K, n_slot)
        m = min(mask.shape[-1], n_max)   # procs in batch have n_part <= n_max; extra cols are 0
        p_prop = torch.einsum("bkn,bnc->bkc", mask[..., :m], momp[:, :m])   # (B,K,4)
        s = p_prop[..., 0] ** 2 - (p_prop[..., 1:] ** 2).sum(-1)
        s = s * (self._pb_mom_div ** 2) - st["m2"][process_ids]             # physical s − m²
        raw = torch.sign(s) * torch.log1p(s.abs())                          # signed-log
        kvalid = st["kvalid"][process_ids]                                  # (B,K)

        if not self._pb_cal_done:                # first-batch standardization stats
            if not bool(self._pb_calibrated):
                sel = raw.detach()[kvalid > 0]
                if sel.numel() > 1:
                    self._pb_mu.copy_(sel.mean())
                    self._pb_sigma.copy_(sel.std().clamp_min(1e-3))
                    self._pb_calibrated.fill_(True)
            self._pb_cal_done = True

        feat = ((raw - self._pb_mu) / self._pb_sigma).clamp(
            -self._pb_clamp, self._pb_clamp).to(dtype).unsqueeze(-1)        # (B,K,1)
        if "pfeat" in st:   # propagator identity (log-mass, massless flag): the same
            # off-shellness means different physics for a γ vs Z vs top propagator
            feat = torch.cat([feat, st["pfeat"][process_ids].to(dtype)], dim=-1)
        scores = self.pair_bias_head(feat)                                  # (B,K,H)
        cap = self._pb_logit_cap
        if cap > 0:   # soft cap: linear for small scores, saturates at ±cap instead
            # of the unbounded hard-mask logits (−17) the uncapped v1 arm learned
            scores = cap * torch.tanh(scores / cap)
        scores = scores * kvalid.unsqueeze(-1)

        tk, tv = st["tk"][process_ids], st["tvalid"][process_ids]           # (B,T)
        vals = scores.gather(1, tk.unsqueeze(-1).expand(-1, -1, H))         # (B,T,H)
        vals = vals * tv.unsqueeze(-1)
        idx = st["ti"][process_ids] * n_max + st["tj"][process_ids]         # (B,T)
        bias = scores.new_zeros(B, n_max * n_max, H)
        bias.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, H), vals)
        return bias.reshape(B, n_max, n_max, H).permute(0, 3, 1, 2)         # (B,H,n,n)

    def setup_diagram_virtuality(self, virt_by_pid, log_scale=0.1,
                                 standardize=True, clamp=4.0, mode="edge", mom_div=1.0):
        """Enable Tier B: per-event propagator virtualities (call after
        setup_diagram_conditioning). ``virt_by_pid`` is a list indexed by process_id
        of the precompute dicts from diagram_graphs.build_process_virtuality (or None
        for processes without a usable per-event leg↔slot map — those events get a
        zero virtuality, i.e. the topology embedding through the same encoder).

        The propagator virtuality is fed as signed-log(s). Its raw distribution is
        very wide and bimodal by channel (s-channel s>0 ~ +9..+19, t-channel s<0 ~
        −8..−19), which destabilises training. With ``standardize=True`` we z-score it
        (mean 0 / std 1, calibrated once on the first training batch and stored as
        buffers so eval/reload are consistent) and clamp the tails to ``±clamp``;
        otherwise the legacy ``signed-log·log_scale`` is used.

        Precomputes each process's static graph tensors padded to the GLOBAL max node
        count so the per-step path can concatenate every (event,diagram) graph into
        ONE batched encoder call (see _diagram_features_virtuality)."""
        self._virt_by_pid = list(virt_by_pid)
        self._virt_log_scale = float(log_scale)
        self._virt_mom_div = float(mom_div)   # physical scale: model momenta are /mom_div
        _pm2 = [vt.get("prop_mass2") for vt in self._virt_by_pid if vt is not None and vt.get("prop_mass2") is not None]
        if _pm2:
            import logging as _lg
            _lg.getLogger("lorentz-gatr").info(
                f"[diag off-shell] mom_div={self._virt_mom_div:.3f}  "
                f"prop masses(GeV)={[round(float(m)**0.5,1) for m in _pm2[0].tolist()]}")
        self._virt_standardize = bool(standardize)
        self._virt_clamp = float(clamp)
        self._virt_mode = str(mode)        # "edge" (per-event encode) | "pool" (cached topology + weighted)
        self.use_diagram_virtuality = True
        # Per-process diagram slice in the concatenated static batch (build_diagram_batch
        # iterates pids in order over non-None pds) — used by the "pool" mode to map
        # events to their process's diagram CLS embeddings.
        offs, cur = {}, 0
        for pid, pd in enumerate(self._pd_by_pid):
            if pd is not None:
                offs[pid] = (cur, pd.n_diagrams); cur += pd.n_diagrams
        self._diag_offsets = offs
        # calibration buffers (persisted in the checkpoint); _virt_cal_done mirrors
        # _virt_calibrated as a Python flag to avoid a GPU→CPU sync after calibration.
        self.register_buffer("_virt_mu", torch.zeros(()))
        self.register_buffer("_virt_sigma", torch.ones(()))
        self.register_buffer("_virt_calibrated", torch.zeros((), dtype=torch.bool))
        self._virt_cal_done = False

        Mstar = max(pd.node_feat.shape[1] for pd in self._pd_by_pid if pd is not None)
        self._virt_Mstar = Mstar
        static = []
        for pd in self._pd_by_pid:
            if pd is None:
                static.append(None); continue
            D, N = pd.node_feat.shape[0], pd.node_feat.shape[1]
            nf = pd.node_feat.new_zeros(D, Mstar, pd.node_feat.shape[-1]); nf[:, :N] = pd.node_feat
            lp = pd.lap_pe.new_zeros(D, Mstar, pd.lap_pe.shape[-1]);       lp[:, :N] = pd.lap_pe
            nm = pd.node_mask.new_zeros(D, Mstar);                         nm[:, :N] = pd.node_mask
            ef = pd.edge_feat.new_zeros(D, Mstar, Mstar, pd.edge_feat.shape[-1]); ef[:, :N, :N] = pd.edge_feat
            em = pd.edge_mask.new_zeros(D, Mstar, Mstar);                  em[:, :N, :N] = pd.edge_mask
            static.append({"node_feat": nf, "lap_pe": lp, "node_mask": nm,
                           "edge_feat": ef, "edge_mask": em, "D": D})
        self._virt_static = static
        self._pd_device = None   # force re-move of the new tensors

    def _diagram_features_virtuality(self, fourmomenta, std, process_ids, ptr, device, dtype):
        """Tier B per-particle diagram embedding (N_total, d_diag), batched.

        Builds every (event, diagram) graph in the batch — static topology tensors
        tiled per event, plus the event's per-propagator virtuality s=(Σ sign·p)²
        (all-outgoing; signed-log compressed) as the encoder's edge feature — into ONE
        flat batch, then a SINGLE encode_grouped call segment-pools per event. The
        per-process loop only does cheap tensor ops (gather/einsum/scatter/tile); the
        expensive graph-transformer forward runs once on the whole batch (vs one
        launch-bound forward_per_event per process). A process with no per-event map
        gets zero virtuality (topology embedding through the same encoder); a process
        with no diagrams leaves its events at zero.
        """
        if self._pd_device != device:
            self._virt_by_pid = [
                ({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vt.items()}
                 if vt is not None else None) for vt in self._virt_by_pid]
            self._virt_static = [
                ({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in st.items()}
                 if st is not None else None) for st in self._virt_static]
            self._pd_device = device

        B = process_ids.shape[0]
        M = self._virt_Mstar

        # --- pass 1: per-process raw signed-log virtuality (cheap einsum, no encoder) ---
        jobs, calib = [], []
        for pid in torch.unique(process_ids).tolist():
            st = self._virt_static[pid] if 0 <= pid < len(self._virt_static) else None
            if st is None:
                continue
            ev_idx = (process_ids == pid).nonzero(as_tuple=True)[0]      # (E_p,)
            vt = self._virt_by_pid[pid]
            raw = None
            if vt is not None:
                n_part = vt["n_part"]
                gather = ptr[ev_idx][:, None] + torch.arange(n_part, device=device)[None, :]
                mom = fourmomenta[gather]                               # (E_p, n_part, 4)
                p_prop = torch.einsum("kn,enc->ekc", vt["mask"], mom)   # (E_p, K, 4)
                s = p_prop[..., 0] ** 2 - (p_prop[..., 1:] ** 2).sum(-1)
                # Off-shellness s − m² (the propagator pole variable 1/(s−m²)),
                # rather than raw s: converts s to physical GeV² (momenta are /mom_div)
                # and subtracts each propagator's (scanned) mass². Massless props (m=0)
                # → unchanged. This is the resonance-aware Tier-B feature.
                pm2 = vt.get("prop_mass2")
                if pm2 is not None:
                    s = s * (self._virt_mom_div ** 2) - pm2[None, :]
                raw = torch.sign(s) * torch.log1p(s.abs())             # (E_p, K) signed-log
                calib.append(raw.detach().reshape(-1))
            jobs.append((ev_idx, st, vt, raw))

        # --- calibrate standardization stats once (first batch); persisted as buffers ---
        if self._virt_standardize and not self._virt_cal_done:
            if not bool(self._virt_calibrated) and calib:    # sync only until calibrated
                allraw = torch.cat(calib)
                self._virt_mu.copy_(allraw.mean())
                self._virt_sigma.copy_(allraw.std().clamp_min(1e-3))
                self._virt_calibrated.fill_(True)
            self._virt_cal_done = True

        # --- pass 2: standardize+clamp the virtuality, scatter, tile static graphs ---
        nf_l, lp_l, nm_l, ef_l, em_l, ee_l, seg_l = [], [], [], [], [], [], []
        for (ev_idx, st, vt, raw) in jobs:
            E_p, D_p = ev_idx.shape[0], st["D"]
            ee = torch.zeros(E_p, D_p, M, M, 1, device=device, dtype=dtype)
            if vt is not None:
                if self._virt_standardize:
                    feat = ((raw - self._virt_mu) / self._virt_sigma).clamp(
                        -self._virt_clamp, self._virt_clamp)
                else:
                    feat = raw * self._virt_log_scale
                feat = feat.to(dtype)
                ee[:, vt["diag_idx"], vt["i_idx"], vt["j_idx"], 0] = feat
                ee[:, vt["diag_idx"], vt["j_idx"], vt["i_idx"], 0] = feat   # symmetric

            def tile(t):   # (D_p, ...) -> (E_p*D_p, ...) event-major, sharing static graph
                return t.unsqueeze(0).expand(E_p, *t.shape).reshape(E_p * D_p, *t.shape[1:])
            nf_l.append(tile(st["node_feat"])); lp_l.append(tile(st["lap_pe"]))
            nm_l.append(tile(st["node_mask"])); ef_l.append(tile(st["edge_feat"]))
            em_l.append(tile(st["edge_mask"])); ee_l.append(ee.reshape(E_p * D_p, M, M, 1))
            seg_l.append(ev_idx.repeat_interleave(D_p))                 # graph -> event id

        if not seg_l:
            E_event = torch.zeros(B, self.d_diag, device=device, dtype=dtype)
        else:
            E_event = self.diagram_encoder.encode_grouped(
                torch.cat(nf_l), torch.cat(lp_l), torch.cat(nm_l),
                torch.cat(ef_l), torch.cat(em_l), torch.cat(seg_l), B,
                edge_extra=torch.cat(ee_l)).to(dtype)                   # (B, d_diag)

        counts = ptr[1:] - ptr[:-1]
        return E_event.repeat_interleave(counts, dim=0)

    def _diagram_features_virtuality_pool(self, fourmomenta, std, process_ids, ptr, device, dtype):
        """Tier B, "pool" mode (the structural speedup): encode every diagram's
        TOPOLOGY once (static, ~#diagrams graphs/step, like Tier A), then let the
        per-event virtuality enter ONLY the pooling weights — a learned ``virt_score``
        of each propagator's virtuality up/down-weights its diagram (near-resonant
        s≈m² → larger weight). The expensive graph transformer no longer runs per
        event, so the cost is ~Tier A's. A different model from "edge" mode (the
        virtuality doesn't change a diagram's internal representation), so it is A/B'd
        separately."""
        if self._pd_device != device:
            self._diag_batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                                for k, v in self._diag_batch.items()} if self._diag_batch else None
            self._virt_by_pid = [
                ({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vt.items()}
                 if vt is not None else None) for vt in self._virt_by_pid]
            self._pd_device = device

        enc = self.diagram_encoder
        cls_all = enc.encode_static(self._diag_batch)                   # (TotalD, d) — ONE forward
        base_all = (cls_all @ enc.pool_query) / (enc.d_model ** 0.5)    # (TotalD,)
        B = process_ids.shape[0]

        # pass 1: raw signed-log virtuality per process (for calibration)
        jobs, calib = [], []
        for pid in torch.unique(process_ids).tolist():
            if pid not in self._diag_offsets:
                continue
            ev_idx = (process_ids == pid).nonzero(as_tuple=True)[0]
            vt = self._virt_by_pid[pid]
            raw = None
            if vt is not None:
                n_part = vt["n_part"]
                gather = ptr[ev_idx][:, None] + torch.arange(n_part, device=device)[None, :]
                p_prop = torch.einsum("kn,enc->ekc", vt["mask"], fourmomenta[gather])
                s = p_prop[..., 0] ** 2 - (p_prop[..., 1:] ** 2).sum(-1)
                raw = torch.sign(s) * torch.log1p(s.abs())             # (E_p, K)
                calib.append(raw.detach().reshape(-1))
            jobs.append((pid, ev_idx, vt, raw))

        if self._virt_standardize and not self._virt_cal_done:
            if not bool(self._virt_calibrated) and calib:
                allraw = torch.cat(calib)
                self._virt_mu.copy_(allraw.mean())
                self._virt_sigma.copy_(allraw.std().clamp_min(1e-3))
                self._virt_calibrated.fill_(True)
            self._virt_cal_done = True

        # pass 2: per-event virtuality-weighted pool over the process's diagrams
        E_event = torch.zeros(B, self.d_diag, device=device, dtype=dtype)
        for (pid, ev_idx, vt, raw) in jobs:
            start, D_p = self._diag_offsets[pid]
            cls_p = cls_all[start:start + D_p]                          # (D_p, d)
            base_p = base_all[start:start + D_p]                        # (D_p,)
            E_p = ev_idx.shape[0]
            if vt is not None:
                if self._virt_standardize:
                    feat = ((raw - self._virt_mu) / self._virt_sigma).clamp(
                        -self._virt_clamp, self._virt_clamp)
                else:
                    feat = raw * self._virt_log_scale
                # [signed-log(s)] ++ propagator property (incl. mass) -> resonance-aware
                ep = vt["edge_prop"].to(feat.dtype).unsqueeze(0).expand(E_p, -1, -1)  # (E_p,K,f_edge)
                vin = torch.cat([feat.unsqueeze(-1), ep], dim=-1)       # (E_p, K, 1+f_edge)
                score = enc.virt_score(vin).squeeze(-1)                # (E_p, K)
                contrib = torch.zeros(E_p, D_p, device=device, dtype=score.dtype)
                contrib.index_add_(1, vt["diag_idx"], score)           # (E_p, D_p)
            else:
                contrib = torch.zeros(E_p, D_p, device=device, dtype=base_p.dtype)
            w = torch.softmax(base_p.unsqueeze(0) + contrib, dim=1)     # (E_p, D_p)
            E_event[ev_idx] = enc.out(w @ cls_p).to(dtype)             # (E_p, d_diag)

        counts = ptr[1:] - ptr[:-1]
        return E_event.repeat_interleave(counts, dim=0)

    def _diagram_features(self, process_ids, ptr, device, dtype):
        """Per-particle diagram embedding (N_total, d_diag).

        Encodes ALL processes' (static) diagram sets in a single batched encoder
        forward → E_all (n_proc, d_diag), gathers per event via process_ids, then
        broadcasts to particles — the order_labels broadcast pattern. The static
        topology embedding only depends on the encoder weights (not the batch), so
        encoding all processes once and gathering is both correct and far cheaper
        than a per-process Python loop (no GPU→CPU unique() sync, one big GPU call).
        """
        if self._diag_batch is None:                                       # no sidecars
            return torch.zeros(int(ptr[-1]), self.d_diag, device=device, dtype=dtype)
        # Move the static batch onto the compute device once (regenerable data, not
        # nn buffers, so it doesn't auto-migrate with .to()).
        if self._pd_device != device:
            self._diag_batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                                for k, v in self._diag_batch.items()}
            self._pd_device = device

        E_all = self.diagram_encoder.encode_all(self._diag_batch).to(dtype)  # (n_proc, d_diag)
        E_event = E_all[process_ids]                                       # (B, d_diag)
        counts = ptr[1:] - ptr[:-1]                                        # (B,)
        return E_event.repeat_interleave(counts, dim=0)                    # (N_total, d_diag)

    def _sign_pair_logit(self, h, p, ptr, seq_lens=None):
        """The pair term of the sign logit, (B, 1): the mean over each event's particle pairs i < j of
        sign_pair_mlp(h_i + h_j, h_i h_j, log(2|p_i.p_j|/s_hat)/5, (a_i + a_j)/10, a_i a_j/100, log s_hat/10), a_i =
        log(|p_i^2|/s_hat) (clamped at 1e-6, the massless legs, as the pair term), s_hat = (p_0 + p_1)^2 of the two beams. The inputs of
        the per-pool sign classifier (prep_signed_pools.py), all Lorentz scalars; symmetric in i, j, so the head is
        permutation invariant. Events padded to the batch's largest multiplicity."""
        cnt = ptr[1:] - ptr[:-1]
        n = int(max(seq_lens)) if seq_lens is not None else int(cnt.max())
        ar = torch.arange(n, device=h.device)
        valid = ar[None, :] < cnt[:, None]                                       # (B, n)
        idx = (ptr[:-1, None] + ar[None, :]).clamp(max=h.shape[0] - 1)
        H = h[idx]                                                               # (B, n, K)
        if self.sign_pair_bound:
            H = torch.tanh(H)
        P = p[idx].to(torch.float32)                                             # (B, n, 4)
        eta = torch.tensor([1.0, -1.0, -1.0, -1.0], device=h.device, dtype=P.dtype)
        dots = torch.einsum("bic,bjc,c->bij", P, P, eta)                         # (B, n, n)
        s_hat = dots[:, 0, 0] + dots[:, 1, 1] + 2 * dots[:, 0, 1]
        s_hat = s_hat.abs().clamp(min=1e-30)
        r = dots / s_hat[:, None, None]
        # floors above float32 rounding of the augmented momenta (a massless leg's p^2/s_hat is ~1e-7 noise that
        # moves with the random Lorentz draw): a leg below 1e-6 reads as massless (the b up to ~4.7 TeV resolved), a pair below 1e-6 as collinear
        a = torch.log(torch.diagonal(r, dim1=1, dim2=2).abs().clamp(min=1e-6))   # (B, n)
        iu = torch.triu_indices(n, n, 1, device=h.device)
        hi, hj = H[:, iu[0]], H[:, iu[1]]                                        # (B, M, K)
        ai, aj = a[:, iu[0]], a[:, iu[1]]
        lp = torch.log((2 * r[:, iu[0], iu[1]]).abs().clamp(min=1e-6))          # (B, M)
        ls = torch.log(s_hat)[:, None].expand_as(lp)
        x = torch.cat([hi + hj, hi * hj, torch.stack([lp / 5, (ai + aj) / 10, ai * aj / 100, ls / 10], -1).to(H.dtype)],
                      dim=-1)
        pm = (valid[:, iu[0]] & valid[:, iu[1]]).to(H.dtype)                    # (B, M)
        z = self.sign_pair_mlp(x).squeeze(-1) * pm
        return (z.sum(1) / pm.sum(1).clamp(min=1)).unsqueeze(-1)

    def forward(self, fourmomenta, particle_type_indices, mean, std, ptr, order_labels=None,
                seq_lens=None, process_ids=None):
        """
        fourmomenta           : (N_total, 4)
        particle_type_indices : (N_total,)              long
        mean, std             : float
        ptr                   : (B+1,)                  event boundaries
        order_labels          : (B, n_order_features)   coupling-order vector per event,
                                e.g. [n_loops, alpha_s_power]. Broadcast to each particle.
        seq_lens              : tuple | None            CPU per-event particle counts;
                                forwarded to the attention-mask builder to skip a
                                GPU→CPU sync (see build_block_diagonal_bias).
        """
        ev_lrs = None
        assert not (self.input_log_skip and self.use_pids), "model.input_log_skip reads the property encoding"
        assert not (self.input_asym_factor and self.use_pids), "model.input_asym_factor reads the property table"
        loops = order_labels[:, 0] if order_labels is not None else None     # L_QCD of the target (experiment._order_row)
        if self.use_pids:
            particle_type = torch.nn.functional.one_hot(
                particle_type_indices, num_classes=self.token_size
            ).to(dtype=self.network_dtype, device=fourmomenta.device)
            if getattr(self, "dimless_pids", False):
                ev_lrs, order_labels = order_labels[:, -1], order_labels[:, :-1]
        else:
            raw_props = self.property_matrix[particle_type_indices]   # (N_total, n_features)
            if self.mass_from_momenta:
                raw_props = self._apply_mass_from_momenta(raw_props, fourmomenta)
            if getattr(self, "dimless_mass", False):
                ev_lrs = order_labels[:, -1]                           # log10 sqrt(s_hat) per event, GeV
                raw_props, order_labels = self._apply_dimless_mass(raw_props, particle_type_indices, ptr, order_labels)
            particle_type = self.particle_encoder(raw_props)           # (N_total, d_particle_hidden)
            if self.input_log_skip:   # coefficients from the static property table, never the per-event mass column
                skip_types = self.particle_encoder(self.property_matrix[particle_type_indices])
            if self.context is not None:
                ctx = self._context_per_particle(process_ids, ptr, particle_type.dtype)
                if ctx is not None:
                    particle_type = particle_type + ctx

        if order_labels is not None:
            # Broadcast each event's coupling-order vector to all its particles.
            # repeat_interleave is vectorised: no Python loop over events.
            counts = ptr[1:] - ptr[:-1]                                # (B,) particles per event
            order_per_particle = order_labels.repeat_interleave(counts, dim=0)  # (N_total, n_order_features)
            particle_type = torch.cat(
                [particle_type, order_per_particle.to(particle_type.dtype)], dim=-1
            )

        # Feynman-diagram conditioning: append the per-process diagram embedding to
        # every particle's scalar features (broadcast like order_labels). E(process)
        # is a Lorentz scalar, so this preserves equivariance. Appended AFTER the
        # order labels so the feature layout is [property | order | diagram], matching
        # the in_channels/num_scalars sizing in experiment._finalize_data_sizing.
        token_add = None
        if self.use_diagrams and process_ids is not None and self.diagram_encoder_impl == "llm4lhc":
            from models.diagram_encoder_llm4lhc import process_token_add
            enc = self.diagram_encoder
            if getattr(enc, "virtuality_pool", False):
                # per-event diagram weighting: s_hat in GeV^2 for the masses, from the hidden log10 sqrt(s_hat)
                # column (dimensionless inputs) or from the model's momenta times the global scale
                if ev_lrs is not None:
                    s_hat_phys = 10.0 ** (2.0 * ev_lrs.to(torch.float64))
                else:
                    p_in = (fourmomenta[ptr[:-1]] + fourmomenta[ptr[:-1] + 1]).to(torch.float64)
                    s_hat_phys = (p_in[:, 0] ** 2 - (p_in[:, 1:] ** 2).sum(-1)) * float(enc.mom_div) ** 2
                token_add = process_token_add(enc.forward_events(fourmomenta, ptr, process_ids, s_hat_phys),
                                              None, ptr, fourmomenta.shape[0], enc.out_dim)
            else:
                token_add = process_token_add(enc(), process_ids, ptr, fourmomenta.shape[0],
                                              enc.out_dim)                     # (N_total, width)
        elif self.use_diagrams and process_ids is not None:
            if self.use_diagram_virtuality:
                feat_fn = (self._diagram_features_virtuality_pool
                           if getattr(self, "_virt_mode", "edge") == "pool"
                           else self._diagram_features_virtuality)
                diag_feat = feat_fn(fourmomenta, std, process_ids, ptr,
                                    particle_type.device, particle_type.dtype)
            else:
                diag_feat = self._diagram_features(
                    process_ids, ptr, particle_type.device, particle_type.dtype)
            particle_type = torch.cat([particle_type, diag_feat], dim=-1)

        # Pairwise diagram-derived attention bias: build the padded-layout context
        # (index maps + key-padding mask) once per forward, then the per-pair bias.
        # The bias is a function of Lorentz scalars (s − m² per propagator), so
        # equivariance is preserved.
        pair_ctx = None
        if self.use_pair_bias and process_ids is not None:
            from models.attention_lloca_mup import build_pair_ctx
            pair_ctx = build_pair_ctx(ptr, fourmomenta.shape[0], seq_lens=seq_lens)
            pair_ctx["bias"] = self._pair_bias(fourmomenta, process_ids, pair_ctx)

        if self.input_pair_logs:   # appended last: layout [property | order | diagram | pair logs]
            particle_type = torch.cat([particle_type, _pair_log_inputs(
                fourmomenta, ptr, self.input_pair_logs, self.input_pair_threshold).to(particle_type.dtype)], dim=-1)
        if self.input_div_form:    # appended after the pair logs
            from particle_ids import GLOBAL_PROPERTY_MATRIX as _G
            G = torch.as_tensor(_G, device=fourmomenta.device)
            particle_type = torch.cat([particle_type, _div_form_inputs(
                fourmomenta, ptr, self.input_pair_logs, G[particle_type_indices, 7],
                G[particle_type_indices, 2] >= 0).to(particle_type.dtype)], dim=-1)
        fm_net = _soft_rescale(fourmomenta, ptr, self.input_soft_eps) if self.input_soft_eps else fourmomenta

        net_kwargs = {} if token_add is None else {"token_add": token_add}
        outputs = self.net(fm_net, particle_type, mean, std, ptr=ptr,
                           seq_lens=seq_lens, pair_ctx=pair_ctx, **net_kwargs)

        if self.input_asym_factor:   # rows [mu, alpha, sigma]: the per-particle coefficient leaves before the pool
            k0 = self.net.out_shape
            alpha, outputs = outputs[:, k0], torch.cat([outputs[:, :k0], outputs[:, k0 + 1:]], dim=-1)
        pair_feats = None
        if self.sign_pair_dim > 0:
            pair_feats, outputs = outputs[:, -self.sign_pair_dim:], outputs[:, :-self.sign_pair_dim]
        # Per-event mean pool (vectorised; see _pool_events / LLOCA_POOL env toggle)
        pooled = _pool_events(outputs, ptr)
        if pair_feats is not None:
            # the last pooled channel is the sign logit (training.sign_head)
            pooled = torch.cat([pooled[:, :-1], pooled[:, -1:] + self._sign_pair_logit(
                pair_feats, fourmomenta, ptr, seq_lens).to(pooled.dtype)], dim=-1)

        if self.input_log_skip:
            skip = _log_skip(fourmomenta, ptr, skip_types, self.input_pair_logs, self.log_skip_head,
                             self.input_pair_threshold, not self.input_log_skip_linear).to(pooled.dtype)
            pooled = torch.cat([pooled[:, :1] + skip[:, None], pooled[:, 1:]], dim=-1)   # mu is column 0
        if self.input_asym_factor:
            from particle_ids import GLOBAL_PROPERTY_MATRIX as _G
            G = torch.as_tensor(_G, device=fourmomenta.device)
            cnt = ptr[1:] - ptr[:-1]
            n = int(max(seq_lens)) if seq_lens is not None else int(cnt.max())
            logD, on, mem = _asym_terms(fourmomenta, ptr, n, G[particle_type_indices, 7],
                                        G[particle_type_indices, 2] >= 0, loops, self.asym_kappa)
            mu = _asym_mu(pooled[:, 0], alpha, ptr, n, logD, on, mem, self.asym_sd)
            pooled = torch.cat([mu[:, None].to(pooled.dtype), pooled[:, 1:]], dim=-1)

        # HETEROSC + sigma_after_pool: the net handed us a RAW per-particle sigma logit, so the
        # positivity nonlinearity is applied here — ONCE, to the pooled per-event value:
        #     sigma = softplus(mean_i z_i)      (matches the reference: softplus on a per-event readout)
        # rather than the old  sigma = mean_i softplus(z_i), which compresses sigma's dynamic range
        # (see models/lloca.py forward). mu is untouched: pooling is linear, so it commutes with
        # its own readout either way.
        if getattr(self.net, "loss", None) == "HETEROSC" and getattr(self.net, "sigma_after_pool", False):
            k = self.net.out_shape
            mu = pooled[..., :-k]          # mu, and the sign logit under sign_channel (rows [mu, sign, sigma])
            sigma = torch.clamp(torch.nn.functional.softplus(pooled[..., -k:]), min=1e-6)
            pooled = torch.cat([mu, sigma], dim=-1)
        return pooled


class _AmplitudeGATrMuPBase(nn.Module):
    """Shared scaffolding for the μP L-GATr amplitude wrappers.

    Matches the AmplitudeLLoCaWrapper interface (same forward signature and
    ``setup_particle_features``) so the experiment treats these models exactly like
    the LLoCa transformer: sparse flat 4-momenta + ``ptr`` event boundaries, per-event
    mean pooling, and the fixed-property particle encoding. Subclasses implement
    ``_run_net`` (embed 4-momenta, call the geometric net, return a scalar per particle).
    """

    def __init__(self, net, token_size, d_particle_hidden: int = 16):
        super().__init__()
        self.net = net
        self.network_dtype = torch.float32
        self.token_size = token_size
        self.d_particle_hidden = d_particle_hidden
        self.use_pids = True
        self.mass_from_momenta = False
        self._mom_div = 1.0
        self._mass_spec = None

    # identical to AmplitudeLLoCaWrapper.setup_mass_from_momenta / _apply_mass_from_momenta
    setup_mass_from_momenta = AmplitudeLLoCaWrapper.setup_mass_from_momenta
    _apply_mass_from_momenta = AmplitudeLLoCaWrapper._apply_mass_from_momenta
    setup_dimless_mass = AmplitudeLLoCaWrapper.setup_dimless_mass
    _apply_dimless_mass = AmplitudeLLoCaWrapper._apply_dimless_mass

    # identical to AmplitudeLLoCaWrapper.setup_particle_features
    def setup_particle_features(self, use_pids: bool, property_matrix=None,
                                encoder_hidden: int = 0):
        self.use_pids = use_pids
        if not use_pids:
            assert property_matrix is not None
            t = torch.tensor(property_matrix, dtype=torch.float32)
            self.register_buffer("property_matrix", t)
            self._register_load_state_dict_pre_hook(_property_matrix_prefix_hook(self))
            n_features = property_matrix.shape[1]
            self.particle_encoder = _build_particle_encoder(
                n_features, self.d_particle_hidden, encoder_hidden
            )

    def _scalars(self, particle_type_indices, fourmomenta, ptr, order_labels):
        if self.use_pids:
            scalars = torch.nn.functional.one_hot(
                particle_type_indices, num_classes=self.token_size
            ).to(dtype=self.network_dtype, device=fourmomenta.device)
        else:
            raw_props = self.property_matrix[particle_type_indices]
            if self.mass_from_momenta:
                raw_props = self._apply_mass_from_momenta(raw_props, fourmomenta)
            if getattr(self, "dimless_mass", False):
                raw_props, order_labels = self._apply_dimless_mass(raw_props, particle_type_indices, ptr, order_labels)
            scalars = self.particle_encoder(raw_props)
        if order_labels is not None:
            counts = ptr[1:] - ptr[:-1]
            order_per_particle = order_labels.repeat_interleave(counts, dim=0)
            scalars = torch.cat([scalars, order_per_particle.to(scalars.dtype)], dim=-1)
        return scalars

    def forward(self, fourmomenta, particle_type_indices, mean, std, ptr,
                order_labels=None, seq_lens=None):
        scalars = self._scalars(particle_type_indices, fourmomenta, ptr, order_labels)
        # Scale by std only (a Lorentz-invariant rescaling); do NOT subtract the mean,
        # which would break equivariance. (The 4-momenta are already preprocessed.)
        fm = fourmomenta / std
        # Prefer the xformers block-diagonal kernel (memory-efficient, O(sum n_i^2)) —
        # passed as `attn_bias`, which lgatr's backend dispatcher routes to xformers
        # `memory_efficient_attention` (same kernel LLoCa uses). The dense O(N_total^2)
        # `attn_mask` path materialises the full cross-event score matrix and OOMs at
        # batchsize 1024. μP's 1/d query pre-scaling is unaffected: it wraps the
        # `scaled_dot_product_attention` dispatcher *above* backend selection, and
        # xformers defaults to the same 1/sqrt(d) scale as native SDPA.
        # `build_block_diagonal_bias` returns None when xformers is unavailable
        # (e.g. login-node CPU) -> fall back to the dense mask + native attention.
        attn_bias = build_block_diagonal_bias(ptr, seq_lens)
        if attn_bias is not None:
            amp_per_particle = self._run_net(fm, scalars, attn_bias=attn_bias)
        else:
            attn_mask = _block_diagonal_attn_mask(ptr, fourmomenta.device)
            amp_per_particle = self._run_net(fm, scalars, attn_mask=attn_mask)
        return _pool_events(amp_per_particle.unsqueeze(-1), ptr)  # (B, 1)


class AmplitudeLGATrMuPWrapper(_AmplitudeGATrMuPBase):
    """μP full L-GATr: 4-momentum -> one multivector, scalar amplitude via extract_scalar."""

    def _run_net(self, fm, scalars, **attn_kwargs):
        mv = embed_vector(fm.unsqueeze(0).unsqueeze(-2))  # (1, N, 1, 16)
        s = scalars.unsqueeze(0)                          # (1, N, n_scalars)
        out_mv, _ = self.net(mv, scalars=s, **attn_kwargs)
        return extract_scalar(out_mv)[0, :, 0, 0]         # (N,)


class AmplitudeLGATrSlimMuPWrapper(_AmplitudeGATrMuPBase):
    """μP L-GATr-slim: 4-momentum -> one vector, scalar amplitude from the scalar output."""

    def _run_net(self, fm, scalars, **attn_kwargs):
        vectors = fm.unsqueeze(0).unsqueeze(-2)           # (1, N, 1, 4)  (v_channels=1)
        s = scalars.unsqueeze(0)                          # (1, N, n_scalars)
        _, out_s = self.net(vectors, s, **attn_kwargs)
        return out_s[0, :, 0]                             # (N,)
