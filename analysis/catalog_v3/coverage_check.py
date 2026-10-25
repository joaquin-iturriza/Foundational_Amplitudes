"""The catalog's coverage (D127, CPU only): what replacing the fiducial cuts by analysis-level acceptance, with the
forward region oversampled, does to every catalog_v3 tree class, before any pool is regenerated.

The cuts every catalog_v3 pool was generated under (mg5_pipeline_final.FIDUCIAL_CUTS, in the partonic rest frame, on
the massless visible finals): pT > 10 GeV, |cos theta| < 0.9 (|eta| < 1.47), Delta R > 0.4, m_ij > 10 GeV. The coverage
proposed to the user (2026-10-25):
  hadronic classes   pT > 10 GeV, |eta| < 5 (|cos theta| < tanh 5), Delta R > 0.4, no pair mass cut: the 13 TeV pp
                     run card's acceptance (recipes/catalog_v3_pp/run_card.dat: pT > 10 GeV, |eta| < 5 on jets, b,
                     photons and leptons; MadEvent's default Delta R > 0.4), read in the partonic frame
  e+e- classes       |cos theta| < 0.99, the other three cuts as now
  b and tau          cut like the other legs (b like a jet, tau like a lepton): the current mask keys on m_final == 0,
                     so the 4.7 GeV b and the 1.777 GeV tau are never cut (481 of the 1586 classes have one, 165 of them
                     no other cut leg, so no cut at all)
  forward component  a share of the events whose angle to the beam is drawn uniform in log(1/(1 - |cos theta|)) up to
                     the coverage's edge: an event is drawn as now, then its final state is rotated (a rotation in the
                     partonic frame keeps momentum conservation and every mass shell) so that one final particle,
                     picked at random, points at that angle. Nothing here knows which process has a t-channel pole.
The mixture policy is otherwise the one the pools were generated with (mg5_pipeline_final.DEFAULT_SAMPLING, catalog_v3's
recipes: mode mixture): a bulk uniform in sqrt(s_hat) and a part kept flat in log|M|^2 from oversampled candidates
(half of them from the IR democratic splitter). The proposed mixture takes FWD_SHARE of the events from the forward
component (out of the bulk) and rotates FWD_CAND of the flat part's candidates, so the flat selection can reach the
forward extremes too. With both shares zero and the current cuts, build() is mg5_pipeline_final.build_mixture_dataset
event for event (checked at the start of `run`).

Three samples per class: `current` (the pools' cuts and measure), `proposed` (the coverage above, the forward
component, sqrt(s_hat) uniform on the class's window as now) and `wide` (the proposed coverage with sqrt(s_hat)
log uniform on [window bottom, max(SQRTS_WIDE, window top)]: what a pp-like falling measure reaching several TeV changes).

The check's own choices, none of them the user's (open for the user; what moves with each):
  b and tau cut (HEAVY_CUT)    the proposed samples of the 481 classes with a b or tau final, their range, the b and tau
                               shares below, the pp share inside the proposed coverage (in_new vs in_new_b_tau_uncut)
  |cos theta| < 0.99 on e+e-   the e+e- classes' proposed samples and range
  FWD_SHARE 0.15, FWD_CAND 1/3 the forward shares, the forward cover of the pp events (u_overlap, the forward ESS and
                               uncovered share), the share outside the current cuts and the range growth at the top
  SQRTS_WIDE 6 TeV             the wide sample and its range, the pp shares up to it (in_new_wide, above_wide) and the
                               *_wide measures' ESS
  the sqrt(s_hat)^-3 measure   its ESS only (lumi_wide)
  the harder-cut probes        keep their own pT threshold (30 GeV) in the proposed coverage, the other cuts as proposed

Every catalog_v3 entry, in two modes:
  sampled  every tree pool generated (recipes/catalog_v3_gen_stage1, 3 and 4: the 1586 tree classes, the 105
           heavy-quark mass points, the 12 BSM entries, the 40 2->5 probes, the 24 alpha_s points, the 40 entries on a
           1-10 TeV window and the 40 at pT > 30 GeV), N events under each coverage on its own window, labelled by its
           compiled backend with alpha_s(sqrt s_hat) as in generation; the 1586 tree classes keep the first run's
           seeds, so their samples are the first run's
  read     the 150 one-loop and 17 loop-induced pools (stage 2): no new sample. Their labels come from MadLoop, which
           datagen runs one process per pool in an isolated subprocess (a process singleton that changes directory), so
           the worker pool here cannot label them, and a 2->3 one-loop event costs 100 to 200 tree units. What needs no
           new label is read: the range of the existing validation pool, and for a hadronic one-loop pool the
           kinematic coverage of its tree's pp events (current cuts and window, proposed, proposed to SQRTS_WIDE, the
           forward angle; the pp labels are the tree's, so no range comparison). The loop-induced pools have no tree
           and no pp pool: range only.
Per entry:
  range      log10|M|^2 quantiles (0.1%, 1%, 50%, 99%, 99.9%), min and max, of the current-coverage sample, the
             proposed sample and the class's existing validation pool (10000 events, $DATA_DIR); the growth is the
             proposed sample's 0.1% to 99.9% width minus the current sample's, at equal N
  new region the share of the proposed sample outside the current cuts
  b and tau  on the current sample, the share of events whose b or tau legs fail the thresholds and the 0.1% to
             99.9% width without those events, under the current thresholds (b_tau_fail_share, width_999_b_tau_cut)
             and under the proposed ones (b_tau_fail_share_proposed, width_999_b_tau_cut_proposed: what the pools now
             hold that the proposed coverage removes)
  forward    u = log(1/(1 - |cos theta|)) of the most forward final particle (any final, massive ones too: small |t|
             and beam-collinear emissions), histogrammed on U_EDGES for each sample and the pp events; the overlap
             sum_bins min(p_pp, p_sample) and the pp share and each sample's share beyond the current edge (|cos| > 0.9)
  cost       per coverage, seconds of sampling and of labelling, events labelled and drawn per kept event; the CPU hours
             of the full pools (n_train + n_val + n_test of the recipe) follow from the seconds per kept event
  pp         for a hadronic class, its 13 TeV pp validation pool (pp13_<class>, 2000 events, boosted to the partonic
             frame; for an alpha_s point, a 1-10 TeV or a pT > 30 GeV entry its base class's pool, the |M|^2 range
             comparison only where the labels are the base's, i.e. not for an alpha_s point; none for a heavy-quark mass
             point, whose top is off the pp events' shell, a BSM entry or a 2->5 probe, which have no pp pool): the share inside the current cuts, inside the proposed coverage, inside the class's sqrt(s_hat)
             window, the share each proposed cut alone removes, and the share of the pp events inside the coverage and
             window whose log|M|^2 lies inside the proposed sample's range (the current sample's for the current cuts);
             the share above the window's top and above SQRTS_WIDE, the share inside the proposed coverage on the wide
             window, and the proposed coverage with b and tau left uncut (their effect on the pp share); for the
             sqrt(s_hat) measure, the effective sample size per event (sum w)^2 / (n sum w^2) of the pp events
             reweighted from each candidate measure (uniform on the window, uniform and log uniform to SQRTS_WIDE, and
             a luminosity-like sqrt(s_hat)^-3 to SQRTS_WIDE; w = p_pp / q, p_pp a histogram of the pp events in
             log sqrt(s_hat); events outside a measure's support count as uncovered, reported apart)
Rows are appended to $SCRATCH/zz_cov_check/ROWS as classes finish (a rerun skips the classes already there).
    python analysis/catalog_v3/coverage_check.py run [--n 2000] [--workers 32]          (CPU job on CC)
    site run --item catalog_coverage ccin2p3 FA -- python analysis/catalog_v3/coverage_check.py collect \
        > analysis/catalog_v3/data/coverage_rows.json
    python3 analysis/catalog_v3/coverage_check.py                                   (medians per group, on the laptop)
"""
import argparse, json, os, sys, time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

COS_ETA5 = float(np.tanh(5.0))       # |eta| < 5 for a massless particle
COVERAGE = {
    "hadronic": {"pt_min": 10.0, "cos_max": COS_ETA5, "dr_min": 0.4, "m_min": -1.0},     # m_min < 0: no pair mass cut
    "ee":       {"pt_min": 10.0, "cos_max": 0.99, "dr_min": 0.4, "m_min": 10.0},
}
FWD_SHARE = 0.15     # share of the dataset from the forward component (taken out of the bulk)
FWD_CAND = 1.0 / 3   # share of the flat part's candidates rotated forward
LEPTON_BEAMS = {11, 13}
HEAVY_CUT = {5, 15}  # b and tau: cut in the proposed coverage like the massless visible legs
NEUTRINOS = {12, 14, 16}
SQRTS_WIDE = 6000.0  # GeV, the wide window's top
U_EDGES = np.linspace(0.0, 10.0, 21)    # u = log(1/(1-|cos theta|)); |cos| = 0.9 is u = 2.30, |eta| = 5 is u = 9.31
QS = (0.001, 0.01, 0.5, 0.99, 0.999)
OUT = lambda: os.path.join(os.environ["SCRATCH"], "zz_cov_check")
ROWS = "rows_v2.jsonl"   # the second run: every entry, the b and tau shares under both thresholds (rows.jsonl: the first)
PT_HARD = 30.0           # the harder-cut probes' own threshold, kept in their proposed coverage


def classes():
    """every catalog_v3 entry: name -> window, group, pool size, type, base and physics; the 1586 tree classes first"""
    R = json.load(open(os.path.join(ROOT, "recipes", "catalog_v3_roles.json")))
    rec = {}
    for st in (1, 2, 3, 4):
        for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v3_gen_stage{st}.yaml")))["processes"]:
            rec.setdefault(e["name"], e)
    axis = {}
    for a in ("H2p5", "H6", "H7E", "H7c"):
        for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v3_heldout_{a}.yaml")))["processes"]:
            axis[e["name"]] = a

    def typ(n, e):
        if n in R["trees"]:
            return "tree"
        if n in R["heavy_quark"]:
            return "heavy_quark"
        if n in R["loops"]:
            return "one_loop"
        if n in R["loop_induced"]:
            return "loop_induced"
        if "__as" in n:
            return "alpha_s"
        if n.endswith("__E10"):
            return "high_energy"
        if (e.get("physics") or {}).get("cuts"):
            return "harder_cut"
        if axis.get(n) == "H2p5":
            return "probe_2to5"
        return "bsm"
    out = {}
    for n in sorted(R["trees"]) + sorted(n for n in rec if n not in R["trees"]):
        e = rec[n]
        t = typ(n, e)
        r = (R["trees"].get(n) or R["heavy_quark"].get(n) or R["loops"].get(n) or R["loop_induced"].get(n) or {})
        g = "train" if r.get("role") == "train" else (r.get("axis") or axis.get(n) or "train")
        out[n] = {"sqrts": [float(x) for x in e["sqrts"]], "group": g, "type": t,
                  "n_pool": int(e["n_train"]) + int(e["n_val"]) + int(e["n_test"]),
                  "base": e.get("base", n), "physics": e.get("physics") or {},
                  "tree": (R["loops"].get(n) or {}).get("tree")}
    return out


def register(C):
    """the scan entries (heavy-quark masses, alpha_s points, the harder cuts, a base other than the name) into
    mg5_pipeline_final.PROCESSES, as generation registers them"""
    import mg5_pipeline_final as mg
    for n, info in C.items():
        if info["type"] in ("one_loop", "loop_induced"):
            continue
        if info["physics"] or info["base"] != n:
            mg.register_scan_process(n, info["base"], info["physics"])


def finals(cfg):
    """the final-state masses as generation reads them (mg5_pipeline_final's recipe rule): m_finals, else m_final (a list,
    or one mass for every final)"""
    mf = cfg.get("m_finals")
    if mf is None:
        mf = cfg.get("m_final", 0.0)
        mf = list(mf) if isinstance(mf, (list, tuple)) else [mf] * int(cfg.get("nfinal", len(cfg["pdg_ids"]) - 2))
    return [float(x) for x in mf]


def kind(cfg):
    return "ee" if abs(int(cfg["pdg_ids"][0])) in LEPTON_BEAMS else "hadronic"


def cut_slots(m_finals, pdg, heavy):
    """the rows the cuts act on: massless visible finals (the pipeline's rule), plus b and tau when heavy"""
    fin = range(2, 2 + len(m_finals))
    return [i for j, i in enumerate(fin) if abs(int(pdg[i])) not in NEUTRINOS
            and (float(m_finals[j]) == 0.0 or (heavy and abs(int(pdg[i])) in HEAVY_CUT))]


def pass_mask(P, slots, cuts):
    """mg5_pipeline_final.fiducial_pass_mask on the given rows"""
    import itertools
    keep = np.ones(P.shape[0], dtype=bool)
    pt = lambda p: np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)
    cost = lambda p: np.abs(p[..., 3]) / np.clip(np.linalg.norm(p[..., 1:], axis=-1), 1e-12, None)
    eta = lambda p: np.arcsinh(p[..., 3] / np.clip(pt(p), 1e-9, None))
    phi = lambda p: np.arctan2(p[..., 2], p[..., 1])
    for s_ in slots:
        keep &= (pt(P[:, s_]) > cuts["pt_min"]) & (cost(P[:, s_]) < cuts["cos_max"])
    for i, j in itertools.combinations(slots, 2):
        pi, pj = P[:, i], P[:, j]
        dphi = np.abs(phi(pi) - phi(pj)); dphi = np.minimum(dphi, 2 * np.pi - dphi)
        q = pi + pj
        m = np.sqrt(np.clip(q[..., 0] ** 2 - (q[..., 1:] ** 2).sum(-1), 0.0, None))
        keep &= (np.sqrt((eta(pi) - eta(pj)) ** 2 + dphi ** 2) > cuts["dr_min"]) & (m > cuts["m_min"])
    return keep


def collect_with_cuts(draw, n_events, mask):
    """mg5_pipeline_final._collect_with_cuts with the mask as an argument (the same batches, so the same events)"""
    accP, accS, got, acc = [], [], 0, 0.5
    while got < n_events:
        need = n_events - got
        batch = int(min(max(need / max(acc, 0.03) * 1.3, need), need * 50)) + 32
        P, sq = draw(batch)
        m = mask(P)
        if m.any():
            accP.append(P[m]); accS.append(sq[m]); got += int(m.sum())
        acc = 0.85 * acc + 0.15 * (float(m.mean()) if m.size else acc)
    return np.concatenate(accP)[:n_events], np.concatenate(accS)[:n_events]


def u_forward(P):
    """u = log(1/(1-|cos theta|)) of the most forward final particle, per event"""
    p = P[:, 2:, 1:]
    c = np.abs(p[..., 2]) / np.clip(np.linalg.norm(p, axis=-1), 1e-300, None)
    return np.log(1.0 / np.clip(1.0 - c.max(1), 1e-300, None))


def u_hist(u):
    h = np.histogram(np.clip(u, U_EDGES[0], U_EDGES[-1] - 1e-9), U_EDGES)[0]
    return [int(x) for x in h]


def rotate_forward(P, cos_cap, rng):
    """P (n, k, 4), beams in rows 0 and 1, the partonic rest frame. Per event one final particle at random is turned to
    an angle with 1 - |cos theta| = exp(-u), u uniform in [0, log(1/(1 - cos_cap))], phi uniform, either hemisphere; the
    whole final state turns with it (one rotation), so momentum conservation and the mass shells hold."""
    n, k = P.shape[:2]
    j = rng.integers(2, k, n)
    p = P[np.arange(n), j, 1:]
    a = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-300, None)
    u = rng.uniform(0.0, np.log(1.0 / (1.0 - cos_cap)), n)
    c = (1.0 - np.exp(-u)) * np.where(rng.random(n) < 0.5, 1.0, -1.0)
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(np.clip(1.0 - c ** 2, 0.0, None))
    b = np.stack([s * np.cos(ph), s * np.sin(ph), c], 1)
    v = np.cross(a, b); sv = np.linalg.norm(v, axis=1); cc = (a * b).sum(1)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    f = np.where(sv > 1e-12, (1.0 - cc) / np.clip(sv ** 2, 1e-300, None), 0.0)
    R = np.eye(3)[None] + K + f[:, None, None] * (K @ K)
    anti = (sv <= 1e-12) & (cc < 0)               # a already opposite b: a half turn about an axis normal to a
    if anti.any():
        e = np.where(np.abs(a[anti, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        w = np.cross(a[anti], e); w /= np.linalg.norm(w, axis=1, keepdims=True)
        R[anti] = 2.0 * w[:, :, None] * w[:, None, :] - np.eye(3)[None]
    Q = P.copy()
    Q[:, 2:, 1:] = np.einsum("nij,nkj->nki", R, P[:, 2:, 1:])
    return Q


def candidates(nb, sqrts_draw, m_finals, pdg, rng, mask, frac_dem, y_lo, frac_fwd, cos_cap, stat):
    """mg5_pipeline_final._candidates with a frac_fwd share of the draws rotated forward (no extra random numbers when
    frac_fwd is zero, so the stream is the pipeline's); stat counts the events drawn"""
    import mg5_pipeline_final as mg
    m_finals = np.asarray(m_finals, float); N = len(m_finals); pdg = np.asarray(pdg, int)
    all_zero = np.all(m_finals == 0.0)

    def draw(n_draw):
        sq = np.asarray(sqrts_draw(n_draw), float)
        beams = mg._beams(sq, pdg)
        if N == 2:
            P = mg.democratic_draw(n_draw, sq, m_finals, y_lo, rng)
        else:
            n_dem = int(round(frac_dem * n_draw)); parts = []
            if n_dem:
                parts.append(mg.democratic_draw(n_dem, sq[:n_dem], m_finals, y_lo, rng))
            if n_draw - n_dem:
                p_ml = mg._rambo_massless_batch(sq[n_dem:], N, rng)
                parts.append(p_ml if all_zero else mg._rambo_massive_batch(sq[n_dem:], m_finals, p_ml))
            P = np.concatenate(parts, axis=0)
        P = np.concatenate([beams, P], axis=1)
        if frac_fwd > 0:
            r = rng.random(n_draw) < frac_fwd
            if r.any():
                P[r] = rotate_forward(P[r], cos_cap, rng)
        stat["drawn"] += n_draw
        return P, sq

    P, sq = collect_with_cuts(draw, nb, mask)
    return [(P[i], pdg) for i in range(nb)], sq


def build(n_events, lo, hi, m_finals, pdg, rng, mask, pol, label, fwd_share=0.0, fwd_cand=0.0, cos_cap=None,
          log_sqrts=False):
    """mg5_pipeline_final.build_mixture_dataset with the forward component and, with log_sqrts, sqrt(s_hat) log uniform
    on [lo, hi] for the bulk and the first candidate pass; returns (P, sqrts, me2, stat)"""
    import mg5_pipeline_final as mg
    stat = {"drawn": 0, "labelled": 0, "t_label": 0.0}

    def lab(ev, sq):
        t = time.time(); me = np.asarray(label(ev, sq), float)
        stat["t_label"] += time.time() - t; stat["labelled"] += len(ev)
        return me

    t0 = time.time()
    n_flat = int(round(pol["f_flat"] * n_events)); n_fwd = int(round(fwd_share * n_events))
    n_bulk = n_events - n_flat - n_fwd
    uni = (lambda n: np.exp(rng.uniform(np.log(lo), np.log(hi), n))) if log_sqrts else (lambda n: rng.uniform(lo, hi, n))
    ev, sq = candidates(n_bulk, uni, m_finals, pdg, rng, mask, 0.0, pol["y_lo"], 0.0, cos_cap, stat)
    me = lab(ev, sq)
    evs, sqs, mes = [ev], [sq], [me]
    if n_fwd:
        ev, sq = candidates(n_fwd, uni, m_finals, pdg, rng, mask, 0.0, pol["y_lo"], 1.0, cos_cap, stat)
        evs.append(ev); sqs.append(sq); mes.append(lab(ev, sq))
    if n_flat:
        n_cand = max(int(pol["oversample"]) * n_flat, n_flat)
        n1 = n_cand // 2 if pol["sqrts_adapt"] else n_cand
        ev1, sq1 = candidates(n1, uni, m_finals, pdg, rng, mask, pol["democratic"], pol["y_lo"], fwd_cand, cos_cap, stat)
        me1 = lab(ev1, sq1)
        cand, sq_c, me_c = ev1, sq1, me1
        if pol["sqrts_adapt"] and n_cand - n1 > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                lg1 = np.log(np.where(me1 > 0, me1, np.nan))
            ok = np.isfinite(lg1)
            nb_s = mg._adaptive_bins(int(ok.sum()), int(pol["n_sqrts_bins"]))
            if ok.any():
                edges, probs = mg._sqrts_proposal_from_pass(sq1[ok], lg1[ok], lo, hi, nb_s,
                                                            n_logm_bins=mg._adaptive_bins(int(ok.sum()), 60, 100))
            else:
                edges, probs = np.linspace(lo, hi, nb_s + 1), np.full(nb_s, 1.0 / nb_s)
            ev2, sq2 = candidates(n_cand - n1, lambda n: mg._draw_sqrts_from_bins(n, edges, probs, rng), m_finals, pdg,
                                  rng, mask, pol["democratic"], pol["y_lo"], fwd_cand, cos_cap, stat)
            me2 = lab(ev2, sq2)
            cand, sq_c, me_c = ev1 + ev2, np.concatenate([sq1, sq2]), np.concatenate([me1, me2])
        with np.errstate(divide="ignore", invalid="ignore"):
            logm = np.log(np.where(me_c > 0, me_c, np.nan))
        keep = mg._flat_logm_select(logm, n_flat, rng, n_bins=mg._adaptive_bins(len(cand), 60, 100))
        if len(keep) < n_flat:
            rest = np.setdiff1d(np.arange(len(cand)), keep)
            keep = np.concatenate([keep, rng.choice(rest, size=n_flat - len(keep), replace=False)])
        evs.append([cand[i] for i in keep]); sqs.append(sq_c[keep]); mes.append(me_c[keep])
    P = np.stack([e[0] for part in evs for e in part])
    sq = np.concatenate(sqs); me = np.concatenate(mes)
    order = rng.permutation(len(P))
    stat["t_total"] = time.time() - t0
    stat["t_sample"] = stat["t_total"] - stat["t_label"]
    return P[order], sq[order], me[order], stat


def quantiles(me):
    me = np.asarray(me, float)
    ok = np.isfinite(me) & (me > 0)
    if not ok.any():
        return {"n_nonpos": int((~ok).sum())}
    l = np.log10(me[ok])
    q = np.quantile(l, QS)
    return {"q": [float(x) for x in q], "min": float(l.min()), "max": float(l.max()), "n": int(ok.sum()),
            "n_nonpos": int((~ok).sum()), "width_999": float(q[-1] - q[0]), "width_99": float(q[-2] - q[1])}


def pool_quantiles(name, sq):
    import glob
    g = sorted(glob.glob(os.path.join(os.environ["DATA_DIR"], f"{name}_{int(sq[0])}-{int(sq[1])}GeV_val_*amplitudes.npy")))
    if not g:
        return {"missing": True}
    a = np.load(g[0], mmap_mode="r")
    return dict(quantiles(np.asarray(a[:, -1], float)), file=os.path.basename(g[0]))


def to_partonic(a, k):
    """pp pool rows -> (P (n, k, 4) in the partonic rest frame, sqrt(s_hat)): a boost along z (pp_compare.py)"""
    P = np.array(a[:, :4 * k], dtype=np.float64).reshape(-1, k, 4)
    tot = P[:, 0] + P[:, 1]
    bz = tot[:, 3] / tot[:, 0]
    g = 1.0 / np.sqrt(1.0 - bz ** 2)
    E, pz = P[..., 0].copy(), P[..., 3].copy()
    P[..., 0] = g[:, None] * (E - bz[:, None] * pz)
    P[..., 3] = g[:, None] * (pz - bz[:, None] * E)
    rs = np.sqrt(np.clip(tot[:, 0] ** 2 - (tot[:, 1:] ** 2).sum(-1), 0, None))
    return P, rs


def single_cut_fails(P, slots, C):
    """the share each cut alone removes (as pass_mask, one cut at a time)"""
    off = {"pt_min": -1.0, "cos_max": 2.0, "dr_min": -1.0, "m_min": -1.0}
    return {key: float(1.0 - pass_mask(P, slots, dict(off, **{key: C[key]})).mean()) for key in off}


def overlap(h_a, h_b):
    a, b = np.asarray(h_a, float), np.asarray(h_b, float)
    return float(np.minimum(a / a.sum(), b / b.sum()).sum()) if a.sum() and b.sum() else None


def ess(rs, lo, hi, density):
    """the pp events in [lo, hi] reweighted to the measure q (density, unnormalised on [lo, hi]): w = p_pp / q with p_pp
    a 40-bin histogram in log sqrt(s_hat); returns (share of pp events in the support, (sum w)^2 / (n sum w^2))"""
    ins = (rs >= lo) & (rs <= hi)
    if ins.sum() < 20:
        return float(ins.mean()), None
    x = rs[ins]
    e = np.geomspace(lo, hi, 41)
    h = np.histogram(x, e)[0].astype(float)
    p_pp = h / h.sum() / np.diff(e)                          # density per GeV
    g = np.geomspace(lo, hi, 4001)
    Z = (getattr(np, "trapezoid", None) or np.trapz)(density(g), g)       # q = density / Z on [lo, hi]
    b = np.clip(np.searchsorted(e, x, side="right") - 1, 0, 39)
    w = p_pp[b] * Z / density(x)
    return float(ins.mean()), float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))


def pp_read(name, sq, cuts_new, rng_cur, rng_new, hists, wide_hi=SQRTS_WIDE):
    """name: the tree class whose pp pool is read; rng_* None: no |M|^2 range comparison (labels not this entry's)"""
    import mg5_pipeline_final as mg
    pn = "pp13_" + name[len("v3_"):]
    f = os.path.join(os.environ["DATA_DIR"], f"{pn}_13000-13000GeV_val_lhe_amplitudes.npy")
    if pn not in mg.PROCESSES or not os.path.exists(f):
        return {"missing": True}
    c = mg.PROCESSES[pn]
    k = len(c["pdg_ids"])
    a = np.load(f, mmap_mode="r")
    P, rs = to_partonic(a, k)
    me = np.asarray(a[:, -1], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        l = np.log10(np.where(me > 0, me, np.nan))
    s_cur, s_new = cut_slots(finals(c), c["pdg_ids"], False), cut_slots(finals(c), c["pdg_ids"], True)
    cur = mg.fiducial_pass_mask(P, finals(c), c["pdg_ids"], mg.FIDUCIAL_CUTS)
    new = pass_mask(P, s_new, cuts_new)
    new_nobt = pass_mask(P, s_cur, cuts_new)                 # the proposed thresholds, b and tau left uncut
    lo, hi = sq
    inw = (rs >= lo) & (rs <= hi)
    inwide = (rs >= lo) & (rs <= wide_hi)
    inr = lambda m, r: float(((l >= r[0]) & (l <= r[1]))[m].mean()) if m.any() and r else None
    u = u_forward(P)
    hp = u_hist(u)
    meas = {"uniform_window": (lo, hi, lambda x: np.ones_like(x)),
            "uniform_wide": (lo, wide_hi, lambda x: np.ones_like(x)),
            "log_wide": (lo, wide_hi, lambda x: 1.0 / x),
            "lumi_wide": (lo, wide_hi, lambda x: x ** -3.0)}
    out = {"n": int(len(P)), "in_cur": float(cur.mean()), "in_new": float(new.mean()),
           "in_new_b_tau_uncut": float(new_nobt.mean()), "in_window": float(inw.mean()),
           "above_window": float((rs > hi).mean()), "above_wide": float((rs > wide_hi).mean()),
           "in_cur_window": float((cur & inw).mean()), "in_new_window": float((new & inw).mean()),
           "in_new_wide": float((new & inwide).mean()),
           "fail_new": single_cut_fails(P, s_new, cuts_new),
           "in_range_cur": inr(cur & inw, rng_cur), "in_range_new": inr(new & inw, rng_new),
           "sqrts_q": [float(x) for x in np.quantile(rs, (0.1, 0.5, 0.9, 0.99))],
           "forward_share": float((u > np.log(10.0)).mean()), "u_hist": hp,
           "u_overlap": {kk: overlap(hp, h) for kk, h in hists.items()},
           "sqrts_measure": {kk: dict(zip(("support", "ess"), ess(rs, *v))) for kk, v in meas.items()}}
    return out


def one(task):
    name, info, n, seed = task
    import datagen
    import mg5_pipeline_final as mg
    if info["type"] in ("one_loop", "loop_induced"):
        return one_loop(task)
    cfg = mg.PROCESSES[name]
    kd = kind(cfg)
    cuts_cur = dict(cfg.get("fiducial_cuts") or mg.FIDUCIAL_CUTS)
    cuts_new = dict(COVERAGE[kd])
    if info["type"] == "harder_cut":
        cuts_new["pt_min"] = max(cuts_new["pt_min"], cuts_cur["pt_min"])
    lo, hi = info["sqrts"]
    wide_hi = max(SQRTS_WIDE, hi)
    sa = datagen.ensure_backend(name)
    _, _, driver_bin, eff_dir = mg.detect_compiled_backend(sa)
    perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
    amz = float(cfg.get("alphas_mz", 0.118))
    pol = mg.sampling_policy({"sampling": {"mode": "mixture"}})
    m, pdg = finals(cfg), cfg["pdg_ids"]
    s_cur, s_new = cut_slots(m, pdg, False), cut_slots(m, pdg, True)
    beams = "".join(sorted("l" if abs(int(p)) in LEPTON_BEAMS else ("g" if int(p) == 21 else "b" if abs(int(p)) == 5
                                                                    else "q") for p in pdg[:2]))
    row = {"name": name, "kind": kd, "group": info["group"], "type": info["type"], "base": info["base"],
           "nfinal": int(cfg["nfinal"]), "sqrts": info["sqrts"], "wide_top": wide_hi, "cuts_current": cuts_cur,
           "n_pool": info["n_pool"], "n": n, "beams": beams, "n_cut_legs_current": len(s_cur),
           "n_cut_legs_proposed": len(s_new), "has_b_tau": any(abs(int(p)) in HEAVY_CUT for p in pdg[2:])}
    m_cur = lambda P: mg.fiducial_pass_mask(P, m, pdg, cuts_cur)
    m_new = lambda P: pass_mask(P, s_new, cuts_new)
    try:
        with mg.CppDriverPipe(driver_bin, eff_dir) as pipe:
            label = lambda ev, sq: pipe.compute(ev, perm=perm, alphas=mg.compute_alphas(np.asarray(sq, float), alphas_mz=amz))
            S = {"current": build(n, lo, hi, m, pdg, np.random.default_rng(seed), m_cur, pol, label),
                 "proposed": build(n, lo, hi, m, pdg, np.random.default_rng(seed + 1), m_new, pol, label,
                                   FWD_SHARE, FWD_CAND, cuts_new["cos_max"]),
                 "wide": build(n, lo, wide_hi, m, pdg, np.random.default_rng(seed + 2), m_new, pol, label,
                               FWD_SHARE, FWD_CAND, cuts_new["cos_max"], log_sqrts=True)}
        hists = {}
        for key, (P, sq, me, st) in S.items():
            u = u_forward(P)
            hists[key] = u_hist(u)
            row[key] = dict(quantiles(me), **st, u_hist=hists[key], forward_share=float((u > np.log(10.0)).mean()),
                            sqrts_q=[float(x) for x in np.quantile(sq, (0.1, 0.5, 0.9, 0.99))])
        for key in ("proposed", "wide"):
            row[key]["outside_current_cuts"] = float(1.0 - m_cur(S[key][0]).mean())
        # b and tau: what the current sample holds that the thresholds would cut on those legs, under the current
        # thresholds and under the proposed ones (the other legs pass both: the current sample is inside the current
        # cuts, which are as tight or tighter on them)
        Pc, _, mc, _ = S["current"]
        bt = [i for i in s_new if i not in s_cur]
        if bt:
            for tag, C in (("", cuts_cur), ("_proposed", cuts_new)):
                keep = pass_mask(Pc, s_new, C)
                row["current"]["b_tau_fail_share" + tag] = float(1.0 - keep.mean())
                row["current"]["width_999_b_tau_cut" + tag] = quantiles(mc[keep]).get("width_999")
        row["pool"] = pool_quantiles(name, info["sqrts"])
        if kd == "hadronic":
            rr = lambda r: (r["min"], r["max"]) if "min" in r else None
            if info["type"] == "tree":
                row["pp"] = pp_read(name, info["sqrts"], cuts_new, rr(row["current"]), rr(row["proposed"]), hists,
                                    wide_hi)
            elif info["type"] in ("high_energy", "harder_cut", "alpha_s"):
                same = info["type"] != "alpha_s"            # an alpha_s point's labels are not its base's
                row["pp"] = pp_read(info["base"], info["sqrts"], cuts_new, rr(row["current"]) if same else None,
                                    rr(row["proposed"]) if same else None, hists, wide_hi)
                row["pp"]["source"] = info["base"]
            else:
                row["pp"] = {"missing": True, "why": "a heavy-quark mass point, a BSM entry or a 2->5 probe: no pp pool"}
    except Exception as e:                    # a class that fails is listed with its error, never dropped silently
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def one_loop(task):
    """a one-loop or loop-induced pool, read without new labels: its validation pool's range, and for a hadronic one-loop
    pool its tree's pp events against the loop pool's window, the current cuts and the proposed coverage"""
    name, info, n, seed = task
    import mg5_pipeline_final as mg
    row = {"name": name, "group": info["group"], "type": info["type"], "tree": info["tree"], "sqrts": info["sqrts"],
           "n_pool": info["n_pool"], "mode": "read"}
    try:
        cfg = mg.PROCESSES.get(name) or mg.PROCESSES[info["tree"]]
        row["kind"] = kind(cfg)
        row["nfinal"] = int(cfg.get("nfinal", len(cfg["pdg_ids"]) - 2))
        row["pool"] = pool_quantiles(name, info["sqrts"])
        if row["kind"] == "hadronic" and info["tree"]:
            row["pp"] = pp_read(info["tree"], info["sqrts"], COVERAGE["hadronic"], None, None, {},
                                max(SQRTS_WIDE, info["sqrts"][1]))
            row["pp"]["source"] = info["tree"]
        else:
            row["pp"] = {"missing": True, "why": "e+e- beams or loop-induced: no tree pp pool"}
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def selftest(name, info):
    """build() with no forward share and the current cuts is the pipeline's build_mixture_dataset, event for event"""
    import datagen
    import mg5_pipeline_final as mg
    cfg = mg.PROCESSES[name]
    sa = datagen.ensure_backend(name)
    _, _, driver_bin, eff_dir = mg.detect_compiled_backend(sa)
    perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
    pol = mg.sampling_policy({"sampling": {"mode": "mixture"}})
    m, pdg = finals(cfg), cfg["pdg_ids"]
    amz = float(cfg.get("alphas_mz", 0.118))
    lo, hi = info["sqrts"]
    with mg.CppDriverPipe(driver_bin, eff_dir) as pipe:
        label = lambda ev, sq: pipe.compute(ev, perm=perm, alphas=mg.compute_alphas(np.asarray(sq, float), alphas_mz=amz))
        ev, _, me_ref = mg.build_mixture_dataset(400, lo, hi, m, pdg, np.random.default_rng(7), dict(mg.FIDUCIAL_CUTS),
                                                  pol, label)
        P, _, me, _ = build(400, lo, hi, m, pdg, np.random.default_rng(7),
                            lambda X: mg.fiducial_pass_mask(X, m, pdg, mg.FIDUCIAL_CUTS), pol, label)
    P_ref = np.stack([e[0] for e in ev])
    assert np.array_equal(P, P_ref) and np.array_equal(me, me_ref), f"build() differs from the pipeline on {name}"
    assert np.array_equal(pass_mask(P, cut_slots(m, pdg, False), mg.FIDUCIAL_CUTS),
                          mg.fiducial_pass_mask(P, m, pdg, mg.FIDUCIAL_CUTS)), "pass_mask differs from the pipeline's"
    rng = np.random.default_rng(3)
    Q = rotate_forward(P, COS_ETA5, rng)
    mass = lambda X: X[..., 0] ** 2 - (X[..., 1:] ** 2).sum(-1)
    assert np.allclose(mass(Q), mass(P), atol=1e-6 * hi ** 2), "rotation broke a mass shell"
    assert np.allclose(Q[:, 2:].sum(1), P[:, 2:].sum(1), atol=1e-8 * hi), "rotation broke momentum conservation"
    print(f"selftest on {name}: build() equals build_mixture_dataset on 400 events; the rotation keeps shells and momenta",
          flush=True)



def run(args):
    from multiprocessing import Pool
    import datagen
    C = classes()
    register(C)
    os.makedirs(OUT(), exist_ok=True)
    path = os.path.join(OUT(), ROWS)
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if "error" not in r:
                done.add(r["name"])
    import mg5_pipeline_final as mg
    loop = lambda n: C[n]["type"] in ("one_loop", "loop_induced")
    nf = lambda n: 0 if loop(n) else mg.PROCESSES[n]["nfinal"]
    todo = sorted((n for n in C if n not in done), key=lambda n: -nf(n))   # the costly first
    print(f"{len(C)} entries ({sum(1 for n in C if not loop(n))} sampled, {sum(1 for n in C if loop(n))} read), "
          f"{len(done)} already done, {len(todo)} to run, {args.n} events per coverage", flush=True)
    datagen.ensure_backends([n for n in todo if not loop(n)], workers=args.workers)
    selftest("v3_bb_abb" if "v3_bb_abb" in C else todo[0], C["v3_bb_abb" if "v3_bb_abb" in C else todo[0]])
    # seeds: the tree classes in sorted order first (the first run's seeds), then every other entry
    tasks = [(n, C[n], args.n, 1000 + 2 * i) for i, n in enumerate(C) if n in set(todo)]
    tasks.sort(key=lambda t: -nf(t[0]))
    t0 = time.time()
    with Pool(args.workers) as pool, open(path, "a") as f:
        for i, row in enumerate(pool.imap_unordered(one, tasks)):
            f.write(json.dumps(row) + "\n"); f.flush()
            msg = row.get("error") or ("read" if row.get("mode") == "read" else
                                       f"width {row['current'].get('width_999', float('nan')):.2f} -> "
                                       f"{row['proposed'].get('width_999', float('nan')):.2f}")
            print(f"[{i + 1}/{len(tasks)} {time.time() - t0:7.0f}s] {row['name']:28s} {msg}", flush=True)
    rows = [json.loads(l) for l in open(path)]
    err = [r["name"] for r in rows if "error" in r]
    print(f"RUN_SUMMARY rows={len({r['name'] for r in rows if 'error' not in r})} expected={len(C)} errors={len(err)}",
          flush=True)


def collect():
    """the latest row per class, as one JSON object on stdout"""
    rows = {}
    for l in open(os.path.join(OUT(), ROWS)):
        r = json.loads(l)
        if r["name"] not in rows or "error" in rows[r["name"]]:
            rows[r["name"]] = r
    json.dump(rows, sys.stdout)


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else None


def u_cover(h_pp, h_s):
    """the pp events' forward angle u against a sample's: (pp share in u bins the sample never reaches, effective sample
    size per event (sum w)^2 / (n sum w^2) of the sample reweighted to the pp u distribution on the bins it reaches)"""
    a, b = np.asarray(h_pp, float), np.asarray(h_s, float)
    a, b = a / a.sum(), b / b.sum()
    m = b > 0
    w = a[m] / b[m]
    return float(a[~m].sum()), float((b[m] * w).sum() ** 2 / (b[m] * w ** 2).sum())


def summary():
    """medians per kind and multiplicity, per held-out axis on pp, b and tau, totals; the last line is the whole
    summary as one JSON object (docs/results.tex's claims, analysis/catalog_v3/coverage_fig.py)"""
    rows = json.load(open(os.path.join(HERE, "data", "coverage_rows.json")))
    keys = ("current", "proposed", "wide")
    typ = lambda r: r.get("type", "tree")
    sampled = {n: r for n, r in rows.items() if "error" not in r and r.get("mode") != "read"
               and all("width_999" in r.get(k, {}) for k in keys)}
    ok = {n: r for n, r in sampled.items() if typ(r) == "tree"}          # the families: the tree classes
    print(f"{len(rows)} rows, {len(sampled)} sampled usable, errors: {sorted(n for n, r in rows.items() if 'error' in r)}")
    out = {"n_rows": len(rows), "n_usable": len(ok), "n_sampled": len(sampled),
           "n_read": sum(1 for r in rows.values() if r.get("mode") == "read" and "error" not in r),
           "n_errors": sum(1 for r in rows.values() if "error" in r),
           "errors": sorted(n for n, r in rows.items() if "error" in r)}
    sec = lambda r, c: r[c]["t_total"] / r["n"]
    pp_ok = lambda r: "pp" in r and not r["pp"].get("missing")
    for kd in ("hadronic", "ee"):
        for nf in (2, 3, 4):
            g = [r for r in ok.values() if r["kind"] == kd and r["nfinal"] == nf]
            if not g:
                continue
            e = {"n": len(g)}
            for k in ("proposed", "wide"):
                e[f"growth_999_{k}"] = med(r[k]["width_999"] - r["current"]["width_999"] for r in g)
                e[f"max_up_{k}"] = med(r[k]["max"] - r["current"]["max"] for r in g)
                e[f"min_down_{k}"] = med(r["current"]["min"] - r[k]["min"] for r in g)
                e[f"outside_current_{k}"] = med(r[k]["outside_current_cuts"] for r in g)
                e[f"cost_ratio_{k}"] = med(sec(r, k) / sec(r, "current") for r in g)
            for k in keys:
                e[f"cpu_h_{k}"] = float(sum(sec(r, k) * r["n_pool"] for r in g) / 3600)
                e[f"forward_share_{k}"] = med(r[k]["forward_share"] for r in g)
                e[f"width_999_{k}"] = med(r[k]["width_999"] for r in g)
            bt = [r for r in g if "b_tau_fail_share" in r["current"]]
            e["b_tau_classes"] = len(bt)
            e["b_tau_no_cut_now"] = sum(1 for r in g if r["has_b_tau"] and r["n_cut_legs_current"] == 0)
            if bt:
                e["b_tau_fail_share"] = med(r["current"]["b_tau_fail_share"] for r in bt)
                e["b_tau_fail_share_proposed"] = med(r["current"].get("b_tau_fail_share_proposed") for r in bt)
            pp = [r for r in g if pp_ok(r)]
            if pp:
                for f in ("in_cur_window", "in_new_window", "in_new_wide", "in_cur", "in_new", "in_new_b_tau_uncut",
                          "in_window", "above_window", "above_wide", "in_range_cur", "in_range_new", "forward_share"):
                    e[f"pp_{f}"] = med(r["pp"][f] for r in pp)
                for k in keys:
                    e[f"pp_u_overlap_{k}"] = med(r["pp"]["u_overlap"].get(k) for r in pp)
                    uc = [u_cover(r["pp"]["u_hist"], r[k]["u_hist"]) for r in pp]
                    e[f"pp_u_uncovered_mean_{k}"] = float(np.mean([x[0] for x in uc]))
                    e[f"pp_u_ess_{k}"] = med(x[1] for x in uc)
                for k in ("uniform_window", "uniform_wide", "log_wide", "lumi_wide"):
                    e[f"pp_ess_{k}"] = med(r["pp"]["sqrts_measure"][k]["ess"] for r in pp)
                e["pp_n"] = len(pp)
            out[f"{kd}_2to{nf}"] = e
            print(f"{kd:8s} 2->{nf}", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}))
    had = [r for r in ok.values() if r["kind"] == "hadronic" and pp_ok(r)]
    out["pp"] = {"n": len(had), "n_missing": sum(1 for r in ok.values() if r["kind"] == "hadronic" and not pp_ok(r)),
                 "in_new_min": float(min(r["pp"]["in_new"] for r in had)),
                 "fail_max": {c: float(max(r["pp"]["fail_new"][c] for r in had)) for c in ("pt_min", "cos_max", "dr_min",
                                                                                         "m_min")}}
    for ax in ("train", "H1", "H2", "H3"):
        g = [r["pp"] for r in had if r["group"] == ax]
        out["pp"][ax] = {"n": len(g), **{f: med(p[f] for p in g) for f in (
            "in_cur_window", "in_new_window", "in_new_wide", "above_window", "above_wide", "in_range_cur",
            "in_range_new")}}
    bt = [r for r in ok.values() if r["has_b_tau"]]
    btc = [r for r in bt if r["current"].get("width_999_b_tau_cut") is not None]
    btp = [r["pp"] for r in bt if pp_ok(r)]
    btq = [r for r in bt if r["current"].get("width_999_b_tau_cut_proposed") is not None]
    out["b_tau"] = {"n": len(bt), "no_cut_now": sum(1 for r in bt if r["n_cut_legs_current"] == 0),
                    "fail_share": med(r["current"].get("b_tau_fail_share") for r in bt),
                    "width_loss": med(r["current"]["width_999"] - r["current"]["width_999_b_tau_cut"] for r in btc),
                    "fail_share_proposed": med(r["current"].get("b_tau_fail_share_proposed") for r in bt),
                    "width_loss_proposed": med(r["current"]["width_999"] - r["current"]["width_999_b_tau_cut_proposed"]
                                               for r in btq),
                    "pp_n": len(btp), "pp_in_new_mean": float(np.mean([p["in_new"] for p in btp])),
                    "pp_in_new_uncut_mean": float(np.mean([p["in_new_b_tau_uncut"] for p in btp]))}
    R = list(ok.values())
    out["cost"] = {k: {"cpu_h": float(sum(sec(r, k) * r["n_pool"] for r in R) / 3600),
                       "labelled_per_kept": med(r[k]["labelled"] / r["n"] for r in R),
                       "drawn_per_kept": med(r[k]["drawn"] / r["n"] for r in R),
                       "label_time_share": float(sum(r[k]["t_label"] for r in R) / sum(r[k]["t_total"] for r in R))}
                   for k in keys}
    out["cost"]["n_pool_events"] = int(sum(r["n_pool"] for r in R))
    X = [r for r in sampled.values() if typ(r) != "tree"]
    out["cost"]["other_sampled"] = {k: float(sum(sec(r, k) * r["n_pool"] for r in X) / 3600) for k in keys}
    out["cost"]["other_n_pool_events"] = int(sum(r["n_pool"] for r in X))
    # the other sampled entries, per type
    out["types"] = {}
    for t in sorted({typ(r) for r in sampled.values()} - {"tree"}):
        g = [r for r in sampled.values() if typ(r) == t]
        e = {"n": len(g), "n_entries": sum(1 for r in rows.values() if typ(r) == t)}
        for k in ("proposed", "wide"):
            e[f"growth_999_{k}"] = med(r[k]["width_999"] - r["current"]["width_999"] for r in g)
            e[f"max_up_{k}"] = med(r[k]["max"] - r["current"]["max"] for r in g)
            e[f"outside_current_{k}"] = med(r[k]["outside_current_cuts"] for r in g)
        for k in keys:
            e[f"forward_share_{k}"] = med(r[k]["forward_share"] for r in g)
        pp = [r["pp"] for r in g if pp_ok(r)]
        if pp:
            e["pp_n"] = len(pp)
            for f in ("in_cur_window", "in_new_window", "in_new_wide", "above_window", "in_range_cur", "in_range_new",
                      "forward_share"):
                e[f"pp_{f}"] = med(p.get(f) for p in pp)
        out["types"][t] = e
        print(f"{t:14s}", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}))
    # the loop pools, read without new labels
    L = [r for r in rows.values() if r.get("mode") == "read" and "error" not in r]
    out["loops"] = {}
    for t in ("one_loop", "loop_induced"):
        for kd in ("hadronic", "ee"):
            g = [r for r in L if r["type"] == t and r.get("kind") == kd]
            if not g:
                continue
            e = {"n": len(g), "pool_width_999": med(r["pool"].get("width_999") for r in g),
                 "n_pool_missing": sum(1 for r in g if r["pool"].get("missing"))}
            pp = [r["pp"] for r in g if pp_ok(r)]
            if pp:
                e["pp_n"] = len(pp)
                for f in ("in_cur_window", "in_new_window", "in_new_wide", "above_window", "forward_share"):
                    e[f"pp_{f}"] = med(p[f] for p in pp)
            out["loops"][f"{t}_{kd}"] = e
            print(f"{t}_{kd:9s}", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}))
    json.dump(out, open(os.path.join(HERE, "data", "coverage_check.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="summary", choices=("run", "collect", "summary"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    {"run": lambda: run(a), "collect": collect, "summary": summary}[a.mode]()
