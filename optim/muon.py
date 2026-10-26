"""Muon with decoupled weight decay and the update RMS matched to AdamW's (Moonshot, "Muon is Scalable for LLM
Training", arXiv:2502.16982), on the muP backbone's hidden matrices, AdamW on everything else, in one optimizer
object (so CosineAnnealingLR, gradient clipping, EMA and the fine-tune lr groups see one param_groups list).

Muon step on a matrix W (fan_out x fan_in) with gradient G, momentum mu (0.95), Nesterov:
    B <- mu B + G;  M = G + mu B;  O = NS5(M)  (5 Newton-Schulz steps, float32, coefficients 3.4445, -4.7750, 2.0315)
    W <- W - lr (s O + wd W),  s = rms sqrt(max(fan_out, fan_in)) / sqrt(m)
rms = 0.2 makes the update's RMS AdamW's typical 0.2 at the base width (O has unit singular values, RMS
1/sqrt(max)); m is muP's width multiplier (infshape.width_mult()). MuAdamW divides a hidden matrix's lr by m (Adam's
update is near low rank, its spectral norm grows like the width); the orthogonalised update's spectral norm grows like
sqrt(m) with this s, so 1/sqrt(m) is the muP rule for Muon (the spectral condition, Yang-Simon-Bernstein), and the lr
number is width-independent.

The Muon set: the parameters muP marks matrix-like (infshape.ninf() == 2): per block qkv, attention out and the two MLP
layers. The AdamW set is everything else, run by a torch AdamW (fused on CUDA) that shares this optimizer's group
dicts, with MuAdamW's grouping (its vector-like and finite parameters get the group lr unchanged, as in the baseline).
"""
import math
from collections import defaultdict

import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(G, steps=5, eps=1e-7):
    """Approximate U V^T of G = U S V^T (singular values brought to ~[0.7, 1.2]), in float32."""
    a, b, c = NS_COEFFS
    X = G.float()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


def is_muon_param(p):
    """muP's matrix-like parameters (both dimensions scale with the width)."""
    return hasattr(p, "infshape") and p.infshape.ninf() == 2


def muon_scale(p, rms=0.2, mup=True):
    fan_out, fan_in = p.shape[0], p[0].numel()
    m = p.infshape.width_mult() if (mup and hasattr(p, "infshape")) else 1.0
    return rms * math.sqrt(max(fan_out, fan_in)) / math.sqrt(m)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, momentum=0.95,
                 nesterov=True, ns_steps=5, muon_weight_decay=0.1, rms=0.2, mup=True, fused=None,
                 muon_filter=is_muon_param):
        groups = list(params)
        if not isinstance(groups[0], dict):
            groups = [{"params": groups}]
        muon_groups, adam_groups = [], []
        for g in groups:
            ps = list(g["params"])
            rest = {k: v for k, v in g.items() if k != "params"}
            rest.setdefault("lr", lr)
            rest.setdefault("weight_decay", weight_decay)
            mu = [p for p in ps if muon_filter(p)]
            if mu:
                muon_groups.append({**rest, "params": mu, "use_muon": True, "weight_decay": muon_weight_decay})
            # MuAdamW's split of the rest: matrix-like by width multiplier (lr / m, wd * m), the others unchanged
            by_m, vec = defaultdict(list), []
            for p in ps:
                if muon_filter(p):
                    continue
                if mup and hasattr(p, "infshape") and p.infshape.ninf() == 2:
                    by_m[p.infshape.width_mult()].append(p)
                else:
                    vec.append(p)
            for m, mp in by_m.items():
                adam_groups.append({**rest, "params": mp, "use_muon": False, "lr": rest["lr"] / m,
                                    "weight_decay": rest["weight_decay"] * m})
            if vec:
                adam_groups.append({**rest, "params": vec, "use_muon": False})
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, rms=rms, use_muon=False)
        self._mup = mup
        super().__init__(muon_groups + adam_groups, defaults)
        kw = {"fused": True} if fused else {}
        self._n_muon = len(muon_groups)
        self._adamw = torch.optim.AdamW(self.param_groups[self._n_muon:], lr=lr, betas=tuple(betas), eps=eps,
                                        weight_decay=weight_decay, **kw) if adam_groups else None
        self._relink()

    def _relink(self):
        # the inner AdamW reads and writes the same group dicts (lr from the scheduler, the fine-tune groups)
        if self._adamw is not None:
            self._adamw.param_groups = self.param_groups[self._n_muon:]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups[:self._n_muon]:
            lr, wd, mu = g["lr"], g["weight_decay"], g["momentum"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                buf = st["momentum_buffer"]
                grad = p.grad.float()
                buf.mul_(mu).add_(grad)
                m = grad.add(buf, alpha=mu) if g["nesterov"] else buf
                o = newton_schulz(m.reshape(p.shape[0], -1), g["ns_steps"]).reshape(p.shape)
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(o.to(p.dtype), alpha=-lr * muon_scale(p, g["rms"], self._mup))
        if self._adamw is not None:
            self._adamw.step()
        return loss

    def state_dict(self):
        sd = super().state_dict()
        sd["adamw"] = self._adamw.state_dict() if self._adamw is not None else None
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        inner = state_dict.pop("adamw", None)
        super().load_state_dict(state_dict)
        self._relink()
        if self._adamw is not None and inner is not None:
            self._adamw.load_state_dict(inner)
            self._relink()
