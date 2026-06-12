"""Fine-tuning utilities: LoRA, EWC, and layer-wise LR decay."""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import LOGGER


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank adapter ΔW = (α/r) * B @ A.

    Initialization: B = 0 so ΔW = 0 at start — pretrained behaviour preserved.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha

        d, k = linear.weight.shape
        # Create the adapters on the wrapped linear's device & dtype: inject_lora runs
        # after the model is already on the GPU, and the default (CPU) tensors would make
        # the fused optimizer reject them ("fused=True requires ... cuda ... tensors").
        dev, dt = linear.weight.device, linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, k, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(d, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # muP: MuAdam/MuAdamW require EVERY trainable parameter to carry an `infshape`.
        # The wrapped linear's weight already has one (set by the muP setup); propagate it
        # to the new LoRA params — rank is a finite (non-width) dim, while in/out inherit
        # the linear's width InfDims. Without this the optimizer raises
        # "parameter ... does not have infshape attribute".
        w_inf = getattr(linear.weight, "infshape", None)
        if w_inf is not None:
            from mup.infshape import InfShape, InfDim
            fin_rank = InfDim(None, rank)                       # finite rank dimension
            self.lora_A.infshape = InfShape([fin_rank, w_inf[1]])   # (rank, in)
            self.lora_B.infshape = InfShape([w_inf[0], fin_rank])   # (out, rank)

        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (self.alpha / self.rank) * F.linear(
            F.linear(x, self.lora_A), self.lora_B
        )

    def merge(self) -> nn.Linear:
        """Return a merged plain Linear (zero inference overhead)."""
        merged = copy.deepcopy(self.linear)
        merged.weight.data.add_((self.alpha / self.rank) * (self.lora_B @ self.lora_A))
        merged.weight.requires_grad_(True)
        return merged


def _wrap_linear(parent: nn.Module, attr: str, rank: int, alpha: float) -> bool:
    linear = getattr(parent, attr, None)
    if isinstance(linear, nn.Linear):
        setattr(parent, attr, LoRALinear(linear, rank, alpha))
        return True
    return False


def inject_lora(model: nn.Module, rank: int, alpha: float, target: list) -> nn.Module:
    """Replace target Linear layers in every transformer block with LoRALinear.

    After injection, all non-LoRA parameters are frozen.

    Parameters
    ----------
    model : AmplitudeLLoCaWrapper
    rank : int
        LoRA bottleneck rank r.
    alpha : float
        LoRA scaling; effective LR multiplier = alpha / rank.
    target : list of str
        Which linears to target. Supported values: 'qkv', 'out'.
    """
    # model.net = LLOCAMuPTransformer; model.net.net = MuPTransformer
    transformer = model.net.net
    count = 0
    for block in transformer.blocks:
        sa = block.attention  # BaselineSelfAttention
        qkv = sa.qkv_linear
        if "qkv" in target:
            # MultiHeadQKVLinear has a single .linear; MultiQueryQKVLinear splits into q/k/v
            if hasattr(qkv, "linear"):
                count += _wrap_linear(qkv, "linear", rank, alpha)
            else:
                for attr in ("q_linear", "k_linear", "v_linear"):
                    count += _wrap_linear(qkv, attr, rank, alpha)
        if "out" in target:
            count += _wrap_linear(sa, "out_linear", rank, alpha)

    # Freeze every non-LoRA parameter
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info(
        f"LoRA injected into {count} linears (rank={rank}, alpha={alpha}). "
        f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )
    return model


# ---------------------------------------------------------------------------
# EWC
# ---------------------------------------------------------------------------

class EWC:
    """Diagonal-Fisher Elastic Weight Consolidation.

    L_total = L_task + (lambda/2) * Σ_i F_i * (θ_i - θ*_i)²

    Fisher is estimated from the fine-tune training set (not pretraining data,
    which may not be available), so it measures parameter importance at the
    pretrained optimum relative to the new task distribution.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader,
        n_fisher_batches: int,
        device: torch.device,
        loss_fn,
    ):
        self._theta_star = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._fisher = self._compute_fisher(model, dataloader, n_fisher_batches, device, loss_fn)
        LOGGER.info("EWC: Fisher diagonal computed.")

    def _compute_fisher(self, model, dataloader, n_batches, device, loss_fn) -> dict:
        fisher = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        model.eval()
        seen = 0
        for batch in dataloader:
            if seen >= n_batches:
                break
            loss = loss_fn(batch)
            model.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name].add_(param.grad.data.pow(2))
            seen += 1
        for name in fisher:
            fisher[name].div_(max(seen, 1))
        model.train()
        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in self._fisher:
                diff = param - self._theta_star[name].to(device)
                loss = loss + (self._fisher[name] * diff.pow(2)).sum()
        return loss


# ---------------------------------------------------------------------------
# Layer-wise LR decay (LLRD)
# ---------------------------------------------------------------------------

def build_ft_param_groups(
    model: nn.Module,
    base_lr: float,
    lr_scale: float,
    layer_decay: float,
) -> list:
    """Build parameter groups with layer-wise LR decay for fine-tuning.

    Depth 0 = output head (highest LR). Depth increases toward FramesNet.
    Group LR = base_lr * lr_scale * layer_decay^depth.

    Parameters
    ----------
    model : AmplitudeLLoCaWrapper
    base_lr : float
        Base LR from config (same as pretraining LR).
    lr_scale : float
        Global fine-tuning LR multiplier.
    layer_decay : float
        Decay factor per depth level. 1.0 = no decay.
    """
    transformer = model.net.net  # MuPTransformer
    n_blocks = len(transformer.blocks)

    # Map named-parameter prefix → depth (0 = shallowest / highest LR)
    prefix_to_depth: dict[str, int] = {}
    prefix_to_depth["net.net.linear_out"] = 0
    for i in range(n_blocks):
        # block index n_blocks-1 is closest to output → depth 1
        prefix_to_depth[f"net.net.blocks.{n_blocks - 1 - i}"] = i + 1
    prefix_to_depth["net.net.linear_in"] = n_blocks + 1
    prefix_to_depth["net.framesnet"] = n_blocks + 2
    prefix_to_depth["net.trafo_fourmomenta"] = n_blocks + 2

    def _depth(param_name: str) -> int:
        for prefix, depth in prefix_to_depth.items():
            if param_name.startswith(prefix):
                return depth
        return n_blocks + 3  # fallback: treat as deepest

    groups: dict[int, list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        groups.setdefault(_depth(name), []).append(param)

    param_groups = []
    for depth in sorted(groups):
        lr = base_lr * lr_scale * (layer_decay ** depth)
        param_groups.append({"params": groups[depth], "lr": lr})
        LOGGER.debug(f"LLRD depth={depth}: lr={lr:.2e}, n_param_tensors={len(groups[depth])}")

    return param_groups
