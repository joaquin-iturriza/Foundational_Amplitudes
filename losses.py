import torch
import math

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)

def rel_l1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    epsilon_tensor = torch.tensor(epsilon, device=y_true.device, dtype=y_true.dtype)
    return torch.mean(torch.abs((y_true - y_pred) / torch.maximum(torch.abs(y_true), epsilon_tensor))) 

class RelL1Loss(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        return rel_l1_loss(y_true, y_pred, self.epsilon)
    
def heteroscedastic_loss(y_pred: torch.Tensor, y_true: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma_clamped = torch.clamp(sigma, min=1e-15, max=1e5)
    loss = ((y_true - y_pred) ** 2) / (2 * sigma_clamped ** 2) + torch.log(sigma_clamped)
    return torch.mean(loss)

class HeteroscedasticLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return heteroscedastic_loss(y_pred, y_true, sigma)