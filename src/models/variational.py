"""Variational head for modelling mean and volatility."""
from __future__ import annotations

import torch
from torch import nn


class VariationalHead(nn.Module):
    def __init__(self, d_model: int, horizons: int = 1) -> None:
        super().__init__()
        self.horizons = horizons
        self.mu = nn.Linear(d_model, horizons)
        self.log_sigma = nn.Linear(d_model, horizons)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(features)
        log_sigma = self.log_sigma(features)
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=3.0)
        return mu, log_sigma

    @staticmethod
    def nll_loss(mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor, l2_weight: float = 1e-4) -> torch.Tensor:
        var = torch.exp(2 * log_sigma)
        base = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
        reg = l2_weight * torch.sum(log_sigma**2, dim=-1)
        return base.mean(dim=-1) + reg
