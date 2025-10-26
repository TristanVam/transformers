"""Variational head for modelling mean and volatility."""
from __future__ import annotations

import torch
from torch import nn


class VariationalHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mu = nn.Linear(d_model, 1)
        self.log_sigma = nn.Linear(d_model, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(features).squeeze(-1)
        log_sigma = self.log_sigma(features).squeeze(-1)
        return mu, log_sigma

    @staticmethod
    def nll_loss(mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2 * log_sigma)
        return 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
