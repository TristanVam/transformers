"""PatchTST-based architecture with stochastic robustness and sentiment fusion."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .sentiment import SentimentFusion
from .variational import VariationalHead


class PatchEmbedding(nn.Module):
    """Convert sliding windows into transformer tokens."""

    def __init__(self, input_dim: int, patch_len: int, stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, input_dim)
        batch, seq_len, input_dim = x.shape
        if seq_len < self.patch_len:
            raise ValueError("Sequence length must be >= patch_len")

        num_patches = 1 + (seq_len - self.patch_len) // self.stride
        unfolded = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        unfolded = unfolded.reshape(batch, num_patches, -1)
        return self.proj(unfolded)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@dataclass
class PatchTSTConfig:
    input_dim: int = 2
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    sentiment_dim: int = 768


class PatchTSTModel(nn.Module):
    """Hybrid stochastic Transformer for ΔP forecasting."""

    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = PatchEmbedding(config.input_dim, config.patch_len, config.stride, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.sentiment_fusion = SentimentFusion(
            d_model=config.d_model,
            sentiment_dim=config.sentiment_dim,
            num_heads=config.nhead,
            dropout=config.dropout,
        )
        self.variational_head = VariationalHead(config.d_model)

    def forward(self, features: torch.Tensor, sentiment: torch.Tensor) -> dict[str, torch.Tensor]:
        # features: (batch, time, input_dim)
        tokens = self.embedding(features)
        tokens = self.positional_encoding(tokens)
        encoded = self.encoder(tokens)
        fused = self.sentiment_fusion(encoded, sentiment)
        mu, log_sigma = self.variational_head(fused)
        return {"mu": mu, "log_sigma": log_sigma}
