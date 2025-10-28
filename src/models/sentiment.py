"""Fusion layers for incorporating FinBERT embeddings via cross-attention."""
from __future__ import annotations

import torch
from torch import nn


class SentimentFusion(nn.Module):
    """Cross-attention layer between time-series tokens and sentiment embeddings."""

    def __init__(self, d_model: int, sentiment_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.sentiment_proj = nn.Linear(sentiment_dim, d_model)
        self.sentiment_dropout = nn.Dropout(dropout)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.gating = nn.Linear(d_model, 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor, sentiment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse sentiment embeddings into the time-series token stream."""

        if sentiment.dim() == 2:
            sentiment = sentiment.unsqueeze(1)
        projected = self.sentiment_dropout(self.sentiment_proj(sentiment))
        batch_size = projected.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        sentiment_tokens = torch.cat([cls_token, projected], dim=1)

        attn_output, attn_weights = self.cross_attention(
            query=tokens,
            key=sentiment_tokens,
            value=sentiment_tokens,
            need_weights=True,
        )
        self.last_attention_weights = attn_weights.detach() if attn_weights is not None else None

        if attn_weights is not None:
            pooled_weights = attn_weights.mean(dim=1, keepdim=True)
            sentiment_summary = torch.bmm(pooled_weights, sentiment_tokens)
            sentiment_cls = sentiment_summary.squeeze(1)
        else:
            sentiment_cls = sentiment_tokens[:, 0, :]

        gating_scalar = torch.sigmoid(self.gating(sentiment_cls)).unsqueeze(1)
        tokens = self.norm1(tokens + gating_scalar * attn_output)
        tokens = self.norm2(tokens + self.ff(tokens))
        return tokens, sentiment_cls
