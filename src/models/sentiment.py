"""Fusion layers for incorporating FinBERT embeddings via cross-attention."""
from __future__ import annotations

import torch
from torch import nn


class SentimentFusion(nn.Module):
    """Cross-attention layer between time-series tokens and sentiment embeddings."""

    def __init__(self, d_model: int, sentiment_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.sentiment_proj = nn.Linear(sentiment_dim, d_model)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, tokens: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:
        """Fuse sentiment embeddings into the time-series token stream.

        Parameters
        ----------
        tokens:
            Tensor of shape (batch, seq_len, d_model) produced by the Transformer encoder.
        sentiment:
            Tensor of shape (batch, sentiment_len, sentiment_dim) or (batch, sentiment_dim).
        """

        if sentiment.dim() == 2:
            sentiment = sentiment.unsqueeze(1)
        sentiment_proj = self.sentiment_proj(sentiment)
        attn_output, _ = self.cross_attention(query=tokens, key=sentiment_proj, value=sentiment_proj)
        tokens = self.norm1(tokens + attn_output)
        tokens = self.norm2(tokens + self.ff(tokens))
        pooled = tokens.mean(dim=1)
        return pooled
