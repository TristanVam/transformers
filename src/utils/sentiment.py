"""Utilities for extracting FinBERT embeddings from short-form text."""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)
FINBERT_MODEL = "ProsusAI/finbert"


@dataclass
class SentimentBatch:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor


class FinBERTEncoder:
    """Wrapper around the FinBERT model to obtain contextual embeddings."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        self.model = AutoModel.from_pretrained(FINBERT_MODEL).to(self.device)
        self.model.eval()

    @functools.lru_cache(maxsize=512)
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text into a pooled FinBERT embedding."""

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=96,
            padding="max_length",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.no_grad():
            outputs = self.model(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze(0).cpu()

    def batch_embed(self, texts: Sequence[str]) -> SentimentBatch:
        """Encode multiple texts into embeddings and attention masks."""

        embeddings = [self.encode_text(text) for text in texts]
        stacked = torch.stack(embeddings, dim=0)
        attention = torch.ones(stacked.size(0), 1)
        return SentimentBatch(embeddings=stacked, attention_mask=attention)


def dummy_sentiment(_: Iterable[str]) -> torch.Tensor:
    """Fallback when sentiment data is unavailable."""

    return torch.zeros(1)
