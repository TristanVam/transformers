"""Dataset utilities combining market data, stochastic augmentation, and sentiment."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import BrownianAugmentor
from .binance import BinanceCandle


@dataclass
class TimeSeriesWindow:
    prices: np.ndarray
    volumes: np.ndarray
    sentiment: np.ndarray
    target_delta: float
    close_time: dt.datetime


class PatchTSTDataset(Dataset[TimeSeriesWindow]):
    """Windowed dataset that exposes Brownian augmentations on the fly."""

    def __init__(
        self,
        candles: Sequence[BinanceCandle],
        window_size: int,
        horizon: int = 1,
        sentiment_fn: Optional[Callable[[dt.datetime], np.ndarray]] = None,
        augmentor: Optional[BrownianAugmentor] = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if len(candles) < window_size + horizon:
            raise ValueError("Not enough candles to build dataset")

        self.candles = list(candles)
        self.window_size = window_size
        self.horizon = horizon
        self.sentiment_fn = sentiment_fn
        self.augmentor = augmentor

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.candles) - self.window_size - self.horizon + 1

    def _extract_window(self, index: int) -> TimeSeriesWindow:
        window = self.candles[index : index + self.window_size]
        future_close = self.candles[index + self.window_size + self.horizon - 1].close
        last_close = window[-1].close
        delta = future_close - last_close

        prices = np.array([candle.close for candle in window], dtype=np.float32)
        volumes = np.array([candle.volume for candle in window], dtype=np.float32)
        if self.augmentor is not None:
            price_paths = np.stack(list(self.augmentor.mix(prices)))
            volume_paths = np.tile(volumes[None, :], (price_paths.shape[0], 1))
        else:
            price_paths = prices[None, :]
            volume_paths = volumes[None, :]

        if self.sentiment_fn is not None:
            sentiment = self.sentiment_fn(window[-1].close_time)
            sentiment = np.asarray(sentiment, dtype=np.float32)
            if sentiment.ndim == 1:
                sentiment = sentiment[None, :]
            if sentiment.shape[0] == 1 and price_paths.shape[0] > 1:
                sentiment = np.repeat(sentiment, price_paths.shape[0], axis=0)
        else:
            sentiment = np.zeros((price_paths.shape[0], 1), dtype=np.float32)

        return TimeSeriesWindow(
            prices=price_paths,
            volumes=volume_paths,
            sentiment=sentiment,
            target_delta=float(delta),
            close_time=window[-1].close_time,
        )

    def __getitem__(self, index: int) -> TimeSeriesWindow:  # type: ignore[override]
        return self._extract_window(index)


def collate_windows(batch: Sequence[TimeSeriesWindow]) -> dict[str, torch.Tensor]:
    """Collate windows into tensors suitable for PatchTST training."""

    price_paths = []
    volume_paths = []
    sentiment_paths = []
    target_values = []

    for window in batch:
        price_paths.append(window.prices)
        volume_paths.append(window.volumes)
        sentiment_paths.append(window.sentiment)
        path_count = window.prices.shape[0]
        target_values.append(np.full(path_count, window.target_delta, dtype=np.float32))

    prices = np.concatenate(price_paths, axis=0)
    volumes = np.concatenate(volume_paths, axis=0)
    sentiment = np.concatenate(sentiment_paths, axis=0)
    targets = np.concatenate(target_values, axis=0)

    features = np.stack([prices, volumes], axis=-1)

    return {
        "features": torch.from_numpy(features),
        "sentiment": torch.from_numpy(sentiment),
        "targets": torch.from_numpy(targets),
    }
