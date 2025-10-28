"""Dataset utilities combining market data, stochastic augmentation, and sentiment."""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    target_delta: np.ndarray
    close_time: dt.datetime
    symbol_id: int
    group_size: int


class PatchTSTDataset(Dataset[TimeSeriesWindow]):
    """Windowed dataset that exposes Brownian augmentations on the fly."""

    def __init__(
        self,
        candles: Sequence[BinanceCandle],
        window_size: int,
        horizons: Sequence[int],
        sentiment_fn: Optional[
            Callable[[str, dt.datetime, dt.datetime], Sequence[Tuple[dt.datetime, np.ndarray]]]
        ] = None,
        augmentor: Optional[BrownianAugmentor] = None,
        sentiment_half_life_hours: float = 6.0,
        max_paths_per_window: int | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not horizons:
            raise ValueError("At least one horizon must be provided")

        sorted_horizons = sorted(set(int(h) for h in horizons if h > 0))
        if len(sorted_horizons) != len(horizons):
            raise ValueError("Horizons must be positive integers")

        if len(candles) < window_size + max(sorted_horizons):
            raise ValueError("Not enough candles to build dataset")

        self.window_size = window_size
        self.horizons = sorted_horizons
        self.sentiment_fn = sentiment_fn
        self.augmentor = augmentor
        self.sentiment_half_life_hours = sentiment_half_life_hours
        self.max_paths_per_window = max_paths_per_window

        grouped: Dict[str, List[BinanceCandle]] = {}
        for candle in candles:
            grouped.setdefault(candle.symbol, []).append(candle)

        self.series_by_symbol: Dict[str, List[BinanceCandle]] = {}
        self.symbol_to_id: Dict[str, int] = {}
        for idx, (symbol, series) in enumerate(sorted(grouped.items())):
            ordered = sorted(series, key=lambda c: c.close_time)
            self.series_by_symbol[symbol] = ordered
            self.symbol_to_id[symbol] = idx

        self.window_refs: List[Tuple[str, int]] = []
        for symbol, series in self.series_by_symbol.items():
            symbol_windows = len(series) - self.window_size - max(self.horizons) + 1
            for start in range(symbol_windows):
                self.window_refs.append((symbol, start))

        if not self.window_refs:
            raise ValueError("No windows available for dataset")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.window_refs)

    def _normalise_series(self, series: np.ndarray) -> np.ndarray:
        transformed = np.log1p(series)
        return (transformed - transformed.mean()) / (transformed.std() + 1e-6)

    def _extract_window(self, index: int) -> TimeSeriesWindow:
        symbol, start = self.window_refs[index]
        series = self.series_by_symbol[symbol]
        window = series[start : start + self.window_size]
        close_time = window[-1].close_time

        closes = np.array([c.close for c in window], dtype=np.float32)
        volumes = np.array([c.volume for c in window], dtype=np.float32)

        price_paths: List[np.ndarray]
        volume_paths: List[np.ndarray]
        group_size = 1

        if self.augmentor is not None:
            augmented = list(self.augmentor.mix(closes, volumes))
            if self.max_paths_per_window is not None:
                augmented = augmented[: self.max_paths_per_window]
            price_paths = [self._normalise_series(path_price) for path_price, _ in augmented]
            volume_paths = [self._normalise_series(path_volume) for _, path_volume in augmented]
            group_size = len(price_paths)
        else:
            price_paths = [self._normalise_series(closes)]
            volume_paths = [self._normalise_series(volumes)]

        prices_arr = np.stack(price_paths, axis=0).astype(np.float32)
        volumes_arr = np.stack(volume_paths, axis=0).astype(np.float32)

        sentiment = np.zeros((prices_arr.shape[0], 1), dtype=np.float32)
        if self.sentiment_fn is not None:
            history_start = window[0].close_time
            entries = self.sentiment_fn(symbol, history_start, close_time)
            if entries:
                decay_seconds = self.sentiment_half_life_hours * 3600.0
                weights: List[float] = []
                vectors: List[np.ndarray] = []
                for timestamp, vector in entries:
                    delta_seconds = max((close_time - timestamp).total_seconds(), 0.0)
                    weight = math.exp(-delta_seconds / decay_seconds) if decay_seconds > 0 else 1.0
                    weights.append(weight)
                    vectors.append(np.asarray(vector, dtype=np.float32))
                stacked = np.stack(vectors, axis=0)
                weight_arr = np.asarray(weights, dtype=np.float32)
                weight_arr /= weight_arr.sum() + 1e-6
                sentiment_vec = np.sum(stacked * weight_arr[:, None], axis=0)
                sentiment = np.repeat(sentiment_vec[None, :], prices_arr.shape[0], axis=0)

        last_close = closes[-1]
        deltas = []
        for horizon in self.horizons:
            future_idx = start + self.window_size + horizon - 1
            future_close = series[future_idx].close
            deltas.append(future_close - last_close)

        return TimeSeriesWindow(
            prices=prices_arr,
            volumes=volumes_arr,
            sentiment=sentiment,
            target_delta=np.asarray(deltas, dtype=np.float32),
            close_time=close_time,
            symbol_id=self.symbol_to_id[symbol],
            group_size=group_size,
        )

    def __getitem__(self, index: int) -> TimeSeriesWindow:  # type: ignore[override]
        return self._extract_window(index)


def collate_windows(batch: Sequence[TimeSeriesWindow]) -> dict[str, torch.Tensor]:
    """Collate windows into tensors suitable for PatchTST training."""

    price_paths = []
    volume_paths = []
    sentiment_paths = []
    target_values = []
    symbol_ids = []
    group_sizes = []

    for window in batch:
        price_paths.append(window.prices)
        volume_paths.append(window.volumes)
        sentiment_paths.append(window.sentiment)
        path_count = window.prices.shape[0]
        target_values.append(np.repeat(window.target_delta[None, :], path_count, axis=0))
        symbol_ids.extend([window.symbol_id] * path_count)
        group_sizes.append(path_count)

    prices = np.concatenate(price_paths, axis=0)
    volumes = np.concatenate(volume_paths, axis=0)
    sentiment = np.concatenate(sentiment_paths, axis=0)
    targets = np.concatenate(target_values, axis=0)

    features = np.stack([prices, volumes], axis=-1)

    return {
        "features": torch.from_numpy(features),
        "sentiment": torch.from_numpy(sentiment),
        "targets": torch.from_numpy(targets),
        "symbol_ids": torch.tensor(symbol_ids, dtype=torch.long),
        "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
    }
