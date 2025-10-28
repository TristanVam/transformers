"""Brownian-style data augmentation utilities for stochastic robustness."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class BrownianAugmentor:
    """Generate geometric Brownian motion trajectories around a base series.

    The augmentor estimates the drift and volatility of the observed log returns and
    samples alternative futures that remain statistically consistent with the recent
    behaviour. This allows the downstream Transformer to experience a diverse set of
    shocks during training, improving robustness to unexpected volatility spikes.
    """

    dt: float = 1.0
    num_paths: int = 8
    clamp_sigma: Tuple[float, float] | None = (1e-5, 2.0)
    rho: float = 0.0
    random_state: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self._rng = self.random_state or np.random.default_rng()

    def _estimate_mu_sigma(self, series: np.ndarray) -> tuple[float, float]:
        if series.ndim != 1:
            raise ValueError("BrownianAugmentor expects a 1-D series")
        window = int(series.size / 3)
        window = max(window, 2)
        recent = series[-window:]
        log_returns = np.diff(np.log(recent + 1e-12))
        if log_returns.size == 0:
            raise ValueError("Series must contain at least two data points.")
        mu = float(np.mean(log_returns)) / self.dt
        sigma = float(np.std(log_returns)) / math.sqrt(self.dt)
        if self.clamp_sigma:
            sigma = float(np.clip(sigma, self.clamp_sigma[0], self.clamp_sigma[1]))
        return mu, sigma

    def sample(self, price_series: np.ndarray, volume_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``num_paths`` Brownian trajectories for price and volume."""

        if price_series.shape != volume_series.shape:
            raise ValueError("Price and volume series must have the same shape")

        mu, sigma = self._estimate_mu_sigma(price_series)
        mu_v, sigma_v = self._estimate_mu_sigma(volume_series + 1e-6)

        steps = len(price_series) - 1
        if steps <= 0:
            raise ValueError("Series must contain at least two data points.")

        dt = self.dt
        price_paths = np.empty((self.num_paths, len(price_series)), dtype=np.float32)
        vol_paths = np.empty_like(price_paths)
        price_paths[:, 0] = price_series[0]
        vol_paths[:, 0] = volume_series[0]

        drift_price = (mu - 0.5 * sigma**2) * dt
        drift_volume = (mu_v - 0.5 * sigma_v**2) * dt
        diffusion_price = sigma * math.sqrt(dt)
        diffusion_volume = sigma_v * math.sqrt(dt)

        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        noise = self._rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=(self.num_paths, steps))
        noise_price = noise[:, :, 0]
        noise_volume = noise[:, :, 1]

        price_increments = np.exp(drift_price + diffusion_price * noise_price)
        volume_increments = np.exp(drift_volume + diffusion_volume * noise_volume)
        price_paths[:, 1:] = price_series[0] * np.cumprod(price_increments, axis=1)
        vol_paths[:, 1:] = volume_series[0] * np.cumprod(volume_increments, axis=1)

        return price_paths, vol_paths

    def mix(self, price_series: np.ndarray, volume_series: np.ndarray) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """Yield the original series followed by simulated perturbations."""

        yield price_series, volume_series
        price_paths, vol_paths = self.sample(price_series, volume_series)
        for path_price, path_volume in zip(price_paths, vol_paths, strict=False):
            yield path_price, path_volume
