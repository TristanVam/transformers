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
    random_state: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self._rng = self.random_state or np.random.default_rng()

    def sample(self, series: np.ndarray) -> np.ndarray:
        """Return ``num_paths`` Brownian trajectories around ``series``.

        Parameters
        ----------
        series:
            Array containing prices for a single trajectory.
        """

        if series.ndim != 1:
            raise ValueError("BrownianAugmentor expects a 1-D price series.")

        log_returns = np.diff(np.log(series + 1e-12))
        if log_returns.size == 0:
            raise ValueError("Series must contain at least two data points.")

        mu = float(np.mean(log_returns)) / self.dt
        sigma = float(np.std(log_returns)) / math.sqrt(self.dt)

        if self.clamp_sigma:
            sigma = float(np.clip(sigma, self.clamp_sigma[0], self.clamp_sigma[1]))

        steps = len(series) - 1
        dt = self.dt
        simulations = np.empty((self.num_paths, len(series)), dtype=np.float32)
        simulations[:, 0] = series[0]

        drift = (mu - 0.5 * sigma**2) * dt
        diffusion_scale = sigma * math.sqrt(dt)

        noise = self._rng.standard_normal(size=(self.num_paths, steps))
        increments = np.exp(drift + diffusion_scale * noise)
        simulations[:, 1:] = series[0] * np.cumprod(increments, axis=1)

        return simulations

    def mix(self, series: np.ndarray) -> Iterable[np.ndarray]:
        """Yield the original series followed by simulated perturbations."""

        yield series
        for path in self.sample(series):
            yield path
