"""Configuration dataclasses for training."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    window_size: int = 64
    horizon: int = 1
    train_hours: int = 24 * 180
    val_hours: int = 24 * 30


@dataclass
class AugmentationConfig:
    enabled: bool = True
    num_paths: int = 8
    dt: float = 1.0


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-3


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    device: str | None = None
    output_dir: Path = Path("checkpoints")
    sentiment_cache: Path = Path("sentiment_cache")


@dataclass
class ExperimentConfig:
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    augmentation: AugmentationConfig = dataclasses.field(default_factory=AugmentationConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
