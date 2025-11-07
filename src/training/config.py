"""Configuration dataclasses for training."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class DataConfig:
    """Market and dataset configuration."""

    symbol: str = "BTCUSDT"
    symbols: List[str] | None = None
    interval: str = "1h"
    window_size: int = 64
    horizons: List[int] = field(default_factory=lambda: [1])
    train_hours: int = 24 * 180
    val_hours: int = 24 * 30
    sentiment_half_life_hours: float = 6.0


@dataclass
class AugmentationConfig:
    """Brownian augmentation parameters."""

    enabled: bool = True
    num_paths: int = 4
    dt: float = 1.0
    rho: float = 0.0
    max_paths_per_batch: int = 64


@dataclass
class OptimizerConfig:
    """Optimizer hyper-parameters."""

    lr: float = 1e-4
    weight_decay: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class TrainingConfig:
    """Training loop configuration and infrastructure options."""

    batch_size: int = 32
    max_epochs: int = 50
    device: str | None = None
    output_dir: Path = Path("checkpoints")
    sentiment_cache: Path = Path("sentiment_cache")
    mixed_precision: bool = True
    grad_clip: float = 1.0
    num_workers: int = 4
    scheduler: str = "cosine_warmup"
    seed: int = 42
    patience: int = 10
    warmup_epochs: int = 5
    log_dir: Path = Path("runs")


@dataclass
class ExperimentConfig:
    """Top-level configuration wiring all sub-components together."""

    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    augmentation: AugmentationConfig = dataclasses.field(default_factory=AugmentationConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        """Instantiate a configuration object from a nested dictionary."""

        def _build(dataclass_type, data: Dict[str, Any] | None):
            if data is None:
                return dataclass_type()
            converted: Dict[str, Any] = {}
            for field in dataclasses.fields(dataclass_type):
                if field.name not in data:
                    continue
                value = data[field.name]
                if value is None:
                    converted[field.name] = None
                    continue
                origin = getattr(field.type, "__origin__", None)
                args = getattr(field.type, "__args__", ())
                if field.type is Path or Path in args or origin is Path:
                    converted[field.name] = Path(value)
                else:
                    converted[field.name] = value
            return dataclass_type(**converted)

        return cls(
            data=_build(DataConfig, payload.get("data")),
            augmentation=_build(AugmentationConfig, payload.get("augmentation")),
            optimizer=_build(OptimizerConfig, payload.get("optimizer")),
            training=_build(TrainingConfig, payload.get("training")),
        )

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(path.read_text()))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: Path, *, indent: int = 2) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=indent))
