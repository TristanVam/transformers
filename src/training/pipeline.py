"""Training pipeline for the hybrid stochastic PatchTST model."""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch import optim
from torch.utils.data import DataLoader

from ..data.augmentation import BrownianAugmentor
from ..data.binance import BinanceDataClient
from ..data.dataset import PatchTSTDataset, collate_windows
from ..models.patchtst import PatchTSTConfig, PatchTSTModel
from ..models.variational import VariationalHead
from ..utils.sentiment import FinBERTEncoder
from .config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


def _prepare_device(config: ExperimentConfig) -> torch.device:
    if config.training.device:
        return torch.device(config.training.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fetch_market_data(config: ExperimentConfig) -> Sequence:
    """Download candles using the Binance REST API."""

    client = BinanceDataClient(symbol=config.data.symbol, interval=config.data.interval)
    end_time = dt.datetime.now(tz=dt.timezone.utc)
    train_start = end_time - dt.timedelta(hours=config.data.train_hours + config.data.val_hours)
    candles = client.get_klines(start_time=train_start, end_time=end_time)
    LOGGER.info("Downloaded %d candles", len(candles))
    return candles


def prepare_sentiment_encoder(cache_dir: Path) -> FinBERTEncoder:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return FinBERTEncoder()


def sentiment_lookup_factory(encoder: FinBERTEncoder) -> Callable[[dt.datetime], torch.Tensor]:
    """Create a callable that returns FinBERT embeddings for a given timestamp."""

    def lookup(timestamp: dt.datetime) -> torch.Tensor:
        # Placeholder: in production you would fetch tweets near ``timestamp``.
        sample_text = f"Bitcoin market close at {timestamp.isoformat()}"
        embedding = encoder.encode_text(sample_text)
        return embedding.unsqueeze(0).numpy()

    return lookup


def build_dataloaders(candles, config: ExperimentConfig, sentiment_fn) -> tuple[DataLoader, DataLoader]:
    split_index = int(len(candles) * 0.8)
    train_candles = candles[:split_index]
    val_candles = candles[split_index:]

    augmentor = None
    if config.augmentation.enabled:
        augmentor = BrownianAugmentor(dt=config.augmentation.dt, num_paths=config.augmentation.num_paths)

    train_dataset = PatchTSTDataset(
        candles=train_candles,
        window_size=config.data.window_size,
        horizon=config.data.horizon,
        sentiment_fn=sentiment_fn,
        augmentor=augmentor,
    )

    val_dataset = PatchTSTDataset(
        candles=val_candles,
        window_size=config.data.window_size,
        horizon=config.data.horizon,
        sentiment_fn=sentiment_fn,
        augmentor=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_windows,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_windows,
    )

    return train_loader, val_loader


def train_epoch(model: PatchTSTModel, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        features = batch["features"].to(device)
        sentiment = batch["sentiment"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        outputs = model(features, sentiment)
        loss = VariationalHead.nll_loss(outputs["mu"], outputs["log_sigma"], targets).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        count += features.size(0)

    return total_loss / max(count, 1)


def evaluate(model: PatchTSTModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            sentiment = batch["sentiment"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(features, sentiment)
            loss = VariationalHead.nll_loss(outputs["mu"], outputs["log_sigma"], targets).mean()

            total_loss += loss.item() * features.size(0)
            count += features.size(0)

    return total_loss / max(count, 1)


def run_experiment(config: ExperimentConfig) -> None:
    device = _prepare_device(config)
    LOGGER.info("Using device %s", device)

    candles = fetch_market_data(config)
    sentiment_encoder = prepare_sentiment_encoder(config.training.sentiment_cache)
    sentiment_fn = sentiment_lookup_factory(sentiment_encoder)
    train_loader, val_loader = build_dataloaders(candles, config, sentiment_fn)

    model_config = PatchTSTConfig()
    model = PatchTSTModel(model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    for epoch in range(config.training.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        LOGGER.info("Epoch %d: train_loss=%.4f val_loss=%.4f", epoch + 1, train_loss, val_loss)

    config.training.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.training.output_dir / "patchtst_hybrid.pt"
    torch.save({"model_state": model.state_dict(), "config": model_config}, checkpoint_path)
    LOGGER.info("Saved checkpoint to %s", checkpoint_path)
