"""Training pipeline for the hybrid stochastic PatchTST model."""
from __future__ import annotations

import datetime as dt
import logging
import random
from collections import deque
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ..data.augmentation import BrownianAugmentor
from ..data.binance import BinanceDataClient, BinanceCandle
from ..data.dataset import PatchTSTDataset, collate_windows
from ..models.patchtst import PatchTSTConfig, PatchTSTModel
from ..models.variational import VariationalHead
from ..utils.sentiment import FinBERTEncoder
from .config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_device(config: ExperimentConfig) -> torch.device:
    if config.training.device:
        return torch.device(config.training.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fetch_market_data(config: ExperimentConfig) -> Sequence[BinanceCandle]:
    """Download candles using the Binance REST API."""

    symbols = config.data.symbols or [config.data.symbol]
    end_time = dt.datetime.now(tz=dt.timezone.utc)
    train_delta = dt.timedelta(hours=config.data.train_hours + config.data.val_hours)
    train_start = end_time - train_delta

    candles: list[BinanceCandle] = []
    cache_dir = config.training.output_dir / "binance_cache"

    for symbol in symbols:
        client = BinanceDataClient(symbol=symbol, interval=config.data.interval, cache_dir=cache_dir)
        symbol_candles = client.get_klines(start_time=train_start, end_time=end_time)
        candles.extend(symbol_candles)
        LOGGER.info("Downloaded %d candles for %s", len(symbol_candles), symbol)
    return candles


def prepare_sentiment_encoder(cache_dir: Path) -> FinBERTEncoder:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return FinBERTEncoder()


def sentiment_lookup_factory(encoder: FinBERTEncoder) -> Callable[[str, dt.datetime, dt.datetime], list[tuple[dt.datetime, np.ndarray]]]:
    """Create a callable that returns FinBERT embeddings for a given timestamp range."""

    def lookup(symbol: str, start: dt.datetime, end: dt.datetime) -> list[tuple[dt.datetime, np.ndarray]]:
        # Placeholder sentiment sourcing: in practice fetch tweets between start/end
        text = f"{symbol} close {end.isoformat()}"
        embedding = encoder.encode_text(text).cpu().numpy()
        return [(end, embedding)]

    return lookup


def build_datasets(candles, config: ExperimentConfig, sentiment_fn) -> tuple[PatchTSTDataset, PatchTSTDataset]:
    cutoff_time = sorted({candle.close_time for candle in candles})
    if len(cutoff_time) < 2:
        raise ValueError("Not enough candles for train/val split")
    cutoff_idx = max(1, min(int(len(cutoff_time) * 0.7), len(cutoff_time) - 1))
    cutoff_timestamp = cutoff_time[cutoff_idx]

    train_candles = [c for c in candles if c.close_time <= cutoff_timestamp]
    val_candles = [c for c in candles if c.close_time > cutoff_timestamp]

    augmentor = None
    if config.augmentation.enabled:
        augmentor = BrownianAugmentor(
            dt=config.augmentation.dt,
            num_paths=config.augmentation.num_paths,
            rho=config.augmentation.rho,
        )

    train_dataset = PatchTSTDataset(
        candles=train_candles,
        window_size=config.data.window_size,
        horizons=config.data.horizons,
        sentiment_fn=sentiment_fn,
        augmentor=augmentor,
        sentiment_half_life_hours=config.data.sentiment_half_life_hours,
        max_paths_per_window=config.augmentation.max_paths_per_batch,
    )

    val_dataset = PatchTSTDataset(
        candles=val_candles,
        window_size=config.data.window_size,
        horizons=config.data.horizons,
        sentiment_fn=sentiment_fn,
        augmentor=None,
        sentiment_half_life_hours=config.data.sentiment_half_life_hours,
    )

    return train_dataset, val_dataset


def build_dataloaders(train_dataset: PatchTSTDataset, val_dataset: PatchTSTDataset, config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=collate_windows,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=collate_windows,
    )
    return train_loader, val_loader


def create_scheduler(optimizer: optim.Optimizer, config: ExperimentConfig) -> SequentialLR | None:
    if config.training.scheduler != "cosine_warmup":
        return None
    warmup_iters = max(config.training.warmup_epochs, 1)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
    cosine = CosineAnnealingLR(optimizer, T_max=max(config.training.max_epochs - warmup_iters, 1))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])


def compute_metrics(mu: torch.Tensor, log_sigma: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    mu = mu.view(-1, mu.size(-1))
    log_sigma = log_sigma.view(-1, log_sigma.size(-1))
    targets = targets.view(-1, targets.size(-1))

    sigma = torch.exp(log_sigma)
    mae = torch.mean(torch.abs(mu - targets)).item()
    rmse = torch.sqrt(torch.mean((mu - targets) ** 2)).item()
    nll = torch.mean(VariationalHead.nll_loss(mu, log_sigma, targets)).item()
    lower = mu - sigma
    upper = mu + sigma
    coverage = torch.mean(((targets >= lower) & (targets <= upper)).float()).item()
    hit = torch.mean((torch.sign(mu) == torch.sign(targets)).float()).item()
    primary_mu = mu[:, 0]
    primary_target = targets[:, 0]
    pnl_values = torch.sign(primary_mu) * primary_target
    pnl = torch.mean(pnl_values).item()
    sharpe = pnl_values.mean() / (pnl_values.std() + 1e-6)
    return {
        "mae": mae,
        "rmse": rmse,
        "nll": nll,
        "coverage": coverage,
        "hit_ratio": hit,
        "pnl": pnl,
        "sharpe": sharpe.item() if isinstance(sharpe, torch.Tensor) else float(sharpe),
    }


class SnapshotEnsembler:
    def __init__(self, max_snapshots: int = 3) -> None:
        self.max_snapshots = max_snapshots
        self.snapshots: deque[dict[str, torch.Tensor]] = deque(maxlen=max_snapshots)

    def update(self, model: PatchTSTModel) -> None:
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.snapshots.append(state)

    def averaged_predictions(self, model: PatchTSTModel, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.snapshots:
            return evaluate_model(model, loader, device)

        mus: list[torch.Tensor] = []
        sigmas: list[torch.Tensor] = []
        targets_accum: list[torch.Tensor] = []
        original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for state in self.snapshots:
            model.load_state_dict(state, strict=False)
            mu, log_sigma, targets = evaluate_model(model, loader, device)
            mus.append(mu)
            sigmas.append(log_sigma)
            targets_accum.append(targets)
        model.load_state_dict(original_state, strict=False)
        target_tensor = targets_accum[0] if targets_accum else torch.empty(0)
        return torch.mean(torch.stack(mus), dim=0), torch.mean(torch.stack(sigmas), dim=0), target_tensor


def evaluate_model(
    model: PatchTSTModel, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_mu = []
    all_log_sigma = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            sentiment = batch["sentiment"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            symbol_ids = batch["symbol_ids"].to(device, non_blocking=True)
            outputs = model(features, sentiment, symbol_ids)
            mu = outputs["mu"]
            log_sigma = outputs["log_sigma"]
            all_mu.append(mu.cpu())
            all_log_sigma.append(log_sigma.cpu())
            all_targets.append(targets.cpu())
    return torch.cat(all_mu, dim=0), torch.cat(all_log_sigma, dim=0), torch.cat(all_targets, dim=0)


def walk_forward_validation(
    model: PatchTSTModel,
    dataset: PatchTSTDataset,
    device: torch.device,
    config: ExperimentConfig,
) -> dict[str, float]:
    folds = max(3, min(10, len(dataset) // max(config.data.horizons)))
    fold_size = max(1, len(dataset) // folds)
    indices = list(range(len(dataset)))
    metrics_accumulator: dict[str, list[float]] = {key: [] for key in ["mae", "rmse", "nll", "coverage", "hit_ratio", "pnl", "sharpe"]}

    for start in range(0, len(dataset), fold_size):
        end = min(start + fold_size, len(dataset))
        subset = Subset(dataset, indices[start:end])
        loader = DataLoader(
            subset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
            collate_fn=collate_windows,
        )
        _, fold_metrics = evaluate(model, loader, device, config)
        for key, value in fold_metrics.items():
            metrics_accumulator[key].append(value)

    return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics_accumulator.items()}


def log_attention_maps(
    model: PatchTSTModel, writer: SummaryWriter, global_step: int, output_dir: Path
) -> None:
    attention = model.sentiment_fusion.last_attention_weights
    if attention is None:
        return
    writer.add_histogram("attention/sentiment", attention, global_step=global_step)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"attention_epoch_{global_step}.pt"
    torch.save(attention.cpu(), path)


def train_epoch(
    model: PatchTSTModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    config: ExperimentConfig,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    consistency_weight = 1e-3

    for batch in tqdm(loader, desc="train", leave=False):
        features = batch["features"].to(device, non_blocking=True)
        sentiment = batch["sentiment"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        symbol_ids = batch["symbol_ids"].to(device, non_blocking=True)
        group_sizes = batch["group_sizes"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=config.training.mixed_precision):
            outputs = model(features, sentiment, symbol_ids)
            base_loss = VariationalHead.nll_loss(outputs["mu"], outputs["log_sigma"], targets).mean()

            offset = 0
            penalties = []
            for group_size in group_sizes:
                group_slice = slice(offset, offset + group_size.item())
                group_mu = outputs["mu"][group_slice]
                base_mu = group_mu[:1]
                penalties.append(torch.mean((group_mu - base_mu) ** 2))
                offset += group_size.item()
            consistency_loss = torch.stack(penalties).mean() if penalties else torch.tensor(0.0, device=device)
            loss = base_loss + consistency_weight * consistency_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.training.grad_clip)
            optimizer.step()

        total_loss += loss.item() * features.size(0)
        count += features.size(0)

    return total_loss / max(count, 1)


def evaluate(
    model: PatchTSTModel,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    count = 0
    all_mu = []
    all_log_sigma = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            features = batch["features"].to(device, non_blocking=True)
            sentiment = batch["sentiment"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            symbol_ids = batch["symbol_ids"].to(device, non_blocking=True)

            outputs = model(features, sentiment, symbol_ids)
            loss = VariationalHead.nll_loss(outputs["mu"], outputs["log_sigma"], targets).mean()

            total_loss += loss.item() * features.size(0)
            count += features.size(0)
            all_mu.append(outputs["mu"].cpu())
            all_log_sigma.append(outputs["log_sigma"].cpu())
            all_targets.append(targets.cpu())

    mu = torch.cat(all_mu, dim=0)
    log_sigma = torch.cat(all_log_sigma, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(mu, log_sigma, targets)
    return total_loss / max(count, 1), metrics


def run_experiment(config: ExperimentConfig) -> None:
    seed_everything(config.training.seed)
    LOGGER.info("Global seed set to %d", config.training.seed)
    device = _prepare_device(config)
    LOGGER.info("Using device %s", device)

    config.training.output_dir.mkdir(parents=True, exist_ok=True)
    config.training.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(config.training.log_dir))

    candles = fetch_market_data(config)
    sentiment_encoder = prepare_sentiment_encoder(config.training.sentiment_cache)
    sentiment_fn = sentiment_lookup_factory(sentiment_encoder)
    train_dataset, val_dataset = build_datasets(candles, config, sentiment_fn)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, config)

    model_config = PatchTSTConfig(
        sentiment_dim=sentiment_encoder.model.config.hidden_size,
        num_symbols=len(train_dataset.symbol_to_id),
        multi_horizon=len(config.data.horizons),
    )
    model = PatchTSTModel(model_config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
    )
    scheduler = create_scheduler(optimizer, config)
    scaler = GradScaler(enabled=config.training.mixed_precision)
    snapshot_ensembler = SnapshotEnsembler()

    best_val_loss = float("inf")
    patience_counter = 0
    best_checkpoint = config.training.output_dir / "best_model.pt"

    for epoch in range(1, config.training.max_epochs + 1):
        LOGGER.info("Epoch %d", epoch)
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler, config)
        val_loss, val_metrics = evaluate(model, val_loader, device, config)
        snapshot_ensembler.update(model)

        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"metrics/{key}", value, epoch)

        log_attention_maps(model, writer, epoch, config.training.output_dir / "attention")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                    "scaler_state": scaler.state_dict() if scaler is not None else None,
                    "epoch": epoch,
                    "config": model_config,
                },
                best_checkpoint,
            )
            LOGGER.info("Saved new best checkpoint to %s", best_checkpoint)
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

    LOGGER.info("Running walk-forward validation")
    wf_metrics = walk_forward_validation(model, val_dataset, device, config)
    for key, value in wf_metrics.items():
        writer.add_scalar(f"walk_forward/{key}", value)

    LOGGER.info("Averaging snapshots for stable predictions")
    ensemble_mu, ensemble_log_sigma, ensemble_targets = snapshot_ensembler.averaged_predictions(
        model, val_loader, device
    )
    ensemble_metrics = compute_metrics(ensemble_mu, ensemble_log_sigma, ensemble_targets)
    for key, value in ensemble_metrics.items():
        writer.add_scalar(f"ensemble/{key}", value)

    writer.flush()
    writer.close()

    final_state_path = config.training.output_dir / "final_state.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "config": model_config,
        },
        final_state_path,
    )
    LOGGER.info("Saved final training state to %s", final_state_path)
