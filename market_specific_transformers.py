"""Market-specific Transformer architectures for financial time-series forecasting.

This module defines three tailored Transformer configurations for
NASDAQ equities, gold futures, and Bitcoin spot markets. Each
configuration reuses the same core data pipeline while adapting the
hyper-parameters, lookback horizon, and technical indicators to match
unique market microstructure characteristics.

Usage (CLI):
    python market_specific_transformers.py --market nasdaq --mode train --csv data.csv

The CLI mirrors the behaviour of the original single-market pipeline:
    * train   - fit and persist the selected market model
    * evaluate/backtest - load a saved model and compute P&L metrics
    * hpsearch - (optional) run Optuna search using the market defaults

Each market configuration predicts the next candle (horizon=1).
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers

# --------------------------------------------------------------------------------------
# Reproducibility and mixed precision toggle
# --------------------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

USE_MIXED_PRECISION = False
if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")


# --------------------------------------------------------------------------------------
# Configuration objects
# --------------------------------------------------------------------------------------
@dataclass
class IndicatorRecipe:
    """Defines custom indicator options per market."""

    ema_spans: Tuple[int, int] = (8, 21)
    rsi_period: int = 14
    atr_period: int = 14
    vwap_window: int = 14
    momentum_windows: Tuple[int, ...] = (10, 20)
    volatility_windows: Tuple[int, ...] = (10, 30)
    volume_windows: Tuple[int, ...] = (10, 30)


@dataclass
class MarketTransformerConfig:
    market: str
    input_len: int
    horizon: int
    d_model: int
    num_heads: int
    num_layers: int
    dff: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    classification_threshold: float
    weight_regression: float
    weight_classification: float
    indicator_recipe: IndicatorRecipe = field(default_factory=IndicatorRecipe)
    include_time_features: bool = True
    description: str = ""


MARKET_CONFIGS: Dict[str, MarketTransformerConfig] = {
    "nasdaq": MarketTransformerConfig(
        market="nasdaq",
        input_len=90,
        horizon=1,
        d_model=192,
        num_heads=6,
        num_layers=4,
        dff=384,
        dropout=0.15,
        batch_size=128,
        epochs=40,
        learning_rate=8e-5,
        classification_threshold=0.0015,
        weight_regression=1.0,
        weight_classification=0.6,
        indicator_recipe=IndicatorRecipe(
            ema_spans=(8, 34),
            rsi_period=21,
            atr_period=14,
            vwap_window=20,
            momentum_windows=(10, 21),
            volatility_windows=(14, 30),
            volume_windows=(10, 40),
        ),
        description="Equity-focused architecture with longer memory and moderate dropout",
    ),
    "gold": MarketTransformerConfig(
        market="gold",
        input_len=120,
        horizon=1,
        d_model=160,
        num_heads=5,
        num_layers=3,
        dff=320,
        dropout=0.1,
        batch_size=96,
        epochs=35,
        learning_rate=1e-4,
        classification_threshold=0.0008,
        weight_regression=1.0,
        weight_classification=0.5,
        indicator_recipe=IndicatorRecipe(
            ema_spans=(13, 55),
            rsi_period=14,
            atr_period=21,
            vwap_window=30,
            momentum_windows=(15, 45),
            volatility_windows=(20, 60),
            volume_windows=(15, 60),
        ),
        include_time_features=False,
        description="Commodity model emphasising smooth momentum and longer ATR window",
    ),
    "btc": MarketTransformerConfig(
        market="btc",
        input_len=72,
        horizon=1,
        d_model=256,
        num_heads=8,
        num_layers=5,
        dff=512,
        dropout=0.2,
        batch_size=160,
        epochs=50,
        learning_rate=5e-5,
        classification_threshold=0.003,
        weight_regression=1.0,
        weight_classification=0.7,
        indicator_recipe=IndicatorRecipe(
            ema_spans=(5, 21),
            rsi_period=12,
            atr_period=14,
            vwap_window=14,
            momentum_windows=(5, 13, 34),
            volatility_windows=(7, 21),
            volume_windows=(7, 21),
        ),
        include_time_features=True,
        description="High-volatility crypto variant with deeper encoder and stronger dropout",
    ),
}


# --------------------------------------------------------------------------------------
# Feature engineering helpers
# --------------------------------------------------------------------------------------
def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def compute_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    num = (typical_price * df["volume"]).rolling(window=window).sum()
    den = df["volume"].rolling(window=window).sum()
    return num / (den + 1e-9)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["dow"] = ts.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    return df


def add_indicator_set(df: pd.DataFrame, recipe: IndicatorRecipe) -> pd.DataFrame:
    ema_fast, ema_slow = recipe.ema_spans
    df[f"ema_{ema_fast}"] = compute_ema(df["close"], ema_fast)
    df[f"ema_{ema_slow}"] = compute_ema(df["close"], ema_slow)
    df["rsi"] = compute_rsi(df["close"], recipe.rsi_period)
    df["atr"] = compute_atr(df, recipe.atr_period)
    df["vwap"] = compute_vwap(df, recipe.vwap_window)

    for window in recipe.momentum_windows:
        df[f"mom_{window}"] = df["close"].pct_change(periods=window)
    for window in recipe.volatility_windows:
        df[f"vol_{window}"] = df["close"].pct_change().rolling(window).std()
    for window in recipe.volume_windows:
        df[f"vol_chg_{window}"] = df["volume"].pct_change(periods=window)

    df["close_minus_ema_fast"] = df["close"] - df[f"ema_{ema_fast}"]
    df["close_minus_ema_slow"] = df["close"] - df[f"ema_{ema_slow}"]
    df["close_div_vwap"] = df["close"] / (df["vwap"] + 1e-9)
    df["high_low_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    return df


def feature_engineer(df: pd.DataFrame, config: MarketTransformerConfig) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic returns and volume features
    df["open_r"] = df["open"].pct_change().fillna(0)
    df["high_r"] = df["high"].pct_change().fillna(0)
    df["low_r"] = df["low"].pct_change().fillna(0)
    df["close_r"] = df["close"].pct_change().fillna(0)
    df["log_volume"] = np.log1p(df["volume"].fillna(0))

    df = add_indicator_set(df, config.indicator_recipe)

    if config.include_time_features:
        df = add_time_features(df)

    base_features = [
        "open_r",
        "high_r",
        "low_r",
        "close_r",
        "log_volume",
        "close_minus_ema_fast",
        "close_minus_ema_slow",
        "close_div_vwap",
        "rsi",
        "atr",
        "high_low_range",
    ]

    momentum_features = [f"mom_{w}" for w in config.indicator_recipe.momentum_windows]
    volatility_features = [f"vol_{w}" for w in config.indicator_recipe.volatility_windows]
    volume_features = [f"vol_chg_{w}" for w in config.indicator_recipe.volume_windows]
    feature_cols = base_features + momentum_features + volatility_features + volume_features

    if config.include_time_features:
        feature_cols += ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, feature_cols


# --------------------------------------------------------------------------------------
# Window creation
# --------------------------------------------------------------------------------------
def make_windows(
    features: np.ndarray,
    targets_reg: np.ndarray,
    signal: np.ndarray,
    window_size: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y_reg, y_cls = [], [], []
    total = len(features)
    for idx in range(total - window_size - horizon + 1):
        X.append(features[idx : idx + window_size])
        y_reg.append(targets_reg[idx + window_size - 1 + horizon])
        y_cls.append(signal[idx + window_size - 1 + horizon])
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y_reg, dtype=np.float32),
        np.asarray(y_cls, dtype=np.int32),
    )


# --------------------------------------------------------------------------------------
# Positional encoding and pooling
# --------------------------------------------------------------------------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / (10000 ** (2 * (i // 2) / np.float32(d_model)))
        angle_rads = pos * angle_rates
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.positional_encoding = tf.constant(pe, dtype=tf.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        seq_len = tf.shape(x)[1]
        return x + self.positional_encoding[:seq_len, :]


class AttentionPooling(layers.Layer):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = layers.Dense(1)

    def call(self, x: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:  # type: ignore[override]
        scores = self.proj(x)
        if mask is not None:
            scores -= 1e9 * (1 - tf.cast(mask[:, :, tf.newaxis], tf.float32))
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(weights * x, axis=1)


# --------------------------------------------------------------------------------------
# Transformer blocks
# --------------------------------------------------------------------------------------
def transformer_encoder_layer(
    d_model: int,
    num_heads: int,
    dff: int,
    dropout: float,
) -> Model:
    inp = layers.Input(shape=(None, d_model))
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(inp, inp)
    attn = layers.Dropout(dropout)(attn)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inp + attn)

    ffn = layers.Dense(dff, activation="relu")(out1)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return Model(inputs=inp, outputs=out2)


def decoder_head(d_model: int, dropout: float) -> Model:
    encoder_out = layers.Input(shape=(None, d_model))
    pooled = AttentionPooling(d_model)(encoder_out)
    last_token = layers.Lambda(lambda t: t[:, -1, :])(encoder_out)
    x = layers.Concatenate()([pooled, last_token])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(d_model, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    return Model(inputs=encoder_out, outputs=x)


def build_model(config: MarketTransformerConfig, n_features: int) -> Model:
    inputs = layers.Input(shape=(config.input_len, n_features), name="price_window")
    x = layers.Dense(config.d_model)(inputs)
    x = PositionalEncoding(config.d_model)(x)

    for _ in range(config.num_layers):
        enc = transformer_encoder_layer(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dff=config.dff,
            dropout=config.dropout,
        )
        x = enc(x)

    dec = decoder_head(config.d_model, config.dropout)(x)

    reg = layers.Dense(config.d_model // 2, activation="relu")(dec)
    reg = layers.Dropout(config.dropout)(reg)
    reg_out = layers.Dense(4, name="next_ohlc", dtype="float32")(reg)

    cls = layers.Dense(config.d_model // 2, activation="relu")(dec)
    cls = layers.Dropout(config.dropout)(cls)
    cls_out = layers.Dense(3, activation="softmax", name="signal", dtype="float32")(cls)

    return Model(inputs=inputs, outputs=[reg_out, cls_out], name=f"transformer_{config.market}")


def compile_model(model: Model, config: MarketTransformerConfig) -> Model:
    losses = {
        "next_ohlc": tf.keras.losses.MeanAbsoluteError(),
        "signal": tf.keras.losses.SparseCategoricalCrossentropy(),
    }
    weights = {
        "next_ohlc": config.weight_regression,
        "signal": config.weight_classification,
    }
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=weights,
        metrics={
            "next_ohlc": [tf.keras.metrics.MeanAbsoluteError()],
            "signal": [tf.keras.metrics.SparseCategoricalAccuracy()],
        },
    )
    return model


# --------------------------------------------------------------------------------------
# Data prep utilities
# --------------------------------------------------------------------------------------
def prepare_data(csv_path: str, config: MarketTransformerConfig):
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column")

    df_fe, feature_cols = feature_engineer(df, config)

    next_close_return = df_fe["close"].pct_change().shift(-1).fillna(0).values
    signal = np.where(
        next_close_return > config.classification_threshold,
        2,
        np.where(next_close_return < -config.classification_threshold, 0, 1),
    )

    target_reg = np.stack(
        [
            df_fe["open"].pct_change().shift(-1).fillna(0).values,
            df_fe["high"].pct_change().shift(-1).fillna(0).values,
            df_fe["low"].pct_change().shift(-1).fillna(0).values,
            next_close_return,
        ],
        axis=-1,
    ).astype(np.float32)

    features = df_fe[feature_cols].astype(np.float32).values
    X, y_reg, y_cls = make_windows(
        features,
        target_reg,
        signal,
        window_size=config.input_len,
        horizon=config.horizon,
    )

    total = len(X)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_reg_train, y_reg_val, y_reg_test = (
        y_reg[:train_end],
        y_reg[train_end:val_end],
        y_reg[val_end:],
    )
    y_cls_train, y_cls_val, y_cls_test = (
        y_cls[:train_end],
        y_cls[train_end:val_end],
        y_cls[val_end:],
    )

    return (
        df_fe,
        feature_cols,
        (X_train, y_reg_train, y_cls_train),
        (X_val, y_reg_val, y_cls_val),
        (X_test, y_reg_test, y_cls_test),
    )


def signals_to_positions(signals: np.ndarray) -> np.ndarray:
    positions = np.zeros_like(signals, dtype=np.int8)
    positions[signals == 2] = 1
    positions[signals == 0] = -1
    return positions


def backtest_from_predictions(
    df: pd.DataFrame,
    preds_reg: np.ndarray,
    preds_cls: np.ndarray,
    entry_fee: float = 0.0005,
    slippage: float = 0.0005,
    initial_capital: float = 1.0,
):
    n = min(len(df), len(preds_cls))
    df = df.iloc[-n:].reset_index(drop=True)
    preds_reg = preds_reg[-n:]
    preds_cls = preds_cls[-n:]

    positions = signals_to_positions(preds_cls)
    future_returns = df["close"].pct_change().shift(-1).fillna(0).values
    pnl = positions * future_returns

    prev_positions = np.concatenate([[0], positions[:-1]])
    trades = (positions != prev_positions).astype(float)
    costs = trades * (entry_fee + slippage)

    pnl_after_costs = pnl - costs
    cumulative = np.cumprod(1 + pnl_after_costs) * initial_capital

    returns = pnl_after_costs
    mean_ret = np.mean(returns)
    vol = np.std(returns)
    sharpe = (mean_ret / (vol + 1e-9)) * math.sqrt(252 * 24)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / (peak + 1e-9)
    max_drawdown = np.min(drawdown)

    return {
        "cumulative": cumulative,
        "pnl_after_costs": pnl_after_costs,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_value": cumulative[-1],
    }


# --------------------------------------------------------------------------------------
# Training / evaluation flows
# --------------------------------------------------------------------------------------
def train_and_save(csv_path: str, config: MarketTransformerConfig, model_dir: str):
    df_fe, feature_cols, train, val, test = prepare_data(csv_path, config)
    X_train, y_reg_train, y_cls_train = train
    X_val, y_reg_val, y_cls_val = val
    X_test, y_reg_test, y_cls_test = test

    model = build_model(config, X_train.shape[2])
    model = compile_model(model, config)
    model.summary()

    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (X_train, {"next_ohlc": y_reg_train, "signal": y_cls_train})
        )
        .shuffle(2048)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            (X_val, {"next_ohlc": y_reg_val, "signal": y_cls_val})
        )
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=config.epochs, callbacks=callbacks)

    os.makedirs(model_dir, exist_ok=True)
    market_dir = os.path.join(model_dir, config.market)
    os.makedirs(market_dir, exist_ok=True)

    model.save(os.path.join(market_dir, f"transformer_{config.market}"), include_optimizer=False)
    np.save(os.path.join(market_dir, "feature_cols.npy"), np.array(feature_cols))
    print(f"Saved {config.market} model to {market_dir}")

    test_ds = (
        tf.data.Dataset.from_tensor_slices(
            (X_test, {"next_ohlc": y_reg_test, "signal": y_cls_test})
        )
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    print("Test set evaluation:")
    print(model.evaluate(test_ds))

    return model, df_fe, test


def evaluate_and_backtest(csv_path: str, config: MarketTransformerConfig, model_dir: str):
    market_dir = os.path.join(model_dir, config.market)
    model = tf.keras.models.load_model(os.path.join(market_dir, f"transformer_{config.market}"))
    feature_cols = np.load(os.path.join(market_dir, "feature_cols.npy"), allow_pickle=True)

    df_fe, _, _, _, test = prepare_data(csv_path, config)
    X_test, _, _ = test

    preds_reg, preds_cls_prob = model.predict(
        X_test, batch_size=config.batch_size, verbose=0
    )
    preds_cls = np.argmax(preds_cls_prob, axis=1)

    results = backtest_from_predictions(df_fe.iloc[-len(preds_cls) :], preds_reg, preds_cls)
    print(
        f"Backtest metrics ({config.market}): Sharpe {results['sharpe']:.3f} | "
        f"MaxDD {results['max_drawdown']:.3f} | Final value {results['final_value']:.3f}"
    )
    return results


def run_hp_search(csv_path: str, config: MarketTransformerConfig, n_trials: int):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed. Please install it to run hpsearch") from exc

    df_fe, feature_cols, train, val, _ = prepare_data(csv_path, config)
    X_train, y_reg_train, y_cls_train = train
    X_val, y_reg_val, y_cls_val = val

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [config.d_model // 2, config.d_model, config.d_model * 2])
        num_layers = trial.suggest_int("num_layers", max(2, config.num_layers - 1), config.num_layers + 1)
        num_heads = trial.suggest_categorical("num_heads", [config.num_heads, max(2, config.num_heads // 2)])
        dff = trial.suggest_categorical("dff", [config.dff, config.dff * 2])
        lr = trial.suggest_loguniform("lr", config.learning_rate / 5, config.learning_rate * 5)
        batch = trial.suggest_categorical("batch_size", [config.batch_size, max(64, config.batch_size // 2)])

        trial_config = MarketTransformerConfig(
            **{**config.__dict__, "d_model": d_model, "num_layers": num_layers, "num_heads": num_heads, "dff": dff, "learning_rate": lr, "batch_size": batch}
        )

        model = build_model(trial_config, X_train.shape[2])
        model = compile_model(model, trial_config)

        train_ds = (
            tf.data.Dataset.from_tensor_slices(
                (X_train, {"next_ohlc": y_reg_train, "signal": y_cls_train})
            )
            .shuffle(2048)
            .batch(trial_config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices(
                (X_val, {"next_ohlc": y_reg_val, "signal": y_cls_val})
            )
            .batch(trial_config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ]
        model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks, verbose=0)
        evaluation = model.evaluate(val_ds, verbose=0)
        return evaluation[0]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best trial:", study.best_trial.params)
    return study


# --------------------------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=MARKET_CONFIGS.keys(), required=True)
    parser.add_argument("--mode", choices=["train", "evaluate", "backtest", "hpsearch"], required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="market_models")
    parser.add_argument("--n_trials", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    config = MARKET_CONFIGS[args.market]
    print(f"Selected market: {config.market} -> {config.description}")

    if args.mode == "train":
        train_and_save(args.csv, config, args.model_dir)
    elif args.mode in {"evaluate", "backtest"}:
        evaluate_and_backtest(args.csv, config, args.model_dir)
    elif args.mode == "hpsearch":
        run_hp_search(args.csv, config, args.n_trials)


if __name__ == "__main__":
    main()
