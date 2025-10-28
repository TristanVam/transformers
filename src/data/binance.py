"""Utilities for interacting with the Binance REST API for market data.

This module only uses the public market-data endpoints, which means it does not require
an authenticated session to download historical candles. The API key parameters are
present so that the same client can be extended to trade or access private endpoints
if desired.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import requests
import numpy as np

LOGGER = logging.getLogger(__name__)


BINANCE_BASE_URL = "https://api.binance.com"


def _to_milliseconds(timestamp: dt.datetime) -> int:
    """Convert a timezone-aware datetime into the integer expected by Binance."""

    if timestamp.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware (UTC recommended).")
    return int(timestamp.timestamp() * 1000)


@dataclass
class BinanceCandle:
    """Simple representation of a single Kline returned by Binance."""

    symbol: str
    open_time: dt.datetime
    close_time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def midpoint(self) -> float:
        """Convenience accessor used by several preprocessing pipelines."""

        return 0.5 * (self.high + self.low)


class BinanceDataClient:
    """Minimal REST client for downloading Binance spot klines.

    The implementation uses only standard library dependencies and handles pagination
    automatically so that multi-month or multi-year downloads can be performed with a
    single call. Results are returned as :class:`BinanceCandle` instances to keep the
    downstream pipeline strongly typed.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        cache_dir: Optional[Path] = None,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.api_key = api_key
        self.session = session or requests.Session()
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        if api_key:
            self.session.headers.update({"X-MBX-APIKEY": api_key})

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, day: dt.date) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        name = f"{self.symbol}_{self.interval}_{day.isoformat()}.npy"
        return self.cache_dir / name

    def _load_cache(self, day: dt.date) -> Optional[List[BinanceCandle]]:
        path = self._cache_path(day)
        if path is None or not path.exists():
            return None
        payload = np.load(path, allow_pickle=False)
        candles = [
            BinanceCandle(
                symbol=str(item["symbol"]),
                open_time=dt.datetime.fromtimestamp(float(item["open_time"]), tz=dt.timezone.utc),
                close_time=dt.datetime.fromtimestamp(float(item["close_time"]), tz=dt.timezone.utc),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item["volume"]),
            )
            for item in payload
        ]
        return candles

    def _store_cache(self, candles: Sequence[BinanceCandle]) -> None:
        if self.cache_dir is None or not candles:
            return
        grouped: dict[dt.date, list[BinanceCandle]] = {}
        for candle in candles:
            grouped.setdefault(candle.open_time.date(), []).append(candle)
        for day, items in grouped.items():
            path = self._cache_path(day)
            if path is None:
                continue
            dtype = np.dtype(
                [
                    ("symbol", "U16"),
                    ("open_time", "f8"),
                    ("close_time", "f8"),
                    ("open", "f8"),
                    ("high", "f8"),
                    ("low", "f8"),
                    ("close", "f8"),
                    ("volume", "f8"),
                ]
            )
            arr = np.zeros(len(items), dtype=dtype)
            arr["symbol"] = [candle.symbol for candle in items]
            arr["open_time"] = [candle.open_time.timestamp() for candle in items]
            arr["close_time"] = [candle.close_time.timestamp() for candle in items]
            arr["open"] = [candle.open for candle in items]
            arr["high"] = [candle.high for candle in items]
            arr["low"] = [candle.low for candle in items]
            arr["close"] = [candle.close for candle in items]
            arr["volume"] = [candle.volume for candle in items]
            np.save(path, arr)

    @staticmethod
    def _validate_timezone(timestamps: Sequence[dt.datetime]) -> None:
        for ts in timestamps:
            if ts.tzinfo is None:
                LOGGER.warning("Timestamp %s missing timezone information; expected UTC", ts)
                break
            if ts.utcoffset() != dt.timedelta(0):
                LOGGER.warning("Timestamp %s not in UTC; downstream processing assumes UTC", ts)
                break

    def _download_range(self, start_ms: int, end_ms: int, limit: int) -> List[BinanceCandle]:
        candles: List[BinanceCandle] = []
        next_start = start_ms
        while next_start < end_ms:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "startTime": next_start,
                "endTime": end_ms,
                "limit": limit,
            }

            LOGGER.debug("Requesting Binance klines", extra=params)
            delay = 1.0
            for attempt in range(self.max_retries):
                try:
                    response = self.session.get(
                        f"{BINANCE_BASE_URL}/api/v3/klines", params=params, timeout=10
                    )
                    response.raise_for_status()
                    payload = response.json()
                    break
                except requests.HTTPError as exc:  # pragma: no cover - network dependent
                    status = exc.response.status_code if exc.response else None
                    if status == 429 and attempt + 1 < self.max_retries:
                        LOGGER.warning("Binance rate limit hit, backing off for %.2fs", delay)
                        time.sleep(delay)
                        delay *= self.backoff_factor
                        continue
                    raise
                except requests.RequestException:  # pragma: no cover - network dependent
                    if attempt + 1 >= self.max_retries:
                        raise
                    LOGGER.warning("Request failed, retrying in %.2fs", delay)
                    time.sleep(delay)
                    delay *= self.backoff_factor
            else:  # pragma: no cover
                payload = []

            if not payload:
                break

            batch = [
                BinanceCandle(
                    symbol=self.symbol,
                    open_time=dt.datetime.fromtimestamp(item[0] / 1000, tz=dt.timezone.utc),
                    close_time=dt.datetime.fromtimestamp(item[6] / 1000, tz=dt.timezone.utc),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                )
                for item in payload
            ]

            candles.extend(batch)
            next_start = int(payload[-1][6]) + 1
        return candles

    def get_klines(
        self,
        start_time: dt.datetime,
        end_time: dt.datetime,
        limit: int = 1000,
    ) -> List[BinanceCandle]:
        """Download OHLCV candles between ``start_time`` and ``end_time``."""

        if end_time <= start_time:
            raise ValueError("end_time must be after start_time")

        candles: List[BinanceCandle] = []
        missing_ranges: List[tuple[dt.datetime, dt.datetime]] = []

        if self.cache_dir is not None:
            day_start = start_time.date()
            day_end = end_time.date()
            current = day_start
            pending_start: Optional[dt.datetime] = None

            while current <= day_end:
                day_open = dt.datetime.combine(current, dt.time.min, tzinfo=start_time.tzinfo)
                day_close = day_open + dt.timedelta(days=1)
                cached = self._load_cache(current)
                if cached is not None:
                    candles.extend(
                        candle
                        for candle in cached
                        if start_time <= candle.close_time <= end_time
                    )
                    if pending_start is not None:
                        missing_ranges.append((pending_start, day_open))
                        pending_start = None
                else:
                    pending_start = pending_start or day_open
                current += dt.timedelta(days=1)

            if pending_start is not None:
                missing_ranges.append((pending_start, end_time))
        else:
            missing_ranges = [(start_time, end_time)]

        for range_start, range_end in missing_ranges:
            start_ms = _to_milliseconds(range_start)
            end_ms = _to_milliseconds(range_end)
            downloaded = self._download_range(start_ms, end_ms, limit)
            candles.extend(downloaded)
            self._store_cache(downloaded)

        candles.sort(key=lambda c: c.close_time)
        self._validate_timezone([c.open_time for c in candles])
        self._validate_timezone([c.close_time for c in candles])
        return [c for c in candles if start_time <= c.close_time <= end_time]

    def stream_klines(
        self,
        start_time: dt.datetime,
        end_time: dt.datetime,
        chunk_hours: int = 24,
    ) -> Iterable[List[BinanceCandle]]:
        """Yield candle batches, which is useful for incremental feature pipelines."""

        current = start_time
        delta = dt.timedelta(hours=chunk_hours)
        while current < end_time:
            batch_end = min(current + delta, end_time)
            yield self.get_klines(current, batch_end)
            current = batch_end
