"""Utilities for interacting with the Binance REST API for market data.

This module only uses the public market-data endpoints, which means it does not require
an authenticated session to download historical candles. The API key parameters are
present so that the same client can be extended to trade or access private endpoints
if desired.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests

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
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.api_key = api_key
        self.session = session or requests.Session()

        if api_key:
            self.session.headers.update({"X-MBX-APIKEY": api_key})

    def get_klines(
        self,
        start_time: dt.datetime,
        end_time: dt.datetime,
        limit: int = 1000,
    ) -> List[BinanceCandle]:
        """Download OHLCV candles between ``start_time`` and ``end_time``.

        Binance caps requests at 1,000 candles per call, therefore this method will
        paginate transparently. The ``limit`` parameter is kept to allow the caller to
        shorten the pagination size in case a different rate-limiting strategy is
        desired.
        """

        start_ms = _to_milliseconds(start_time)
        end_ms = _to_milliseconds(end_time)

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
            response = self.session.get(f"{BINANCE_BASE_URL}/api/v3/klines", params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()

            if not payload:
                break

            batch = [
                BinanceCandle(
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
            next_start = int(payload[-1][6]) + 1  # move past the last close time

        return candles

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
