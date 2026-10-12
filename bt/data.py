# -*- coding: utf-8 -*-
"""Свечи Bybit linear BTCUSDT из кэша ORB. Время бара — open, UTC, без таймзоны."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BTC_5M = Path(r"D:\PYTHON\Opening Range Breakout (ORB)\data\BTCUSDT_5m_bybit.csv")
SANITY_TS = pd.Timestamp("2024-06-12 13:30:00")
SANITY_CLOSE = 69374.1


def load_5m(path: Path = BTC_5M) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df.set_index("timestamp")
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    hit = df.loc[SANITY_TS, "close"] if SANITY_TS in df.index else None
    if hit is None or abs(float(hit) - SANITY_CLOSE) > 0.05:
        raise RuntimeError(f"свечи не сходятся с Bybit: {SANITY_TS} close={hit}")
    return df


def load_ms(path: Path) -> pd.DataFrame:
    """Минутный OHLCV, open_time в миллисекундах, как в D:\\databtcusdt."""
    df = pd.read_csv(path, usecols=["open_time", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["open_time"], unit="ms")
    df = df.drop(columns=["open_time"])
    df.index = ts
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


def resample(df: pd.DataFrame, rule: str, need: int) -> pd.DataFrame:
    """label=left, closed=left, как в прогонах cisdbot. Неполный крайний бакет выбрасываем."""
    g = df.resample(rule, label="left", closed="left")
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
        }
    )
    cnt = g["close"].count()
    return out.loc[cnt >= need].dropna(subset=["open"])
