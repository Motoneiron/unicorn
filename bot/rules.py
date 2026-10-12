# -*- coding: utf-8 -*-
"""Правила входа и выхода. Совпадают с бэктестом: филл важнее 2R на той же минутке, стоп раньше тейка."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from bot.config import ENTRY_MIN, LTF_MIN, MIN_NOTIONAL, MARGIN_RESERVE, RISK_PCT, LEVERAGE, RR, SLIP, MAKER, TAKER


def utc_naive() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))


def mss_close(mss_open: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(mss_open) + pd.Timedelta(minutes=LTF_MIN)


def entry_deadline(mss_open: pd.Timestamp) -> pd.Timestamp:
    return mss_close(mss_open) + pd.Timedelta(minutes=ENTRY_MIN)


def setup_id(pair: str, side: int, mss_open: pd.Timestamp) -> str:
    return f"{pair}|{int(side)}|{pd.Timestamp(mss_open).isoformat()}"


def link_id(pair: str, side: int, mss_open: pd.Timestamp) -> str:
    stamp = pd.Timestamp(mss_open).strftime("%Y%m%d%H%M")
    mark = "L" if int(side) == 1 else "S"
    return f"uni{pair}{mark}{stamp}"[:36]


def side_name(side: int) -> str:
    return "лонг" if int(side) == 1 else "шорт"


def drop_forming(df: pd.DataFrame, minutes: int, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Последний бар, который ещё не закрылся, в расчёт не берём."""
    if df is None or len(df) == 0:
        return df
    now = utc_naive() if now is None else pd.Timestamp(now)
    last = pd.Timestamp(df.index[-1])
    if last.tzinfo is not None:
        last = last.tz_convert("UTC").tz_localize(None)
    if last + pd.Timedelta(minutes=minutes) > now:
        return df.iloc[:-1]
    return df


def window_bars(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    return df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]


def tp2_price(side: int, entry: float, stop: float, rr: float = RR) -> float:
    return float(entry) + int(side) * float(rr) * abs(float(entry) - float(stop))


def first_entry_event(
    side: int,
    entry: float,
    stop: float,
    lows,
    highs,
) -> tuple[str, int]:
    """Как _fill_limit: на одной минутке филл важнее касания 2R."""
    tp2 = tp2_price(side, entry, stop, 2.0)
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        if int(side) == 1:
            filled = float(lo) <= entry
            saw = float(hi) >= tp2
        else:
            filled = float(hi) >= entry
            saw = float(lo) <= tp2
        if saw and not filled:
            return "ran", i
        if filled:
            return "fill", i
    return "miss", -1


def position_hit(side: int, stop: float, tp: float, low: float, high: float) -> str | None:
    """На одной минутке стоп раньше тейка."""
    if int(side) == 1:
        if float(low) <= stop:
            return "sl"
        if float(high) >= tp:
            return "tp"
    else:
        if float(high) >= stop:
            return "sl"
        if float(low) <= tp:
            return "tp"
    return None


def _nearest(px: float, tick: float) -> float:
    if tick <= 0:
        return float(px)
    return round(round(float(px) / tick) * tick, 10)


def _floor(px: float, tick: float) -> float:
    if tick <= 0:
        return float(px)
    return round(math.floor(float(px) / tick + 1e-9) * tick, 10)


def _ceil(px: float, tick: float) -> float:
    if tick <= 0:
        return float(px)
    return round(math.ceil(float(px) / tick - 1e-9) * tick, 10)


def round_levels(side: int, entry: float, stop: float, tick: float) -> tuple[float, float, float] | None:
    """Вход к ближайшему шагу. Стоп округляем от цены, чтобы не заузить риск."""
    entry_r = _nearest(entry, tick)
    if int(side) == 1:
        stop_r = _floor(stop, tick)
        if stop_r >= entry_r:
            return None
    else:
        stop_r = _ceil(stop, tick)
        if stop_r <= entry_r:
            return None
    tp = _nearest(tp2_price(side, entry_r, stop_r, RR), tick)
    if int(side) == 1 and tp <= entry_r:
        return None
    if int(side) == -1 and tp >= entry_r:
        return None
    return entry_r, stop_r, tp


def position_qty(
    balance: float,
    entry: float,
    stop: float,
    qty_step: float,
    risk_pct: float = RISK_PCT,
    leverage: int = LEVERAGE,
) -> float:
    """Столько монет, чтобы стоп стоил risk_pct процентов баланса. Плечо только ограничивает маржу."""
    risk = abs(float(entry) - float(stop))
    if risk <= 0 or entry <= 0 or balance <= 0 or qty_step <= 0:
        return 0.0
    qty = (balance * risk_pct / 100.0) / risk
    free = balance * (1.0 - MARGIN_RESERVE)
    margin = qty * entry / leverage
    if margin > free > 0:
        qty = free * leverage / entry
    steps = math.floor(qty / qty_step + 1e-9)
    if steps <= 0:
        return 0.0
    qty = steps * qty_step
    if qty_step < 1:
        decimals = max(0, int(round(-math.log10(qty_step))))
        qty = round(qty, decimals)
    if qty * entry < MIN_NOTIONAL:
        return 0.0
    return float(qty)


def exit_price(side: int, stop: float, tp: float, reason: str) -> tuple[float, float]:
    """Цена выхода и ставка комиссии. Стоп со слипом 1 б.п., тейк по лимитке."""
    if reason == "tp":
        return float(tp), MAKER
    slip_px = float(stop) * (1.0 - int(side) * SLIP)
    return slip_px, TAKER


def pnl_usdt(side: int, entry: float, exit_px: float, qty: float, fee_out: float) -> tuple[float, float]:
    gross = int(side) * (float(exit_px) - float(entry)) * float(qty)
    fees = MAKER * float(entry) * float(qty) + fee_out * float(exit_px) * float(qty)
    return gross - fees, fees


def fmt_px(px: float, tick: float) -> str:
    if tick <= 0:
        return f"{px:.8f}".rstrip("0").rstrip(".")
    if tick >= 1:
        decimals = 0
    else:
        decimals = max(0, int(round(-math.log10(tick))))
    text = f"{px:.{decimals}f}"
    return text


def fmt_num(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")
