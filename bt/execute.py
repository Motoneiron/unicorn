# -*- coding: utf-8 -*-
"""Исполнение как в прогонах cisdbot, под лимитку Единорога.

Вход — лимитка, комиссия мейкер 0,036%. Тейк — лимитка, тоже мейкер.
Стоп и выход по концу данных — тейкер 0,10% и 1 б.п. проскальзывания против нас.
На одной свече стоп проверяется раньше тейка.

Рядом считается плоская комиссия из ТЗ: 0,06% на вход и 0,06% на выход,
с тем же проскальзыванием на стопе.

Одна заявка или позиция за раз. Жизнь зоны — 24 бара M15 после свечи слома.
Лимитка снимается, если 2R случился раньше филла. На одной минутке филл важнее.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAKER = 0.00036
TAKER = 0.00100
SLIP = 0.00010
FEE06 = 0.00060
LIFE = 24


def r_multiples(
    side: int,
    entry: float,
    stop: float,
    exit_px: float,
    fee_out: float,
    fee_in: float = MAKER,
    fill_px: float | None = None,
) -> tuple[float, float, float]:
    """R от диапазона вход–стоп. fill_px отличается от entry, если вход со слипом."""
    fill = entry if fill_px is None else fill_px
    risk = abs(entry - stop)
    sp = risk / entry
    gross = side * (exit_px - fill) / risk
    r = gross - fee_in / sp - fee_out / sp
    r06 = gross - FEE06 / sp - FEE06 / sp
    return float(r), float(r06), float(gross)


def run_book(
    setups: list[dict],
    m15: pd.DataFrame,
    path: pd.DataFrame,
    rr: float,
    need_h4: bool,
    entry_kind: str = "limit",
    step_min: int = 5,
    skip_ran_2r: bool = True,
) -> tuple[list[dict], dict]:
    ts = path.index.to_numpy()
    hi = path["high"].to_numpy(float)
    lo = path["low"].to_numpy(float)
    cl = path["close"].to_numpy(float)
    opens = m15.index.to_numpy()
    free = np.datetime64("1970-01-01")
    step = np.timedelta64(step_min, "m")
    skipped_h4 = blocked = expired = invalid = ran_2r = 0
    trades: list[dict] = []
    for s in setups:
        if need_h4 and s["h4"] != s["side"]:
            skipped_h4 += 1
            continue
        known_close = opens[s["known_i"]] + np.timedelta64(15, "m")
        if entry_kind == "stop":
            deadline = known_close + np.timedelta64(15 * LIFE, "m")
        else:
            deadline = opens[s["form_i"]] + np.timedelta64(15 * (LIFE + 1), "m")
        if known_close < free:
            blocked += 1
            continue
        if known_close >= deadline:
            expired += 1
            continue
        side = int(s["side"])
        entry = float(s["entry"])
        stop = float(s["stop"])
        risk = abs(entry - stop)
        tp = entry + side * rr * risk
        i0 = int(np.searchsorted(ts, known_close, side="left"))
        i1 = int(np.searchsorted(ts, deadline, side="left"))
        if entry_kind == "stop":
            kind, filled = _fill_stop(side, entry, stop, lo, hi, i0, i1)
            if kind == "invalid":
                invalid += 1
                free = ts[filled] + step
                continue
            if kind != "fill":
                expired += 1
                free = deadline
                continue
            fill_px = entry * (1.0 + side * SLIP)
            fee_in = TAKER
        else:
            if skip_ran_2r:
                kind, filled = _fill_limit(side, entry, stop, lo, hi, i0, i1)
            else:
                filled = _fill(side, entry, lo, hi, i0, i1)
                kind = "fill" if filled >= 0 else "miss"
            if kind == "ran":
                ran_2r += 1
                free = ts[filled] + step
                continue
            if kind != "fill":
                expired += 1
                free = deadline
                continue
            fill_px = entry
            fee_in = MAKER
        outcome, exit_i, exit_px, fee = _walk(side, stop, tp, lo, hi, cl, filled)
        r, r06, gross = r_multiples(side, entry, stop, exit_px, fee, fee_in=fee_in, fill_px=fill_px)
        free = ts[exit_i] + step
        name = "long" if side == 1 else "short"
        trades.append(
            {
                "side": name,
                "rr": rr,
                "h4": int(s["h4"]),
                "outcome": outcome,
                "r": r,
                "r06": r06,
                "gross": gross,
                "fill_px": fill_px,
                "entry": entry,
                "stop": stop,
                "tp": tp,
                "exit": exit_px,
                "stop_pct": risk / entry * 100.0,
                "formed": pd.Timestamp(opens[s["form_i"]]),
                "known": pd.Timestamp(known_close),
                "sweep_time": pd.Timestamp(opens[s["sweep_i"]]),
                "h4_sweep_at": s.get("h4_sweep_at"),
                "fill": pd.Timestamp(ts[filled]),
                "exit_time": pd.Timestamp(ts[exit_i]),
                "hold_min": int(exit_i - filled) * step_min,
                "zone_bot": s["zone_bot"],
                "zone_top": s["zone_top"],
            }
        )
    stats = {
        "skipped_h4": skipped_h4,
        "blocked": blocked,
        "expired": expired,
        "invalid": invalid,
        "ran_2r": ran_2r,
        "filled": len(trades),
    }
    return trades, stats


def _fill_stop(side: int, entry: float, stop: float, lo: np.ndarray, hi: np.ndarray, i0: int, i1: int):
    """Покупка по пробою хая (лонг) или продажа по пробою лоу (шорт).

    Если раньше пробоя цена снимает стоп, заявка снимается.
    Если на одной минутке и пробой, и стоп — это филл, дальше стоп проверяется первым.
    """
    for i in range(i0, i1):
        if side == 1:
            if lo[i] <= stop and hi[i] < entry:
                return "invalid", i
            if hi[i] >= entry:
                return "fill", i
        else:
            if hi[i] >= stop and lo[i] > entry:
                return "invalid", i
            if lo[i] <= entry:
                return "fill", i
    return "miss", -1


def _fill_limit(
    side: int,
    entry: float,
    stop: float,
    lo: np.ndarray,
    hi: np.ndarray,
    i0: int,
    i1: int,
):
    """Лимитка. Если 2R случился на более ранней минутке, чем филл, заявку снимаем.

    На одной минутке филл важнее 2R: фитиль мог сначала задеть лимитку, потом уйти к тейку.
    """
    tp2 = entry + side * 2.0 * abs(entry - stop)
    for i in range(i0, i1):
        if side == 1:
            filled = lo[i] <= entry
            saw_2r = hi[i] >= tp2
        else:
            filled = hi[i] >= entry
            saw_2r = lo[i] <= tp2
        if saw_2r and not filled:
            return "ran", i
        if filled:
            return "fill", i
    return "miss", -1


def _fill(side: int, entry: float, lo: np.ndarray, hi: np.ndarray, i0: int, i1: int) -> int:
    if side == 1:
        for i in range(i0, i1):
            if lo[i] <= entry:
                return i
    else:
        for i in range(i0, i1):
            if hi[i] >= entry:
                return i
    return -1


def _walk(side, stop, tp, lo, hi, cl, filled):
    n = len(lo)
    for j in range(filled, n):
        if side == 1:
            sl_hit = lo[j] <= stop
            tp_hit = hi[j] >= tp
        else:
            sl_hit = hi[j] >= stop
            tp_hit = lo[j] <= tp
        if sl_hit:
            return "sl", j, stop * (1.0 - side * SLIP), TAKER
        if tp_hit:
            return "tp", j, tp, MAKER
    last = n - 1
    return "eod", last, float(cl[last]) * (1.0 - side * SLIP), TAKER
