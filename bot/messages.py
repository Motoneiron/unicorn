# -*- coding: utf-8 -*-
"""Карточки в Telegram в том же виде, что у cisdbot, без его фильтров."""
from __future__ import annotations

import pandas as pd

from bot.config import LEVERAGE, PAIRS, RISK_PCT
from bot.rules import fmt_px

TF = "15m → 4H"


def balance_line(balance: float) -> str:
    return f"💼 Баланс: {float(balance):.2f} USDT"


def mode_prefix(paper: bool) -> str:
    return "🧪 SHADOW" if paper else "🔴 LIVE"


def direction(side: int) -> str:
    return "LONG" if int(side) == 1 else "SHORT"


def arrow(side: int) -> str:
    return "📈" if int(side) == 1 else "📉"


def instrument(pair: str) -> str:
    return f"{pair}/USDT"


def price(pair: str, px: float) -> str:
    tick = PAIRS[pair]["price_tick"]
    return fmt_px(float(px), tick)


def clock(raw: str) -> str:
    if not raw:
        return ""
    try:
        ts = pd.Timestamp(raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(raw)[:16]


def moscow_hm(raw: str) -> str:
    if not raw:
        return ""
    try:
        ts = pd.Timestamp(raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.tz_convert("Europe/Moscow").strftime("%H:%M")
    except Exception:
        return ""


def _signal_line(rec: dict) -> str:
    sig = clock(str(rec.get("signal_time") or rec.get("mss_open") or ""))
    htf = clock(str(rec.get("htf_time") or ""))
    if htf:
        return f"Сигнал: {sig} (HTF {htf})"
    return f"Сигнал: {sig}" if sig else "Сигнал: —"


def _rr(entry: float, stop: float, tp: float) -> float:
    risk = abs(float(entry) - float(stop))
    if risk <= 0:
        return 0.0
    return abs(float(tp) - float(entry)) / risk


def _num(rec: dict) -> str:
    n = rec.get("trade_num")
    return str(n) if n else "?"


def limit_card(rec: dict, balance: float | None = None) -> str:
    pair = rec["symbol"]
    side = int(rec["side"])
    entry, stop, tp = float(rec["entry"]), float(rec["stop"]), float(rec["tp"])
    text = (
        f"{mode_prefix(bool(rec.get('paper', True)))} {arrow(side)} {direction(side)} "
        f"#{_num(rec)} — заявка\n"
        f"Сигнал найден, заявка выставлена, ожидаем вход\n"
        f"Инструмент: {instrument(pair)}\n"
        f"ТФ: {TF}\n"
        f"{_signal_line(rec)}\n"
        f"Вход: {price(pair, entry)} | Стоп: {price(pair, stop)} | Тейк: {price(pair, tp)}\n"
        f"R:R={_rr(entry, stop, tp):.2f}\n"
        f"Ожидание входа: 2ч  |  позиция: до стопа или тейка"
    )
    if balance is not None:
        text += f"\n{balance_line(balance)}"
    return text


def fill_card(pos: dict, balance: float | None = None) -> str:
    pair = pos["symbol"]
    side = int(pos["side"])
    entry, stop, tp = float(pos["entry"]), float(pos["stop"]), float(pos["tp"])
    risk = abs(entry - stop) * float(pos["qty"])
    when = moscow_hm(str(pos.get("opened") or pos.get("last_1m") or ""))
    enter = f"вход ({when} МСК)" if when else "вход"
    text = (
        f"🟢 {mode_prefix(bool(pos.get('paper', True)))} {arrow(side)} {direction(side)} "
        f"#{_num(pos)} — {enter}\n"
        f"Инструмент: {instrument(pair)}\n"
        f"ТФ: {TF}\n"
        f"{_signal_line(pos)}\n"
        f"Вход: {price(pair, entry)} | Стоп: {price(pair, stop)} | Тейк: {price(pair, tp)}\n"
        f"Риск: {risk:.2f} USDT ({float(RISK_PCT):.1f}%)"
    )
    if not pos.get("tpsl_ok", True):
        sl_pct = abs(stop - entry) / entry * 100 if entry else 0.0
        tp_pct = abs(tp - entry) / entry * 100 if entry else 0.0
        text += (
            f"\n🛑 Стоп: {price(pair, stop)} ({sl_pct:.2f}%) — ❌ НЕ выставлен!"
            f"\n💰 Тейк: {price(pair, tp)} ({tp_pct:.2f}%) — ❌ НЕ выставлен!"
        )
    if balance is not None:
        text += f"\n{balance_line(balance)}"
    return text


def protection_alert(pos: dict) -> str:
    return (
        f"🚨 #{_num(pos)} стоп и тейк не встали. Проверь Bybit.\n"
        f"Инструмент: {instrument(pos['symbol'])}"
    )


def close_card(
    pos: dict,
    reason: str,
    exit_px: float,
    gross: float,
    fees: float,
    net: float,
    balance: float | None = None,
) -> str:
    pair = pos["symbol"]
    side = int(pos["side"])
    entry = float(pos["entry"])
    qty = float(pos["qty"])
    margin = entry * qty / LEVERAGE if LEVERAGE else 0.0
    g_pct = gross / margin * 100 if margin > 0 else 0.0
    n_pct = net / margin * 100 if margin > 0 else 0.0
    ico = {"tp": "🟢", "sl": "🔴"}.get(reason, "⚪")
    title = {"tp": "ТЕЙК ПРОФИТ", "sl": "СТОП ЛОСС"}.get(reason, "ЗАКРЫТА НА БИРЖЕ")
    text = (
        f"{mode_prefix(bool(pos.get('paper', True)))} {ico} #{_num(pos)} {title} "
        f"{arrow(side)} {direction(side)}\n"
        f"Инструмент: {instrument(pair)}\n"
        f"ТФ: {TF}\n"
        f"Вход: {price(pair, entry)}  |  Выход: {price(pair, exit_px)}\n"
        f"PnL:   {gross:+.2f} USDT  ({g_pct:+.1f}%)\n"
        f"Fees:  −{fees:.2f} USDT\n"
        f"Net:   {net:+.2f} USDT  ({n_pct:+.1f}%)"
    )
    if balance is not None:
        text += f"\n{balance_line(balance)}"
    return text


def cancel_card(rec: dict, why: str) -> str:
    pair = rec["symbol"]
    side = int(rec["side"])
    return (
        f"{mode_prefix(bool(rec.get('paper', True)))} 🗑 ЛИМИТКА #{_num(rec)} отменена — {why} [{TF}]\n"
        f"Инструмент: {instrument(pair)}\n"
        f"{direction(side)}  |  Вход: {price(pair, rec['entry'])}  |  "
        f"Тейк: {price(pair, rec['tp'])}"
    )


def missing_position(pos: dict) -> str:
    return (
        f"🚨 #{_num(pos)} ордер исполнен, позиции на бирже не вижу.\n"
        f"Инструмент: {instrument(pos['symbol'])}  |  {direction(int(pos['side']))}"
    )
