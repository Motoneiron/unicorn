# -*- coding: utf-8 -*-
"""Рабочий сет: 15м → 4ч, девять пар без BTC, лимитка 2 часа, тейк 2R.

Стоп за бар-триггер. Если цена дошла до 2R раньше филла — лимитку снимаем.
Выход по стопу, тейку или рынку через 6ч.
Одна заявка или позиция на пару. Пока PAPER_TRADING=0 не стоит — тень.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"


def load_env(path: Path | None = None) -> None:
    path = path or (ROOT / ".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_env()

# Тень, пока явно не выключили. Живой режим после перезапуска только с PAPER_TRADING=0.
_paper_raw = os.environ.get("PAPER_TRADING", "1").strip().lower()
PAPER_TRADING = _paper_raw not in ("0", "false", "no")

BYBIT_API_KEY = os.environ.get("BYBIT_API_KEY", "").strip()
BYBIT_API_SECRET = os.environ.get("BYBIT_API_SECRET", "").strip()
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = int(os.environ.get("TELEGRAM_CHAT_ID", "0") or "0")
TELEGRAM_PROXY = os.environ.get("TELEGRAM_PROXY", "").strip()

MAKER = 0.00036
TAKER = 0.001
SLIP = 0.00010

RISK_PCT = float(os.environ.get("RISK_PCT", "1"))
# Сайзинг по компактности сетапа (stop/ATR): границы квинтилей по IS 2020-23,
# веса из исследования (results/ROUND2_RESULTS.md). SIZING=0 в env — выключить.
SIZING_ON = os.environ.get("SIZING", "1").strip().lower() not in ("0", "false", "no")
SIZING_BOUNDS = (1.89, 2.27, 2.72, 3.38)
SIZING_WEIGHTS = (1.5, 1.25, 1.0, 0.75, 0.5)
LEVERAGE = int(os.environ.get("LEVERAGE", "20"))
PAPER_BALANCE = float(os.environ.get("PAPER_BALANCE", "100"))
MIN_NOTIONAL = 5.0
MARGIN_RESERVE = 0.05

LTF = "15m"
LTF_MIN = 15
HTF_MIN = 240
ENTRY_MIN = 120
RR = 2.0
# Выход по времени: если за это время не сработали стоп/тейк — закрыть по рынку.
# 6ч — оптимум бэктеста (клетка 6–8ч; 6ч даёт лучшую просадку). См. results/STOP_VARIANTS.md.
TIME_EXIT_H = float(os.environ.get("TIME_EXIT_H", "6"))
OHLCV_BARS = 600
M1_BARS = 180
# Для сопровождения тени достаточно нескольких минуток; 180 на каждый тик ломают лимит Bybit.
WATCH_M1 = 8
SCAN_S = 5

_FEES = {"maker_fee": MAKER, "taker_fee": TAKER}

# qty_step и price_tick сверены с cisdbot (Bybit linear).
PAIRS: dict[str, dict] = {
    "SOL": {"symbol": "SOL/USDT:USDT", "qty_step": 0.1, "price_tick": 0.01, **_FEES},
    "DOGE": {"symbol": "DOGE/USDT:USDT", "qty_step": 1.0, "price_tick": 0.00001, **_FEES},
    "KAS": {"symbol": "KAS/USDT:USDT", "qty_step": 10.0, "price_tick": 0.00001, **_FEES},
    "ADA": {"symbol": "ADA/USDT:USDT", "qty_step": 1.0, "price_tick": 0.0001, **_FEES},
    "LINK": {"symbol": "LINK/USDT:USDT", "qty_step": 0.1, "price_tick": 0.001, **_FEES},
    "LTC": {"symbol": "LTC/USDT:USDT", "qty_step": 0.1, "price_tick": 0.01, **_FEES},
    "XRP": {"symbol": "XRP/USDT:USDT", "qty_step": 0.1, "price_tick": 0.0001, **_FEES},
    "AVAX": {"symbol": "AVAX/USDT:USDT", "qty_step": 0.1, "price_tick": 0.001, **_FEES},
    "GRAM": {"symbol": "GRAM/USDT:USDT", "qty_step": 0.1, "price_tick": 0.001, **_FEES},
}
PAIR_KEYS = tuple(PAIRS)
