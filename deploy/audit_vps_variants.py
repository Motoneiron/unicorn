# -*- coding: utf-8 -*-
"""Сверка paper-сделок VPS с финальной историей сетапов.

Для каждой сделки после деплоя fix2:
1. Восстанавливает причинный сетап на 600 закрытых M15-барах, доступных боту.
2. Проверяет тот же sid и уровни на полной истории после закрытия окна.
3. Классифицирует сделку как core, variant, immature или unmatched.

Свечи загружаются из публичного Bybit API, журнал — с VPS по SSH.
Секреты и .env не читаются.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.config import PAIRS  # noqa: E402
from bot.rules import drop_forming, mss_close, round_levels, trigger_stop  # noqa: E402
from bt.cisd_setup import find_setups  # noqa: E402

FIX2_DEPLOY_UTC = pd.Timestamp("2026-10-13 21:35:36")
LTF_MIN = 15
BOT_WINDOW = 600
MAX_SIGNAL_TO_EXIT_H = 9


def load_vps_trades(host: str) -> pd.DataFrame:
    code = (
        "import json,pathlib,sys;"
        "p=pathlib.Path(sys.argv[1]);"
        "print(json.dumps([json.loads(x) for x in p.read_text().splitlines() if x.strip()]))"
    )
    remote = (
        f"python3 -c {shlex.quote(code)} "
        f"{shlex.quote('/opt/unicorn/runtime/trades.jsonl')}"
    )
    raw = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, remote],
        text=True,
        encoding="utf-8",
    )
    rows = json.loads(raw)
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"])
    frame["side_i"] = frame["side"].map({"лонг": 1, "шорт": -1})
    if frame["side_i"].isna().any():
        raise ValueError("Неизвестное направление в журнале VPS")
    return frame.sort_values("ts").reset_index(drop=True)


def fetch_m15(
    exchange: ccxt.Exchange,
    symbol: str,
    since: pd.Timestamp,
    until: pd.Timestamp,
) -> pd.DataFrame:
    step_ms = LTF_MIN * 60_000
    cursor = int(since.timestamp() * 1000)
    end_ms = int(until.timestamp() * 1000)
    rows: list[list] = []
    while cursor < end_ms:
        batch = exchange.fetch_ohlcv(symbol, "15m", cursor, 1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + step_ms
        if nxt <= cursor:
            break
        cursor = nxt
        if len(batch) < 1000 and cursor >= end_ms:
            break
        time.sleep(exchange.rateLimit / 1000)
    frame = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"],
    )
    if frame.empty:
        return frame
    frame.index = pd.to_datetime(frame.pop("ts"), unit="ms")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame[(frame.index >= since) & (frame.index <= until)]
    return drop_forming(frame, LTF_MIN, now=until)


def setup_record(pair: str, df: pd.DataFrame, setup: dict) -> dict | None:
    side = int(setup["side"])
    try:
        levels = round_levels(
            side,
            float(setup["entry"]),
            trigger_stop(df, setup),
            float(PAIRS[pair]["price_tick"]),
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if levels is None:
        return None
    mss_open = pd.Timestamp(df.index[int(setup["known_i"])])
    anchor = pd.Timestamp(setup["h4_sweep_at"])
    return {
        "side": side,
        "entry": levels[0],
        "stop": levels[1],
        "tp": levels[2],
        "mss_open": mss_open,
        "anchor": anchor,
        "sid": f"{pair}|{side}|{mss_open.isoformat()}",
    }


def same_levels(trade: pd.Series, record: dict, tick: float) -> bool:
    return (
        int(trade["side_i"]) == record["side"]
        and abs(float(trade["entry"]) - record["entry"]) <= tick * 0.1
        and abs(float(trade["stop"]) - record["stop"]) <= tick * 0.1
    )


def causal_match(
    pair: str,
    m15: pd.DataFrame,
    trade: pd.Series,
) -> dict | None:
    tick = float(PAIRS[pair]["price_tick"])
    exit_ts = pd.Timestamp(trade["ts"]).floor("15min")
    first = exit_ts - pd.Timedelta(hours=MAX_SIGNAL_TO_EXIT_H)
    i0 = max(BOT_WINDOW - 1, int(m15.index.searchsorted(first)))
    i1 = min(len(m15), int(m15.index.searchsorted(exit_ts, side="right")))
    matches: list[dict] = []
    seen: set[str] = set()
    for j in range(i0, i1):
        window = m15.iloc[j - BOT_WINDOW + 1:j + 1]
        setups, _stats = find_setups(window, fvg_in_sweep=False)
        for setup in setups:
            rec = setup_record(pair, window, setup)
            if rec is None or rec["sid"] in seen:
                continue
            seen.add(rec["sid"])
            if same_levels(trade, rec, tick):
                detected_at = pd.Timestamp(window.index[-1]) + pd.Timedelta(minutes=15)
                if mss_close(rec["mss_open"]) <= detected_at:
                    rec["detected_at"] = detected_at
                    matches.append(rec)
    if not matches:
        return None
    matches.sort(
        key=lambda rec: (
            abs((pd.Timestamp(trade["ts"]) - rec["mss_open"]).total_seconds()),
            rec["detected_at"],
        )
    )
    return matches[0]


def final_records(pair: str, m15: pd.DataFrame) -> list[dict]:
    setups, _stats = find_setups(m15, fvg_in_sweep=False)
    records = []
    for setup in setups:
        rec = setup_record(pair, m15, setup)
        if rec is not None:
            records.append(rec)
    return records


def classify_pair(
    pair: str,
    trades: pd.DataFrame,
    m15: pd.DataFrame,
    now: pd.Timestamp,
) -> list[dict]:
    final = final_records(pair, m15)
    by_sid = {rec["sid"]: rec for rec in final}
    rows = []
    for _, trade in trades.iterrows():
        base = {
            "ts": trade["ts"],
            "symbol": pair,
            "side": int(trade["side_i"]),
            "entry": float(trade["entry"]),
            "stop": float(trade["stop"]),
            "outcome": trade["reason"],
            "r": float(trade["r"]),
        }
        if pd.Timestamp(trade["ts"]) < FIX2_DEPLOY_UTC:
            rows.append({**base, "class": "pre_fix2", "note": "старая версия бота"})
            continue
        causal = causal_match(pair, m15, trade)
        if causal is None:
            rows.append({
                **base,
                "class": "unmatched",
                "note": "не восстановлен сигнал на окне 600",
            })
            continue
        mature_at = causal["anchor"] + pd.Timedelta(hours=8)
        common = {
            **base,
            "mss_open": causal["mss_open"],
            "anchor": causal["anchor"],
            "detected_at": causal["detected_at"],
            "mature_at": mature_at,
        }
        if now < mature_at:
            rows.append({
                **common,
                "class": "immature",
                "note": "окно триггера ещё не закрылось",
            })
            continue
        final_rec = by_sid.get(causal["sid"])
        tick = float(PAIRS[pair]["price_tick"])
        if final_rec is not None and same_levels(trade, final_rec, tick):
            rows.append({
                **common,
                "class": "core",
                "note": "сохранилась в финальной истории",
            })
        else:
            rows.append({
                **common,
                "class": "variant",
                "note": "была видна боту, исчезла в финальной истории",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="cisdbot")
    parser.add_argument(
        "--out",
        default=str(
            Path(r"D:\hermes-agent\RUNS\2026-10-21\unicorn_vps_variant_audit")
        ),
    )
    args = parser.parse_args()

    trades = load_vps_trades(args.host)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    since = trades["ts"].min().floor("D") - pd.Timedelta(days=14)
    until = now.floor("min")
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultSubType": "linear"},
    })
    exchange.load_markets()

    rows = []
    for pair, group in trades.groupby("symbol", sort=True):
        symbol = PAIRS[pair]["symbol"]
        print(f"[{pair}] свечи {since} — {until}", flush=True)
        m15 = fetch_m15(exchange, symbol, since, until)
        if len(m15) < BOT_WINDOW:
            raise RuntimeError(f"{pair}: получено только {len(m15)} M15-баров")
        rows.extend(classify_pair(pair, group, m15, until))
        print(f"[{pair}] готово: {len(group)} сделок", flush=True)

    result = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "vps_trade_classes.csv", index=False)

    lines = [
        f"VPS trades: {len(result)}",
        f"fix2 cutoff UTC: {FIX2_DEPLOY_UTC}",
        f"market data through UTC: {until}",
        "",
        "classes:",
    ]
    for klass, group in result.groupby("class"):
        lines.append(
            f"  {klass}: n={len(group)}, sumR={group['r'].sum():+.3f}"
        )
    lines.extend(["", result.to_string(index=False)])
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
