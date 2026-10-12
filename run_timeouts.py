# -*- coding: utf-8 -*-
"""Таймаут лимитки и таймаут открытой позиции.

Сетап: 9 пар без BTC, тейк 2R, не входим, если 2R был до филла.
Вход снимаем через 15, 30, 45 или 60 минут.
Позицию закрываем по рынку через 4–16 часов, если стоп или тейк ещё не сработали.
На минутке таймаута выход по цене открытия, без её фитиля.

Запуск: python run_timeouts.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from bt.cisd_setup import find_setups
from bt.data import load_ms, resample
from bt.execute import MAKER, SLIP, TAKER, _fill_limit, r_multiples
from run_pairs import PAIRS, _dd, _pf

ENTRY_MIN = (15, 30, 45, 60)
POS_H = (4, 6, 8, 10, 12, 14, 16)
OUT = Path(__file__).resolve().parent / "results"


def _check() -> None:
    """Филл внутри 15 минут живёт. Стоп раньше 4 часов важнее таймаута."""
    m15 = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.date_range("2024-01-01 10:00", periods=4, freq="15min"),
    )
    idx = pd.date_range("2024-01-01 10:15", periods=20 * 60, freq="1min")
    n = len(idx)
    hi = np.full(n, 102.0)
    lo = np.full(n, 101.0)
    op = np.full(n, 102.0)
    cl = np.full(n, 102.0)
    # 10:20 — филл. 12:00 — стоп.
    lo[5] = 100.0
    op[5] = 101.0
    lo[105] = 90.0
    m1 = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": cl, "volume": 1.0}, index=idx)
    setup = [{
        "side": 1, "form_i": 0, "known_i": 0, "entry": 100.0, "stop": 90.0,
        "sweep_i": 0, "h4": 0, "zone_bot": 1.0, "zone_top": 2.0,
    }]
    recs, ts, opn, close = _prepare(m15, m1, setup)
    trades = _simulate(recs, ts, opn, close, 15, 4)
    if len(trades) != 1 or trades[0]["outcome"] != "sl":
        raise AssertionError(f"стоп должен быть раньше 4ч, {trades}")
    lo[105] = 99.0
    op[4 * 60] = 105.0
    m1 = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": cl, "volume": 1.0}, index=idx)
    recs, ts, opn, close = _prepare(m15, m1, setup)
    trades = _simulate(recs, ts, opn, close, 15, 4)
    if len(trades) != 1 or trades[0]["outcome"] != "time":
        raise AssertionError(f"ждал выход по времени, {trades}")
    late = _simulate(recs, ts, opn, close, 15, 6)
    if late[0]["outcome"] != "time" or late[0]["exit_i"] <= trades[0]["exit_i"]:
        raise AssertionError("6ч должно закрывать позже 4ч")
    missed = _simulate(recs, ts, opn, close, 15, 4)
    # филл на 5-й минутке от 10:15, это внутри 15 минут
    if missed[0]["outcome"] != "time":
        raise AssertionError(missed)
    no = _simulate(recs, ts, opn, close, 4, 4)
    if no:
        raise AssertionError("4 минуты не должны успеть на филл в 10:20")


def _prepare(
    m15: pd.DataFrame,
    m1: pd.DataFrame,
    setups: list[dict],
    entry_span_min: int = 60,
    pos_until_end: bool = False,
    bar_min: int = 15,
):
    ts = m1.index.to_numpy()
    hi = m1["high"].to_numpy(float)
    lo = m1["low"].to_numpy(float)
    op = m1["open"].to_numpy(float)
    cl = m1["close"].to_numpy(float)
    opens = m15.index.to_numpy()
    n = len(ts)
    recs = []
    span = np.timedelta64(entry_span_min, "m")
    horizon = np.timedelta64(16, "h")
    for s in setups:
        known = opens[s["known_i"]] + np.timedelta64(int(bar_min), "m")
        side = int(s["side"])
        entry = float(s["entry"])
        stop = float(s["stop"])
        i0 = int(np.searchsorted(ts, known, side="left"))
        i1 = int(np.searchsorted(ts, known + span, side="left"))
        kind, idx = _fill_limit(side, entry, stop, lo, hi, i0, min(i1, n))
        sl_i = tp_i = -1
        if kind == "fill":
            risk = abs(entry - stop)
            tp = entry + side * 2.0 * risk
            fill_ts = ts[idx]
            end = n if pos_until_end else int(np.searchsorted(ts, fill_ts + horizon, side="left"))
            for i in range(idx, min(end, n)):
                if side == 1:
                    if lo[i] <= stop:
                        sl_i = i
                        break
                    if hi[i] >= tp:
                        tp_i = i
                        break
                else:
                    if hi[i] >= stop:
                        sl_i = i
                        break
                    if lo[i] <= tp:
                        tp_i = i
                        break
        recs.append({
            "known": known,
            "side": side,
            "entry": entry,
            "stop": stop,
            "kind": kind,
            "idx": int(idx),
            "sl_i": sl_i,
            "tp_i": tp_i,
        })
    return recs, ts, op, cl


def _simulate_hold(recs, ts, op, cl, entry_min: int) -> list[dict]:
    """Позиция без таймера: только стоп, тейк или конец данных."""
    free = np.datetime64("1970-01-01")
    step = np.timedelta64(1, "m")
    wait = np.timedelta64(entry_min, "m")
    n = len(ts)
    trades = []
    for rec in recs:
        known = rec["known"]
        if known < free:
            continue
        entry_end = known + wait
        kind = rec["kind"]
        idx = rec["idx"]
        if kind == "miss" or ts[idx] >= entry_end:
            free = entry_end
            continue
        if kind == "ran":
            free = ts[idx] + step
            continue
        side = rec["side"]
        entry = rec["entry"]
        stop = rec["stop"]
        sl_i = rec["sl_i"]
        tp_i = rec["tp_i"]
        if sl_i >= 0:
            exit_i = sl_i
            exit_px = stop * (1.0 - side * SLIP)
            fee = TAKER
            outcome = "sl"
        elif tp_i >= 0:
            exit_i = tp_i
            exit_px = entry + side * 2.0 * abs(entry - stop)
            fee = MAKER
            outcome = "tp"
        else:
            exit_i = n - 1
            exit_px = float(cl[exit_i]) * (1.0 - side * SLIP)
            fee = TAKER
            outcome = "eod"
        r, _r06, gross = r_multiples(side, entry, stop, exit_px, fee)
        free = ts[exit_i] + step
        trades.append({
            "outcome": outcome,
            "r": r,
            "gross": gross,
            "fill": ts[idx],
            "exit_i": exit_i,
            "hold_min": int(exit_i - idx),
        })
    return trades


def _simulate(recs, ts, op, cl, entry_min: int, pos_h: int) -> list[dict]:
    free = np.datetime64("1970-01-01")
    step = np.timedelta64(1, "m")
    wait = np.timedelta64(entry_min, "m")
    hold = np.timedelta64(pos_h, "h")
    n = len(ts)
    trades = []
    for rec in recs:
        known = rec["known"]
        if known < free:
            continue
        entry_end = known + wait
        kind = rec["kind"]
        idx = rec["idx"]
        if kind == "miss" or ts[idx] >= entry_end:
            free = entry_end
            continue
        if kind == "ran":
            free = ts[idx] + step
            continue
        side = rec["side"]
        entry = rec["entry"]
        stop = rec["stop"]
        fill_ts = ts[idx]
        end_ts = fill_ts + hold
        sl_i = rec["sl_i"]
        tp_i = rec["tp_i"]
        if sl_i >= 0 and ts[sl_i] < end_ts:
            exit_i = sl_i
            exit_px = stop * (1.0 - side * SLIP)
            fee = TAKER
            outcome = "sl"
        elif tp_i >= 0 and ts[tp_i] < end_ts:
            exit_i = tp_i
            exit_px = entry + side * 2.0 * abs(entry - stop)
            fee = MAKER
            outcome = "tp"
        else:
            exit_i = int(np.searchsorted(ts, end_ts, side="left"))
            if exit_i >= n:
                exit_i = n - 1
                exit_px = float(cl[exit_i]) * (1.0 - side * SLIP)
                outcome = "eod"
            else:
                exit_px = float(op[exit_i]) * (1.0 - side * SLIP)
                outcome = "time"
            fee = TAKER
        r, _r06, gross = r_multiples(side, entry, stop, exit_px, fee)
        free = ts[exit_i] + step
        trades.append({
            "outcome": outcome,
            "r": r,
            "gross": gross,
            "fill": fill_ts,
            "exit_i": exit_i,
            "hold_min": int(exit_i - idx),
        })
    return trades


def _stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "sum": 0.0, "avg": 0.0, "pf": 0.0, "dd": 0.0,
                "tp": 0, "sl": 0, "time": 0, "time_r": 0.0, "sl_r": 0.0, "tp_r": 0.0}
    df = pd.DataFrame(trades).sort_values("fill")
    r = df["r"].to_numpy(float)
    tp = df["outcome"] == "tp"
    sl = df["outcome"] == "sl"
    tm = df["outcome"] == "time"
    return {
        "n": len(df),
        "wr": float(tp.mean() * 100.0),
        "sum": float(r.sum()),
        "avg": float(r.mean()),
        "pf": _pf(r),
        "dd": _dd(r),
        "tp": int(tp.sum()),
        "sl": int(sl.sum()),
        "time": int(tm.sum()),
        "time_r": float(df.loc[tm, "r"].sum()) if tm.any() else 0.0,
        "sl_r": float(df.loc[sl, "r"].sum()) if sl.any() else 0.0,
        "tp_r": float(df.loc[tp, "r"].sum()) if tp.any() else 0.0,
    }


def main() -> None:
    t0 = time.perf_counter()
    _check()
    print("самотест ок", flush=True)
    OUT.mkdir(exist_ok=True)
    books: dict[tuple[int, int], list[dict]] = {(e, h): [] for e in ENTRY_MIN for h in POS_H}
    for key, label, path in PAIRS:
        if key == "BTC":
            continue
        t1 = time.perf_counter()
        print(f"== {label} ==", flush=True)
        m1 = load_ms(path)
        m15 = resample(m1, "15min", 15)
        setups, gate = find_setups(m15)
        recs, ts, op, cl = _prepare(m15, m1, setups)
        print(f"сетапы {gate['kept']}  разбор {time.perf_counter() - t1:.0f} с", flush=True)
        for entry_min in ENTRY_MIN:
            for pos_h in POS_H:
                trades = _simulate(recs, ts, op, cl, entry_min, pos_h)
                for t in trades:
                    t["symbol"] = key
                books[(entry_min, pos_h)].extend(trades)
        del m1, m15, setups, recs

    rows = []
    lines = [
        "9 пар без BTC. Тейк 2R. Снимаем лимитку, если 2R был до филла.",
        "Вход: отмена через 15/30/45/60 мин. Позиция: рынок через 4–16 ч, если нет стопа и тейка.",
        "База без таймаута позиции и со входом 6 ч: 4873 сделки, +1025 R, просадка −35 R, стопов 2546.",
        "",
    ]
    for entry_min in ENTRY_MIN:
        lines.append(f"вход {entry_min} мин")
        for pos_h in POS_H:
            st = _stats(books[(entry_min, pos_h)])
            rows.append({"entry_min": entry_min, "pos_h": pos_h, **st})
            lines.append(
                f"  выход {pos_h:2d}ч  n={st['n']}  WR={st['wr']:.1f}%  "
                f"sumR={st['sum']:+.1f}  avgR={st['avg']:+.3f}  PF={st['pf']:.2f}  "
                f"maxDD={st['dd']:+.1f}  тейк {st['tp']} ({st['tp_r']:+.0f}R)  "
                f"стоп {st['sl']} ({st['sl_r']:+.0f}R)  время {st['time']} ({st['time_r']:+.0f}R)"
            )
        lines.append("")
    text = "\n".join(lines)
    (OUT / "timeouts_summary.txt").write_text(text, encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "timeouts_grid.csv", index=False)
    print("\n" + text, flush=True)
    print(f"всё за {time.perf_counter() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
