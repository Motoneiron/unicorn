# -*- coding: utf-8 -*-
"""Причинный live-реплей сетки «стоп × тейк» для Единорога.

Четыре независимых портфеля на каждой паре проходят один и тот же поток:

- trigger_rr2 — стоп за баром-триггером, тейк 2R (контроль текущего реплея);
- trigger_rr1 — стоп за баром-триггером, тейк 1R;
- block_rr2 — стоп за противоположным краем блока-единорога, тейк 2R;
- block_rr1 — стоп за противоположным краем блока-единорога, тейк 1R.

Скан использует только закрытые M15-бары. Исполнение: лимитка 120 минут,
выход через 6 часов, maker/taker-комиссии и проскальзывание как в боте,
стоп раньше тейка на одной минутке, одна заявка/позиция на пару.
Сетап берётся с FVG внутри свечи свипа. Лимитка снимается, если 2R
случился раньше филла; на одной минутке филл важнее.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS = Path(r"D:\hermes-agent\RUNS\2026-10-20\unicorn_live_stop_tp_grid")
sys.path.insert(0, str(ROOT))

from bot.config import PAIRS as BOT_PAIRS  # noqa: E402
from bot.rules import entry_deadline, mss_close, position_hit, round_levels, setup_id  # noqa: E402
from bt.cisd_setup import find_setups  # noqa: E402
from bt.data import load_ms, resample  # noqa: E402
from bt.execute import MAKER, SLIP, TAKER, r_multiples  # noqa: E402
from run_pairs import PAIRS as DATA_PAIRS  # noqa: E402

W_DEFAULT = 224
TIME_EXIT_H = 6.0
CONFIGS = (
    ("trigger_rr2", "trigger", 2.0),
    ("trigger_rr1", "trigger", 1.0),
    ("block_rr2", "block", 2.0),
    ("block_rr1", "block", 1.0),
)


@dataclass
class Portfolio:
    name: str
    stop_kind: str
    rr: float
    seen: set[str] = field(default_factory=set)
    pending: dict | None = None
    position: dict | None = None
    trades: list[dict] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=lambda: {
        "scan": 0,
        "guard": 0,
        "seen_skip": 0,
        "dogon": 0,
        "placed": 0,
        "expired": 0,
        "fills": 0,
        "blocked": 0,
        "ran_2r": 0,
    })


def _search(ts_arr, value) -> int:
    return int(np.searchsorted(ts_arr, value, side="left"))


def _nearest(px: float, tick: float) -> float:
    if tick <= 0:
        return float(px)
    return round(round(float(px) / tick) * tick, 10)


def _levels(side: int, entry: float, stop: float, tick: float, rr: float):
    base = round_levels(side, entry, stop, tick)
    if base is None:
        return None
    entry_r, stop_r, _tp2 = base
    tp_r = _nearest(entry_r + side * rr * abs(entry_r - stop_r), tick)
    if side == 1 and tp_r <= entry_r:
        return None
    if side == -1 and tp_r >= entry_r:
        return None
    return entry_r, stop_r, tp_r


def _trade_row(
    key: str,
    portfolio: Portfolio,
    rec: dict,
    fill_ts,
    exit_ts,
    outcome: str,
    exit_px: float,
    fee_out: float,
    hold_min: int,
) -> dict:
    r, _r06, gross = r_multiples(
        rec["side"], rec["entry"], rec["stop"], exit_px, fee_out,
    )
    return {
        "config": portfolio.name,
        "stop_kind": portfolio.stop_kind,
        "rr": portfolio.rr,
        "symbol": key,
        "side": rec["side"],
        "entry": rec["entry"],
        "stop": rec["stop"],
        "tp": rec["tp"],
        "mss_open": rec["mss_open"],
        "trigger_at": rec["trigger_at"],
        "anchor": rec["anchor"],
        "fill": pd.Timestamp(fill_ts),
        "exit_time": pd.Timestamp(exit_ts),
        "outcome": outcome,
        "r": r,
        "gross": gross,
        "hold_min": hold_min,
        "fill_delay_min": float(
            (fill_ts - np.datetime64(mss_close(rec["mss_open"])))
            / np.timedelta64(1, "m")
        ),
    }


def _walk_minute(
    key: str,
    portfolio: Portfolio,
    ts_i,
    open_i: float,
    low_i: float,
    high_i: float,
) -> None:
    pending = portfolio.pending
    if pending is not None:
        if ts_i >= pending["deadline"]:
            portfolio.counters["expired"] += 1
            portfolio.pending = None
            return
        side = pending["side"]
        entry = pending["entry"]
        tp2 = entry + side * 2.0 * abs(entry - pending["stop"])
        filled = low_i <= entry if side == 1 else high_i >= entry
        saw_2r = high_i >= tp2 if side == 1 else low_i <= tp2
        if saw_2r and not filled:
            portfolio.counters["ran_2r"] += 1
            portfolio.pending = None
            return
        if not filled:
            return
        hit = position_hit(side, pending["stop"], pending["tp"], low_i, high_i)
        if hit:
            exit_px = (
                pending["stop"] * (1 - side * SLIP)
                if hit == "sl"
                else pending["tp"]
            )
            fee = TAKER if hit == "sl" else MAKER
            portfolio.trades.append(
                _trade_row(
                    key, portfolio, pending, ts_i, ts_i, hit, exit_px, fee, 0,
                )
            )
        else:
            portfolio.position = {
                **pending,
                "fill_ts": ts_i,
            }
            portfolio.counters["fills"] += 1
        portfolio.pending = None
        return

    position = portfolio.position
    if position is None:
        return
    side = position["side"]
    end_ts = position["fill_ts"] + np.timedelta64(int(TIME_EXIT_H * 60), "m")
    if ts_i >= end_ts:
        exit_px = float(open_i) * (1 - side * SLIP)
        hold = int((ts_i - position["fill_ts"]) / np.timedelta64(1, "m"))
        portfolio.trades.append(
            _trade_row(
                key, portfolio, position, position["fill_ts"], ts_i,
                "time", exit_px, TAKER, hold,
            )
        )
        portfolio.position = None
        return
    hit = position_hit(side, position["stop"], position["tp"], low_i, high_i)
    if hit:
        exit_px = (
            position["stop"] * (1 - side * SLIP)
            if hit == "sl"
            else position["tp"]
        )
        fee = TAKER if hit == "sl" else MAKER
        hold = int((ts_i - position["fill_ts"]) / np.timedelta64(1, "m"))
        portfolio.trades.append(
            _trade_row(
                key, portfolio, position, position["fill_ts"], ts_i,
                hit, exit_px, fee, hold,
            )
        )
        portfolio.position = None


def _consider_setups(
    key: str,
    portfolio: Portfolio,
    setups: list[dict],
    sl_df: pd.DataFrame,
    now_ts: pd.Timestamp,
    now64,
    ts: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    tick: float,
) -> None:
    portfolio.counters["scan"] += 1
    found = []
    for setup in setups:
        mss_open = pd.Timestamp(sl_df.index[int(setup["known_i"])])
        if mss_close(mss_open) > now_ts or entry_deadline(mss_open) <= now_ts:
            portfolio.counters["guard"] += 1
            continue
        sid = setup_id(key, int(setup["side"]), mss_open)
        if sid in portfolio.seen:
            portfolio.counters["seen_skip"] += 1
            continue
        found.append((mss_open, setup, sid))
    found.sort(key=lambda item: item[0])

    for mss_open, setup, sid in found:
        if portfolio.pending is not None or portfolio.position is not None:
            portfolio.counters["blocked"] += 1
            break
        side = int(setup["side"])
        try:
            if portfolio.stop_kind == "trigger":
                row = sl_df.iloc[int(setup["sweep_i"])]
                raw_stop = float(row["low"] if side == 1 else row["high"])
            else:
                raw_stop = float(setup["stop"])
            levels = _levels(
                side, float(setup["entry"]), raw_stop, tick, portfolio.rr,
            )
        except (IndexError, KeyError, TypeError, ValueError):
            levels = None
        if levels is None:
            portfolio.seen.add(sid)
            continue
        entry_r, stop_r, tp_r = levels
        deadline = entry_deadline(mss_open)
        a0 = _search(ts, np.datetime64(mss_close(mss_open)))
        a1 = _search(ts, min(np.datetime64(deadline), now64))
        tp2 = entry_r + side * 2.0 * abs(entry_r - stop_r)
        ran = touched = False
        for i in range(a0, a1):
            if side == 1:
                got_fill = lo[i] <= entry_r
                saw_2r = hi[i] >= tp2
            else:
                got_fill = hi[i] >= entry_r
                saw_2r = lo[i] <= tp2
            if saw_2r and not got_fill:
                ran = True
                break
            if got_fill:
                touched = True
                break
        if ran:
            portfolio.counters["ran_2r"] += 1
            portfolio.seen.add(sid)
            continue
        if touched:
            portfolio.counters["dogon"] += 1
            portfolio.seen.add(sid)
            continue
        portfolio.pending = {
            "side": side,
            "entry": entry_r,
            "stop": stop_r,
            "tp": tp_r,
            "deadline": np.datetime64(deadline),
            "mss_open": mss_open,
            "trigger_at": pd.Timestamp(sl_df.index[int(setup["sweep_i"])]),
            "anchor": setup.get("h4_sweep_at"),
        }
        portfolio.counters["placed"] += 1
        portfolio.seen.add(sid)


def run_pair(
    key: str,
    path: Path,
    window: int,
    out_dir: Path,
    from_ts: str | None = None,
    to_ts: str | None = None,
) -> list[dict]:
    m1 = load_ms(path)
    m15 = resample(m1, "15min", 15)
    op15 = m15.index.to_numpy()
    ts = m1.index.to_numpy()
    op = m1["open"].to_numpy(float)
    lo = m1["low"].to_numpy(float)
    hi = m1["high"].to_numpy(float)
    cl = m1["close"].to_numpy(float)
    del m1

    tick = float(BOT_PAIRS[key]["price_tick"])
    min15 = np.timedelta64(15, "m")
    start_j = window - 1
    if from_ts:
        start_j = max(start_j, _search(op15, np.datetime64(pd.Timestamp(from_ts))))
    end_j = len(m15)
    if to_ts:
        end_j = min(end_j, _search(op15, np.datetime64(pd.Timestamp(to_ts))))

    portfolios = [Portfolio(*cfg) for cfg in CONFIGS]
    t0 = time.perf_counter()
    for j in range(start_j, end_j):
        now64 = op15[j] + min15
        now_ts = pd.Timestamp(now64)
        i0 = _search(ts, op15[j])
        i1 = _search(ts, now64)

        for i in range(i0, i1):
            for portfolio in portfolios:
                _walk_minute(
                    key, portfolio, ts[i], op[i], lo[i], hi[i],
                )

        for portfolio in portfolios:
            if (
                portfolio.pending is not None
                and now_ts >= pd.Timestamp(portfolio.pending["deadline"])
            ):
                portfolio.counters["expired"] += 1
                portfolio.pending = None

        free = [
            p for p in portfolios
            if p.pending is None and p.position is None
        ]
        if free and j >= window - 1:
            sl_df = m15.iloc[j - window + 1:j + 1]
            setups, _stats = find_setups(sl_df)
            for portfolio in free:
                _consider_setups(
                    key, portfolio, setups, sl_df, now_ts, now64,
                    ts, lo, hi, tick,
                )

        if (j - start_j) % 20000 == 0 and j > start_j:
            elapsed = time.perf_counter() - t0
            print(
                f"[{key}] {j-start_j}/{end_j-start_j}, {elapsed:.0f}s",
                flush=True,
            )

    last_i = _search(ts, op15[end_j - 1] + min15) - 1
    for portfolio in portfolios:
        if portfolio.position is not None:
            position = portfolio.position
            side = position["side"]
            exit_px = float(cl[last_i]) * (1 - side * SLIP)
            hold = int(
                (ts[last_i] - position["fill_ts"]) / np.timedelta64(1, "m")
            )
            portfolio.trades.append(
                _trade_row(
                    key, portfolio, position, position["fill_ts"], ts[last_i],
                    "eod", exit_px, TAKER, hold,
                )
            )
            portfolio.position = None

    pair_dir = out_dir / "books"
    pair_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for portfolio in portfolios:
        frame = pd.DataFrame(portfolio.trades)
        frame.to_csv(pair_dir / f"{portfolio.name}_{key}.csv", index=False)
        results.append({
            "key": key,
            "config": portfolio.name,
            "stop_kind": portfolio.stop_kind,
            "rr": portfolio.rr,
            "n": len(frame),
            "sumR": float(frame["r"].sum()) if len(frame) else 0.0,
            "gross": float(frame["gross"].sum()) if len(frame) else 0.0,
            "tp": int((frame["outcome"] == "tp").sum()) if len(frame) else 0,
            "sl": int((frame["outcome"] == "sl").sum()) if len(frame) else 0,
            "time": int((frame["outcome"] == "time").sum()) if len(frame) else 0,
            "eod": int((frame["outcome"] == "eod").sum()) if len(frame) else 0,
            **portfolio.counters,
        })
    print(
        f"[{key}] DONE "
        + " | ".join(f"{r['config']} {r['n']}/{r['sumR']:+.1f}R" for r in results),
        flush=True,
    )
    return results


def _worker(args):
    try:
        return run_pair(*args)
    except Exception as exc:
        traceback.print_exc()
        return [{"key": args[0], "error": f"{type(exc).__name__}: {exc}"}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="")
    parser.add_argument("--window", type=int, default=W_DEFAULT)
    parser.add_argument("--from", dest="from_ts", default=None)
    parser.add_argument("--to", dest="to_ts", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(RUNS))
    args = parser.parse_args()

    selected = {x.strip().upper() for x in args.pairs.split(",") if x.strip()}
    out_dir = Path(args.out)
    tasks = []
    for key, _label, path in DATA_PAIRS:
        if key == "BTC" or (selected and key not in selected):
            continue
        tasks.append((key, Path(path), args.window, out_dir, args.from_ts, args.to_ts))
    if not tasks:
        raise SystemExit("Нет выбранных пар")

    if len(tasks) == 1:
        nested = [_worker(tasks[0])]
    else:
        with mp.Pool(processes=min(args.workers, len(tasks))) as pool:
            nested = pool.map(_worker, tasks)
    results = [row for group in nested for row in group]
    summary = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "pair_summary.csv", index=False)

    lines = [
        "Причинный live-реплей: стоп за триггером/блоком x тейк 1R/2R.",
        f"W={args.window}, лимитка 120 минут, выход 6 часов.",
        "",
    ]
    if any("error" in row for row in results):
        for row in results:
            if "error" in row:
                lines.append(f"{row['key']}: ОШИБКА {row['error']}")
    else:
        for config, group in summary.groupby("config", sort=False):
            lines.append(
                f"{config}: n={int(group['n'].sum())} "
                f"sumR={group['sumR'].sum():+.3f} "
                f"gross={group['gross'].sum():+.3f} "
                f"tp={int(group['tp'].sum())} "
                f"sl={int(group['sl'].sum())} "
                f"time={int(group['time'].sum())}"
            )
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
