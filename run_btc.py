# -*- coding: utf-8 -*-
"""Прогон Единорога по BTCUSDT, Bybit linear, 5m → M15.

Запуск из папки проекта: python run_btc.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from bt.check import check
from bt.data import load_5m, resample
from bt.execute import run_book
from bt.pattern import attach_h4, find_unicorns

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def _pf(r: np.ndarray) -> float:
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return pos / neg


def _dd(r: np.ndarray) -> float:
    if len(r) == 0:
        return 0.0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def _line(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return f"{label}: сделок нет"
    r = df["r"].to_numpy(float)
    r06 = df["r06"].to_numpy(float)
    wr = float((df["outcome"] == "tp").mean() * 100.0)
    return (
        f"{label}: n={len(df)}  WR={wr:.1f}%  "
        f"sumR={r.sum():+.1f}  avgR={r.mean():+.3f}  PF={_pf(r):.2f}  maxDD={_dd(r):+.1f}R  "
        f"| плоские 0.06%: sum={r06.sum():+.1f} avg={r06.mean():+.3f}"
    )


def _report(trades: list[dict], stats: dict, title: str) -> str:
    lines = [title, _line(pd.DataFrame(trades), "все")]
    if not trades:
        lines.append(
            f"мимо H4={stats['skipped_h4']}  занято={stats['blocked']}  без ретеста={stats['expired']}"
        )
        return "\n".join(lines)
    df = pd.DataFrame(trades)
    lines.append(
        f"мимо H4={stats['skipped_h4']}  занято={stats['blocked']}  без ретеста={stats['expired']}  "
        f"филл={stats['filled']}"
    )
    for side, g in df.groupby("side"):
        lines.append(_line(g, str(side)))
    df["year"] = pd.to_datetime(df["fill"]).dt.year
    for year, g in df.groupby("year"):
        lines.append(_line(g, str(year)))
    wide = df[df["stop_pct"] >= 0.30]
    lines.append(_line(wide, "стоп от 0.30%"))
    lines.append(
        f"стоп медиана {df['stop_pct'].median():.2f}%  "
        f"удержание медиана {df['hold_min'].median():.0f} мин  "
        f"исходы {df['outcome'].value_counts().to_dict()}"
    )
    return "\n".join(lines)


def main() -> None:
    t0 = time.perf_counter()
    check()
    print("самотест ок")
    m5 = load_5m()
    m15 = resample(m5, "15min", 3)
    h4 = resample(m5, "4h", 48)
    print(
        f"5m {len(m5)}  {m5.index[0]} .. {m5.index[-1]}  |  "
        f"M15 {len(m15)}  H4 {len(h4)}"
    )
    setups, funnel = find_unicorns(m15)
    attach_h4(setups, m15, h4)
    n_long = sum(s["side"] == 1 for s in setups)
    n_short = len(setups) - n_long
    h4_match = sum(s["h4"] == s["side"] for s in setups)
    print(f"сетапы {len(setups)}  long {n_long}  short {n_short}  H4 в сторону {h4_match}")
    print("воронка", funnel)

    OUT.mkdir(exist_ok=True)
    chunks = ["Единорог BTCUSDT Bybit. M15 паттерн, филл по 5m.", ""]
    for rr in (2.0, 3.0):
        for need_h4, tag in ((False, "все"), (True, "h4")):
            trades, stats = run_book(setups, m15, m5, rr=rr, need_h4=need_h4)
            title = f"RR 1:{rr:.0f}, фильтр H4: {tag}"
            text = _report(trades, stats, title)
            print("\n" + text)
            chunks.append(text)
            chunks.append("")
            name = f"trades_rr{int(rr)}_{tag}.csv"
            pd.DataFrame(trades).to_csv(OUT / name, index=False)
    chunks.append(f"воронка: {funnel}")
    chunks.append(f"сетапы {len(setups)} long {n_long} short {n_short} h4 {h4_match}")
    (OUT / "summary.txt").write_text("\n".join(chunks), encoding="utf-8")
    print(f"\nготово за {time.perf_counter() - t0:.1f} с, файлы в {OUT}")


if __name__ == "__main__":
    main()
