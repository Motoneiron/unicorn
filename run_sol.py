# -*- coding: utf-8 -*-
"""SOLUSDT. Свип соседних 4ч как в CISD, потом Единорог на M15.

Единорог — свеча, от которой серия ушла в свип.
Лимитка на её хай (лонг) или лоу (шорт), стоп за другим краем.
Слом — закрытие за этим краем, на нём FVG пересекает диапазон свечи.
Тейки 1R, 1.5R, 2R. Путь по 1m.

Запуск: python run_sol.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from bt.check import check
from bt.cisd_setup import check_setup, find_setups
from bt.data import load_ms, resample
from bt.execute import run_book

SOL_1M = Path(r"D:\databtcusdt\SOLUSDT-1m\SOLUSDT_1m_full_hdr.csv")
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
    g = df["gross"].to_numpy(float)
    wr = float((df["outcome"] == "tp").mean() * 100.0)
    return (
        f"{label}: n={len(df)}  WR={wr:.1f}%  "
        f"sumR={r.sum():+.1f}  avgR={r.mean():+.3f}  PF={_pf(r):.2f}  maxDD={_dd(r):+.1f}R  "
        f"| до комиссии sum={g.sum():+.1f} avg={g.mean():+.3f}"
    )


def _report(trades: list[dict], stats: dict, title: str) -> str:
    lines = [title, _line(pd.DataFrame(trades) if trades else pd.DataFrame(), "все")]
    lines.append(
        f"занято={stats['blocked']}  лимитка не сработала={stats['expired']}  филл={stats['filled']}"
    )
    if not trades:
        return "\n".join(lines)
    df = pd.DataFrame(trades)
    for side, g in df.groupby("side"):
        lines.append(_line(g, str(side)))
    df["year"] = pd.to_datetime(df["fill"]).dt.year
    for year, g in df.groupby("year"):
        lines.append(_line(g, str(year)))
    lines.append(
        f"стоп медиана {df['stop_pct'].median():.2f}%  "
        f"удержание медиана {df['hold_min'].median():.0f} мин  "
        f"исходы {df['outcome'].value_counts().to_dict()}"
    )
    return "\n".join(lines)


def main() -> None:
    t0 = time.perf_counter()
    check()
    check_setup()
    print("самотест ок", flush=True)
    m1 = load_ms(SOL_1M)
    m15 = resample(m1, "15min", 15)
    print(
        f"1m {len(m1)}  {m1.index[0]} .. {m1.index[-1]}  |  M15 {len(m15)}",
        flush=True,
    )
    setups, gate = find_setups(m15)
    n_long = sum(s["side"] == 1 for s in setups)
    print(
        f"свипы 4ч {gate['sweeps']}  низ {gate['down']}  верх {gate['up']}",
        flush=True,
    )
    print(
        f"после свипа единорогов {gate['kept']}  long {n_long}  short {len(setups) - n_long}",
        flush=True,
    )
    print(
        f"нет серии {gate['no_series']}  нет свечи {gate['no_origin']}  "
        f"нет слома {gate['no_mss']}  экстремум обновлён {gate['extremum']}  "
        f"нет FVG или пересечения {gate['no_fvg']}",
        flush=True,
    )

    OUT.mkdir(exist_ok=True)
    chunks = [
        "SOLUSDT. Свип соседних 4ч как в CISD, потом Единорог на M15.",
        "Единорог — свеча перед серией в свип. Лимитка на её хай (лонг) или лоу (шорт).",
        "Слом — закрытие за этим краем, FVG на свече слома пересекает диапазон свечи.",
        "Путь по 1m. Вход и тейк — мейкер 0.036%, стоп — тейкер 0.10% + 1 б.п.",
        "",
        f"1m {m1.index[0]} .. {m1.index[-1]}",
        f"свипы 4ч {gate['sweeps']} низ {gate['down']} верх {gate['up']}",
        f"после свипа единорогов {gate['kept']} long {n_long} short {len(setups) - n_long}",
        f"нет серии {gate['no_series']} нет свечи {gate['no_origin']} "
        f"нет слома {gate['no_mss']} экстремум {gate['extremum']} нет FVG {gate['no_fvg']}",
        "",
    ]
    for rr in (1.0, 1.5, 2.0):
        trades, stats = run_book(
            setups, m15, m1, rr=rr, need_h4=False, entry_kind="limit", step_min=1,
        )
        tag = str(rr).replace(".", "")
        title = f"RR 1:{rr:g}"
        text = _report(trades, stats, title)
        print("\n" + text, flush=True)
        chunks.append(text)
        chunks.append("")
        pd.DataFrame(trades).to_csv(OUT / f"sol_rr{tag}.csv", index=False)
    (OUT / "sol_summary.txt").write_text("\n".join(chunks), encoding="utf-8")
    print(f"\nготово за {time.perf_counter() - t0:.1f} с", flush=True)


if __name__ == "__main__":
    main()
