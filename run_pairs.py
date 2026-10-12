# -*- coding: utf-8 -*-
"""Десять пар из cisdbot. Тот же Единорог, что на SOL.

Пары — ACTIVE_SYMBOL_KEYS в cisdbot.py. GRAM в боте, файл свечей ещё TONUSDT.
Каждая пара со своей книгой: одна заявка на пару. Между парами сделки не блокируют друг друга.

Запуск: python run_pairs.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from bt.cisd_setup import find_setups
from bt.data import load_ms, resample
from bt.execute import run_book

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
DATA = Path(r"D:\databtcusdt")

# Ключ как в боте, подпись, файл минуток.
PAIRS = [
    ("SOL", "SOL", DATA / "SOLUSDT-1m" / "SOLUSDT_1m_full_hdr.csv"),
    ("DOGE", "DOGE", DATA / "DOGEUSDT-1m" / "DOGEUSDT_1m_full_hdr.csv"),
    ("KAS", "KAS", DATA / "KASUSDT-1m" / "KASUSDT_1m_full_hdr.csv"),
    ("ADA", "ADA", DATA / "ADAUSDT-1m-merged.csv"),
    ("LINK", "LINK", DATA / "LINKUSDT-1m-merged.csv"),
    ("LTC", "LTC", DATA / "LTCUSDT-1m-merged.csv"),
    ("XRP", "XRP", DATA / "XRPUSDT-1m-merged.csv"),
    ("BTC", "BTC", DATA / "BTCUSDT-1m" / "BTCUSDT_1m_2020_2026_conv.csv"),
    ("AVAX", "AVAX", DATA / "AVAXUSDT-1m-merged.csv"),
    ("GRAM", "GRAM (файл TON)", DATA / "TONUSDT-1m-merged.csv"),
]
RRS = (1.0, 1.5, 2.0)


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


def _tag(rr: float) -> str:
    return str(rr).replace(".", "")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fill"] = pd.to_datetime(df["fill"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values(["fill", "symbol"]).reset_index(drop=True)
    return df


def _counts(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    """Число сделок на каждый бакет диапазона, включая пустые."""
    if len(index) == 0:
        return pd.Series(dtype=int)
    start = index.min()
    end = index.max()
    if freq == "D":
        full = pd.date_range(start.normalize(), end.normalize(), freq="D")
        key = index.normalize()
    elif freq == "W":
        full = pd.period_range(start.to_period("W-SUN"), end.to_period("W-SUN"), freq="W-SUN")
        key = index.to_period("W-SUN")
        counted = key.value_counts()
        return counted.reindex(full, fill_value=0).astype(int)
    else:
        full = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
        key = index.to_period("M")
        counted = key.value_counts()
        return counted.reindex(full, fill_value=0).astype(int)
    counted = key.value_counts()
    return counted.reindex(full, fill_value=0).astype(int)


def _bucket_line(name: str, counts: pd.Series, r_by) -> str:
    if len(counts) == 0:
        return f"{name}: нет"
    active = counts[counts > 0]
    return (
        f"{name}: бакетов {len(counts)}  среднее {counts.mean():.2f}  "
        f"медиана {counts.median():.1f}  в дни/недели со сделками среднее {active.mean():.2f}  "
        f"макс {int(counts.max())}  пустых {(counts == 0).mean() * 100:.0f}%  "
        f"R среднее {r_by.mean():+.2f}  лучший {r_by.max():+.1f}  худший {r_by.min():+.1f}"
    )


def _r_by(df: pd.DataFrame, freq: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if freq == "D":
        key = df["fill"].dt.normalize()
    elif freq == "W":
        key = df["fill"].dt.to_period("W-SUN")
    else:
        key = df["fill"].dt.to_period("M")
    return df.groupby(key)["r"].sum()


def _block(df: pd.DataFrame, title: str) -> list[str]:
    lines = [title]
    if df.empty:
        lines.append("сделок нет")
        return lines
    r = df["r"].to_numpy(float)
    g = df["gross"].to_numpy(float)
    wr = float((df["outcome"] == "tp").mean() * 100.0)
    lines.append(
        f"все: n={len(df)}  WR={wr:.1f}%  sumR={r.sum():+.1f}  avgR={r.mean():+.3f}  "
        f"PF={_pf(r):.2f}  maxDD={_dd(r):+.1f}R  | до комиссии sum={g.sum():+.1f} avg={g.mean():+.3f}"
    )
    lines.append(
        f"стоп медиана {df['stop_pct'].median():.2f}%  "
        f"удержание медиана {df['hold_min'].median():.0f} мин  "
        f"среднее {df['hold_min'].mean():.0f} мин"
    )
    for name, mask in (
        ("тейк", df["outcome"] == "tp"),
        ("стоп", df["outcome"] == "sl"),
        ("конец данных", df["outcome"] == "eod"),
    ):
        sub = df.loc[mask]
        if sub.empty:
            lines.append(f"{name}: 0")
            continue
        rr = sub["r"].to_numpy(float)
        lines.append(
            f"{name}: n={len(sub)}  доля {len(sub) / len(df) * 100:.1f}%  "
            f"sumR={rr.sum():+.1f}  avgR={rr.mean():+.3f}  "
            f"медиана {np.median(rr):+.3f}  до комиссии {sub['gross'].sum():+.1f}"
        )
    for side, sub in df.groupby("side"):
        rr = sub["r"].to_numpy(float)
        lines.append(
            f"{side}: n={len(sub)}  WR={(sub['outcome'] == 'tp').mean() * 100:.1f}%  "
            f"sumR={rr.sum():+.1f}  avgR={rr.mean():+.3f}  PF={_pf(rr):.2f}  maxDD={_dd(rr):+.1f}R"
        )
    fills = pd.DatetimeIndex(df["fill"])
    span_days = max((fills.max() - fills.min()).total_seconds() / 86400.0, 1.0)
    lines.append(
        f"темп по диапазону филлов {fills.min()} .. {fills.max()}  "
        f"{span_days:.0f} суток  "
        f"{len(df) / span_days:.2f} в день  {len(df) / span_days * 7:.1f} в неделю  "
        f"{len(df) / span_days * 30.44:.1f} в месяц"
    )
    for freq, name in (("D", "день"), ("W", "неделя"), ("M", "месяц")):
        counts = _counts(fills, freq)
        rsum = _r_by(df, freq).reindex(counts.index, fill_value=0.0)
        lines.append(_bucket_line(name, counts, rsum))
    df = df.copy()
    df["year"] = df["fill"].dt.year
    for year, sub in df.groupby("year"):
        rr = sub["r"].to_numpy(float)
        lines.append(
            f"{year}: n={len(sub)}  WR={(sub['outcome'] == 'tp').mean() * 100:.1f}%  "
            f"sumR={rr.sum():+.1f}  avgR={rr.mean():+.3f}  PF={_pf(rr):.2f}  "
            f"maxDD={_dd(rr):+.1f}R  тейк {(sub['outcome'] == 'tp').sum()}  "
            f"стоп {(sub['outcome'] == 'sl').sum()}"
        )
    return lines


def _pair_table(df: pd.DataFrame) -> list[str]:
    lines = ["пара | с | по | n | WR | sumR | avgR | PF | maxDD | тейк | стоп | в месяц | медиана стопа"]
    if df.empty:
        return lines
    order = [key for key, _label, _path in PAIRS]
    for sym in order:
        sub = df[df["symbol"] == sym]
        if sub.empty:
            lines.append(f"{sym} | — | — | 0 | — | — | — | — | — | 0 | 0 | — | —")
            continue
        rr = sub["r"].to_numpy(float)
        fills = pd.DatetimeIndex(sub["fill"])
        span = max((fills.max() - fills.min()).total_seconds() / 86400.0, 1.0)
        per_m = len(sub) / span * 30.44
        lines.append(
            f"{sym} | {fills.min():%Y-%m-%d} | {fills.max():%Y-%m-%d} | {len(sub)} | "
            f"{(sub['outcome'] == 'tp').mean() * 100:.1f}% | {rr.sum():+.1f} | {rr.mean():+.3f} | "
            f"{_pf(rr):.2f} | {_dd(rr):+.1f} | {(sub['outcome'] == 'tp').sum()} | "
            f"{(sub['outcome'] == 'sl').sum()} | {per_m:.1f} | {sub['stop_pct'].median():.2f}%"
        )
    return lines


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    books: dict[float, list[pd.DataFrame]] = {rr: [] for rr in RRS}
    funnels = []
    for key, label, path in PAIRS:
        t1 = time.perf_counter()
        print(f"\n== {label} ==", flush=True)
        m1 = load_ms(path)
        if m1.index[0].year < 2017:
            raise RuntimeError(f"{label}: странное время {m1.index[0]}")
        m15 = resample(m1, "15min", 15)
        print(
            f"1m {len(m1)}  {m1.index[0]} .. {m1.index[-1]}  M15 {len(m15)}",
            flush=True,
        )
        setups, gate = find_setups(m15)
        n_long = sum(s["side"] == 1 for s in setups)
        print(
            f"свипы {gate['sweeps']} низ {gate['down']} верх {gate['up']}  "
            f"сетапы {gate['kept']} long {n_long} short {len(setups) - n_long}",
            flush=True,
        )
        funnels.append(
            {
                "symbol": key,
                "label": label,
                "start": str(m1.index[0]),
                "end": str(m1.index[-1]),
                "bars_1m": len(m1),
                "bars_15m": len(m15),
                **gate,
                "long": n_long,
                "short": len(setups) - n_long,
            }
        )
        for rr in RRS:
            trades, stats = run_book(
                setups, m15, m1, rr=rr, need_h4=False, entry_kind="limit", step_min=1,
            )
            df = pd.DataFrame(trades)
            if not df.empty:
                df.insert(0, "symbol", key)
            books[rr].append(df)
            print(
                f"RR {rr:g}: филл {stats['filled']}  занято {stats['blocked']}  "
                f"лимитка мимо {stats['expired']}",
                flush=True,
            )
        del m1, m15, setups
        print(f"готово за {time.perf_counter() - t1:.0f} с", flush=True)

    pd.DataFrame(funnels).to_csv(OUT / "pairs_funnel.csv", index=False)
    chunks = [
        "Десять пар cisdbot. Единорог: свип соседних 4ч, свеча перед серией, FVG.",
        "Лимитка на край этой свечи, стоп за другим краем. Путь по 1m.",
        "Книга отдельная на каждую пару. GRAM — файл TONUSDT.",
        "",
    ]
    for row in funnels:
        chunks.append(
            f"{row['label']}: {row['start']} .. {row['end']}  "
            f"свипы {row['sweeps']}  сетапы {row['kept']} long {row['long']} short {row['short']}  "
            f"нет серии {row['no_series']} нет свечи {row['no_origin']} "
            f"нет слома {row['no_mss']} нет FVG {row['no_fvg']}"
        )
    chunks.append("")
    for rr in RRS:
        frames = [d for d in books[rr] if not d.empty]
        df = _prepare(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
        df.to_csv(OUT / f"pairs_rr{_tag(rr)}.csv", index=False)
        chunks.append("")
        chunks.extend(_block(df, f"RR 1:{rr:g}  все 10 пар вместе"))
        chunks.append("")
        chunks.extend(_pair_table(df))
        if not df.empty:
            month = (
                df.assign(month=df["fill"].dt.to_period("M").astype(str))
                .groupby(["month", "symbol"], sort=True)["r"]
                .agg(n="count", sumR="sum")
                .reset_index()
            )
            month.to_csv(OUT / f"pairs_month_rr{_tag(rr)}.csv", index=False)
    text = "\n".join(chunks)
    (OUT / "pairs_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)
    print(f"\nвсё за {time.perf_counter() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
