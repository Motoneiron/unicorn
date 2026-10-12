# -*- coding: utf-8 -*-
"""Единорог без BTC. Не входим, если после FVG и до филла цена уже была на 2R.

Проверка по минутке. На одной минутке филл важнее касания 2R.
Тейки 1R, 1.5R, 2R. Уровень отмены всегда 2R от входа до стопа.

Запуск: python run_skip2r.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from bt.cisd_setup import find_setups
from bt.data import load_ms, resample
from bt.execute import run_book
from run_pairs import PAIRS, RRS, _block, _pair_table, _prepare, _tag

OUT_NAME = "pairs_skip2r_summary.txt"


def _check() -> None:
    """2R до филла снимает заявку. Филл и 2R на одной минутке — вход остаётся."""
    m15 = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.date_range("2024-01-01 10:00", periods=3, freq="15min"),
    )
    idx = pd.date_range("2024-01-01 10:15", periods=20, freq="1min")
    setup = {
        "side": 1, "form_i": 0, "known_i": 0, "entry": 100.0, "stop": 90.0,
        "sweep_i": 0, "h4": 0, "zone_bot": 1.0, "zone_top": 2.0,
    }

    def path(high, low) -> pd.DataFrame:
        n = len(idx)
        return pd.DataFrame(
            {
                "open": np.full(n, 110.0),
                "high": np.asarray(high, float),
                "low": np.asarray(low, float),
                "close": np.full(n, 110.0),
            },
            index=idx,
        )

    hi = np.full(20, 111.0)
    lo = np.full(20, 109.0)
    hi[1] = 120.0
    trades, stats = run_book(
        [setup], m15, path(hi, lo), rr=2, need_h4=False,
        entry_kind="limit", step_min=1, skip_ran_2r=True,
    )
    if stats["ran_2r"] != 1 or trades:
        raise AssertionError(f"ждал съём по 2R, {stats}")

    hi = np.full(20, 111.0)
    lo = np.full(20, 109.0)
    lo[0] = 100.0
    hi[5] = 130.0
    trades, stats = run_book(
        [setup], m15, path(hi, lo), rr=2, need_h4=False,
        entry_kind="limit", step_min=1, skip_ran_2r=True,
    )
    if stats["filled"] != 1 or stats["ran_2r"] != 0 or trades[0]["outcome"] != "tp":
        raise AssertionError(f"филл раньше 2R должен жить, {stats} {trades}")

    hi = np.full(20, 111.0)
    lo = np.full(20, 109.0)
    lo[0] = 100.0
    hi[0] = 120.0
    trades, stats = run_book(
        [setup], m15, path(hi, lo), rr=2, need_h4=False,
        entry_kind="limit", step_min=1, skip_ran_2r=True,
    )
    if stats["filled"] != 1 or stats["ran_2r"] != 0 or trades[0]["outcome"] != "tp":
        raise AssertionError(f"одна минутка филл+2R не должна снимать вход, {stats} {trades}")


def main() -> None:
    from pathlib import Path

    t0 = time.perf_counter()
    _check()
    print("самотест ок", flush=True)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    pairs = [row for row in PAIRS if row[0] != "BTC"]
    books: dict[float, list[pd.DataFrame]] = {rr: [] for rr in RRS}
    notes = []
    for key, label, path in pairs:
        t1 = time.perf_counter()
        print(f"\n== {label} ==", flush=True)
        m1 = load_ms(path)
        m15 = resample(m1, "15min", 15)
        setups, gate = find_setups(m15)
        print(f"сетапы {gate['kept']}", flush=True)
        for rr in RRS:
            trades, stats = run_book(
                setups, m15, m1, rr=rr, need_h4=False,
                entry_kind="limit", step_min=1, skip_ran_2r=True,
            )
            df = pd.DataFrame(trades)
            if not df.empty:
                df.insert(0, "symbol", key)
            books[rr].append(df)
            notes.append(
                f"{key} RR {rr:g}: филл {stats['filled']}  снято до входа, цена была на 2R "
                f"{stats['ran_2r']}  занято {stats['blocked']}  лимитка мимо {stats['expired']}"
            )
            print(notes[-1], flush=True)
        del m1, m15, setups
        print(f"готово за {time.perf_counter() - t1:.0f} с", flush=True)

    chunks = [
        "Девять пар без BTC. После FVG не входим, если до филла цена на минутке уже была на 2R.",
        "На одной минутке филл важнее 2R. Тейк сделки отдельно: 1R, 1.5R, 2R.",
        "",
        *notes,
        "",
    ]
    for rr in RRS:
        frames = [d for d in books[rr] if not d.empty]
        df = _prepare(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
        df.to_csv(out / f"pairs_skip2r_rr{_tag(rr)}.csv", index=False)
        chunks.append("")
        chunks.extend(_block(df, f"RR 1:{rr:g}  без BTC, фильтр 2R до входа"))
        chunks.append("")
        chunks.extend(ln for ln in _pair_table(df) if not ln.startswith("BTC"))
    text = "\n".join(chunks)
    (out / OUT_NAME).write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)
    print(f"\nвсё за {time.perf_counter() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
