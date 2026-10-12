# -*- coding: utf-8 -*-
"""Единорог на других ТФ. Вход 6 часов, позиция до стопа или 2R.

5м → 1ч, 10м → 2ч, 30м → 8ч, 1ч → день.
Девять пар без BTC. 2R до филла отменяет лимитку.

Запуск: python run_tfs.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from bt.cisd_setup import find_setups
from bt.data import load_ms, resample
from run_pairs import PAIRS
from run_timeouts import _prepare, _simulate_hold, _stats

# Подпись, правило ресемпла, сколько минуток в баре, старший ТФ в минутах.
TFS = (
    ("5м-1ч", "5min", 5, 60),
    ("10м-2ч", "10min", 10, 120),
    ("30м-8ч", "30min", 30, 480),
    ("1ч-день", "60min", 60, 1440),
)
ENTRY_MIN = 360
OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    lines = [
        "9 пар без BTC. Тейк 2R, лимитка 6 часов, позиция до стопа или тейка.",
        "2R до филла отменяет вход. База 15м → 4ч при тех же правилах: "
        "4873 сделки, WR 47.8%, +1024.7 R, PF 1.31, просадка -34.8 R, стопов 2546.",
        "",
    ]
    rows = []
    for label, rule, bar_min, htf_min in TFS:
        books: list[dict] = []
        lines.append(label)
        print(f"\n==== {label} ====", flush=True)
        for key, name, path in PAIRS:
            if key == "BTC":
                continue
            t1 = time.perf_counter()
            print(f"== {name} ==", flush=True)
            m1 = load_ms(path)
            ltf = resample(m1, rule, bar_min)
            setups, gate = find_setups(ltf, htf_min=htf_min)
            recs, ts, op, cl = _prepare(
                ltf, m1, setups,
                entry_span_min=ENTRY_MIN,
                pos_until_end=True,
                bar_min=bar_min,
            )
            trades = _simulate_hold(recs, ts, op, cl, ENTRY_MIN)
            for trade in trades:
                trade["symbol"] = key
            books.extend(trades)
            st = _stats(trades)
            row = f"{key}: сетапы {gate['kept']}  n={st['n']}  sumR={st['sum']:+.1f}  стоп {st['sl']}"
            lines.append("  " + row)
            print(row, f"  {time.perf_counter() - t1:.0f} с", flush=True)
            del m1, ltf, setups, recs
        st = _stats(books)
        rows.append({"tf": label, **st})
        lines.append(
            f"ИТОГО {label}: n={st['n']}  WR={st['wr']:.1f}%  sumR={st['sum']:+.1f}  "
            f"avgR={st['avg']:+.3f}  PF={st['pf']:.2f}  maxDD={st['dd']:+.1f}  "
            f"тейк {st['tp']} ({st['tp_r']:+.0f}R)  стоп {st['sl']} ({st['sl_r']:+.0f}R)"
        )
        lines.append("")
        text = "\n".join(lines)
        (OUT / "tf_summary.txt").write_text(text, encoding="utf-8")
        pd.DataFrame(rows).to_csv(OUT / "tf_grid.csv", index=False)
        print(lines[-2], flush=True)
    print(f"\nвсё за {time.perf_counter() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
