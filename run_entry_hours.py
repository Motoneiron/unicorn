# -*- coding: utf-8 -*-
"""Жизнь лимитки от 1 до 6 часов. Позиция без таймера, до стопа или 2R.

Девять пар без BTC. Не входим, если 2R случился раньше филла.

Запуск: python run_entry_hours.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from bt.cisd_setup import find_setups
from bt.data import load_ms, resample
from run_pairs import PAIRS
from run_timeouts import _prepare, _simulate_hold, _stats

HOURS = (1, 2, 3, 4, 5, 6)
OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    books = {h: [] for h in HOURS}
    for key, label, path in PAIRS:
        if key == "BTC":
            continue
        t1 = time.perf_counter()
        print(f"== {label} ==", flush=True)
        m1 = load_ms(path)
        m15 = resample(m1, "15min", 15)
        setups, gate = find_setups(m15)
        recs, ts, op, cl = _prepare(m15, m1, setups, entry_span_min=360, pos_until_end=True)
        print(f"сетапы {gate['kept']}  разбор {time.perf_counter() - t1:.0f} с", flush=True)
        for hour in HOURS:
            trades = _simulate_hold(recs, ts, op, cl, hour * 60)
            for trade in trades:
                trade["symbol"] = key
            books[hour].extend(trades)
        del m1, m15, setups, recs

    lines = [
        "9 пар без BTC. Тейк 2R. Позиция до стопа или тейка, таймера на выход нет.",
        "Лимитку снимаем, если её не зафилило за 1–6 часов. 2R до филла по-прежнему отменяет вход.",
        "",
    ]
    rows = []
    for hour in HOURS:
        st = _stats(books[hour])
        rows.append({"hours": hour, **st})
        lines.append(
            f"{hour} ч  n={st['n']}  WR={st['wr']:.1f}%  sumR={st['sum']:+.1f}  "
            f"avgR={st['avg']:+.3f}  PF={st['pf']:.2f}  maxDD={st['dd']:+.1f}  "
            f"тейк {st['tp']} ({st['tp_r']:+.0f}R)  стоп {st['sl']} ({st['sl_r']:+.0f}R)"
        )
    text = "\n".join(lines)
    (OUT / "entry_hours_summary.txt").write_text(text, encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "entry_hours.csv", index=False)
    print(text, flush=True)
    print(f"всё за {time.perf_counter() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
