# -*- coding: utf-8 -*-
"""Короткий самотест геометрии и комиссии. Без файла свечей."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bt.execute import MAKER, SLIP, TAKER, r_multiples, run_book
from bt.pattern import find_unicorns


def toy_bullish() -> pd.DataFrame:
    """Один бычий Единорог: лоу 90, хай 120, снятие 80, слом на баре 29, зона 114–118."""
    n = 40
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l = np.full(n, 100.0)
    c = np.full(n, 100.0)
    # Плавный спуск в свинг-лоу 10 (90), подъём в свинг-хай 18 (120), снятие на 26 (80).
    path = {
        8: (99, 102, 98, 99),
        9: (98, 101, 94, 97),
        10: (96, 100, 90, 95),
        11: (95, 104, 93, 103),
        12: (103, 108, 97, 107),
        13: (107, 111, 100, 110),
        14: (110, 113, 103, 112),
        15: (112, 115, 106, 114),
        16: (114, 116, 109, 115),
        17: (115, 117, 111, 116),
        18: (114, 120, 111, 118),
        19: (117, 118, 108, 109),
        20: (109, 112, 104, 105),
        21: (105, 108, 100, 101),
        22: (101, 104, 96, 97),
        23: (97, 100, 93, 94),
        24: (94, 98, 90.5, 92),
        25: (92, 96, 88, 89),
        26: (89, 95, 80, 86),
        27: (86, 108, 81, 100),
        28: (100, 118, 99, 115),
        29: (116, 130, 125, 128),
    }
    for i, (oo, hh, ll, cc) in path.items():
        o[i], h[i], l[i], c[i] = oo, hh, ll, cc
    for i in range(30, n):
        o[i], h[i], l[i], c[i] = 128, 129, 127, 128
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": 1.0},
        index=pd.date_range("2024-01-01", periods=n, freq="15min"),
    )


def check() -> None:
    r, r06, _g = r_multiples(1, 100.0, 99.0, 102.0, MAKER)
    if abs(r - 1.928) > 1e-9:
        raise AssertionError(f"тейк R {r}")
    if abs(r06 - 1.88) > 1e-9:
        raise AssertionError(f"тейк R06 {r06}")
    exit_sl = 99.0 * (1.0 - SLIP)
    gross = (exit_sl - 100.0) / 1.0
    expect = gross - MAKER / 0.01 - TAKER / 0.01
    r_sl, _, _g2 = r_multiples(1, 100.0, 99.0, exit_sl, TAKER)
    if abs(r_sl - expect) > 1e-9:
        raise AssertionError(f"стоп R {r_sl} != {expect}")

    df = toy_bullish()
    setups, funnel = find_unicorns(df)
    bulls = [s for s in setups if s["side"] == 1]
    if len(bulls) != 1:
        raise AssertionError(f"ждал 1 бычий сетап, получил {len(bulls)}; воронка {funnel}")
    s = bulls[0]
    if abs(s["entry"] - 118.0) > 1e-9 or abs(s["stop"] - 80.0) > 1e-9:
        raise AssertionError(f"зона/стоп {s['entry']} {s['stop']}")
    if s["form_i"] != 29:
        raise AssertionError(f"свеча слома {s['form_i']}")

    # Лимитка 100, стоп 99, тейк 102. Первая 5m касается входа, следующая — тейка.
    t0 = pd.Timestamp("2024-01-02")
    m15 = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0]},
        index=[t0],
    )
    m5_idx = pd.date_range(t0 + pd.Timedelta(minutes=15), periods=3, freq="5min")
    m5 = pd.DataFrame(
        {
            "open": [100.5, 100.2, 101.0],
            "high": [100.6, 102.2, 102.0],
            "low": [100.0, 100.1, 100.5],
            "close": [100.2, 102.0, 101.5],
        },
        index=m5_idx,
    )
    setup = {
        "side": 1,
        "form_i": 0,
        "known_i": 0,
        "entry": 100.0,
        "stop": 99.0,
        "zone_bot": 99.5,
        "zone_top": 100.0,
        "sweep": 99.0,
        "ob_i": 0,
        "sweep_i": 0,
        "pivot_i": 0,
        "h4": 1,
    }
    trades, _ = run_book([setup], m15, m5, rr=2.0, need_h4=False)
    if len(trades) != 1 or trades[0]["outcome"] != "tp":
        raise AssertionError(f"исполнение {trades}")
    if abs(trades[0]["r"] - 1.928) > 1e-9:
        raise AssertionError(f"R сделки {trades[0]['r']}")

    # Стоп-заявка на хай: первая 5m не доходит, вторая пробивает 100, третья берёт тейк 102.
    stop_idx = pd.date_range(t0 + pd.Timedelta(minutes=15), periods=3, freq="5min")
    stop_path = pd.DataFrame(
        {
            "open": [99.6, 99.7, 100.2],
            "high": [99.8, 100.2, 102.4],
            "low": [99.4, 99.6, 100.0],
            "close": [99.7, 100.1, 102.2],
        },
        index=stop_idx,
    )
    stop_setup = dict(setup)
    stop_setup["entry"] = 100.0
    stop_setup["stop"] = 99.0
    got, st = run_book([stop_setup], m15, stop_path, rr=2.0, need_h4=False, entry_kind="stop", step_min=5)
    if st["filled"] != 1 or got[0]["outcome"] != "tp":
        raise AssertionError(f"стоп-вход {got} {st}")
    fill = 100.0 * (1.0 + SLIP)
    risk = 1.0
    gross_tp = (102.0 - fill) / risk
    expect_stop = gross_tp - TAKER / 0.01 - MAKER / 0.01
    if abs(got[0]["r"] - expect_stop) > 1e-9:
        raise AssertionError(f"R стоп-входа {got[0]['r']} != {expect_stop}")

    # 2R раньше лимитки: первая свеча пути бьёт 102 и не касается 100.
    ran_idx = pd.date_range(t0 + pd.Timedelta(minutes=15), periods=2, freq="5min")
    ran_path = pd.DataFrame(
        {
            "open": [100.5, 100.2],
            "high": [102.0, 100.4],
            "low": [100.4, 99.8],
            "close": [101.5, 100.0],
        },
        index=ran_idx,
    )
    ran_trades, ran_st = run_book([setup], m15, ran_path, rr=2.0, need_h4=False)
    if ran_trades or ran_st["ran_2r"] != 1:
        raise AssertionError(f"отмена по 2R {ran_trades} {ran_st}")
