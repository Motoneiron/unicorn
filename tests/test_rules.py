# -*- coding: utf-8 -*-
"""Правила входа совпадают с бэктестом нового стека."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from bot.rules import (
    atr14,
    drop_forming,
    entry_deadline,
    exit_price,
    first_entry_event,
    position_hit,
    position_qty,
    round_levels,
    sizing_weight,
    time_exit_due,
    trigger_stop,
)
from bt.execute import TAKER, _fill_limit, _walk


def test_parity():
    rng = np.random.default_rng(7)
    for side, entry, stop in ((1, 100.0, 99.0), (-1, 100.0, 101.0)):
        for _ in range(30):
            span = abs(entry - stop)
            lows = entry + rng.normal(0, span, 40)
            highs = lows + rng.random(40) * span
            got = first_entry_event(side, entry, stop, lows, highs)
            kind, idx = _fill_limit(
                side, entry, stop, np.asarray(lows, float), np.asarray(highs, float), 0, len(lows),
            )
            assert got == (kind, idx), (got, kind, idx)


def test_same_bar_fill_beats_2r():
    kind, idx = first_entry_event(1, 100.0, 99.0, [100.0], [102.0])
    assert (kind, idx) == ("fill", 0)


def test_2r_before_fill_cancels():
    kind, idx = first_entry_event(1, 100.0, 99.0, [100.5, 100.0], [102.0, 100.2])
    assert (kind, idx) == ("ran", 0)
    kind, idx = first_entry_event(-1, 100.0, 101.0, [98.0, 99.8], [99.5, 100.0])
    assert (kind, idx) == ("ran", 0)


def test_stop_before_tp():
    assert position_hit(1, 99.0, 102.0, 98.0, 103.0) == "sl"
    lows = np.array([99.5, 98.0])
    highs = np.array([101.0, 103.0])
    outcome, j, px, fee = _walk(1, 99.0, 102.0, lows, highs, np.array([101.0, 100.0]), 0)
    assert outcome == "sl" and j == 1


def test_deadline_is_two_hours_after_close():
    mss = pd.Timestamp("2026-01-01 12:00")
    assert entry_deadline(mss) == pd.Timestamp("2026-01-01 14:15")


def test_drop_forming():
    idx = pd.date_range("2026-01-01 12:00", periods=3, freq="15min")
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1}, index=idx)
    now = pd.Timestamp("2026-01-01 12:40")
    out = drop_forming(df, 15, now)
    assert list(out.index) == list(idx[:2])


def test_qty_and_levels():
    assert position_qty(100, 100, 99, 0.1) == 1.0
    assert round_levels(1, 100.014, 99.006, 0.01) == (100.01, 99.0, 102.03)
    assert position_qty(1000, 100, 99, 0.1, risk_pct=1.0) == 10.0
    assert position_qty(1000, 100, 99, 0.1, risk_pct=1.5) == 15.0


def test_sizing_weight():
    cases = [
        (1.5, 1.5),
        (1.89, 1.5),
        (1.9, 1.25),
        (2.27, 1.25),
        (2.28, 1.0),
        (2.72, 1.0),
        (2.73, 0.75),
        (3.38, 0.75),
        (3.39, 0.5),
        (10, 0.5),
        (None, 1.0),
        (float("nan"), 1.0),
        (0, 1.0),
        (-3, 1.0),
    ]
    for raw, expect in cases:
        assert sizing_weight(raw) == expect, (raw, sizing_weight(raw), expect)


def test_atr14_flat_and_spike():
    n = 20
    high = np.full(n, 2.0)
    low = np.full(n, 1.0)
    close = np.full(n, 1.5)
    atr = atr14(high, low, close)
    assert np.isnan(atr[12])
    assert abs(atr[13] - 1.0) < 1e-12
    assert abs(atr[19] - 1.0) < 1e-12

    high2 = high.copy()
    high2[15] = 5.0  # TR = max(4, |5-1.5|, |1-1.5|) = 4
    atr2 = atr14(high2, low, close)
    # RMA: atr[14]=1; atr[15]=(1*13+4)/14
    expect15 = (1.0 * 13 + 4.0) / 14
    assert abs(atr2[15] - expect15) < 1e-12
    expect16 = (expect15 * 13 + 1.0) / 14
    assert abs(atr2[16] - expect16) < 1e-12


def test_trigger_stop():
    df = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2],
            "high": [10.5, 10.4, 10.8],
            "low": [9.8, 9.5, 10.0],
            "close": [10.2, 9.7, 10.6],
        }
    )
    long_setup = {"side": 1, "sweep_i": 1}
    short_setup = {"side": -1, "sweep_i": 2}
    assert trigger_stop(df, long_setup) == 9.5
    assert trigger_stop(df, short_setup) == 10.8


def test_time_exit_due():
    opened = pd.Timestamp("2026-01-01 12:00")
    assert time_exit_due(opened, now=opened + pd.Timedelta(hours=5, minutes=59)) is False
    assert time_exit_due(opened, now=opened + pd.Timedelta(hours=6, minutes=1)) is True


def test_time_exit_price_is_taker_with_slip():
    px, fee = exit_price(1, 99.0, 102.0, "time", mark=100.0)
    assert fee == TAKER
    assert abs(px - 100.0 * (1.0 - 0.0001)) < 1e-12


def main():
    test_parity()
    test_same_bar_fill_beats_2r()
    test_2r_before_fill_cancels()
    test_stop_before_tp()
    test_deadline_is_two_hours_after_close()
    test_drop_forming()
    test_qty_and_levels()
    test_sizing_weight()
    test_atr14_flat_and_spike()
    test_trigger_stop()
    test_time_exit_due()
    test_time_exit_price_is_taker_with_slip()
    print("ok")


if __name__ == "__main__":
    main()
