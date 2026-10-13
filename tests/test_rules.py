# -*- coding: utf-8 -*-
"""Правила входа совпадают с бэктестом нового стека."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from bot.rules import (
    drop_forming,
    entry_deadline,
    exit_price,
    first_entry_event,
    position_hit,
    position_qty,
    round_levels,
    time_exit_due,
    trigger_stop,
)
from bt.execute import TAKER, _fill, _walk


def test_parity():
    rng = np.random.default_rng(7)
    for side, entry, stop in ((1, 100.0, 99.0), (-1, 100.0, 101.0)):
        for _ in range(30):
            span = abs(entry - stop)
            lows = entry + rng.normal(0, span, 40)
            highs = lows + rng.random(40) * span
            got = first_entry_event(side, entry, stop, lows, highs)
            exp_i = _fill(side, entry, np.asarray(lows, float), np.asarray(highs, float), 0, len(lows))
            if exp_i < 0:
                assert got == ("miss", -1), (got, exp_i)
            else:
                assert got == ("fill", exp_i), (got, exp_i)


def test_same_bar_fill_beats_2r():
    kind, idx = first_entry_event(1, 100.0, 99.0, [100.0], [102.0])
    assert (kind, idx) == ("fill", 0)


def test_2r_before_fill_does_not_cancel():
    kind, idx = first_entry_event(1, 100.0, 99.0, [100.5, 100.0], [102.0, 100.2])
    assert (kind, idx) == ("fill", 1)


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
    test_2r_before_fill_does_not_cancel()
    test_stop_before_tp()
    test_deadline_is_two_hours_after_close()
    test_drop_forming()
    test_qty_and_levels()
    test_trigger_stop()
    test_time_exit_due()
    test_time_exit_price_is_taker_with_slip()
    print("ok")


if __name__ == "__main__":
    main()
