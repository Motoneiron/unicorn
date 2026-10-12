# -*- coding: utf-8 -*-
"""Правила входа совпадают с бэктестом."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from bot.rules import (
    drop_forming,
    entry_deadline,
    first_entry_event,
    position_hit,
    position_qty,
    round_levels,
)
from bt.execute import _fill_limit, _walk


def test_parity():
    rng = np.random.default_rng(7)
    for side, entry, stop in ((1, 100.0, 99.0), (-1, 100.0, 101.0)):
        for _ in range(30):
            span = abs(entry - stop)
            lows = entry + rng.normal(0, span, 40)
            highs = lows + rng.random(40) * span
            got = first_entry_event(side, entry, stop, lows, highs)
            exp = _fill_limit(side, entry, stop, lows, highs, 0, len(lows))
            assert got == exp, (got, exp)


def test_same_bar_fill_beats_2r():
    kind, idx = first_entry_event(1, 100.0, 99.0, [100.0], [102.0])
    assert (kind, idx) == ("fill", 0)


def test_2r_before_fill_cancels():
    kind, idx = first_entry_event(1, 100.0, 99.0, [100.5, 100.0], [102.0, 100.2])
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


def main():
    test_parity()
    test_same_bar_fill_beats_2r()
    test_2r_before_fill_cancels()
    test_stop_before_tp()
    test_deadline_is_two_hours_after_close()
    test_drop_forming()
    test_qty_and_levels()
    print("ok")


if __name__ == "__main__":
    main()
