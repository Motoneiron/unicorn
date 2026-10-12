# -*- coding: utf-8 -*-
"""Единорог на M15.

Свинг — фрактал из 5 свечей (две слева, две справа), строгие неравенства.
Известен только после закрытия правой свечи (индекс свинг+2).

Бычий брейкер:
  предыдущий свинг-лоу, затем свинг-хай, затем более низкий свинг-лоу (снятие).
  Ордер-блок — последняя растущая свеча на отрезке от лоу до этого хая.
  Слом — первая свеча после снятия, чей close выше и хая структуры, и верха тела.
  На этой же свече обязан быть бычий FVG, и он пересекается с телом брейкера.

Медвежья сторона — зеркало. Импульс ищем не дальше IMPULSE баров M15:
дальше это уже не тот выстрел, а новый заход.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IMPULSE = 16
LIFE = 24
WING = 2


def fractals(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(low)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(WING, n - WING):
        hv = high[i]
        lv = low[i]
        if (
            hv > high[i - 1]
            and hv > high[i - 2]
            and hv > high[i + 1]
            and hv > high[i + 2]
        ):
            sh[i] = True
        if (
            lv < low[i - 1]
            and lv < low[i - 2]
            and lv < low[i + 1]
            and lv < low[i + 2]
        ):
            sl[i] = True
    return sh, sl


def _idx(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def _last_body(open_: np.ndarray, close: np.ndarray, i0: int, i1: int, bull: bool) -> int:
    for k in range(i1, i0 - 1, -1):
        if bull and close[k] > open_[k]:
            return k
        if not bull and close[k] < open_[k]:
            return k
    return -1


def _overlap(a_bot: float, a_top: float, b_bot: float, b_top: float):
    top = min(a_top, b_top)
    bot = max(a_bot, b_bot)
    if top > bot:
        return bot, top
    return None


def _between(sorted_idx: np.ndarray, left: int, right: int) -> int:
    """Последний индекс из sorted_idx строго внутри (left, right). -1 если нет."""
    lo = int(np.searchsorted(sorted_idx, left, side="right"))
    hi = int(np.searchsorted(sorted_idx, right, side="left"))
    if lo >= hi:
        return -1
    return int(sorted_idx[hi - 1])


def bias_with_open(high, low, open_, close, sh, sl) -> np.ndarray:
    """+1 после закрытия выше последнего неснятого свинг-хая, −1 ниже свинг-лоу."""
    n = len(close)
    bias = np.zeros(n, dtype=np.int8)
    last_sh = -1
    last_sl = -1
    broken_sh = -1
    broken_sl = -1
    cur = 0
    for i in range(n):
        j = i - WING
        if j >= WING and sh[j]:
            last_sh = j
        if j >= WING and sl[j]:
            last_sl = j
        up = last_sh >= 0 and close[i] > high[last_sh] and last_sh != broken_sh
        dn = last_sl >= 0 and close[i] < low[last_sl] and last_sl != broken_sl
        if up and not dn:
            cur = 1
            broken_sh = last_sh
        elif dn and not up:
            cur = -1
            broken_sl = last_sl
        elif up and dn:
            if close[i] >= open_[i]:
                cur = 1
                broken_sh = last_sh
            else:
                cur = -1
                broken_sl = last_sl
        bias[i] = cur
    return bias


def find_unicorns(df: pd.DataFrame) -> tuple[list[dict], dict]:
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    open_ = df["open"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(df)
    sh_m, sl_m = fractals(high, low)
    sh = _idx(sh_m)
    sl = _idx(sl_m)
    funnel = {
        "swing_high": int(len(sh)),
        "swing_low": int(len(sl)),
        "pairs": 0,
        "no_pivot": 0,
        "no_ob": 0,
        "no_mss": 0,
        "mss_if_48": 0,
        "mss_no_fvg": 0,
        "no_overlap": 0,
        "bad_risk": 0,
        "no_life": 0,
        "ok": 0,
    }
    out: list[dict] = []
    out.extend(_side(1, sl, sh, high, low, open_, close, n, funnel))
    out.extend(_side(-1, sh, sl, high, low, open_, close, n, funnel))
    out.sort(key=lambda s: (s["known_i"], s["form_i"], -s["side"]))
    return out, funnel


def _side(side, sweeps, pivots, high, low, open_, close, n, funnel) -> list[dict]:
    """side +1: sweeps = лоу, pivots = хаи. side −1: наоборот."""
    found = []
    bull = side == 1
    for a, b in zip(sweeps[:-1], sweeps[1:]):
        swept = low[b] < low[a] if bull else high[b] > high[a]
        if not swept:
            continue
        funnel["pairs"] += 1
        mid = _between(pivots, int(a), int(b))
        if mid < 0:
            funnel["no_pivot"] += 1
            continue
        ob = _last_body(open_, close, int(a) + 1, mid, bull)
        if ob < 0:
            funnel["no_ob"] += 1
            continue
        bb_top = float(max(open_[ob], close[ob]))
        bb_bot = float(min(open_[ob], close[ob]))
        if bull:
            level = max(float(high[mid]), bb_top)
        else:
            level = min(float(low[mid]), bb_bot)
        t = _break(close, int(b), n, level, bull, IMPULSE)
        if t < 0:
            if _break(close, int(b), n, level, bull, 48) >= 0:
                funnel["mss_if_48"] += 1
            funnel["no_mss"] += 1
            continue
        gap = _fvg(high, low, t, bull)
        if gap is None:
            funnel["mss_no_fvg"] += 1
            continue
        zone = _overlap(gap[0], gap[1], bb_bot, bb_top)
        if zone is None:
            funnel["no_overlap"] += 1
            continue
        zone_bot, zone_top = zone
        entry = zone_top if bull else zone_bot
        stop = float(low[b] if bull else high[b])
        ok_risk = entry > stop if bull else entry < stop
        if not ok_risk:
            funnel["bad_risk"] += 1
            continue
        known = max(t, int(b) + WING)
        if known >= t + LIFE or known >= n:
            funnel["no_life"] += 1
            continue
        funnel["ok"] += 1
        found.append(
            {
                "side": side,
                "form_i": t,
                "known_i": known,
                "entry": float(entry),
                "stop": stop,
                "zone_bot": float(zone_bot),
                "zone_top": float(zone_top),
                "sweep": stop,
                "ob_i": ob,
                "sweep_i": int(b),
                "pivot_i": mid,
                "h4": 0,
            }
        )
    return found


def _break(close: np.ndarray, sweep_i: int, n: int, level: float, bull: bool, window: int) -> int:
    end = min(n, sweep_i + 1 + window)
    for k in range(sweep_i + 1, end):
        if bull and close[k] > level:
            return k
        if not bull and close[k] < level:
            return k
    return -1


def _fvg(high, low, t: int, bull: bool):
    if t < 2:
        return None
    if bull and low[t] > high[t - 2]:
        return float(high[t - 2]), float(low[t])
    if not bull and high[t] < low[t - 2]:
        return float(high[t]), float(low[t - 2])
    return None


def h4_candle_sweeps(h4: pd.DataFrame, m15: pd.DataFrame) -> list[tuple[int, np.datetime64, np.datetime64]]:
    """Свип между соседними 4-часовыми свечами.

    Опорные хай и лоу — уже закрытая свеча. Следующая 4-часовая снимает их
    фитилём 15-минутки. Окно для Единорога: до конца этой 4-часовой и ещё
    одной следом, чтобы слом успел закрыться.
    Нижнее снятие → side +1, верхнее → side −1.
    """
    h_open = h4.index.to_numpy()
    h_high = h4["high"].to_numpy(float)
    h_low = h4["low"].to_numpy(float)
    m_open = m15.index.to_numpy()
    m_high = m15["high"].to_numpy(float)
    m_low = m15["low"].to_numpy(float)
    step = np.timedelta64(240, "m")
    bar = np.timedelta64(15, "m")
    events: list[tuple[int, np.datetime64, np.datetime64]] = []
    for i in range(1, len(h4)):
        t0 = h_open[i]
        t1 = t0 + step
        i0 = int(np.searchsorted(m_open, t0, side="left"))
        i1 = int(np.searchsorted(m_open, t1, side="left"))
        if i0 >= i1:
            continue
        end = t1 + step
        down = np.flatnonzero(m_low[i0:i1] < h_low[i - 1])
        up = np.flatnonzero(m_high[i0:i1] > h_high[i - 1])
        if len(down):
            start = m_open[i0 + int(down[0])] + bar
            events.append((1, start, end))
        if len(up):
            start = m_open[i0 + int(up[0])] + bar
            events.append((-1, start, end))
    return events


def _cover_event(starts: np.ndarray, ends: np.ndarray, t: np.datetime64):
    """Окно свипа, которое уже открыто в момент t. None, если свип ещё не был."""
    i = int(np.searchsorted(starts, t, side="right")) - 1
    for j in (i, i - 1):
        if j >= 0 and starts[j] <= t <= ends[j]:
            return starts[j], ends[j]
    return None


def select_h4_sweep(setups: list[dict], m15: pd.DataFrame, h4: pd.DataFrame) -> tuple[list[dict], dict]:
    """Единорог только после свипа соседней 4-часовой в ту же сторону.

    Сначала свип. Снятие ликвидности на 15 минутах и слом должны попасть
    в окно этого свипа, а не наоборот. Вход — лимитка на хай свечи слома
    (лонг) или на её лоу (шорт). Стоп — за другим краем той же свечи.
    """
    events = h4_candle_sweeps(h4, m15)
    by_side: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for side in (1, -1):
        sub = [e for e in events if e[0] == side]
        if sub:
            by_side[side] = (
                np.array([e[1] for e in sub]),
                np.array([e[2] for e in sub]),
            )
        else:
            by_side[side] = (np.array([], dtype="datetime64[ns]"), np.array([], dtype="datetime64[ns]"))
    high = m15["high"].to_numpy(float)
    low = m15["low"].to_numpy(float)
    opens = m15.index.to_numpy()
    bar = np.timedelta64(15, "m")
    kept: list[dict] = []
    stats = {
        "h4_sweeps": len(events),
        "h4_down": sum(e[0] == 1 for e in events),
        "h4_up": sum(e[0] == -1 for e in events),
        "before_sweep": 0,
        "late_break": 0,
        "flat_bar": 0,
        "killed": 0,
        "kept": 0,
    }
    for src in setups:
        side = int(src["side"])
        sweep_close = opens[src["sweep_i"]] + bar
        form_close = opens[src["form_i"]] + bar
        starts, ends = by_side[side]
        event = _cover_event(starts, ends, sweep_close) if len(starts) else None
        if event is None:
            stats["before_sweep"] += 1
            continue
        sweep_at, sweep_end = event
        if form_close > sweep_end:
            stats["late_break"] += 1
            continue
        hi = float(high[src["form_i"]])
        lo = float(low[src["form_i"]])
        if side == 1:
            entry, stop = hi, lo
        else:
            entry, stop = lo, hi
        if entry == stop or (side == 1 and entry <= stop) or (side == -1 and entry >= stop):
            stats["flat_bar"] += 1
            continue
        if src["known_i"] > src["form_i"]:
            slc = slice(src["form_i"] + 1, src["known_i"] + 1)
            if side == 1 and float(low[slc].min()) <= stop:
                stats["killed"] += 1
                continue
            if side == -1 and float(high[slc].max()) >= stop:
                stats["killed"] += 1
                continue
        row = dict(src)
        row["entry"] = entry
        row["stop"] = stop
        row["bar_high"] = hi
        row["bar_low"] = lo
        row["h4_sweep_at"] = pd.Timestamp(sweep_at)
        stats["kept"] += 1
        kept.append(row)
    return kept, stats


def attach_h4(setups: list[dict], m15: pd.DataFrame, h4: pd.DataFrame) -> None:
    hh = h4["high"].to_numpy(float)
    hl = h4["low"].to_numpy(float)
    ho = h4["open"].to_numpy(float)
    hc = h4["close"].to_numpy(float)
    sh, sl = fractals(hh, hl)
    bias = bias_with_open(hh, hl, ho, hc, sh, sl)
    h4_close = h4.index.to_numpy() + np.timedelta64(240, "m")
    m15_open = m15.index.to_numpy()
    for s in setups:
        known_close = m15_open[s["known_i"]] + np.timedelta64(15, "m")
        k = int(np.searchsorted(h4_close, known_close, side="right")) - 1
        s["h4"] = int(bias[k]) if k >= 0 else 0
