# -*- coding: utf-8 -*-
"""Сетап Единорога поверх движка CISD (15m → 4h).

Свип соседних 4ч, триггер по фитилю и серия свечей в свип — как
CISDDetector в cisdbot, профиль tf1. Сверху две вещи:

- свеча, от которой серия пошла в свип (она и есть Единорог);
- FVG на свече слома и пересечение с диапазоном этой свечи.

Лонг: нижний свип, медвежья серия, бычья свеча слева от неё.
Слом — бычье закрытие выше хая всей пачки от свечи Единорога
до бара свипа, оба края включительно. Лимитка остаётся на хай
самой свечи Единорога, стоп за её лоу. Шорт — зеркало.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.cisd_htf import CISDDetector  # noqa: E402

HTF_MIN = 240
MAX_RANGE = 50
CONFIRM_WINDOWS = 2


def _series_extreme(df: pd.DataFrame, uni: int, trigger: int, long_side: bool) -> float:
    """Хай лонга или лоу шорта от свечи Единорога до бара свипа, оба края включительно."""
    window = df.iloc[int(uni):int(trigger) + 1]
    if long_side:
        return float(window["high"].max())
    return float(window["low"].min())


def _origin(df: pd.DataFrame, start_index, long_side: bool) -> int | None:
    """Свеча сразу перед серией. Для лонга она бычья, для шорта медвежья."""
    if start_index is None:
        return None
    i = int(start_index) - 1
    if i < 0:
        return None
    o = float(df.iloc[i]["open"])
    c = float(df.iloc[i]["close"])
    if long_side and c > o:
        return i
    if not long_side and c < o:
        return i
    return None


def _fvg_overlap(
    df: pd.DataFrame, mss_i: int, uni_i: int, long_side: bool,
) -> tuple[float, float] | None:
    """FVG на свече слома и кусок, общий с диапазоном Единорога."""
    if mss_i < 2:
        return None
    hi = df["high"]
    lo = df["low"]
    if long_side:
        if float(lo.iloc[mss_i]) <= float(hi.iloc[mss_i - 2]):
            return None
        f_lo = float(hi.iloc[mss_i - 2])
        f_hi = float(lo.iloc[mss_i])
    else:
        if float(hi.iloc[mss_i]) >= float(lo.iloc[mss_i - 2]):
            return None
        f_lo = float(hi.iloc[mss_i])
        f_hi = float(lo.iloc[mss_i - 2])
    u_lo = float(lo.iloc[uni_i])
    u_hi = float(hi.iloc[uni_i])
    top = min(f_hi, u_hi)
    bot = max(f_lo, u_lo)
    if top <= bot:
        return None
    return bot, top


def _fvg_inside_sweep(m15: pd.DataFrame, htf_time, mss_i: int, htf_min: int) -> bool:
    """Три бара подтверждающего FVG открылись внутри свечи свипа."""
    if mss_i < 2:
        return False
    start = pd.Timestamp(htf_time)
    end = start + pd.Timedelta(minutes=int(htf_min))
    first = pd.Timestamp(m15.index[mss_i - 2])
    last = pd.Timestamp(m15.index[mss_i])
    return start <= first and last < end


def find_setups(
    m15: pd.DataFrame,
    htf_min: int = HTF_MIN,
    max_range: int = MAX_RANGE,
    confirm_windows: int = CONFIRM_WINDOWS,
    unicorn_mid_swing: bool = False,
    unicorn_clean_body: bool = False,
    fvg_in_sweep: bool = True,
) -> tuple[list[dict], dict]:
    """Сначала свипы старшего ТФ, потом Единорог на серии после каждого свипа.

    ПОПРАВКИ 14.10.2026 (флаги, дефолт False — канон не меняется):
    unicorn_mid_swing — единорог обязан быть средней (второй) свечой трехсвечного
    свинга (лонг: хай выше максимумов обоих соседей — первой и третьей свечи;
    шорт зеркально: лоу ниже минимумов обоих соседей).
    unicorn_clean_body — после свипа и до бара слома (исключая оба) ни одна свеча
    не должна перекрывать ТЕЛОМ тело единорога.
    fvg_in_sweep — подтверждающий FVG целиком внутри 4ч-свечи свипа,
    той же, где лежит триггер. В бэктесте включено. Бот передаёт False,
    пока прогон не покажет, что правило стоит переносить.
    """
    det = CISDDetector(
        htf_ratio=htf_min,
        max_range=max_range,
        htf_confirm_windows=confirm_windows,
    )
    htf = det.resample_to_htf(m15)
    sweeps = det.detect_htf_sweeps(htf, m15)

    # ФИКС 13.10.2026: для каждого бара-триггера — САМЫЙ СВЕЖИЙ свип (каскады),
    # и правило: триггер обязан лежать внутри своей свечи свипа.
    newest_by_trig: dict[tuple, pd.Timestamp] = {}
    for w in sweeps:
        k = (w["type"], int(w["trigger_idx"]))
        t = pd.Timestamp(w["htf_time"])
        if k not in newest_by_trig or t > newest_by_trig[k]:
            newest_by_trig[k] = t

    stats = {
        "sweeps": len(sweeps),
        "down": 0,
        "up": 0,
        "no_series": 0,
        "no_origin": 0,
        "no_midswing": 0,
        "no_mss": 0,
        "dirty_body": 0,
        "extremum": 0,
        "no_fvg": 0,
        "fvg_out": 0,
        "skip_out": 0,
        "kept": 0,
    }
    setups: list[dict] = []
    seen: set[tuple] = set()
    last_long = None
    last_short = None

    for sweep in sweeps:
        long_side = sweep["type"] == "bullish"
        if long_side:
            stats["down"] += 1
        else:
            stats["up"] += 1
        trigger = int(sweep["trigger_idx"])
        # ФИКС: якорь = самый свежий свип этого триггера; триггер должен быть в его свече.
        htf_time = newest_by_trig[(sweep["type"], trigger)]
        a_i = int(m15.index.searchsorted(htf_time, side="left"))
        if not (0 <= trigger - a_i < 16):
            stats["skip_out"] += 1
            continue
        if long_side:
            series = det.find_bearish_series(m15, trigger)
            level_key = "max_open"
        else:
            series = det.find_bullish_series(m15, trigger)
            level_key = "min_open"
        if not series["found"] or series[level_key] is None or series["start_index"] is None:
            stats["no_series"] += 1
            continue
        uni = _origin(m15, series["start_index"], long_side)
        if uni is None or uni >= trigger:
            stats["no_origin"] += 1
            continue
        if unicorn_mid_swing:
            # ПОПРАВКА 14.10.2026: единорог — средняя (вторая) свеча трехсвечного свинга.
            if uni < 1:
                stats["no_midswing"] += 1
                continue
            cp, lf, rt = m15.iloc[uni], m15.iloc[uni - 1], m15.iloc[uni + 1]
            if long_side:
                ok_swing = (float(cp["high"]) > float(lf["high"])
                            and float(cp["high"]) > float(rt["high"]))
            else:
                ok_swing = (float(cp["low"]) < float(lf["low"])
                            and float(cp["low"]) < float(rt["low"]))
            if not ok_swing:
                stats["no_midswing"] += 1
                continue
        level = _series_extreme(m15, uni, trigger, long_side)
        mss = det.find_confirmation(
            m15,
            trigger,
            level,
            is_bearish=not long_side,
            htf_anchor=htf_time,
        )
        if mss is None:
            stats["no_mss"] += 1
            continue
        if unicorn_clean_body:
            # ПОПРАВКА 14.10.2026: после свипа и до бара слома (оба края исключая)
            # ни одна свеча не перекрывает ТЕЛОМ тело единорога.
            u_lo_b = min(float(m15.iloc[uni]["open"]), float(m15.iloc[uni]["close"]))
            u_hi_b = max(float(m15.iloc[uni]["open"]), float(m15.iloc[uni]["close"]))
            dirty = False
            for k in range(trigger + 1, int(mss)):
                kb_lo = min(float(m15.iloc[k]["open"]), float(m15.iloc[k]["close"]))
                kb_hi = max(float(m15.iloc[k]["open"]), float(m15.iloc[k]["close"]))
                if kb_hi > u_lo_b and kb_lo < u_hi_b:
                    dirty = True
                    break
            if dirty:
                stats["dirty_body"] += 1
                continue
        if det.is_extremum_updated(m15, trigger, int(mss), not long_side):
            stats["extremum"] += 1
            continue
        if fvg_in_sweep and not _fvg_inside_sweep(m15, htf_time, int(mss), htf_min):
            stats["fvg_out"] += 1
            continue
        zone = _fvg_overlap(m15, int(mss), uni, long_side)
        if zone is None:
            stats["no_fvg"] += 1
            continue
        key = (1 if long_side else -1, uni, int(mss))
        if key in seen:
            continue
        dedup = (int(series["start_index"]), int(mss))
        if long_side:
            if dedup == last_long:
                continue
            last_long = dedup
        else:
            if dedup == last_short:
                continue
            last_short = dedup
        seen.add(key)
        entry = float(m15.iloc[uni]["high"] if long_side else m15.iloc[uni]["low"])
        stop = float(m15.iloc[uni]["low"] if long_side else m15.iloc[uni]["high"])
        if entry == stop:
            continue
        setups.append(
            {
                "side": 1 if long_side else -1,
                "form_i": int(mss),
                "known_i": int(mss),
                "entry": entry,
                "stop": stop,
                "sweep_i": trigger,
                "ob_i": uni,
                "h4": 0,
                "zone_bot": zone[0],
                "zone_top": zone[1],
                "h4_sweep_at": htf_time,
            }
        )
        stats["kept"] += 1

    setups.sort(key=lambda s: (s["known_i"], s["side"]))
    return setups, stats


def check_setup() -> None:
    """Один нижний свип 4ч, бычья свеча перед медвежьей серией, FVG на сломе."""
    n = 16 * 6
    o = [125.0] * n
    h = [126.0] * n
    l = [124.0] * n
    c = [125.0] * n
    # 12:00–16:00: лоу 100, закрытие 4ч ниже предыдущего лоу — это не свип.
    l[55] = 100.0
    c[63] = 105.0
    o[63] = 108.0
    # 16:00+: бычья свеча 70, медвежья серия, снятие на 74, слом с FVG на 77.
    o[70], h[70], l[70], c[70] = 102.0, 112.0, 101.0, 108.0
    o[71], h[71], l[71], c[71] = 108.0, 109.0, 104.0, 105.0
    o[72], h[72], l[72], c[72] = 105.0, 106.0, 100.0, 101.0
    o[73], h[73], l[73], c[73] = 101.0, 102.0, 96.0, 97.0
    o[74], h[74], l[74], c[74] = 97.0, 98.0, 90.0, 94.0
    o[75], h[75], l[75], c[75] = 94.0, 100.0, 93.0, 99.0
    o[76], h[76], l[76], c[76] = 99.0, 110.0, 98.0, 109.0
    o[77], h[77], l[77], c[77] = 109.0, 120.0, 111.0, 118.0
    c[79] = 115.0
    o[79] = 114.0
    df = pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": 1.0},
        index=pd.date_range("2024-01-01", periods=n, freq="15min"),
    )
    setups, stats = find_setups(df)
    longs = [s for s in setups if s["side"] == 1 and s["ob_i"] == 70]
    if len(longs) != 1:
        raise AssertionError(f"ждал единорога на баре 70, {stats} {setups}")
    s = longs[0]
    if s["form_i"] != 77 or abs(s["entry"] - 112.0) > 1e-9 or abs(s["stop"] - 101.0) > 1e-9:
        raise AssertionError(f"слом/лимитка {s}")
    if not (s["zone_bot"] < s["zone_top"]):
        raise AssertionError(f"зона {s['zone_bot']} {s['zone_top']}")
    kept, _ = find_setups(df, fvg_in_sweep=True)
    inside = [s for s in kept if s["side"] == 1 and s["ob_i"] == 70]
    if len(inside) != 1:
        raise AssertionError(f"FVG этого примера внутри свечи свипа, {inside}")
    # Бар после единорога выше его хая: закрытие 118 над 112 уже не слом.
    higher = df.copy()
    higher.iloc[71, higher.columns.get_loc("high")] = 119.0
    bumped, _ = find_setups(higher)
    if any(s["side"] == 1 and s["ob_i"] == 70 for s in bumped):
        raise AssertionError(f"слом взял хай единорога, а не пачки, {bumped}")
