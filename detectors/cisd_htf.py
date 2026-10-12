"""
Детектор CISD (Change in State of Delivery)
Портировано с ICT FRACTAL MODEL Pine Script v2.8.6

ИСПРАВЛЕНИЯ v3:
1. Проверка "свежести" от конца HTF свечи, а не от конца датасета
2. Проверка КАЖДОЙ пары 1→2, 2→3, 3→4 независимо (убран if not found)
3. htf_ratio теперь в минутах (дефолт 60 для 1H)
4. Улучшена дедупликация (добавлен trigger_idx в ключ)

ИСПРАВЛЕНИЯ v4:
1. Добавлена валидация свипа через 2 следующих HTF диапазона
2. Проверка обновления экстремума триггер-бара в следующих диапазонах
3. Логика "один валидный свип за раз" (флаги high_sweep_found/low_sweep_found)
4. Пары 2→3 и 3→4 проверяются только если не найден свип в предыдущих парах

ИСПРАВЛЕНИЯ v5 (проекции как Pine v2.8.6):
1. База Wick: отрезок start_bar — sweep_trigger (не до confirm); вторая граница — high/low триггера свипа
2. Опционально projections_source='body'; уровни из projection_levels (дефолт 1,2,2.5,4,4.5)
3. Колонка cisd_sweep_trigger_idx

ИСПРАВЛЕНИЯ v6 (заменено v7):

ИСПРАВЛЕНИЯ v7 (база проекций):
1. Bearish Wick: base_low = low старт-бара, base_high = high триггер-бара; start==trigger → low/high той же свечи.
2. Bullish Wick: base_low = low триггер-бара, base_high = high старт-бара (зеркально); start==trigger → low/high той же свечи.
3. Body: нижняя/верхняя граница тел — с тех же баров (min/max open,close соответственно).

ИСПРАВЛЕНИЯ v8:
1. Поиск конфирма ограничен по времени: только бары < floor_htf(trigger) + 2 * htf_ratio
   (HTF-диапазон триггера + следующий), плюс прежний лимит max_range.

ИСПРАВЛЕНИЯ v9 (старт-серия = Pine v2.8.6):
1. Каскад баров −2/−3 при i=1 против серии и триггере против серии.
2. start_bar на каждом баре серии (при равных open — самый ранний).

ИСПРАВЛЕНИЯ v12 (doji в старт-серии):
1. open==close не прерывает серию: для верхнего свипа (бычья серия) — бычий бар;
   для нижнего свипа (медвежья серия) — медвежий бар.

ИСПРАВЛЕНИЯ v11 (свип: wick trigger + отдельный close-arm):
1. trigger_idx — wick экстремум как Pine f_find_extreme_sweep (tie → правый).
2. find_first_sweep_close_confirm_bar — close rejection от trigger вперёд; сигнал только после него.
3. confirm (MSS/CISD) по-прежнему отдельно от trigger; confirm_on_sweep если индексы совпали.
4. find_sweep_trigger_bar — без формирующейся свечи; после close попадёт в trigger, если экстремум выше.
"""
import logging
import pandas as pd
import numpy as np
from .base import BaseDetector

logger = logging.getLogger("cisd_bot")


# Как в Pine: f_parse_projection_levels + дефолтная строка уровней
DEFAULT_PROJECTION_LEVELS: tuple[float, ...] = (1.0, 2.0, 2.5, 4.0, 4.5)


class CISDDetector(BaseDetector):
    """Детектор CISD паттернов, портированный с Pine Script v2.8.6"""

    def __init__(
        self,
        htf_ratio: int = 60,
        max_range: int = 50,
        projections_source: str = "wick",
        projection_levels: list[float] | None = None,
        htf_confirm_windows: int = 2,
    ):
        """
        Args:
            htf_ratio: длительность HTF в минутах (60 для 5m→1H)
            max_range: максимум баров для поиска серии и подтверждения
            projections_source: "wick" или "body" — как в Pine cisd_projections_source
            projection_levels: уровни k для множителя (k+1); по умолчанию как в Pine
            htf_confirm_windows: HTF-окон от anchor для close-arm / MSS (дефолт 2)
        """
        super().__init__("CISD")
        self.htf_ratio = htf_ratio
        self.max_range = max_range
        self.htf_confirm_windows = max(1, int(htf_confirm_windows))
        self.projections_source = projections_source
        lv = projection_levels if projection_levels is not None else list(DEFAULT_PROJECTION_LEVELS)
        self.projection_levels = lv[:5] if len(lv) > 5 else lv
        # MSS: закрытие конфирма должно быть строго за экстремумом старт-бара (low/high), не только за уровнем серии
        self.confirm_must_close_past_start_bar = False

    @staticmethod
    def compute_projection_base(
        df: pd.DataFrame,
        start_idx: int,
        trigger_idx: int,
        is_bearish: bool,
        source: str,
    ) -> tuple[float, float]:
        """
        База для проекций: только экстремумы двух баров — старт и триггер (confirm не участвует).

        Wick Bearish: base_low = low старт-бара, base_high = high триггер-бара.
        Wick Bullish: base_low = low триггер-бара, base_high = high старт-бара.
        Если start_idx == trigger_idx — одна свеча: её low и high.

        Body: те же два бара, границы по телам (min/max open и close на соответствующей стороне).
        """
        row_s = df.iloc[start_idx]
        row_t = df.iloc[trigger_idx]

        if source == "body":
            if is_bearish:
                base_low = float(min(row_s["open"], row_s["close"]))
                base_high = float(max(row_t["open"], row_t["close"]))
            else:
                base_low = float(min(row_t["open"], row_t["close"]))
                base_high = float(max(row_s["open"], row_s["close"]))
            return base_high, base_low

        if is_bearish:
            base_low = float(row_s["low"])
            base_high = float(row_t["high"])
        else:
            base_low = float(row_t["low"])
            base_high = float(row_s["high"])
        return base_high, base_low

    def _fib_prices(self, base_high: float, base_low: float, is_bearish: bool) -> list[float]:
        price_range = base_high - base_low
        out: list[float] = []
        for level in self.projection_levels:
            m = level + 1.0
            if is_bearish:
                out.append(base_high - m * price_range)
            else:
                out.append(base_low + m * price_range)
        while len(out) < 5:
            out.append(float("nan"))
        return out[:5]

    def resample_to_htf(self, df: pd.DataFrame) -> pd.DataFrame:
        rule = f'{self.htf_ratio}min'
        htf = pd.DataFrame({
            'open':   df['open'].resample(rule).first(),
            'high':   df['high'].resample(rule).max(),
            'low':    df['low'].resample(rule).min(),
            'close':  df['close'].resample(rule).last(),
            'volume': df['volume'].resample(rule).sum(),
        })
        return htf.dropna()

    def get_ltf_candles_inside_htf(
        self, df_ltf: pd.DataFrame, htf_time: pd.Timestamp
    ) -> pd.DataFrame:
        htf_end = htf_time + pd.Timedelta(minutes=self.htf_ratio)
        mask = (df_ltf.index >= htf_time) & (df_ltf.index < htf_end)
        return df_ltf[mask].copy()

    def get_ltf_candles_in_htf_windows(
        self,
        df_ltf: pd.DataFrame,
        htf_anchor: pd.Timestamp,
        n_htf: int = 2,
    ) -> pd.DataFrame:
        """LTF-бары в n HTF-окнах от floor(htf_anchor) (для trigger/confirm)."""
        rule = f"{int(self.htf_ratio)}min"
        start = pd.Timestamp(htf_anchor).floor(rule)
        end = start + pd.Timedelta(minutes=int(self.htf_ratio) * n_htf)
        mask = (df_ltf.index >= start) & (df_ltf.index < end)
        return df_ltf[mask].copy()

    def _ltf_for_sweep_trigger(
        self, ltf_candles: pd.DataFrame, df_ltf: pd.DataFrame,
    ) -> pd.DataFrame:
        """LTF в HTF без формирующейся свечи (ccxt: последний бар df — open)."""
        if len(ltf_candles) == 0:
            return ltf_candles
        if len(df_ltf) >= 2 and ltf_candles.index[-1] == df_ltf.index[-1]:
            return ltf_candles.iloc[:-1]
        return ltf_candles

    def _bar_is_sweep_close_arm(
        self, row: pd.Series, sweep_level: float, is_bearish: bool, eps: float = 1e-8,
    ) -> bool:
        """Close-rejection свипа на одном баре (как find_first_sweep_close_confirm_bar)."""
        lvl = float(sweep_level)
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        if is_bearish:
            return h >= lvl - eps and c < lvl
        return h >= lvl - eps and l <= lvl + eps and c > lvl

    def find_sweep_trigger_bar(
        self,
        ltf_candles: pd.DataFrame,
        sweep_level: float,
        is_bearish: bool,
        df_ltf: pd.DataFrame,
    ) -> int | None:
        """
        LTF trigger: экстремум фитиля как Pine f_find_extreme_sweep.
        Bearish: max high среди баров с high >= level.
        Bullish: min low среди баров с low <= level.
        При равенстве — правый бар. Формирующуюся свечу не учитываем.
        Close-rejection — отдельно в find_first_sweep_close_confirm_bar.
        """
        eps = 1e-8
        ltf = self._ltf_for_sweep_trigger(ltf_candles, df_ltf)
        if len(ltf) == 0:
            return None

        best_val = None
        best_ts = None

        for ts, row in ltf.iterrows():
            if is_bearish:
                v = float(row["high"])
                if v < float(sweep_level) - eps:
                    continue
                if (
                    best_val is None
                    or v > best_val + eps
                    or (abs(v - best_val) <= eps and (best_ts is None or ts > best_ts))
                ):
                    best_val = v
                    best_ts = ts
            else:
                v = float(row["low"])
                if v > float(sweep_level) + eps:
                    continue
                if (
                    best_val is None
                    or v < best_val - eps
                    or (abs(v - best_val) <= eps and (best_ts is None or ts > best_ts))
                ):
                    best_val = v
                    best_ts = ts

        if best_ts is None:
            return None

        return df_ltf.index.get_loc(best_ts)

    def find_first_sweep_close_confirm_bar(
        self,
        df: pd.DataFrame,
        sweep_trigger_idx: int,
        sweep_level: float,
        is_bearish: bool,
        *,
        htf_anchor: pd.Timestamp | None = None,
    ) -> int | None:
        """
        Первый бар от trigger вперёд с close-подтверждением свипа (rejection).
        htf_anchor — час регистрации свипа; окно поиска <= floor(anchor)+2 HTF.
        """
        eps = 1e-8
        closed_end = len(df) - 1 if len(df) >= 2 else len(df)
        end = min(
            sweep_trigger_idx + self.max_range + 1,
            closed_end,
            self._confirm_end_iloc_exclusive_by_htf(
                df, sweep_trigger_idx, htf_anchor=htf_anchor,
            ),
        )
        for i in range(sweep_trigger_idx, end):
            row = df.iloc[i]
            if self._bar_is_sweep_close_arm(row, sweep_level, is_bearish, eps):
                return i
        return None

    def find_sweep_origin_bar(
        self,
        ltf_candles: pd.DataFrame,
        sweep_level: float,
        is_bearish: bool,
        df_ltf: pd.DataFrame,
    ) -> int | None:
        """
        Находит бар в предыдущем HTF диапазоне, где находится свипнутый экстремум.
        Это бар, который создал high/low, который был свипнут.
        """
        eps = 1e-8
        best_val = None
        best_ts = None

        for ts, row in ltf_candles.iterrows():
            if is_bearish:
                # Ищем бар с high на уровне sweep_level
                if abs(row['high'] - sweep_level) <= eps:
                    if best_val is None or row['high'] >= sweep_level - eps:
                        best_val = row['high']
                        best_ts = ts
            else:
                # Ищем бар с low на уровне sweep_level
                if abs(row['low'] - sweep_level) <= eps:
                    if best_val is None or row['low'] <= sweep_level + eps:
                        best_val = row['low']
                        best_ts = ts

        if best_ts is None:
            return None

        return df_ltf.index.get_loc(best_ts)

    @staticmethod
    def htf_candles_sweep(c_later, c_earlier, bearish: bool) -> bool:
        """Pine f_is_sweep: high/low пробой + close rejection между двумя HTF-свечами."""
        if bearish:
            return (
                float(c_later["high"]) > float(c_earlier["high"])
                and float(c_later["close"]) < float(c_earlier["high"])
            )
        return (
            float(c_later["low"]) < float(c_earlier["low"])
            and float(c_later["close"]) > float(c_earlier["low"])
        )

    def validate_sweep_with_next_htf_ranges(
        self,
        sweep_candle_index: int,
        sweep_trigger_idx: int,
        is_bearish: bool,
        df_htf: pd.DataFrame,
        df_ltf: pd.DataFrame,
        htf_index: int
    ) -> bool:
        """ОТКЛЮЧЕНО при детекте: всегда True. Валидация 2→3 — в боте при закрытии HTF-окна."""
        return True

    def detect_htf_sweeps(self, df_htf: pd.DataFrame, df_ltf: pd.DataFrame) -> list:
        """
        Детекция HTF sweeps с валидацией:
        1. HTF условие
        2. Поиск LTF trigger bar
        3. Проверка "свежести" от КОНЦА HTF СВЕЧИ (не от конца датасета!)
        4. НОВОЕ: Валидация через 2 следующих HTF диапазона (экстремум не должен обновляться)
        5. НОВОЕ: Один валидный свип за раз (если нашли свип 1-2, не проверяем 2-3 и 3-4)
        """
        sweeps = []

        for i in range(3, len(df_htf)):
            candle_1 = df_htf.iloc[i]
            candle_2 = df_htf.iloc[i - 1]
            candle_3 = df_htf.iloc[i - 2]
            candle_4 = df_htf.iloc[i - 3]

            # Флаги для отслеживания найденных свипов (как в Pine Script)
            high_sweep_found = False
            low_sweep_found = False

            # Пара 1→2 (проверяем всегда)
            if candle_1['high'] > candle_2['high'] and candle_1['close'] < candle_2['high']:
                htf_time = df_htf.index[i]
                ltf_candles = self.get_ltf_candles_in_htf_windows(df_ltf, htf_time, n_htf=2)
                
                if len(ltf_candles) > 0:
                    sweep_level = candle_2['high']
                    trigger_idx = self.find_sweep_trigger_bar(ltf_candles, sweep_level, True, df_ltf)
                    
                    # Проверка "свежести" от КОНЦА HTF СВЕЧИ
                    if trigger_idx is not None:
                        htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                        if htf_end_idx - trigger_idx <= 60:
                            # НОВОЕ: Валидация через 2 следующих HTF диапазона
                            is_valid = self.validate_sweep_with_next_htf_ranges(
                                sweep_candle_index=1,
                                sweep_trigger_idx=trigger_idx,
                                is_bearish=True,
                                df_htf=df_htf,
                                df_ltf=df_ltf,
                                htf_index=i
                            )
                            
                            if is_valid:
                                htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                                
                                # Находим origin bar (свипнутый максимум в диапазоне candle_2)
                                htf_time_origin = df_htf.index[i - 1]
                                ltf_candles_origin = self.get_ltf_candles_inside_htf(df_ltf, htf_time_origin)
                                origin_idx = self.find_sweep_origin_bar(ltf_candles_origin, sweep_level, True, df_ltf)
                                
                                sweeps.append({
                                    'type': 'bearish',
                                    'htf_time': htf_time,
                                    'sweep_level': sweep_level,
                                    'trigger_idx': trigger_idx,
                                    'origin_idx': origin_idx,
                                    'pair': '1-2',
                                    'sweep_htf_end_idx': htf_end_idx,
                                })
                                high_sweep_found = True

            if candle_1['low'] < candle_2['low'] and candle_1['close'] > candle_2['low']:
                htf_time = df_htf.index[i]
                ltf_candles = self.get_ltf_candles_in_htf_windows(df_ltf, htf_time, n_htf=2)
                
                if len(ltf_candles) > 0:
                    sweep_level = candle_2['low']
                    trigger_idx = self.find_sweep_trigger_bar(ltf_candles, sweep_level, False, df_ltf)
                    
                    if trigger_idx is not None:
                        htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                        if htf_end_idx - trigger_idx <= 60:
                            # НОВОЕ: Валидация через 2 следующих HTF диапазона
                            is_valid = self.validate_sweep_with_next_htf_ranges(
                                sweep_candle_index=1,
                                sweep_trigger_idx=trigger_idx,
                                is_bearish=False,
                                df_htf=df_htf,
                                df_ltf=df_ltf,
                                htf_index=i
                            )
                            
                            if is_valid:
                                htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                                
                                # Находим origin bar (свипнутый минимум в диапазоне candle_2)
                                htf_time_origin = df_htf.index[i - 1]
                                ltf_candles_origin = self.get_ltf_candles_inside_htf(df_ltf, htf_time_origin)
                                origin_idx = self.find_sweep_origin_bar(ltf_candles_origin, sweep_level, False, df_ltf)
                                
                                sweeps.append({
                                    'type': 'bullish',
                                    'htf_time': htf_time,
                                    'sweep_level': sweep_level,
                                    'trigger_idx': trigger_idx,
                                    'origin_idx': origin_idx,
                                    'pair': '1-2',
                                    'sweep_htf_end_idx': htf_end_idx,
                                })
                                low_sweep_found = True

            # Пара 2→3 (проверяем ТОЛЬКО если не нашли свип в паре 1→2)
            if not high_sweep_found:
                if candle_2['high'] > candle_3['high'] and candle_2['close'] < candle_3['high']:
                    htf_time = df_htf.index[i - 1]
                    ltf_candles = self.get_ltf_candles_in_htf_windows(df_ltf, htf_time, n_htf=2)
                    
                    if len(ltf_candles) > 0:
                        sweep_level = candle_3['high']
                        trigger_idx = self.find_sweep_trigger_bar(ltf_candles, sweep_level, True, df_ltf)
                        
                        if trigger_idx is not None:
                            htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                            if htf_end_idx - trigger_idx <= 60:
                                # НОВОЕ: Валидация через 2 следующих HTF диапазона
                                is_valid = self.validate_sweep_with_next_htf_ranges(
                                    sweep_candle_index=2,
                                    sweep_trigger_idx=trigger_idx,
                                    is_bearish=True,
                                    df_htf=df_htf,
                                    df_ltf=df_ltf,
                                    htf_index=i
                                )
                                
                                if is_valid:
                                    htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                                    
                                    # Находим origin bar (свипнутый максимум в диапазоне candle_3)
                                    htf_time_origin = df_htf.index[i - 2]
                                    ltf_candles_origin = self.get_ltf_candles_inside_htf(df_ltf, htf_time_origin)
                                    origin_idx = self.find_sweep_origin_bar(ltf_candles_origin, sweep_level, True, df_ltf)
                                    
                                    sweeps.append({
                                        'type': 'bearish',
                                        'htf_time': htf_time,
                                        'sweep_level': sweep_level,
                                        'trigger_idx': trigger_idx,
                                        'origin_idx': origin_idx,
                                        'pair': '2-3',
                                        'sweep_htf_end_idx': htf_end_idx,
                                    })
                                    high_sweep_found = True

            if not low_sweep_found:
                if candle_2['low'] < candle_3['low'] and candle_2['close'] > candle_3['low']:
                    htf_time = df_htf.index[i - 1]
                    ltf_candles = self.get_ltf_candles_in_htf_windows(df_ltf, htf_time, n_htf=2)
                    
                    if len(ltf_candles) > 0:
                        sweep_level = candle_3['low']
                        trigger_idx = self.find_sweep_trigger_bar(ltf_candles, sweep_level, False, df_ltf)
                        
                        if trigger_idx is not None:
                            htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                            if htf_end_idx - trigger_idx <= 60:
                                # НОВОЕ: Валидация через 2 следующих HTF диапазона
                                is_valid = self.validate_sweep_with_next_htf_ranges(
                                    sweep_candle_index=2,
                                    sweep_trigger_idx=trigger_idx,
                                    is_bearish=False,
                                    df_htf=df_htf,
                                    df_ltf=df_ltf,
                                    htf_index=i
                                )
                                
                                if is_valid:
                                    htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                                    
                                    # Находим origin bar (свипнутый минимум в диапазоне candle_3)
                                    htf_time_origin = df_htf.index[i - 2]
                                    ltf_candles_origin = self.get_ltf_candles_inside_htf(df_ltf, htf_time_origin)
                                    origin_idx = self.find_sweep_origin_bar(ltf_candles_origin, sweep_level, False, df_ltf)
                                    
                                    sweeps.append({
                                        'type': 'bullish',
                                        'htf_time': htf_time,
                                        'sweep_level': sweep_level,
                                        'trigger_idx': trigger_idx,
                                        'origin_idx': origin_idx,
                                        'pair': '2-3',
                                        'sweep_htf_end_idx': htf_end_idx,
                                    })
                                    low_sweep_found = True

            # Пара 3→4 (проверяем ТОЛЬКО если не нашли свип в парах 1→2 и 2→3)
            if not high_sweep_found:
                if candle_3['high'] > candle_4['high'] and candle_3['close'] < candle_4['high']:
                    htf_time = df_htf.index[i - 2]
                    ltf_candles = self.get_ltf_candles_in_htf_windows(df_ltf, htf_time, n_htf=2)
                    
                    if len(ltf_candles) > 0:
                        sweep_level = candle_4['high']
                        trigger_idx = self.find_sweep_trigger_bar(ltf_candles, sweep_level, True, df_ltf)
                        
                        if trigger_idx is not None:
                            htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                            if htf_end_idx - trigger_idx <= 60:
                                # НОВОЕ: Валидация через 2 следующих HTF диапазона
                                is_valid = self.validate_sweep_with_next_htf_ranges(
                                    sweep_candle_index=3,
                                    sweep_trigger_idx=trigger_idx,
                                    is_bearish=True,
                                    df_htf=df_htf,
                                    df_ltf=df_ltf,
                                    htf_index=i
                                )
                                
                                if is_valid:
                                    htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                                    
                                    # Находим origin bar (свипнутый максимум в диапазоне candle_4)
                                    htf_time_origin = df_htf.index[i - 3]
                                    ltf_candles_origin = self.get_ltf_candles_inside_htf(df_ltf, htf_time_origin)
                                    origin_idx = self.find_sweep_origin_bar(ltf_candles_origin, sweep_level, True, df_ltf)
                                    
                                    sweeps.append({
                                        'type': 'bearish',
                                        'htf_time': htf_time,
                                        'sweep_level': sweep_level,
                                        'trigger_idx': trigger_idx,
                                        'origin_idx': origin_idx,
                                        'pair': '3-4',
                                        'sweep_htf_end_idx': htf_end_idx,
                                    })

            if not low_sweep_found:
                if candle_3['low'] < candle_4['low'] and candle_3['close'] > candle_4['low']:
                    htf_time = df_htf.index[i - 2]
                    ltf_candles = self.get_ltf_candles_in_htf_windows(df_ltf, htf_time, n_htf=2)
                    
                    if len(ltf_candles) > 0:
                        sweep_level = candle_4['low']
                        trigger_idx = self.find_sweep_trigger_bar(ltf_candles, sweep_level, False, df_ltf)
                        
                        if trigger_idx is not None:
                            htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                            if htf_end_idx - trigger_idx <= 60:
                                # НОВОЕ: Валидация через 2 следующих HTF диапазона
                                is_valid = self.validate_sweep_with_next_htf_ranges(
                                    sweep_candle_index=3,
                                    sweep_trigger_idx=trigger_idx,
                                    is_bearish=False,
                                    df_htf=df_htf,
                                    df_ltf=df_ltf,
                                    htf_index=i
                                )
                                
                                if is_valid:
                                    htf_end_idx = df_ltf.index.get_loc(ltf_candles.index[-1])
                                    
                                    # Находим origin bar (свипнутый минимум в диапазоне candle_4)
                                    htf_time_origin = df_htf.index[i - 3]
                                    ltf_candles_origin = self.get_ltf_candles_inside_htf(df_ltf, htf_time_origin)
                                    origin_idx = self.find_sweep_origin_bar(ltf_candles_origin, sweep_level, False, df_ltf)
                                    
                                    sweeps.append({
                                        'type': 'bullish',
                                        'htf_time': htf_time,
                                        'sweep_level': sweep_level,
                                        'trigger_idx': trigger_idx,
                                        'origin_idx': origin_idx,
                                        'pair': '3-4',
                                        'sweep_htf_end_idx': htf_end_idx,
                                    })

        # Улучшенная дедупликация (с trigger_idx)
        seen = set()
        unique_sweeps = []
        for s in sweeps:
            key = (s['type'], s['htf_time'], s['sweep_level'], s['trigger_idx'])
            if key not in seen:
                seen.add(key)
                unique_sweeps.append(s)

        return unique_sweeps

    def find_bullish_series(self, df: pd.DataFrame, sweep_trigger_idx: int) -> dict:
        """Серия бычьих баров для Bearish CISD — Pine v2.8.6 (1185–1250)."""
        min_open = None
        start_bar = None
        found_bearish = False

        range_size = min(self.max_range, sweep_trigger_idx)
        trig_row = df.iloc[sweep_trigger_idx]

        for i in range(1, range_size + 1):
            idx = sweep_trigger_idx - i
            if idx < 0:
                break

            o = float(df.iloc[idx]["open"])
            c = float(df.iloc[idx]["close"])

            if i == 1:
                if c >= o:
                    min_open = o
                    start_bar = idx
                    continue
                o_t = float(trig_row["open"])
                c_t = float(trig_row["close"])
                if c_t >= o_t:
                    min_open = o_t
                    start_bar = sweep_trigger_idx
                    found_bearish = False
                    break
                second_idx = sweep_trigger_idx - 2
                if second_idx < 0:
                    found_bearish = True
                    break
                o2 = float(df.iloc[second_idx]["open"])
                c2 = float(df.iloc[second_idx]["close"])
                if c2 < o2:
                    third_idx = sweep_trigger_idx - 3
                    if third_idx < 0:
                        found_bearish = True
                        break
                    o3 = float(df.iloc[third_idx]["open"])
                    c3 = float(df.iloc[third_idx]["close"])
                    if c3 >= o3:
                        min_open = o3
                        start_bar = third_idx
                        found_bearish = False
                        continue
                    found_bearish = True
                    break
                min_open = o2
                start_bar = second_idx
                found_bearish = False
                continue

            if c < o:
                found_bearish = True
                break
            if c >= o:
                min_open = o if min_open is None else min(min_open, o)
                start_bar = idx

        if not found_bearish and min_open is None:
            o_t = float(trig_row["open"])
            c_t = float(trig_row["close"])
            if c_t >= o_t:
                min_open = o_t
                start_bar = sweep_trigger_idx
                found_bearish = True

        return {
            "found": found_bearish or (min_open is not None),
            "start_index": start_bar,
            "min_open": min_open,
        }

    def find_bearish_series(self, df: pd.DataFrame, sweep_trigger_idx: int) -> dict:
        """Серия медвежьих баров для Bullish CISD — зеркало Pine v2.8.6 (1706–1770)."""
        max_open = None
        start_bar = None
        found_bullish = False

        range_size = min(self.max_range, sweep_trigger_idx)
        trig_row = df.iloc[sweep_trigger_idx]

        for i in range(1, range_size + 1):
            idx = sweep_trigger_idx - i
            if idx < 0:
                break

            o = float(df.iloc[idx]["open"])
            c = float(df.iloc[idx]["close"])

            if i == 1:
                if c <= o:
                    max_open = o
                    start_bar = idx
                    continue
                o_t = float(trig_row["open"])
                c_t = float(trig_row["close"])
                if c_t <= o_t:
                    max_open = o_t
                    start_bar = sweep_trigger_idx
                    found_bullish = False
                    break
                second_idx = sweep_trigger_idx - 2
                if second_idx < 0:
                    found_bullish = True
                    break
                o2 = float(df.iloc[second_idx]["open"])
                c2 = float(df.iloc[second_idx]["close"])
                if c2 > o2:
                    third_idx = sweep_trigger_idx - 3
                    if third_idx < 0:
                        found_bullish = True
                        break
                    o3 = float(df.iloc[third_idx]["open"])
                    c3 = float(df.iloc[third_idx]["close"])
                    if c3 <= o3:
                        max_open = o3
                        start_bar = third_idx
                        found_bullish = False
                        continue
                    found_bullish = True
                    break
                max_open = o2
                start_bar = second_idx
                found_bullish = False
                continue

            if c > o:
                found_bullish = True
                break
            if c <= o:
                max_open = o if max_open is None else max(max_open, o)
                start_bar = idx

        if not found_bullish and max_open is None:
            o_t = float(trig_row["open"])
            c_t = float(trig_row["close"])
            if c_t <= o_t:
                max_open = o_t
                start_bar = sweep_trigger_idx
                found_bullish = True

        return {
            "found": found_bullish or (max_open is not None),
            "start_index": start_bar,
            "max_open": max_open,
        }

    def _confirm_end_iloc_exclusive_by_htf(
        self,
        df: pd.DataFrame,
        sweep_trigger_idx: int,
        *,
        htf_anchor: pd.Timestamp | None = None,
    ) -> int:
        """
        Первый iloc, с которого поиск конфирма заканчивается (бары не включаются).
        Окно: floor(htf_anchor или trigger) + htf_confirm_windows * htf_ratio.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return len(df)
        td = pd.Timedelta(minutes=int(self.htf_ratio))
        rule = f"{int(self.htf_ratio)}min"
        if htf_anchor is not None:
            htf_start = pd.Timestamp(htf_anchor).floor(rule)
        else:
            t0 = pd.Timestamp(df.index[sweep_trigger_idx])
            htf_start = t0.floor(rule)
        n_win = int(getattr(self, "htf_confirm_windows", 2) or 2)
        window_end_exclusive = htf_start + n_win * td
        pos = int(df.index.searchsorted(window_end_exclusive, side="left"))
        return max(0, min(pos, len(df)))

    def find_confirmation(
        self,
        df: pd.DataFrame,
        sweep_trigger_idx: int,
        level: float,
        is_bearish: bool,
        start_idx: int | None = None,
        require_close_past_start_extreme: bool | None = None,
        *,
        htf_anchor: pd.Timestamp | None = None,
    ) -> int | None:
        end = min(
            sweep_trigger_idx + self.max_range + 1,
            len(df),
            self._confirm_end_iloc_exclusive_by_htf(
                df, sweep_trigger_idx, htf_anchor=htf_anchor,
            ),
        )
        past_start = (
            require_close_past_start_extreme
            if require_close_past_start_extreme is not None
            else getattr(self, "confirm_must_close_past_start_bar", False)
        )

        for i in range(sweep_trigger_idx, end):
            o = df.iloc[i]["open"]
            c = df.iloc[i]["close"]
            if is_bearish and c < o and c < level:
                if past_start and start_idx is not None:
                    if float(c) >= float(df.iloc[start_idx]["low"]):
                        continue
                return i
            if not is_bearish and c > o and c > level:
                if past_start and start_idx is not None:
                    if float(c) <= float(df.iloc[start_idx]["high"]):
                        continue
                return i

        return None

    def is_extremum_updated(
        self, df: pd.DataFrame, sweep_trigger_idx: int, confirm_idx: int, is_bearish: bool
    ) -> bool:
        if confirm_idx <= sweep_trigger_idx:
            return False

        sweep_extreme = (
            df.iloc[sweep_trigger_idx]['high'] if is_bearish
            else df.iloc[sweep_trigger_idx]['low']
        )

        for i in range(sweep_trigger_idx, confirm_idx + 1):
            if is_bearish and df.iloc[i]['high'] > sweep_extreme:
                return True
            if not is_bearish and df.iloc[i]['low'] < sweep_extreme:
                return True

        return False

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_dataframe(df)

        df = df.copy()
        df['cisd_bullish'] = 0
        df['cisd_bearish'] = 0
        df['cisd_level'] = np.nan
        df['cisd_start_bar'] = np.nan
        df['cisd_confirm_bar'] = np.nan
        df['cisd_htf_sweep_level'] = np.nan
        df['cisd_sweep_htf_end_idx'] = np.nan
        df['cisd_sweep_origin_idx'] = np.nan
        df['cisd_sweep_trigger_idx'] = np.nan
        
        # Добавляем колонки для проекций Фибоначчи
        df['cisd_base_high'] = np.nan
        df['cisd_base_low'] = np.nan
        df['cisd_fib_0'] = np.nan  # level 1
        df['cisd_fib_1'] = np.nan  # level 2
        df['cisd_fib_2'] = np.nan  # level 2.5
        df['cisd_fib_3'] = np.nan  # level 4
        df['cisd_fib_4'] = np.nan  # level 4.5

        last_bearish_key = None
        last_bullish_key = None

        df_htf = self.resample_to_htf(df)
        htf_sweeps = self.detect_htf_sweeps(df_htf, df)
        logger.debug(f"HTF sweeps found: {len(htf_sweeps)}")

        for sweep in htf_sweeps:
            is_bearish = sweep['type'] == 'bearish'
            sweep_trigger_idx = sweep['trigger_idx']

            if is_bearish:
                # Для Bearish CISD (верхний свип) ищем серию БЫЧЬИХ баров
                # Берём минимальный open из всех бычьих баров
                series = self.find_bullish_series(df, sweep_trigger_idx)

                if not series['found'] or series['min_open'] is None:
                    continue

                start_i = series["start_index"]
                start_i_int = int(start_i) if start_i is not None else None
                confirm_idx = self.find_confirmation(
                    df,
                    sweep_trigger_idx,
                    series["min_open"],
                    is_bearish=True,
                    start_idx=start_i_int,
                )
                if confirm_idx is None:
                    continue

                if self.is_extremum_updated(df, sweep_trigger_idx, confirm_idx, True):
                    continue

                key = (series['start_index'], confirm_idx)
                if key == last_bearish_key:
                    continue
                last_bearish_key = key
                
                start_idx = int(series['start_index'])
                base_high, base_low = self.compute_projection_base(
                    df, start_idx, sweep_trigger_idx, True, self.projections_source
                )
                price_range = base_high - base_low
                if price_range <= 0:
                    continue

                fib_1, fib_2, fib_2_5, fib_4, fib_4_5 = self._fib_prices(
                    base_high, base_low, True
                )

                df.loc[df.index[confirm_idx], 'cisd_bearish'] = 1
                df.loc[df.index[confirm_idx], 'cisd_level'] = series['min_open']
                df.loc[df.index[confirm_idx], 'cisd_start_bar'] = series['start_index']
                df.loc[df.index[confirm_idx], 'cisd_confirm_bar'] = confirm_idx
                df.loc[df.index[confirm_idx], 'cisd_htf_sweep_level'] = sweep['sweep_level']
                df.loc[df.index[confirm_idx], 'cisd_sweep_htf_end_idx'] = sweep.get('sweep_htf_end_idx', np.nan)
                df.loc[df.index[confirm_idx], 'cisd_sweep_origin_idx'] = sweep.get('origin_idx', np.nan)
                df.loc[df.index[confirm_idx], 'cisd_sweep_trigger_idx'] = sweep_trigger_idx
                df.loc[df.index[confirm_idx], 'cisd_base_high'] = base_high
                df.loc[df.index[confirm_idx], 'cisd_base_low'] = base_low
                df.loc[df.index[confirm_idx], 'cisd_fib_0'] = fib_1
                df.loc[df.index[confirm_idx], 'cisd_fib_1'] = fib_2
                df.loc[df.index[confirm_idx], 'cisd_fib_2'] = fib_2_5
                df.loc[df.index[confirm_idx], 'cisd_fib_3'] = fib_4
                df.loc[df.index[confirm_idx], 'cisd_fib_4'] = fib_4_5

            else:
                # Для Bullish CISD (нижний свип) ищем серию МЕДВЕЖЬИХ баров
                # Берём максимальный open из всех медвежьих баров
                series = self.find_bearish_series(df, sweep_trigger_idx)

                if not series['found'] or series['max_open'] is None:
                    continue

                start_i = series["start_index"]
                start_i_int = int(start_i) if start_i is not None else None
                confirm_idx = self.find_confirmation(
                    df,
                    sweep_trigger_idx,
                    series["max_open"],
                    is_bearish=False,
                    start_idx=start_i_int,
                )
                if confirm_idx is None:
                    continue

                if self.is_extremum_updated(df, sweep_trigger_idx, confirm_idx, False):
                    continue

                key = (series['start_index'], confirm_idx)
                if key == last_bullish_key:
                    continue
                last_bullish_key = key
                
                start_idx = int(series['start_index'])
                base_high, base_low = self.compute_projection_base(
                    df, start_idx, sweep_trigger_idx, False, self.projections_source
                )
                price_range = base_high - base_low
                if price_range <= 0:
                    continue

                fib_1, fib_2, fib_2_5, fib_4, fib_4_5 = self._fib_prices(
                    base_high, base_low, False
                )

                df.loc[df.index[confirm_idx], 'cisd_bullish'] = 1
                df.loc[df.index[confirm_idx], 'cisd_level'] = series['max_open']
                df.loc[df.index[confirm_idx], 'cisd_start_bar'] = series['start_index']
                df.loc[df.index[confirm_idx], 'cisd_confirm_bar'] = confirm_idx
                df.loc[df.index[confirm_idx], 'cisd_htf_sweep_level'] = sweep['sweep_level']
                df.loc[df.index[confirm_idx], 'cisd_sweep_htf_end_idx'] = sweep.get('sweep_htf_end_idx', np.nan)
                df.loc[df.index[confirm_idx], 'cisd_sweep_origin_idx'] = sweep.get('origin_idx', np.nan)
                df.loc[df.index[confirm_idx], 'cisd_sweep_trigger_idx'] = sweep_trigger_idx
                df.loc[df.index[confirm_idx], 'cisd_base_high'] = base_high
                df.loc[df.index[confirm_idx], 'cisd_base_low'] = base_low
                df.loc[df.index[confirm_idx], 'cisd_fib_0'] = fib_1
                df.loc[df.index[confirm_idx], 'cisd_fib_1'] = fib_2
                df.loc[df.index[confirm_idx], 'cisd_fib_2'] = fib_2_5
                df.loc[df.index[confirm_idx], 'cisd_fib_3'] = fib_4
                df.loc[df.index[confirm_idx], 'cisd_fib_4'] = fib_4_5

        return df

    def get_summary(self, df: pd.DataFrame) -> dict:
        bullish_count = int(df['cisd_bullish'].sum())
        bearish_count = int(df['cisd_bearish'].sum())
        total = bullish_count + bearish_count

        return {
            'pattern': 'CISD',
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'total': total,
            'frequency': f"{(total / len(df) * 100):.2f}%" if len(df) > 0 else "0.00%",
        }
