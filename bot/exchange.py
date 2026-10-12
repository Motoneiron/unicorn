# -*- coding: utf-8 -*-
"""Bybit USDT linear через ccxt. Публичные свечи работают без ключей, ордера только в живом режиме."""
from __future__ import annotations

import logging

import ccxt
import pandas as pd

from bot.config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    LEVERAGE,
    PAIRS,
)

logger = logging.getLogger("unicorn")


class Bybit:
    def __init__(self):
        self.ex = ccxt.bybit({
            "apiKey": BYBIT_API_KEY,
            "secret": BYBIT_API_SECRET,
            "timeout": 30000,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "defaultSubType": "linear",
                "recvWindow": 10000,
                "adjustForTimeDifference": True,
                "aiohttp_trust_env": False,
            },
        })
        sess = getattr(self.ex, "session", None)
        if sess is not None:
            sess.trust_env = False
        self._ready = False

    async def prepare(self) -> None:
        if self._ready:
            return
        await self._call(self.ex.load_markets)
        self._ready = True

    async def _call(self, fn, *args, **kwargs):
        import asyncio
        return await asyncio.to_thread(fn, *args, **kwargs)

    def _symbol(self, pair: str) -> str:
        return PAIRS[pair]["symbol"]

    def _market_id(self, pair: str) -> str:
        return self.ex.market(self._symbol(pair))["id"]

    async def ohlcv(self, pair: str, timeframe: str, limit: int) -> pd.DataFrame:
        rows = await self._call(self.ex.fetch_ohlcv, self._symbol(pair), timeframe, None, limit)
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df["ts"], unit="ms")
        df = df.drop(columns=["ts"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    async def last_price(self, pair: str) -> float | None:
        try:
            tick = await self._call(self.ex.fetch_ticker, self._symbol(pair))
        except Exception as e:
            logger.warning("цена %s: %s", pair, e)
            return None
        px = tick.get("last") or tick.get("close")
        return float(px) if px else None

    async def balance(self) -> float:
        bal = await self._call(self.ex.fetch_balance)
        info = bal.get("info") or {}
        result = info.get("result") if isinstance(info.get("result"), dict) else {}
        rows = result.get("list") if isinstance(result, dict) else None
        if isinstance(rows, list):
            for acc in rows:
                for coin in acc.get("coin") or []:
                    if str(coin.get("coin") or "").upper() != "USDT":
                        continue
                    raw = coin.get("equity") or coin.get("walletBalance")
                    if raw not in (None, ""):
                        return float(raw)
        total = (bal.get("total") or {}).get("USDT")
        if total not in (None, ""):
            return float(total)
        free = (bal.get("free") or {}).get("USDT")
        return float(free or 0)

    async def set_leverage(self, pair: str) -> None:
        sym = self._symbol(pair)
        try:
            await self._call(self.ex.set_margin_mode, "isolated", sym)
        except Exception as e:
            logger.info("маржа %s: %s", pair, e)
        try:
            await self._call(
                self.ex.set_leverage,
                LEVERAGE,
                sym,
                {"buyLeverage": str(LEVERAGE), "sellLeverage": str(LEVERAGE)},
            )
            logger.info("плечо %sx %s", LEVERAGE, pair)
        except Exception as e:
            logger.info("плечо %s: %s", pair, e)

    def _qty(self, pair: str, qty: float) -> float:
        return float(self.ex.amount_to_precision(self._symbol(pair), qty))

    def _px(self, pair: str, px: float) -> str:
        return self.ex.price_to_precision(self._symbol(pair), px)

    async def place_limit(self, pair: str, side: int, qty: float, price: float, link: str) -> dict:
        params = {
            "positionIdx": 0,
            "timeInForce": "PostOnly",
            "orderLinkId": link,
        }
        order_side = "buy" if int(side) == 1 else "sell"
        return await self._call(
            self.ex.create_order,
            self._symbol(pair),
            "limit",
            order_side,
            self._qty(pair, qty),
            float(self._px(pair, price)),
            params,
        )

    async def cancel(self, pair: str, order_id: str) -> None:
        await self._call(self.ex.cancel_order, order_id, self._symbol(pair))

    async def fetch_order(self, pair: str, order_id: str) -> dict | None:
        try:
            return await self._call(self.ex.fetch_order, order_id, self._symbol(pair))
        except Exception as e:
            logger.warning("ордер %s %s: %s", pair, order_id, e)
            return None

    async def find_open_by_link(self, pair: str, link: str) -> dict | None:
        try:
            rows = await self._call(self.ex.fetch_open_orders, self._symbol(pair))
        except Exception as e:
            logger.warning("открытые ордера %s: %s", pair, e)
            return None
        for row in rows or []:
            info = row.get("info") or {}
            if str(row.get("clientOrderId") or "") == link or str(info.get("orderLinkId") or "") == link:
                return row
        return None

    async def position_qty(self, pair: str, side: int) -> float:
        try:
            rows = await self._call(
                self.ex.fetch_positions,
                [self._symbol(pair)],
                {"category": "linear"},
            )
        except Exception as e:
            logger.warning("позиция %s: %s", pair, e)
            return -1.0
        want = "long" if int(side) == 1 else "short"
        for row in rows or []:
            contracts = float(row.get("contracts") or 0)
            if contracts <= 0:
                continue
            got = str(row.get("side") or "").lower()
            if got == want:
                return contracts
        return 0.0

    async def set_tpsl(self, pair: str, stop: float, tp: float) -> None:
        """Полный стоп маркетом и тейк лимиткой на всю позицию."""
        body = {
            "category": "linear",
            "symbol": self._market_id(pair),
            "positionIdx": 0,
            "tpslMode": "Full",
            "stopLoss": self._px(pair, stop),
            "slOrderType": "Market",
            "takeProfit": self._px(pair, tp),
            "tpOrderType": "Limit",
            "tpLimitPrice": self._px(pair, tp),
        }
        await self._call(self.ex.privatePostV5PositionTradingStop, body)

    async def market_close(self, pair: str, side: int, qty: float) -> None:
        close_side = "sell" if int(side) == 1 else "buy"
        await self._call(
            self.ex.create_order,
            self._symbol(pair),
            "market",
            close_side,
            self._qty(pair, qty),
            None,
            {"positionIdx": 0, "reduceOnly": True},
        )

    async def recent_close(self, pair: str, entry: float, opened_ms: int) -> dict | None:
        try:
            raw = await self._call(
                self.ex.privateGetV5PositionClosedPnl,
                {"category": "linear", "symbol": self._market_id(pair), "limit": 20},
            )
        except Exception as e:
            logger.warning("закрытия %s: %s", pair, e)
            return None
        result = raw.get("result") if isinstance(raw, dict) else None
        rows = (result or {}).get("list") if isinstance(result, dict) else None
        best = None
        for row in rows or []:
            try:
                ts = int(row.get("updatedTime") or row.get("createdTime") or 0)
                avg_entry = float(row.get("avgEntryPrice") or 0)
            except (TypeError, ValueError):
                continue
            if ts < opened_ms - 5000:
                continue
            if entry > 0 and abs(avg_entry - entry) / entry > 0.01:
                continue
            if best is None or ts > int(best.get("updatedTime") or 0):
                best = row
        return best
