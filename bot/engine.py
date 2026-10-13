# -*- coding: utf-8 -*-
"""Сканер Единорога и сопровождение лимитки до стопа или тейка 2R."""
from __future__ import annotations

import logging
import uuid

import pandas as pd

from bot.config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    ENTRY_MIN,
    LEVERAGE,
    LTF,
    MAKER,
    TAKER,
    LTF_MIN,
    M1_BARS,
    OHLCV_BARS,
    PAIR_KEYS,
    PAIRS,
    PAPER_TRADING,
    RISK_PCT,
    RR,
    SCAN_S,
)
from bot.exchange import Bybit
from bot.messages import (
    cancel_card,
    close_card,
    fill_card,
    limit_card,
    missing_position,
    protection_alert,
)
from bot.rules import (
    drop_forming,
    entry_deadline,
    exit_price,
    first_entry_event,
    fmt_num,
    link_id,
    mss_close,
    pnl_usdt,
    position_hit,
    position_qty,
    round_levels,
    setup_id,
    side_name,
    utc_naive,
    window_bars,
)
from bot.store import Store
from bt.cisd_setup import find_setups
from bt.execute import r_multiples

logger = logging.getLogger("unicorn")


def _iso(ts) -> str:
    return pd.Timestamp(ts).isoformat()


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


class Engine:
    def __init__(self, ex: Bybit, store: Store, notify):
        self.ex = ex
        self.store = store
        self.notify = notify
        self.paper = PAPER_TRADING
        self.paused = False
        self._last_ltf: dict[str, str] = {}
        self._levered: set[str] = set()

    def mode_label(self) -> str:
        return "тень" if self.paper else "живой"

    async def run(self) -> None:
        import asyncio
        await self.ex.prepare()
        self.store.ensure_nums()
        await self.notify(self._hello())
        if self.store.pending or self.store.positions:
            await self.notify(self.status_text())
        while True:
            try:
                await self.tick()
            except Exception:
                logger.exception("цикл")
            await asyncio.sleep(SCAN_S)

    def _hello(self) -> str:
        mode = "Тень, ордеров на биржу нет" if self.paper else "ЖИВОЙ режим, ордера реальные"
        return (
            f"Единорог запущен. {mode}.\n"
            f"9 пар, 15м на 4ч, лимитка {ENTRY_MIN // 60} ч, тейк {fmt_num(RR, 0)}R, "
            f"выход только по стопу или тейку.\n"
            f"Риск {fmt_num(RISK_PCT, 1)}% на сделку, плечо {LEVERAGE}x."
        )

    async def tick(self) -> None:
        await self._watch_pending()
        await self._watch_positions()
        if not self.paused:
            await self.scan()

    async def scan(self) -> None:
        for pair in PAIR_KEYS:
            if self.store.busy(pair):
                continue
            try:
                await self._scan_pair(pair)
            except Exception:
                logger.exception("скан %s", pair)

    async def _scan_pair(self, pair: str) -> None:
        import asyncio
        probe = drop_forming(await self.ex.ohlcv(pair, LTF, 3), LTF_MIN)
        if probe is None or len(probe) == 0:
            return
        last = _iso(probe.index[-1])
        if self._last_ltf.get(pair) == last:
            return
        df = drop_forming(await self.ex.ohlcv(pair, LTF, OHLCV_BARS), LTF_MIN)
        if df is None or len(df) < 50:
            return
        self._last_ltf[pair] = _iso(df.index[-1])
        setups, stats = await asyncio.to_thread(find_setups, df)
        logger.info("%s закрытый 15м %s, сетапов в окне %s", pair, self._last_ltf[pair], stats.get("kept"))
        now = utc_naive()
        found = []
        for setup in setups:
            mss_open = pd.Timestamp(df.index[int(setup["known_i"])])
            if mss_close(mss_open) > now or entry_deadline(mss_open) <= now:
                continue
            sid = setup_id(pair, int(setup["side"]), mss_open)
            if self.store.seen_setup(sid):
                continue
            found.append((mss_open, setup, sid))
        found.sort(key=lambda item: item[0])
        for mss_open, setup, sid in found:
            if self.store.busy(pair):
                break
            await self._consider(pair, setup, mss_open, sid)

    async def _consider(self, pair: str, setup: dict, mss_open: pd.Timestamp, sid: str) -> None:
        side = int(setup["side"])
        profile = PAIRS[pair]
        levels = round_levels(side, float(setup["entry"]), float(setup["stop"]), profile["price_tick"])
        if levels is None:
            self.store.mark_seen(sid, utc_naive().isoformat())
            return
        entry, stop, tp = levels
        m1 = drop_forming(await self.ex.ohlcv(pair, "1m", M1_BARS), 1)
        bars = window_bars(m1, mss_close(mss_open), entry_deadline(mss_open))
        kind, _idx = first_entry_event(side, entry, stop, bars["low"], bars["high"])
        if kind == "ran":
            self.store.mark_seen(sid, utc_naive().isoformat())
            logger.info("%s 2R уже был до входа, пропуск", pair)
            return
        if kind == "fill":
            self.store.mark_seen(sid, utc_naive().isoformat())
            logger.info("%s цена уже забирала лимитку, пока бот не смотрел. Не догоняю", pair)
            return
        bal = await self._balance()
        qty = position_qty(bal, entry, stop, profile["qty_step"])
        if qty <= 0:
            logger.info("%s размер меньше минимума биржи, баланс %s", pair, round(bal, 2))
            return
        htf = setup.get("h4_sweep_at")
        htf_s = ""
        if htf is not None and str(htf) not in ("NaT", "None", "nan"):
            htf_s = _iso(htf)
        last_1m = _iso(bars.index[-1]) if len(bars) else ""
        rec = {
            "id": uuid.uuid4().hex[:12],
            "trade_num": self.store.allocate_num(),
            "symbol": pair,
            "side": side,
            "entry": entry,
            "stop": stop,
            "tp": tp,
            "qty": qty,
            "mss_open": _iso(mss_open),
            "signal_time": _iso(mss_open),
            "htf_time": htf_s,
            "deadline": _iso(entry_deadline(mss_open)),
            "link": link_id(pair, side, mss_open),
            "order_id": "",
            "paper": self.paper,
            "last_1m": last_1m,
            "created": utc_naive().isoformat(),
        }
        self.store.mark_seen(sid, rec["created"])
        self.store.add_pending(rec)
        if not rec["paper"]:
            placed = await self._place_live(rec)
            if not placed:
                self.store.drop_pending(rec["id"])
                self.store.unsee(sid)
                self._last_ltf.pop(pair, None)
                return
        await self.notify(limit_card(rec, await self._balance()))

    async def _place_live(self, rec: dict) -> bool:
        pair = rec["symbol"]
        if pair not in self._levered:
            await self.ex.set_leverage(pair)
            self._levered.add(pair)
        try:
            existing = await self.ex.find_open_by_link(pair, rec["link"])
            order = existing or await self.ex.place_limit(
                pair, rec["side"], rec["qty"], rec["entry"], rec["link"],
            )
        except Exception:
            logger.exception("не поставил лимитку %s", pair)
            return False
        rec["order_id"] = str(order.get("id") or "")
        self.store.save()
        if not rec["order_id"]:
            logger.error("биржа не вернула id ордера %s", pair)
            return False
        return True

    async def _watch_pending(self) -> None:
        for rec in list(self.store.pending):
            try:
                await self._watch_limit(rec)
            except Exception:
                logger.exception("лимитка %s", rec.get("symbol"))

    async def _watch_limit(self, rec: dict) -> None:
        if not rec["paper"]:
            fill = await self._live_fill(rec)
            if fill is not None:
                await self._open_live(rec, fill[0], fill[1])
                return
        m1 = drop_forming(await self.ex.ohlcv(rec["symbol"], "1m", M1_BARS), 1)
        start = _ts(rec["last_1m"]) + pd.Timedelta(minutes=1) if rec.get("last_1m") else mss_close(_ts(rec["mss_open"]))
        bars = window_bars(m1, start, _ts(rec["deadline"]))
        if bars is not None and len(bars):
            await self._walk_limit_bars(rec, bars)
        if any(p["id"] == rec["id"] for p in self.store.pending) and utc_naive() >= _ts(rec["deadline"]):
            await self._expire(rec)

    async def _walk_limit_bars(self, rec: dict, bars: pd.DataFrame) -> None:
        for ts, row in bars.iterrows():
            if not rec["paper"]:
                fill = await self._live_fill(rec)
                if fill is not None:
                    await self._open_live(rec, fill[0], fill[1])
                    return
            kind, _idx = first_entry_event(
                rec["side"], rec["entry"], rec["stop"], [row["low"]], [row["high"]],
            )
            rec["last_1m"] = _iso(ts)
            if kind == "ran":
                await self._cancel_ran(rec)
                return
            if kind == "fill" and rec["paper"]:
                hit = position_hit(rec["side"], rec["stop"], rec["tp"], row["low"], row["high"])
                if hit:
                    await self._paper_roundtrip(rec, hit, _iso(ts))
                else:
                    await self._open_paper(rec, _iso(ts))
                return
            self.store.save()

    async def _live_fill(self, rec: dict) -> tuple[float, float] | None:
        if not rec.get("order_id"):
            found = await self.ex.find_open_by_link(rec["symbol"], rec["link"])
            if found:
                rec["order_id"] = str(found.get("id") or "")
                self.store.save()
            if not rec.get("order_id"):
                return None
        order = await self.ex.fetch_order(rec["symbol"], rec["order_id"])
        if not order:
            return None
        status = str(order.get("status") or "").lower()
        filled = float(order.get("filled") or 0)
        if filled <= 0:
            return None
        if status == "open":
            try:
                await self.ex.cancel(rec["symbol"], rec["order_id"])
            except Exception:
                logger.info("хвост лимитки %s уже не снять", rec["symbol"])
        elif status not in ("closed", "filled", "canceled", "cancelled"):
            return None
        avg = float(order.get("average") or 0) or float(rec["entry"])
        return filled, avg

    async def _expire(self, rec: dict) -> None:
        if not rec["paper"] and rec.get("order_id"):
            fill = await self._live_fill(rec)
            if fill is not None:
                await self._open_live(rec, fill[0], fill[1])
                return
            try:
                await self.ex.cancel(rec["symbol"], rec["order_id"])
            except Exception:
                fill = await self._live_fill(rec)
                if fill is not None:
                    await self._open_live(rec, fill[0], fill[1])
                    return
                logger.exception("не снял просроченную лимитку %s", rec["symbol"])
                return
        self.store.drop_pending(rec["id"])
        await self.notify(cancel_card(rec, "2 часа вышли"))

    async def _cancel_ran(self, rec: dict) -> None:
        if not rec["paper"] and rec.get("order_id"):
            fill = await self._live_fill(rec)
            if fill is not None:
                await self._open_live(rec, fill[0], fill[1])
                return
            try:
                await self.ex.cancel(rec["symbol"], rec["order_id"])
            except Exception:
                fill = await self._live_fill(rec)
                if fill is not None:
                    await self._open_live(rec, fill[0], fill[1])
                    return
                logger.exception("не снял лимитку после 2R %s", rec["symbol"])
                return
        self.store.drop_pending(rec["id"])
        await self.notify(cancel_card(rec, "2R без входа"))

    async def _open_paper(self, rec: dict, bar_iso: str) -> None:
        pos = self._position_from(rec, rec["qty"], rec["entry"], paper=True)
        pos["opened"] = bar_iso
        pos["last_1m"] = bar_iso
        pos["tpsl_ok"] = True
        self.store.drop_pending(rec["id"])
        self.store.add_position(pos)
        await self.notify(fill_card(pos, await self._balance()))

    async def _paper_roundtrip(self, rec: dict, reason: str, bar_iso: str) -> None:
        pos = self._position_from(rec, rec["qty"], rec["entry"], paper=True)
        pos["opened"] = bar_iso
        self.store.drop_pending(rec["id"])
        await self._close_paper(pos, reason, keep=False)

    async def _open_live(self, rec: dict, qty: float, price: float) -> None:
        pos = self._position_from(rec, qty, price, paper=False)
        pos["tpsl_ok"] = False
        pos["tpsl_since"] = utc_naive().isoformat()
        self.store.drop_pending(rec["id"])
        self.store.add_position(pos)
        await self._arm_tpsl(pos)
        await self.notify(fill_card(pos, await self._balance()))
        if not pos.get("tpsl_ok"):
            await self.notify(protection_alert(pos))

    def _position_from(self, rec: dict, qty: float, price: float, paper: bool) -> dict:
        return {
            "id": uuid.uuid4().hex[:12],
            "symbol": rec["symbol"],
            "side": rec["side"],
            "entry": float(price),
            "stop": rec["stop"],
            "tp": rec["tp"],
            "qty": float(qty),
            "opened": utc_naive().isoformat(),
            "paper": paper,
            "last_1m": rec.get("last_1m") or "",
            "tpsl_ok": paper,
            "order_id": rec.get("order_id") or "",
            "trade_num": rec.get("trade_num"),
            "signal_time": rec.get("signal_time") or rec.get("mss_open") or "",
            "htf_time": rec.get("htf_time") or "",
            "mss_open": rec.get("mss_open") or "",
        }

    async def _watch_positions(self) -> None:
        for pos in list(self.store.positions):
            try:
                if pos["paper"]:
                    await self._watch_paper_pos(pos)
                else:
                    await self._watch_live_pos(pos)
            except Exception:
                logger.exception("позиция %s", pos.get("symbol"))

    async def _watch_paper_pos(self, pos: dict) -> None:
        m1 = drop_forming(await self.ex.ohlcv(pos["symbol"], "1m", M1_BARS), 1)
        start = _ts(pos["last_1m"]) + pd.Timedelta(minutes=1) if pos.get("last_1m") else _ts(pos["opened"])
        bars = window_bars(m1, start, utc_naive() + pd.Timedelta(minutes=1))
        if bars is None or len(bars) == 0:
            return
        for ts, row in bars.iterrows():
            hit = position_hit(pos["side"], pos["stop"], pos["tp"], row["low"], row["high"])
            pos["last_1m"] = _iso(ts)
            if hit:
                await self._close_paper(pos, hit, keep=True)
                return
        self.store.save()

    async def _close_paper(self, pos: dict, reason: str, keep: bool) -> None:
        exit_px, fee_out = exit_price(pos["side"], pos["stop"], pos["tp"], reason)
        net, fees = pnl_usdt(pos["side"], pos["entry"], exit_px, pos["qty"], fee_out)
        gross = net + fees
        self.store.balance = self.store.balance + net
        if keep:
            self.store.drop_position(pos["id"])
        else:
            self.store.save()
        self._journal(pos, reason, exit_px, net, fees)
        await self.notify(
            close_card(pos, reason, exit_px, gross, fees, net, await self._balance())
        )

    async def _watch_live_pos(self, pos: dict) -> None:
        if not pos.get("tpsl_ok"):
            await self._arm_tpsl(pos)
            await self._emergency(pos)
        qty = await self.ex.position_qty(pos["symbol"], pos["side"])
        if qty < 0:
            return
        if qty > 0:
            if not pos.get("seen_on_exchange"):
                pos["seen_on_exchange"] = True
                self.store.save()
            return
        if not pos.get("seen_on_exchange"):
            age_s = (utc_naive() - _ts(pos["opened"])).total_seconds()
            if age_s > 180 and not pos.get("missing_alert"):
                pos["missing_alert"] = True
                self.store.save()
                await self.notify(missing_position(pos))
            return
        await self._close_live(pos)

    async def _arm_tpsl(self, pos: dict) -> None:
        try:
            await self.ex.set_tpsl(pos["symbol"], pos["stop"], pos["tp"])
            pos["tpsl_ok"] = True
            self.store.save()
            logger.info("стоп и тейк стоят %s", pos["symbol"])
        except Exception:
            logger.exception("не поставил стоп/тейк %s", pos["symbol"])

    async def _emergency(self, pos: dict) -> None:
        if pos.get("tpsl_ok"):
            return
        since = _ts(pos.get("tpsl_since") or pos["opened"])
        if (utc_naive() - since).total_seconds() < 15:
            return
        px = await self.ex.last_price(pos["symbol"])
        if px is None:
            return
        through = (pos["side"] == 1 and px <= pos["stop"]) or (pos["side"] == -1 and px >= pos["stop"])
        if not through:
            return
        try:
            live_qty = await self.ex.position_qty(pos["symbol"], pos["side"])
            if live_qty <= 0:
                return
            await self.ex.market_close(pos["symbol"], pos["side"], live_qty)
            logger.error("аварийно закрыл %s: стоп не встал, цена за ним", pos["symbol"])
        except Exception:
            logger.exception("аварийное закрытие %s", pos["symbol"])

    async def _close_live(self, pos: dict) -> None:
        opened_ms = int(_ts(pos["opened"]).timestamp() * 1000)
        row = await self.ex.recent_close(pos["symbol"], pos["entry"], opened_ms)
        reason = "close"
        exit_px = float(pos["entry"])
        net = 0.0
        fees = 0.0
        if row:
            try:
                exit_px = float(row.get("avgExitPrice") or exit_px)
            except (TypeError, ValueError):
                exit_px = float(pos["entry"])
            try:
                net = float(row.get("closedPnl") or 0)
            except (TypeError, ValueError):
                net = 0.0
            reason = self._reason_from_exit(pos, exit_px)
        else:
            px = await self.ex.last_price(pos["symbol"])
            if px:
                exit_px = px
                reason = self._reason_from_exit(pos, exit_px)
        fee_out = MAKER if reason == "tp" else TAKER
        model_net, fees = pnl_usdt(pos["side"], pos["entry"], exit_px, pos["qty"], fee_out)
        if row is None:
            net = model_net
        gross = net + fees
        self.store.drop_position(pos["id"])
        self._journal(pos, reason, exit_px, net, fees)
        await self.notify(
            close_card(pos, reason, exit_px, gross, fees, net, await self._balance())
        )

    def _reason_from_exit(self, pos: dict, exit_px: float) -> str:
        if pos["side"] == 1:
            if exit_px <= pos["stop"] * 1.002:
                return "sl"
            if exit_px >= pos["tp"] * 0.998:
                return "tp"
        else:
            if exit_px >= pos["stop"] * 0.998:
                return "sl"
            if exit_px <= pos["tp"] * 1.002:
                return "tp"
        return "close"

    def _journal(self, pos: dict, reason: str, exit_px: float, net: float, fees: float) -> None:
        fee_out = MAKER if reason == "tp" else TAKER
        r, _r06, gross = r_multiples(pos["side"], pos["entry"], pos["stop"], exit_px, fee_out)
        self.store.append_trade({
            "ts": utc_naive().isoformat(),
            "symbol": pos["symbol"],
            "side": side_name(pos["side"]),
            "entry": pos["entry"],
            "exit": exit_px,
            "stop": pos["stop"],
            "tp": pos["tp"],
            "qty": pos["qty"],
            "reason": reason,
            "pnl": net,
            "fees": fees,
            "r": r,
            "gross_r": gross,
            "paper": pos["paper"],
        })

    async def _balance(self) -> float:
        if self.paper:
            return float(self.store.balance)
        return await self.ex.balance()

    def status_text(self) -> str:
        from bot.messages import balance_line, direction, instrument, mode_prefix, price
        head = mode_prefix(self.paper)
        if self.paused:
            head += "  сканер на паузе"
        lines = [head, balance_line(self.store.balance)]
        if not self.store.pending and not self.store.positions:
            lines.append("Лимиток и позиций нет.")
        for rec in self.store.pending:
            lines.append(
                f"#{rec.get('trade_num')} {instrument(rec['symbol'])} {direction(rec['side'])} — заявка\n"
                f"Вход: {price(rec['symbol'], rec['entry'])} | "
                f"Стоп: {price(rec['symbol'], rec['stop'])} | "
                f"Тейк: {price(rec['symbol'], rec['tp'])}\n"
                f"до {_ts(rec['deadline']).strftime('%H:%M')} UTC"
            )
        for pos in self.store.positions:
            lines.append(
                f"#{pos.get('trade_num')} {instrument(pos['symbol'])} {direction(pos['side'])} — в позиции\n"
                f"Вход: {price(pos['symbol'], pos['entry'])} | "
                f"Стоп: {price(pos['symbol'], pos['stop'])} | "
                f"Тейк: {price(pos['symbol'], pos['tp'])}"
            )
        return "\n".join(lines)

    def set_mode(self, paper: bool) -> str:
        if not paper:
            if not BYBIT_API_KEY or not BYBIT_API_SECRET:
                return "Ключей Bybit нет, живой режим не включаю."
            self.paper = False
            return "Живой режим до перезапуска. Новые сетапы пойдут ордерами на Bybit."
        self.paper = True
        return "Снова тень. Уже стоящие живые ордера не трогаю, новые сетапы без ордеров."
