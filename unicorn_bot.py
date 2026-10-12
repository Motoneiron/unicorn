# -*- coding: utf-8 -*-
"""Бот Единорога.

Запуск из папки проекта: python unicorn_bot.py

По умолчанию тень: свечи настоящие, ордеров нет. Живой режим на той же
учётке Bybit, где уже работает другой бот, трогать нельзя: счёт one-way,
одна позиция на пару. Для живых ордеров нужен свой ключ и PAPER_TRADING=0,
либо команда /mode live на время этого запуска.

Токен телеграма должен быть отдельный, если второй бот ещё запущен.
На сервере Telegram идёт через локальный прокси 127.0.0.1:8790.
"""
from __future__ import annotations

import asyncio
import logging
import time

from bot.config import (
    PAIR_KEYS,
    PAPER_TRADING,
    RUNTIME,
    TELEGRAM_CHAT_ID,
    TELEGRAM_PROXY,
    TELEGRAM_TOKEN,
)
from bot.engine import Engine
from bot.exchange import Bybit
from bot.store import Store

logger = logging.getLogger("unicorn")
_confirm_until = 0.0
_queue: asyncio.Queue[str] | None = None
_sender_started = False
_tg_bot = None


def _setup_log() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(RUNTIME / "bot.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("cisd_bot").setLevel(logging.WARNING)


def _own(update) -> bool:
    chat = update.effective_chat
    return chat is not None and chat.id == TELEGRAM_CHAT_ID


async def _sender() -> None:
    assert _queue is not None
    while True:
        text = await _queue.get()
        bot = _tg_bot
        if bot is None:
            await asyncio.sleep(1)
            await _queue.put(text)
            continue
        delay = 2
        while True:
            try:
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text[:4096])
                logger.info("телеграм: сообщение ушло")
                break
            except Exception as exc:
                logger.warning("телеграм не ушёл, повтор через %s с: %s", delay, exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
                bot = _tg_bot or bot


async def _telegram_loop(engine: Engine) -> None:
    """Телеграм через локальный прокси. Обрыв не останавливает сканер."""
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes

    global _sender_started

    async def post_init(app: Application) -> None:
        global _sender_started, _tg_bot
        _tg_bot = app.bot
        if not _sender_started:
            _sender_started = True
            asyncio.create_task(_sender())

    async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _own(update) or update.message is None:
            return
        await update.message.reply_text(engine.status_text())

    async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _own(update) or update.message is None:
            return
        engine.paused = True
        await update.message.reply_text("Сканер на паузе. Открытые лимитки и позиции досматриваю.")

    async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _own(update) or update.message is None:
            return
        engine.paused = False
        await update.message.reply_text("Сканер снова ищет сетапы.")

    async def cmd_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _own(update) or update.message is None:
            return
        await update.message.reply_text("Пары: " + ", ".join(PAIR_KEYS))

    async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        global _confirm_until
        if not _own(update) or update.message is None:
            return
        arg = (context.args[0].lower() if context.args else "")
        if arg == "confirm":
            if time.time() > _confirm_until:
                await update.message.reply_text("Сначала /mode live")
                return
            _confirm_until = 0.0
            await update.message.reply_text(engine.set_mode(False))
            return
        if arg == "live":
            _confirm_until = time.time() + 60
            await update.message.reply_text(
                "Живые ордера на Bybit. В течение минуты: /mode confirm"
            )
            return
        if arg in ("shadow", "paper", "тень"):
            _confirm_until = 0.0
            await update.message.reply_text(engine.set_mode(True))
            return
        now = "тень" if engine.paper else "живой"
        await update.message.reply_text(
            f"Сейчас {now}. После перезапуска снова {'тень' if PAPER_TRADING else 'живой'}.\n"
            "/mode live, /mode shadow"
        )

    async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _own(update) or update.message is None:
            return
        await update.message.reply_text(
            "/status - лимитки и позиции\n"
            "/stop - не искать новые\n"
            "/resume - снова искать\n"
            "/pairs - список пар\n"
            "/mode live - реальные ордера, с подтверждением\n"
            "/mode shadow - обратно в тень"
        )

    commands = (
        ("start", cmd_help),
        ("help", cmd_help),
        ("status", cmd_status),
        ("stop", cmd_stop),
        ("resume", cmd_resume),
        ("pairs", cmd_pairs),
        ("mode", cmd_mode),
    )

    while True:
        builder = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init)
        if TELEGRAM_PROXY:
            builder = builder.proxy(TELEGRAM_PROXY).get_updates_proxy(TELEGRAM_PROXY)
            logger.info("телеграм через прокси")
        app = builder.build()
        for name, fn in commands:
            app.add_handler(CommandHandler(name, fn))
        try:
            await app.initialize()
            await app.start()
            # post_init сам вызывается только из run_polling. Здесь стартуем отправку руками.
            await post_init(app)
            if app.updater is None:
                raise RuntimeError("нет updater")
            await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            logger.info("телеграм подключен")
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("телеграм, повтор через 15 с")
        finally:
            try:
                if app.updater is not None and app.updater.running:
                    await app.updater.stop()
                if app.running:
                    await app.stop()
                await app.shutdown()
            except Exception:
                logger.info("телеграм закрыт перед повтором")
        await asyncio.sleep(15)


async def _run() -> None:
    global _queue
    store = Store()
    engine = Engine(Bybit(), store, _notify)
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("телеграм выключен, бот пишет только в лог")
        await engine.run()
        return

    _queue = asyncio.Queue()
    engine_task = asyncio.create_task(engine.run())
    try:
        await _telegram_loop(engine)
    finally:
        engine_task.cancel()


async def _notify(text: str) -> None:
    logger.info("%s", text.replace("\n", " | "))
    if _queue is not None:
        await _queue.put(text)


def main() -> None:
    _setup_log()
    asyncio.run(_run())


if __name__ == "__main__":
    main()
