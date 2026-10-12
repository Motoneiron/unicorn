# -*- coding: utf-8 -*-
"""Состояние бота: лимитки, позиции, уже виденные сломы. Не коммитить."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from bot.config import PAPER_BALANCE, RUNTIME


class Store:
    def __init__(self, path: Path | None = None):
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.path = path or (RUNTIME / "state.json")
        self.balance = PAPER_BALANCE
        self.seen: dict[str, str] = {}
        self.pending: list[dict] = []
        self.positions: list[dict] = []
        self.trade_num = 0
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.balance = float(data.get("balance", PAPER_BALANCE))
        self.seen = {str(k): str(v) for k, v in (data.get("seen") or {}).items()}
        self.pending = list(data.get("pending") or [])
        self.positions = list(data.get("positions") or [])
        self.trade_num = int(data.get("trade_num") or 0)

    def save(self) -> None:
        self._prune_seen()
        payload = {
            "balance": self.balance,
            "seen": self.seen,
            "pending": self.pending,
            "positions": self.positions,
            "trade_num": self.trade_num,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _prune_seen(self) -> None:
        cutoff = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=12))
        keep = {}
        for key, stamp in self.seen.items():
            try:
                if pd.Timestamp(stamp) >= cutoff:
                    keep[key] = stamp
            except (TypeError, ValueError):
                continue
        self.seen = keep

    def allocate_num(self) -> int:
        self.trade_num += 1
        return self.trade_num

    def ensure_nums(self) -> None:
        changed = False
        for rec in list(self.pending) + list(self.positions):
            if not rec.get("trade_num"):
                rec["trade_num"] = self.allocate_num()
                changed = True
        if changed:
            self.save()

    def busy(self, pair: str) -> bool:
        return any(p["symbol"] == pair for p in self.pending) or any(
            p["symbol"] == pair for p in self.positions
        )

    def seen_setup(self, sid: str) -> bool:
        return sid in self.seen

    def mark_seen(self, sid: str, when: str) -> None:
        self.seen[sid] = when
        self.save()

    def unsee(self, sid: str) -> None:
        self.seen.pop(sid, None)
        self.save()

    def add_pending(self, rec: dict) -> None:
        self.pending.append(rec)
        self.save()

    def drop_pending(self, pid: str) -> None:
        self.pending = [p for p in self.pending if p["id"] != pid]
        self.save()

    def add_position(self, rec: dict) -> None:
        self.positions.append(rec)
        self.save()

    def drop_position(self, pid: str) -> None:
        self.positions = [p for p in self.positions if p["id"] != pid]
        self.save()

    def append_trade(self, row: dict) -> None:
        path = RUNTIME / "trades.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
