"""Notification/logging helpers."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class Notifier:
    """Dispatches messages to stdout and persists basic trade analytics."""

    def __init__(self, storage_dir: Path | None = Path("storage")) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path.cwd()
        self.trades_path = self.storage_dir / "trades.csv"
        self.returns_path = self.storage_dir / "returns.csv"
        self.logs_dir = self.storage_dir / "logs"
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for file_path in (self.trades_path, self.returns_path):
            if not file_path.exists():
                file_path.touch()

    def info(self, message: str) -> None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO {timestamp} UTC] {message}")
        log_file = self.logs_dir / f"log_{timestamp.split(' ')[0]}.txt"
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"[INFO {timestamp} UTC] {message}\n")

    def persist_trade(self, trade: Dict[str, Any] | None) -> None:
        if not trade:
            return
        self._append_dict_as_csv_row(self.trades_path, trade)

    def persist_return(self, returns: Dict[str, Any] | None) -> None:
        if not returns:
            return
        self._append_dict_as_csv_row(self.returns_path, returns)

    def _append_dict_as_csv_row(self, file_path: Path, payload: Dict[str, Any]) -> None:
        with file_path.open("a", newline="", encoding="utf-8") as csvfile:
            timestamp = datetime.utcnow().isoformat()
            if csvfile.tell() == 0:
                csvfile.write("timestamp,payload\n")
            payload_json = json.dumps(payload, default=str)
            csvfile.write(f"{timestamp},{payload_json}\n")
