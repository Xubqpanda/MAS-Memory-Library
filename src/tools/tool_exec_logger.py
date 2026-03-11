# src/tools/tool_exec_logger.py
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ToolExecLogger:
    """Record tool execution traces independent from llm_io logs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = False
        self._log_dir: Path | None = None
        self._jsonl_fh = None
        self._readable_fh = None
        self._index = 0

    def setup(self, log_dir: str) -> None:
        with self._lock:
            self.close()
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_fh = open(self._log_dir / "tool_exec.jsonl", "a", encoding="utf-8")
            self._readable_fh = open(self._log_dir / "tool_exec.log", "a", encoding="utf-8")
            self._index = 0
            self._enabled = True

    def close(self) -> None:
        with self._lock:
            if self._jsonl_fh is not None:
                try:
                    self._jsonl_fh.close()
                except Exception:
                    pass
            if self._readable_fh is not None:
                try:
                    self._readable_fh.close()
                except Exception:
                    pass
            self._jsonl_fh = None
            self._readable_fh = None
            self._enabled = False

    def log(self, name: str, args: dict[str, Any], output: str, ok: bool, duration_ms: int) -> None:
        if not self._enabled:
            return
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with self._lock:
            idx = self._index
            self._index += 1
            row = {
                "index": idx,
                "timestamp": ts,
                "tool": name,
                "args": args,
                "ok": ok,
                "duration_ms": duration_ms,
                "output": output,
            }
            self._jsonl_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._jsonl_fh.flush()
            self._readable_fh.write(
                f"[#{idx:04d}] {ts} tool={name} ok={ok} dur={duration_ms}ms\n"
                f"args={args}\n"
                f"output={output[:2000]}\n"
                f"{'-' * 72}\n"
            )
            self._readable_fh.flush()


tool_exec_logger = ToolExecLogger()
