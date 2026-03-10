# src/llm/llm_io_logger.py
"""
LLMIOLogger：记录所有 LLM 调用的完整输入输出。

与 token_tracker 完全对称的单例设计，由 ModelCaller 在每次成功调用后自动写入，
调用方（solver / memory / env）无需任何改动。

输出文件（均在 log_dir 下）：
  ┌─ 机器可读 ──────────────────────────────────────────────────────────┐
  │  solver.jsonl  memory.jsonl  env.jsonl                              │
  │  每行一条完整 JSON 记录，含 messages / output / usage 等全量字段。   │
  └─────────────────────────────────────────────────────────────────────┘
  ┌─ 人类可读（实时）────────────────────────────────────────────────────┐
  │  llm_io.log                                                         │
  │  所有 role 的调用按时间顺序混排，格式化输出，可 tail -f 实时跟踪。   │
  └─────────────────────────────────────────────────────────────────────┘

每条 JSONL 记录的字段：
    call_index  (int)  : 全局调用序号，从 0 开始，单调递增
    timestamp   (str)  : ISO 8601 格式，精确到毫秒
    role        (str)  : "solver" / "memory" / "env"
    model       (str)  : 实际调用的模型名
    messages    (list) : 输入消息列表，[{"role": ..., "content": ...}, ...]
    output      (str)  : 模型输出文本
    usage       (dict) : {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    finish_reason (str): 模型停止原因

使用方式：
  # 1. runner.py 实验开始时初始化（指定输出目录）
  from src.llm.llm_io_logger import llm_io_logger
  llm_io_logger.setup(log_dir="experiments/logs/hle/single_agent_emptymemory/20240101_120000")

  # 2. ModelCaller 每次调用后自动写入，无需手动调用
  llm_io_logger.log(role="solver", model="gpt-4o", messages=[...], output="...", usage={...})

  # 3. 实验结束后可选择关闭文件句柄
  llm_io_logger.close()

  # 4. 实时跟踪（另开终端）
  tail -f experiments/logs/hle/single_agent_emptymemory/<ts>/llm_io/llm_io.log

线程安全：使用 threading.Lock，并行评测下不会产生写入竞争。
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# role → ANSI 颜色（仅写入 llm_io.log，JSONL 不含颜色码）
_ROLE_COLOR = {
    "solver": "\033[36m",   # cyan
    "memory": "\033[35m",   # magenta
    "env":    "\033[33m",   # yellow
}
_RESET = "\033[0m"
_DIVIDER = "─" * 72


class LLMIOLogger:
    """
    LLM 输入输出全量日志记录器。

    核心设计：
      - 单例，全库共享同一个实例（见模块末尾的 llm_io_logger）。
      - 延迟初始化：不调用 setup() 时静默跳过所有写入，不影响正常运行。
      - 每个 role 独立 JSONL 文件（机器可读）+ 统一 llm_io.log（人类可读）。
      - 两个文件同步写入，flush() 保证实时落盘，tail -f 可即时看到输出。
    """

    def __init__(self):
        self._lock:        threading.Lock    = threading.Lock()
        self._log_dir:     Optional[Path]    = None
        self._jsonl_files: Dict[str, object] = {}   # role → JSONL file handle
        self._readable_fh: Optional[object]  = None  # llm_io.log file handle
        self._call_index:  int               = 0
        self._enabled:     bool              = False

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def setup(self, log_dir: str) -> None:
        """
        指定日志输出目录，开启记录。每个实验开始时调用一次。

        会在 log_dir 下创建：
          solver.jsonl / memory.jsonl / env.jsonl  （机器可读，按 role 分文件）
          llm_io.log                               （人类可读，所有 role 混排）

        若目录不存在会自动创建。
        若上一个实验的文件句柄还开着，会先调用 _close_files() 关闭。

        Args:
            log_dir: 日志目录路径（绝对路径或相对路径均可）。
        """
        with self._lock:
            self._close_files()
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._call_index = 0
            self._enabled    = True
            # 预先创建三个标准 role 的 JSONL 句柄
            for role in ("solver", "memory", "env"):
                self._get_or_create_jsonl(role)
            # 统一的人类可读 log
            self._readable_fh = open(self._log_dir / "llm_io.log", "a", encoding="utf-8")

    def close(self) -> None:
        """关闭所有文件句柄。实验结束时调用（可选）。"""
        with self._lock:
            self._close_files()
            self._enabled = False

    def reset(self) -> None:
        """
        关闭现有句柄并清空状态（不删除已写入的文件）。
        setup() 内部会自动调用，外部一般不需要直接调用。
        """
        with self._lock:
            self._close_files()
            self._log_dir    = None
            self._call_index = 0
            self._enabled    = False

    # ── 写入 ──────────────────────────────────────────────────────────────────

    def log(
        self,
        role:          str,
        model:         str,
        messages:      List[Dict],
        output:        str,
        usage:         Dict,
        finish_reason: str = "",
    ) -> None:
        """
        记录一次 LLM 调用。由 ModelCaller 在每次成功调用后自动调用。

        若 setup() 尚未调用，此方法静默返回，不做任何事。

        同时写入两份：
          - {role}.jsonl      机器可读，全量字段
          - llm_io.log        人类可读，格式化排版，可 tail -f 实时跟踪

        Args:
            role          : 调用方身份，"solver" / "memory" / "env"（或自定义）。
            model         : 实际调用的模型名（来自 litellm response.model）。
            messages      : 输入消息列表，格式同 litellm：[{"role": ..., "content": ...}]。
            output        : 模型输出文本。
            usage         : token 统计 dict，含 prompt_tokens / completion_tokens / total_tokens。
            finish_reason : 模型停止原因（来自 response.choices[0].finish_reason）。
        """
        if not self._enabled:
            return

        idx = self._next_index()
        ts  = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        # ── 1. JSONL（机器可读）──────────────────────────────────────────────
        record = {
            "call_index":    idx,
            "timestamp":     ts,
            "role":          role,
            "model":         model,
            "messages":      messages,
            "output":        output,
            "usage":         usage,
            "finish_reason": finish_reason,
        }
        jsonl_line = json.dumps(record, ensure_ascii=False)

        # ── 2. llm_io.log（人类可读）─────────────────────────────────────────
        readable = self._format_readable(idx, ts, role, model, messages, output, usage)

        with self._lock:
            # JSONL
            fh = self._get_or_create_jsonl(role)
            fh.write(jsonl_line + "\n")
            fh.flush()
            # 人类可读 log
            if self._readable_fh is not None:
                self._readable_fh.write(readable)
                self._readable_fh.flush()   # 实时落盘，tail -f 可即时看到

    # ── 格式化 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_readable(
        idx:      int,
        ts:       str,
        role:     str,
        model:    str,
        messages: List[Dict],
        output:   str,
        usage:    Dict,
    ) -> str:
        """
        生成人类可读的文本块。示例输出：

        ────────────────────────────────────────────────────────────────────────
        [#0042]  2024-01-01T12:00:05.123+00:00  SOLVER  gpt-4o
        ── INPUT ──
        [system]
        Your response should be in the following format: ...

        [user]
        Question: What is the capital of France?

        ── OUTPUT ──
        Explanation: France is a country in Europe.
        Answer: Paris

        ── USAGE ──  prompt=512  completion=64  total=576
        ────────────────────────────────────────────────────────────────────────
        """
        color = _ROLE_COLOR.get(role, "")
        header = (
            f"\n{_DIVIDER}\n"
            f"{color}[#{idx:04d}]  {ts}  {role.upper()}  {model}{_RESET}\n"
        )

        input_block = "── INPUT ──\n"
        for msg in messages:
            r = msg.get("role", "?")
            c = msg.get("content", "")
            input_block += f"[{r}]\n{c}\n\n"

        output_block = f"── OUTPUT ──\n{output}\n"

        p = usage.get("prompt_tokens", 0)
        c = usage.get("completion_tokens", 0)
        t = usage.get("total_tokens", p + c)
        usage_block = f"\n── USAGE ──  prompt={p}  completion={c}  total={t}\n"

        return header + input_block + output_block + usage_block

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _next_index(self) -> int:
        with self._lock:
            idx = self._call_index
            self._call_index += 1
        return idx

    def _get_or_create_jsonl(self, role: str):
        """获取 role 对应的 JSONL 文件句柄，不存在时创建（追加模式）。调用前需持有 _lock。"""
        if role not in self._jsonl_files:
            path = self._log_dir / f"{role}.jsonl"
            self._jsonl_files[role] = open(path, "a", encoding="utf-8")
        return self._jsonl_files[role]

    def _close_files(self) -> None:
        """关闭所有文件句柄。调用前需持有 _lock。"""
        for fh in self._jsonl_files.values():
            try:
                fh.close()
            except Exception:
                pass
        self._jsonl_files.clear()
        if self._readable_fh is not None:
            try:
                self._readable_fh.close()
            except Exception:
                pass
            self._readable_fh = None


# ── 全局单例 ──────────────────────────────────────────────────────────────────
llm_io_logger = LLMIOLogger()