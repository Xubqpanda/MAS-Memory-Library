# src/llm/token_tracker.py
"""
TokenTracker：按 role 分桶统计全局 token 消耗。

三个内置 role：
  solver  — MAS 框架调用 LLM 解题（working space）
  memory  — Memory 方法内部的 LLM 调用（摘要、insight 提取、skill 生成等）
  env     — 环境侧的 judge 调用
  tool    — Tool 内部的 LLM 调用（如网页摘要、图像理解等）

设计原则：
  - 单例（模块级 `token_tracker`），全库共享同一个实例。
  - role 开放扩展：传入未预设的 role 字符串时自动创建新桶，不报错。
    例如后续加 embedding 统计时直接用 role="embedding" 即可。
  - 线程安全：使用 threading.Lock 保护计数器，并行评测时不会漏计。
  - 每个实验结束后调用 reset() 清零，避免跨实验污染。
"""

import threading
from typing import Dict


# 预设 role，文档用，实际不做强制校验
ROLES = ("solver", "memory", "env", "tool")


class TokenTracker:

    def __init__(self):
        self._lock:   threading.Lock = threading.Lock()
        self._counts: Dict[str, Dict[str, int]] = {}
        self._init_roles()

    def _init_roles(self):
        for role in ROLES:
            self._counts[role] = {"prompt": 0, "completion": 0}

    def reset(self):
        """清零所有计数器，每个实验开始前调用。"""
        with self._lock:
            self._counts.clear()
            self._init_roles()

    def add(self, role: str, prompt_tokens: int, completion_tokens: int):
        """
        记录一次 LLM 调用的 token 消耗。

        Args:
            role              : 调用方身份，如 "solver" / "memory" / "env"。
            prompt_tokens     : 本次调用的输入 token 数。
            completion_tokens : 本次调用的输出 token 数。
        """
        with self._lock:
            if role not in self._counts:
                self._counts[role] = {"prompt": 0, "completion": 0}
            self._counts[role]["prompt"]     += prompt_tokens
            self._counts[role]["completion"] += completion_tokens

    def summary(self) -> Dict:
        """
        返回各 role 及总计的 token 统计。

        Returns:
            {
                "solver":  {"prompt": 100, "completion": 50, "total": 150},
                "memory":  {"prompt": 200, "completion": 80, "total": 280},
                "env":     {"prompt":  60, "completion": 20, "total":  80},
                "total":   {"prompt": 360, "completion": 150, "total": 510},
            }
        """
        with self._lock:
            result: Dict = {}
            total_p = total_c = 0
            for role, counts in self._counts.items():
                p, c = counts["prompt"], counts["completion"]
                result[role] = {"prompt": p, "completion": c, "total": p + c}
                total_p += p
                total_c += c
            result["total"] = {
                "prompt":     total_p,
                "completion": total_c,
                "total":      total_p + total_c,
            }
        return result


# ── 全局单例 ──────────────────────────────────────────────────────────────────
token_tracker = TokenTracker()
