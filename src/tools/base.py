# src/tools/base.py
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .tool_exec_logger import tool_exec_logger


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


class Tool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs) -> str:
        raise NotImplementedError


class ToolExecutor:
    """
    Minimal tool runtime.

    Expected tool-call block in model output:
    TOOL_CALL
    {"name":"web_search","args":{"query":"..."}}
    """

    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def get_tools_prompt(self) -> str:
        if not self._tools:
            return ""
        lines = [
            "You can call tools. If needed, output exactly:",
            "TOOL_CALL",
            '{"name":"<tool_name>","args":{...}}',
            "Available tools:",
        ]
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        lines.append("If no tool is needed, provide final answer directly.")
        return "\n".join(lines)

    def parse_tool_call(self, text: str) -> ToolCall | None:
        marker = "TOOL_CALL"
        idx = text.find(marker)
        if idx < 0:
            return None

        payload = text[idx + len(marker):].strip()
        if not payload:
            return None

        first_line = payload.splitlines()[0].strip()
        if first_line.startswith("{") and first_line.endswith("}"):
            raw = first_line
        else:
            start = payload.find("{")
            end = payload.rfind("}")
            if start < 0 or end < 0 or end <= start:
                return None
            raw = payload[start:end + 1]

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None

        name = obj.get("name")
        args = obj.get("args", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        return ToolCall(name=name, args=args)

    def execute(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            output = f"ToolError: unknown tool '{call.name}'. Available: {self.tool_names}"
            tool_exec_logger.log(call.name, call.args, output=output, ok=False, duration_ms=0)
            return output
        start = time.time()
        try:
            output = tool.run(**call.args)
            duration_ms = int((time.time() - start) * 1000)
            tool_exec_logger.log(call.name, call.args, output=output, ok=True, duration_ms=duration_ms)
            return output
        except Exception as exc:
            output = f"ToolError: {exc}"
            duration_ms = int((time.time() - start) * 1000)
            tool_exec_logger.log(call.name, call.args, output=output, ok=False, duration_ms=duration_ms)
            return output
