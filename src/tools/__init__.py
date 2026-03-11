# src/tools/__init__.py
from .base import Tool, ToolCall, ToolExecutor
from .search_tools import (
    ReadPageTool,
    WebSearchTool,
    WikiSearchTool,
    build_default_search_tools,
)
from .mm_tools import (
    AudioInspectorTool,
    EncodeFileBase64Tool,
    TextInspectorTool,
    VisualInspectorTool,
    build_default_mm_tools,
)
from .tool_exec_logger import tool_exec_logger


def build_default_tools() -> list[Tool]:
    return [
        *build_default_search_tools(),
        *build_default_mm_tools(),
    ]


__all__ = [
    "Tool",
    "ToolCall",
    "ToolExecutor",
    "WebSearchTool",
    "ReadPageTool",
    "WikiSearchTool",
    "TextInspectorTool",
    "VisualInspectorTool",
    "AudioInspectorTool",
    "EncodeFileBase64Tool",
    "build_default_search_tools",
    "build_default_mm_tools",
    "build_default_tools",
    "tool_exec_logger",
]
