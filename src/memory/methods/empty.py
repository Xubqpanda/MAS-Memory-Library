# src/memory/methods/empty.py
from dataclasses import dataclass
from src.memory.base import MemoryBase


@dataclass
class EmptyMemory(MemoryBase):
    """
    No-memory baseline。

    完全继承 MemoryBase 的默认实现：
      - init_working_memory           : 简单初始化 task context
      - add_working_memory            : 写入 StateChain（agent 输出 或 env 反馈）
      - retrieve_working_memory       : 全量返回 task_description + task_trajectory
      - retrieve_experiential_memory  : 返回空列表（无历史经验）
      - add_experiential_memory       : 仅打标签，不持久化
    """

    def __post_init__(self):
        super().__post_init__()