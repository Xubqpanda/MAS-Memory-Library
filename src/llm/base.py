# src/llm/base.py
"""
LLM 层基础定义。

只保留三件事：
  Message      — 全库通用的消息数据结构
  LLMCallable  — 鸭子类型 Protocol，供类型注解使用
  LLMBase      — 抽象基类，具体实现见 model_caller.py

GPTChat 和模块级 token global 已移除：
  - GPTChat 被 ModelCaller 统一替代（src/llm/model_caller.py）
  - token 统计由 TokenTracker 接管（src/llm/token_tracker.py）
"""

from dataclasses import dataclass
from typing import Protocol, Literal, Optional, List, Union
from abc import ABC, abstractmethod


# ── Message ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Message:
    """
    单条对话消息，全库通用。

    content 类型：
      str  : 纯文本消息（绝大多数情况）
      list : 多模态消息，格式为 Responses API content blocks，如：
             [
               {"type": "input_text",  "text": "..."},
               {"type": "input_image", "image_url": {"url": "data:image/jpeg;base64,..."}}
             ]
    """
    role:    Literal["system", "user", "assistant"]
    content: Union[str, list]


# ── LLMCallable Protocol ──────────────────────────────────────────────────────

class LLMCallable(Protocol):
    """
    鸭子类型 Protocol，任何实现了此签名的 callable 都满足接口。
    供 ReasoningBase、MASMemoryBase 等的类型注解使用。
    """
    def __call__(self, messages: List[Message]) -> str: ...


# ── LLMBase ABC ───────────────────────────────────────────────────────────────

class LLMBase(ABC):
    """
    LLM 实现的抽象基类。

    子类只需实现 __call__，接收 Message 列表，返回模型的文本输出。
    temperature、max_tokens 等推理超参由子类在初始化时或 call 内部处理，
    不暴露在基类接口上——这样 ReasoningConfig 的超参可以在 ModelCaller
    层面统一管理，而不是散落在每个子类的签名里。
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def __call__(self, messages: List[Message]) -> str: ...