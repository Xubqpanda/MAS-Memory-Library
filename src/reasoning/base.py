# src/reasoning/base.py
"""
Reasoning 模块：封装 agent 的推理策略。

当前只有一种实现 ReasoningIO（直接 IO），后续可扩展：
  - ReasoningCoT  : Chain-of-Thought，在 prompt 里注入思维链引导
  - ReasoningReAct: ReAct 模式，交替输出 Thought / Action / Observation

ReasoningBase 是统一抽象，MAS 框架只感知这个接口，
切换推理策略时只需替换传入的 reasoning 实例，框架代码不动。
"""

from dataclasses import dataclass
from typing import Optional, List

from src.llm.base        import LLMCallable, Message
from src.llm.model_caller import ModelCaller


@dataclass
class ReasoningConfig:
    """
    控制推理行为的超参配置。

    None 表示使用 ModelCaller 实例的默认值（不覆盖）。
    这样 ReasoningConfig 只需要关心"想覆盖什么"，
    不用每次都把所有参数都写一遍。
    """
    temperature: Optional[float] = None
    max_tokens:  Optional[int]   = None
    stop_strs:   Optional[List[str]] = None
    num_comps:   Optional[int]   = None


class ReasoningBase:
    """推理模块抽象基类，子类实现 __call__。"""

    def __init__(self, llm_model: LLMCallable):
        self.llm_model = llm_model

    def __call__(self, messages: List[Message], config: ReasoningConfig) -> str:
        raise NotImplementedError


class ReasoningIO(ReasoningBase):
    """
    最简直接 IO 推理：将 messages 原样传给 LLM，返回输出。

    这是 no-reasoning baseline，所有复杂推理策略（CoT、ReAct 等）
    都通过继承 ReasoningBase 实现，不影响此类。
    """

    def __call__(self, messages: List[Message], config: ReasoningConfig) -> str:
        return self.llm_model(
            messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop_strs=config.stop_strs,
            num_comps=config.num_comps if config.num_comps is not None else 1,
        )