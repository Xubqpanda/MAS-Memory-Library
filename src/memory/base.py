# src/memory/base.py
"""
MemoryBase：所有 memory 方法的统一抽象基类。

两层记忆模型
────────────
Working Memory（inside-trial，任务执行中的实时上下文）
  当前任务的短期记忆，任务结束后固化为经验记忆。
  上下文策略（全量 history、摘要、压缩等）由 memory system 自主决定。

  init_working_memory         任务启动，初始化 working memory
  add_working_memory          写入一条记录（agent 输出 或 env 反馈，统一接口）
  retrieve_working_memory     读取 working memory，返回可注入 prompt 的文本（上下文策略由 memory 内部决定：全量 / 摘要 / 压缩）

Experiential Memory（cross-trial，跨任务积累的经验）
  历史经验的持久化存储，供后续任务检索参考。

  retrieve_experiential_memory  任务执行中检索历史经验（调用时机 inside-trial，数据来源 cross-trial）
  add_experiential_memory       任务结束时调用，固化 working memory 并更新经验权重 ← inside-trial / cross-trial 的边界（合并原 save_task_context + backward）

接口总览（共 5 个）
────────────────────
  init_working_memory          working memory 初始化
  add_working_memory           写入 working memory
  retrieve_working_memory      读取 working memory
  retrieve_experiential_memory 检索历史经验
  add_experiential_memory      固化经验 + 更新权重（任务结束时调用）

设计约束
────────
- add_working_memory 统一承接原 add_agent_node（agent 输出）和 move_memory_state（env 反馈）两种写入，通过 content 类型区分。
- retrieve_working_memory 取代原 summarize，上下文策略完全由 memory 内部决定。
- add_experiential_memory 合并原 save_task_context 和 backward，子类在此方法内自由决定固化轨迹与更新权重的顺序和策略。
- 子类只需覆盖需要扩展的方法，其余继承基类默认实现。
"""

import os
from dataclasses import dataclass
from abc import ABC
from typing import Optional, Union

from src.common import AgentMessage, MASMessage, StorageNameSpace
from src.llm import LLMCallable
from src.utils import EmbeddingFunc

# add_working_memory 的 content 类型：
#   AgentMessage    → agent 的单步输出（原 add_agent_node）
#   tuple[str, str] → (action, observation)，env 反馈（原 move_memory_state）
WorkingMemoryContent = Union[AgentMessage, tuple[str, str]]


@dataclass
class MemoryBase(StorageNameSpace, ABC):
    """
    Memory 抽象基类。

    Attributes:
        llm_model      : 用于生成摘要、洞见等的 LLM 调用接口。
        embedding_func : 向量化函数，用于语义检索。
    """

    llm_model: LLMCallable
    embedding_func: EmbeddingFunc

    def __post_init__(self):
        self.persist_dir: str = os.path.join(self.global_config["working_dir"], self.namespace)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.current_task_context: Optional[MASMessage] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Working Memory（inside-trial）
    # ─────────────────────────────────────────────────────────────────────────

    def init_working_memory(
        self,
        task_main: str,
        task_description: str = None,
        context_hint: Optional[dict] = None,
    ) -> MASMessage:
        """
        任务启动时调用，初始化 working memory。每个任务开始时调用一次。

        Args:
            task_main        : 题目核心内容，用于经验检索的 key。
            task_description : 完整题目描述（含解题指令），默认同 task_main。
            context_hint     : 可选任务元信息（如 subject、task_type、difficulty），
                               子类可用于辅助经验检索或 skill 激活决策，基类忽略。
        """
        self.current_task_context = MASMessage(
            task_main=task_main,
            task_description=task_description or task_main,
        )
        return self.current_task_context

    def add_working_memory(
        self,
        content: WorkingMemoryContent,
        upstream_ids: Optional[list[str]] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        向 working memory 写入一条记录，统一承接两种写入场景：

        场景一：agent 输出（原 add_agent_node）
            content      = AgentMessage(...)
            upstream_ids = ["node-0", ...]   # 无依赖时传 []
            返回值：当前节点在 StateChain 中的 node_id（str）

        场景二：env 反馈（原 move_memory_state，interactive task 专用）
            content      = (action, observation)
            upstream_ids 忽略
            **kwargs     透传给 StateChain（如 reward=0.5）
            返回值：None

        Args:
            content      : AgentMessage 或 (action, observation) tuple。
            upstream_ids : agent 输出场景下的上游节点 ID 列表。
            **kwargs     : env 反馈场景下的额外参数（如 reward）。

        Returns:
            str  : agent 输出场景下的 node_id。
            None : env 反馈场景。
        """
        if isinstance(content, AgentMessage):
            return self.current_task_context.add_message_to_current_state(
                content, upstream_ids or []
            )
        elif isinstance(content, tuple) and len(content) == 2:
            action, observation = content
            self.current_task_context.move_state(action, observation, **kwargs)
            return None
        else:
            raise TypeError(
                "content must be AgentMessage or tuple[str, str] (action, observation). "
                f"Got: {type(content)}"
            )

    def retrieve_working_memory(self, **kwargs) -> str:
        """
        读取当前 working memory，返回可直接注入 prompt 的文本。

        上下文策略完全由 memory 内部决定：
          - 基类默认：全量返回 task_description + task_trajectory
          - 子类可覆盖：摘要压缩、滑动窗口、关键步骤抽取等

        取代原 summarize() 接口。
        """
        ctx = self.current_task_context
        return (ctx.task_description or "") + (ctx.task_trajectory or "")

    # ─────────────────────────────────────────────────────────────────────────
    # Experiential Memory（cross-trial）
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_experiential_memory(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
        **kwargs,
    ) -> tuple[list[MASMessage], list[MASMessage], list]:
        """
        任务执行中检索历史经验，为当前任务提供参考。

        注意：调用时机是 inside-trial，但数据来源是 cross-trial 积累的结果。

        Args:
            query_task      : 检索查询文本（通常是 task_main）。
            successful_topk : 返回成功案例数上限。
            failed_topk     : 返回失败案例数上限。
            **kwargs        : 子类扩展参数（如 threshold、insight_topk 等）。

        Returns:
            tuple: (successful_trajectories, failed_trajectories, insights)
        """
        return [], [], []

    def add_experiential_memory(
        self,
        label: Union[bool, float],
        feedback: str = None,
    ) -> None:
        """
        任务结束时调用，固化 working memory 并更新经验权重。
        这是 inside-trial → cross-trial 的边界。

        合并了原 save_task_context（固化轨迹）和 backward（更新权重）两个操作。
        子类可自由决定固化与更新的顺序和策略，例如：
          - 先固化轨迹，再调整 insight 分数
          - 先提炼 skill，再写入向量库

        基类默认实现：仅打标签，不做任何持久化（no-memory baseline）。

        Args:
            label    : olympiad 传 bool；research 传 float rubric score。
            feedback : 可选的额外反馈文本（如 env 的 feedback_str、judge 分析）。
        """
        if self.current_task_context is None:
            raise RuntimeError("working memory 为空，请先调用 init_working_memory。")
        self.current_task_context.label = label
        if feedback is not None:
            self.current_task_context.task_description += f"\n- Environment feedback\n{feedback}\n"