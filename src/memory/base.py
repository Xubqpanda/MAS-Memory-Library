# src/memory/base.py
"""
MemoryBase：所有 memory 方法的统一抽象基类。

两层记忆模型
────────────
Working Memory（inside-trial，任务执行中的实时上下文）
  当前任务的短期记忆，任务结束后固化为经验记忆。
  上下文管理策略（全量 history、摘要、压缩等）完全由 memory 内部决定。

  init_working_memory         任务启动时调用一次：初始化 working memory，
                              同时按 memory 自身策略决定是否预加载 experiential memory
                              （如 SkillMem 在此加载第一、二层：元数据 + SKILL.md）
  add_working_memory          写入一条记录（agent 输出 或 env 反馈，统一接口）
  retrieve_working_memory     读取 working memory，返回可直接注入 prompt 的文本，
                              solver 无需感知其内部构造逻辑

Experiential Memory（cross-trial，跨任务积累的经验）
  历史经验的持久化存储。

  retrieve_experiential_memory  agent 主动按需调用的深层检索接口。
                                基类默认返回空，普通 memory 方法不重写。
                                专为 SkillMem 第三层（references/scripts）设计：
                                  - 第一、二层（元数据 + SKILL.md）由 init_working_memory 加载
                                  - 第三层（详细逻辑/代码）由 agent 在执行中按需触发此方法加载
                                其他 memory 方法（GenerativeMemory、MemoryBank 等）的
                                经验检索在 init_working_memory 内部完成，不暴露给 solver。

  add_experiential_memory       任务结束时调用，固化 working memory 并更新经验权重。
                                合并了原 save_task_context + backward 两个操作。

接口总览（共 5 个）
────────────────────
  init_working_memory          working memory 初始化（+ 按需预加载经验）
  add_working_memory           写入 working memory
  retrieve_working_memory      读取 working memory → 可用 prompt 字符串
  retrieve_experiential_memory agent 按需调用的深层检索（SkillMem 第三层专用）
  add_experiential_memory      固化经验 + 更新权重（任务结束时调用）
"""

import os
from dataclasses import dataclass
from abc import ABC
from typing import Optional, Union

from src.common import AgentMessage, MASMessage, StorageNameSpace
from src.llm import LLMCallable
from src.utils import EmbeddingFunc

# add_working_memory 的 content 类型：
#   AgentMessage    → agent 的单步输出
#   tuple[str, str] → (action, observation)，env 反馈
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
    ) -> None:
        """
        任务启动时调用一次，完成两件事：
          1. 初始化 working memory（设置 current_task_context）
          2. 按 memory 自身策略决定是否预加载 experiential memory

        子类覆盖此方法时，可在 super().__post_init__() 之后追加经验加载逻辑：
          - EmptyMemory    : 只做初始化，不加载任何经验
          - GenerativeMemory: 检索 insights / few-shots，拼入初始 prompt
          - SkillMem       : 加载第一层（元数据摘要）+ 第二层（SKILL.md），
                             第三层由 agent 按需调用 retrieve_experiential_memory 加载

        Args:
            task_main        : 题目核心内容，用于经验检索的 query key。
            task_description : 完整任务描述（含解题指令），默认同 task_main。
            context_hint     : 可选任务元信息（如 category、answer_type 等），
                               子类可用于辅助检索或 skill 激活，基类忽略。
        """
        self.current_task_context = MASMessage(
            task_main=task_main,
            task_description=task_description or task_main,
        )

    def add_working_memory(
        self,
        content: WorkingMemoryContent,
        upstream_ids: Optional[list[str]] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        向 working memory 写入一条记录，统一承接两种写入场景：

        场景一：agent 输出
            content      = AgentMessage(...)
            upstream_ids = ["node-0", ...]   # 无依赖时传 []
            返回值：当前节点在 StateChain 中的 node_id（str）

        场景二：env 反馈（interactive task 专用）
            content      = (action, observation)
            upstream_ids 忽略
            **kwargs     透传给 StateChain（如 reward=0.5）
                         推荐携带 event_type 区分来源：
                           - event_type="env"   : 环境反馈
                           - event_type="tool"  : 工具调用轨迹
                           - event_type="system": 系统提示/约束提示
            返回值：None
        """
        if isinstance(content, AgentMessage):
            return self.current_task_context.add_message_to_current_state(
                content, upstream_ids or []
            )
        elif isinstance(content, tuple) and len(content) == 2:
            action, observation = content
            kwargs.setdefault("event_type", "env")
            self.current_task_context.move_state(action, observation, **kwargs)
            return None
        else:
            raise TypeError(
                "content must be AgentMessage or tuple[str, str] (action, observation). "
                f"Got: {type(content)}"
            )

    def retrieve_working_memory(self, **kwargs) -> str:
        """
        读取当前 working memory，返回可直接注入 prompt 的字符串。

        基类默认：返回 task_description（不含任何空章节噪声）。
        子类可覆盖：拼接 trajectory history、摘要压缩、滑动窗口等。

        Returns:
            str: 可直接作为 user message 内容的 prompt 字符串。
        """
        ctx = self.current_task_context
        base = ctx.task_description or ""
        traj = ctx.task_trajectory or ""
        return base + traj if traj else base

    # ─────────────────────────────────────────────────────────────────────────
    # Experiential Memory（cross-trial）
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_experiential_memory(
        self,
        query: str,
        **kwargs,
    ) -> str:
        """
        agent 主动按需调用的深层检索接口。

        设计意图：专为 SkillMem 第三层（references / scripts）设计。
          - 第一、二层（元数据 + SKILL.md）在 init_working_memory 时已加载进 prompt。
          - 第三层在 agent 判断需要更多细节时，主动调用此方法获取。

        其他 memory 方法（EmptyMemory、GenerativeMemory 等）的经验检索
        全部在 init_working_memory 内部完成，不重写此方法（返回空字符串）。

        Args:
            query  : 检索 query（通常是 agent 当前的意图描述或子任务）。
            **kwargs: 子类扩展参数（如 topk、threshold 等）。

        Returns:
            str: 检索到的经验文本，可直接追加到 prompt 中。
                 基类默认返回空字符串。
        """
        return ""

    def add_experiential_memory(
        self,
        label: Union[bool, float],
        feedback: str = None,
    ) -> None:
        """
        任务结束时调用，固化 working memory 并更新经验权重。
        这是 inside-trial → cross-trial 的边界。

        基类默认实现：仅打标签，不做任何持久化（no-memory baseline）。

        Args:
            label    : 成功/失败标签（bool）或质量评分（float）。
            feedback : 可选的额外反馈文本（如 env feedback、judge 分析）。
        """
        if self.current_task_context is None:
            raise RuntimeError("working memory 为空，请先调用 init_working_memory。")
        self.current_task_context.label = label
        if feedback is not None:
            self.current_task_context.task_description += f"\n- Environment feedback\n{feedback}\n"
