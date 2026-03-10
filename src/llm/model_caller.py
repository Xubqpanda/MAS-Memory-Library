# src/llm/model_caller.py
"""
ModelCaller：全库统一的 LLM 调用接口。

原 experiments/benchmarks/FrontierScience/src/model_caller.py 迁移至此，
同时承接原 src/llm/base.py 中 GPTChat 的职责，成为唯一的 LLM 调用入口。

设计要点：
  - 底层用 litellm，支持 OpenAI / Anthropic / Gemini 等任意 provider。
  - 初始化时指定 role（"solver" / "memory" / "env"），每次调用后自动向
    全局 TokenTracker 汇报，无需调用方手动统计。
  - 实现 LLMBase 接口（__call__ 接收 List[Message]），可直接传给
    ReasoningBase，与原 GPTChat 完全替换。
  - 同时保留 call(prompt: str) 方法供 HLEEnv 等只需要单轮 prompt 的场景使用。

用法示例：
    from src.llm import ModelCaller

    solver_llm = ModelCaller(model="gpt-4o",         role="solver")
    judge_llm  = ModelCaller(model="gpt-4o",         role="env")
    memory_llm = ModelCaller(model="gpt-4o-mini",    role="memory")

    # ReasoningIO 路径（List[Message] → str）
    reasoning = ReasoningIO(llm_model=solver_llm)

    # 直接单轮调用路径（str → dict）
    result = judge_llm.call(prompt="Is this correct?")
"""

import os
import time
import logging
from typing import Any, Dict, List, Literal, Optional

import litellm
from dotenv import load_dotenv

from src.llm.base import LLMBase, Message
from src.llm.token_tracker import token_tracker

load_dotenv()
litellm.drop_params = True   # 不同 provider 不支持的参数自动忽略

logger = logging.getLogger("emams")

RoleType = Literal["solver", "memory", "env"]


class ModelCaller(LLMBase):
    """
    统一 LLM 调用器，实现 LLMBase 接口。

    Attributes:
        model             : litellm 格式的模型名，如 "gpt-4o" / "claude-3-5-sonnet-20241022"。
        role              : 调用方身份，用于 TokenTracker 分桶统计。
        temperature       : 默认采样温度，可在 __call__ 时覆盖。
        max_tokens        : 默认最大输出 token 数，可在 __call__ 时覆盖。
        reasoning_effort  : 推理模型（o1/o3）专用，"low" / "medium" / "high"。
        max_retries       : 遇到 rate limit 时的最大重试次数。
        retry_wait        : 每次重试的等待秒数（指数退避）。
    """

    def __init__(
        self,
        model: str,
        role: RoleType = "solver",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        reasoning_effort: Optional[str] = None,
        max_retries: int = 5,
        retry_wait: float = 1.0,
    ):
        super().__init__(model_name=model)
        self.model            = model
        self.role             = role
        self.temperature      = temperature
        self.max_tokens       = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retries      = max_retries
        self.retry_wait       = retry_wait

        self._validate_api_key()

    # ── LLMBase 接口（供 ReasoningIO 使用）───────────────────────────────────

    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1,
    ) -> str:
        """
        接收 List[Message]，返回模型的文本输出。
        供 ReasoningIO 等 MAS 框架内部调用。

        Args:
            messages    : 对话消息列表（system / user / assistant）。
            temperature : 覆盖实例默认值；None 时使用实例默认值。
            max_tokens  : 覆盖实例默认值；None 时使用实例默认值。
            stop_strs   : 停止词列表，触发时提前截断输出。
            num_comps   : 保留参数，当前固定为 1（不支持多路采样）。
        """
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        result    = self._call_with_retry(
            messages=formatted,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stop=stop_strs,
        )
        return result["content"]

    # ── 单轮 prompt 接口（供 HLEEnv、FrontierScienceEvaluator 等使用）─────────

    def call(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        接收单条 prompt 字符串，返回包含 content / usage / model 的 dict。
        供 Env 侧 judge 调用，与原 ModelCaller.call() 接口保持一致。

        Returns:
            {
                "content":      str,          # 模型输出文本
                "usage":        {             # token 消耗明细
                    "prompt_tokens":     int,
                    "completion_tokens": int,
                    "total_tokens":      int,
                },
                "model":        str,          # 实际调用的模型名
                "finish_reason": str,
            }
        """
        return self._call_with_retry(
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.pop("temperature", self.temperature),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            **kwargs,
        )

    # ── 内部实现 ──────────────────────────────────────────────────────────────

    def _call_with_retry(self, messages: list, **kwargs) -> Dict[str, Any]:
        """
        带指数退避重试的 litellm 调用。
        调用成功后向全局 TokenTracker 汇报本次消耗。
        """
        kwargs = {k: v for k, v in kwargs.items() if v is not None}   # 过滤 None

        # 推理模型的 reasoning_effort 参数
        if self.reasoning_effort and any(
            k in self.model.lower() for k in ("o1", "o3", "gpt-5")
        ):
            kwargs["reasoning_effort"] = self.reasoning_effort

        wait = self.retry_wait
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
                content = response.choices[0].message.content or ""
                usage   = response.usage

                # 向 TokenTracker 汇报
                token_tracker.add(
                    role=self.role,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )

                return {
                    "content":      content,
                    "usage": {
                        "prompt_tokens":     usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens":      usage.total_tokens,
                    },
                    "model":        response.model,
                    "finish_reason": response.choices[0].finish_reason,
                }

            except Exception as e:
                last_error = e
                err_str    = str(e).lower()
                if "rate limit" in err_str or "429" in err_str:
                    logger.warning(
                        f"[ModelCaller] Rate limit，{wait:.1f}s 后重试 "
                        f"(attempt {attempt + 1}/{self.max_retries}) ..."
                    )
                    time.sleep(wait)
                    wait *= 2   # 指数退避
                else:
                    logger.error(f"[ModelCaller] API error (model={self.model}): {e}")
                    break

        raise RuntimeError(
            f"ModelCaller 调用失败（model={self.model}, role={self.role}），"
            f"已重试 {self.max_retries} 次。最后错误: {last_error}"
        )

    def _validate_api_key(self):
        """检查对应 provider 的 API key 是否已设置。"""
        m = self.model.lower()
        checks = {
            "openai":    ("gpt", "o1", "o3"),
            "anthropic": ("claude",),
            "google":    ("gemini",),
        }
        env_keys = {
            "openai":    "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google":    "GOOGLE_API_KEY",
        }
        for provider, prefixes in checks.items():
            if any(p in m for p in prefixes):
                key = env_keys[provider]
                if not os.getenv(key):
                    raise ValueError(
                        f"使用 {provider} 模型（{self.model}）需要设置环境变量 {key}。"
                    )
                return