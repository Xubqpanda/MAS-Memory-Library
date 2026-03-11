# src/llm/model_caller.py
"""
ModelCaller：全库统一的 LLM 调用接口。

支持两种后端：
  1. litellm（默认）：标准 OpenAI chat completions 格式，支持多 provider。
  2. Responses API  ：OpenAI /v1/responses 格式，用于自定义端点（如 gmn.chuangzuoli.com）。
     触发条件：初始化时传入 base_url，自动走此路径，无需额外配置。

路由逻辑：
  base_url 为 None → litellm（chat completions）
  base_url 非 None → requests 直接调用（responses API）
"""

import os
import time
import logging
import requests
from typing import Any, Dict, List, Literal, Optional

import litellm
from dotenv import load_dotenv

from src.llm.base import LLMBase, Message
from src.llm.token_tracker import token_tracker
from src.llm.llm_io_logger import llm_io_logger

load_dotenv()
litellm.drop_params = True

logger = logging.getLogger("emams")

RoleType = Literal["solver", "memory", "env", "tool"]


class ModelCaller(LLMBase):
    """
    统一 LLM 调用器。

    Args:
        model            : 模型名，如 "gpt-5.3" / "gpt-4o" / "claude-3-5-sonnet-20241022"。
        role             : 调用方身份（"solver" / "memory" / "env" / "tool"），用于 TokenTracker 分桶。
        temperature      : 默认采样温度。
        max_tokens       : 默认最大输出 token 数。
        reasoning_effort : 推理模型专用，"low" / "medium" / "high"。
        max_retries      : rate limit 时最大重试次数。
        retry_wait       : 首次重试等待秒数（指数退避）。
        base_url         : 自定义 API 端点根地址（如 "https://gmn.chuangzuoli.com"）。
                           非 None 时自动走 Responses API，不经过 litellm。
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
        base_url: Optional[str] = None,
    ):
        super().__init__(model_name=model)
        self.model            = model
        self.role             = role
        self.temperature      = temperature
        self.max_tokens       = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retries      = max_retries
        self.retry_wait       = retry_wait
        self.base_url         = base_url

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
        # content 可以是 str（纯文本）或 list（多模态），原样透传
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        result = self._call_with_retry(
            messages=formatted,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stop=stop_strs,
        )
        return result["content"]

    # ── 单轮 prompt 接口（供 HLEEnv 等使用）──────────────────────────────────

    def call(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return self._call_with_retry(
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.pop("temperature", self.temperature),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            **kwargs,
        )

    # ── 路由层 ────────────────────────────────────────────────────────────────

    def _call_with_retry(self, messages: list, **kwargs) -> Dict[str, Any]:
        """base_url 存在时走 Responses API，否则走 litellm。"""
        if self.base_url:
            return self._call_responses_api(messages, **kwargs)
        return self._call_litellm(messages, **kwargs)

    # ── Responses API（自定义端点）────────────────────────────────────────────

    def _call_responses_api(self, messages: list, **kwargs) -> Dict[str, Any]:
        """
        OpenAI Responses API 适配器。

        格式转换：
          messages[{role, content}]  →  input[{type, role, content:[{type, text}]}]
          system role                →  developer role（Responses API 规范）
          output[0].content[0].text  →  content
          input_tokens/output_tokens →  prompt_tokens/completion_tokens（统一字段名）
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        url     = self.base_url.rstrip("/") + "/v1/responses"

        input_messages = []
        for m in messages:
            role = "developer" if m["role"] == "system" else m["role"]
            # content 可以是字符串（纯文本）或列表（文字+图片）
            if isinstance(m["content"], list):
                # 已经是结构化 content，直接使用（由 SingleAgentSolver 构造）
                content_blocks = m["content"]
            else:
                content_blocks = [{"type": "input_text", "text": m["content"]}]
            input_messages.append({
                "type": "message",
                "role": role,
                "content": content_blocks,
            })

        body: Dict[str, Any] = {
            "model":       self.model,
            "input":       input_messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        wait       = self.retry_wait
        last_error = None

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=120)
                resp.raise_for_status()
                data = resp.json()

                content      = data["output"][0]["content"][0]["text"]
                usage        = data.get("usage", {})
                prompt_tok   = usage.get("input_tokens", 0)
                complete_tok = usage.get("output_tokens", 0)

                token_tracker.add(
                    role=self.role,
                    prompt_tokens=prompt_tok,
                    completion_tokens=complete_tok,
                )
                llm_io_logger.log(
                    role=self.role, model=data.get("model", self.model),
                    messages=messages, output=content,
                    usage={"prompt_tokens": prompt_tok, "completion_tokens": complete_tok,
                        "total_tokens": prompt_tok + complete_tok},
                    finish_reason=data["output"][0].get("status", "completed"),
                )
                return {
                    "content": content,
                    "usage": {
                        "prompt_tokens":     prompt_tok,
                        "completion_tokens": complete_tok,
                        "total_tokens":      usage.get("total_tokens", prompt_tok + complete_tok),
                    },
                    "model":        data.get("model", self.model),
                    "finish_reason": data["output"][0].get("status", "completed"),
                }

            except requests.HTTPError as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str:
                    logger.warning(
                        f"[ModelCaller] Rate limit，{wait:.1f}s 后重试 "
                        f"(attempt {attempt+1}/{self.max_retries}) ..."
                    )
                    time.sleep(wait)
                    wait *= 2
                else:
                    logger.error(f"[ModelCaller] HTTP error (model={self.model}): {e}")
                    last_error = e
                    break
            except Exception as e:
                logger.error(f"[ModelCaller] Responses API error (model={self.model}): {e}")
                last_error = e
                break

        raise RuntimeError(
            f"ModelCaller 调用失败（model={self.model}, role={self.role}），"
            f"已重试 {self.max_retries} 次。最后错误: {last_error}"
        )

    # ── litellm（标准 chat completions）──────────────────────────────────────

    def _call_litellm(self, messages: list, **kwargs) -> Dict[str, Any]:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if self.reasoning_effort and any(
            k in self.model.lower() for k in ("o1", "o3", "gpt-5")
        ):
            kwargs["reasoning_effort"] = self.reasoning_effort

        wait       = self.retry_wait
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

                token_tracker.add(
                    role=self.role,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
                llm_io_logger.log(
                    role=self.role, model=response.model,
                    messages=messages, output=content,
                    usage={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens},
                    finish_reason=response.choices[0].finish_reason,
                )
                return {
                    "content": content,
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
                        f"(attempt {attempt+1}/{self.max_retries}) ..."
                    )
                    time.sleep(wait)
                    wait *= 2
                else:
                    logger.error(f"[ModelCaller] API error (model={self.model}): {e}")
                    break

        raise RuntimeError(
            f"ModelCaller 调用失败（model={self.model}, role={self.role}），"
            f"已重试 {self.max_retries} 次。最后错误: {last_error}"
        )

    # ── API key 检查 ──────────────────────────────────────────────────────────

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
