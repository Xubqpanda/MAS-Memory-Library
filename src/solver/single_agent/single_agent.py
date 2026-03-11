# src/solver/single_agent/single_agent.py
"""
SingleAgentSolver：单 agent solver baseline。

继承 MetaSolver，run_task 主循环精简为：

    init_working_memory(task_main, task_description, context_hint)
        └── memory 内部决定是否预加载 experiential memory（第一、二层）

    for i in range(max_trials):
        user_prompt = retrieve_working_memory()   # memory 决定 prompt 内容
        answer      = reasoning(messages)
        add_working_memory(AgentMessage)
        observation, reward, done = env.step(answer)
        add_working_memory((answer, observation), reward=reward)
        if done: break

    final_reward, final_done, final_feedback = env.feedback()
    add_experiential_memory(label=final_done, feedback=final_feedback)

设计原则：
  - solver 只负责驱动推理和 env 交互，完全不感知 prompt 构造细节。
  - retrieve_experiential_memory 是 agent 的主动按需调用接口（SkillMem 第三层），
    solver 在此不调用，由继承此 solver 的 SkillMem-aware solver 决定何时触发。
  - 检索超参（topk、threshold 等）属于各 memory 方法的内部配置，solver 不传递。
"""

import json
from typing import Optional
from dataclasses import dataclass

from src.envs.base      import Env
from src.solver.base    import MetaSolver
from src.reasoning      import ReasoningBase, ReasoningConfig
from src.memory.base    import MemoryBase
from src.common.message import AgentMessage
from src.llm            import Message, token_tracker
from src.tools          import ToolExecutor, build_default_tools

FINAL_ANSWER_TOOL_NAME = "final_answer"

SINGLE_AGENT_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


@dataclass
class SingleAgentSolver(MetaSolver):
    """单 agent solver baseline，继承 MetaSolver。"""

    def __post_init__(self):
        self.observers        = []
        self.reasoning_config = ReasoningConfig(temperature=0)

    # ── build_system ──────────────────────────────────────────────────────────

    def build_system(
        self,
        reasoning: ReasoningBase,
        solver_memory: MemoryBase,
        env: Env,
        config: dict,
    ) -> None:
        """
        注入推理模块、memory、环境，完成初始化。

        config 字段：
          system_prompt            (str)  : 覆盖默认 system prompt（可选）
          enable_tools             (bool) : 是否启用内置 tool runtime（默认 False）
          max_tool_steps           (int)  : 单次任务内工具调用最大轮次（默认 4）
          max_working_memory_chars (int)  : 可选字符预算，超出后截断（默认不限）
          require_final_answer     (bool) : 工具模式下是否要求显式 final_answer（默认 True）
        """
        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning must be an instance of ReasoningBase")
        if not isinstance(solver_memory, MemoryBase):
            raise TypeError("solver_memory must be an instance of MemoryBase")
        if not isinstance(env, Env):
            raise TypeError("env must be an instance of Env")

        self._system_prompt: str = config.get("system_prompt", SINGLE_AGENT_SYSTEM_PROMPT)
        self._reasoning          = reasoning
        self.meta_memory         = solver_memory
        self.set_env(env)
        self._tool_executor: Optional[ToolExecutor] = None
        self._max_tool_steps: int = int(config.get("max_tool_steps", 10))
        self._max_working_memory_chars: Optional[int] = config.get("max_working_memory_chars")
        self._require_final_answer: bool = bool(config.get("require_final_answer", True))
        if bool(config.get("enable_tools", False)):
            self._tool_executor = ToolExecutor(build_default_tools())

    # ── run_task ──────────────────────────────────────────────────────────────

    def run_task(self, task_config: dict) -> tuple[float, bool]:
        """
        执行单个任务。

        task_config 字段：
          task_main        (str)  : 题目核心内容，必填
          task_description (str)  : 完整任务描述，默认同 task_main
          context_hint     (dict) : 可选任务元信息，透传给 memory
                                    特殊 key：image_b64 / image_media_type（多模态）
          max_trials       (int)  : 最大交互步数；
                                    未指定时优先读 env.max_trials，默认 1（QA 场景）
        """
        if task_config.get("task_main") is None:
            raise ValueError("Missing required key 'task_main' in task_config")

        task_main:        str  = task_config["task_main"]
        task_description: str  = task_config.get("task_description", task_main)
        context_hint:     dict = task_config.get("context_hint", {})
        max_trials:       int  = task_config.get(
            "max_trials", getattr(self.env, "max_trials", 1)
        )

        env = self.env
        env.reset()

        # ── 初始化 working memory ──────────────────────────────────────────
        # memory 内部决定是否预加载 experiential memory（及加载哪几层）
        self.meta_memory.init_working_memory(
            task_main=task_main,
            task_description=task_description,
            context_hint=context_hint,
        )

        # ── 多模态图片（与 memory 无关，在 solver 层处理）─────────────────
        image_b64:        Optional[str] = context_hint.get("image_b64")
        image_media_type: str           = context_hint.get("image_media_type", "image/jpeg")

        # ── 主循环 ─────────────────────────────────────────────────────────
        for i in range(max_trials):
            if self._tool_executor:
                answer, prompt_snapshot = self._reason_with_tools(
                    image_b64=image_b64,
                    image_media_type=image_media_type,
                )
            else:
                answer, prompt_snapshot = self._reason_without_tools(
                    image_b64=image_b64,
                    image_media_type=image_media_type,
                )
                self.meta_memory.add_working_memory(
                    AgentMessage(
                        agent_name="solver",
                        user_instruction=prompt_snapshot,
                        message=answer,
                    ),
                    upstream_ids=[],
                )
            self.notify_observers(f"Step {i+1} Answer: {answer}")

            observation, reward, done = env.step(answer)
            self.notify_observers(f"Act {i+1}: {answer}\nObs {i+1}: {observation}")

            self.meta_memory.add_working_memory(
                (answer, observation),
                event_type="env",
                reward=reward,
            )

            if done:
                break

        # ── 结尾 ──────────────────────────────────────────────────────────
        final_reward, final_done, final_feedback = env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.add_experiential_memory(
            label=final_done,
            feedback=final_feedback,
        ) 

        # ── Token 快照 ────────────────────────────────────────────────────
        t = token_tracker.summary()
        self.notify_observers(
            f"Token usage — "
            f"solver: {t['solver']['total']}  "
            f"env: {t['env']['total']}  "
            f"memory: {t['memory']['total']}  "
            f"tool: {t.get('tool', {}).get('total', 0)}  "
            f"total: {t['total']['total']}"
        )

        return final_reward, final_done

    def _reason_without_tools(
        self,
        image_b64: Optional[str] = None,
        image_media_type: str = "image/jpeg",
    ) -> tuple[str, str]:
        user_prompt: str = self._retrieve_working_prompt()
        self.notify_observers(user_prompt)
        user_content = self._build_user_content(
            user_prompt=user_prompt,
            image_b64=image_b64,
            image_media_type=image_media_type,
        )
        messages = [
            Message("system", self._build_system_prompt(enable_tools=False)),
            Message("user", user_content),
        ]
        answer = self._reasoning(messages, self.reasoning_config)
        return answer, user_prompt

    def _reason_with_tools(
        self,
        image_b64: Optional[str] = None,
        image_media_type: str = "image/jpeg",
    ) -> tuple[str, str]:
        """
        Tool loop driven by working memory:
          1) retrieve_working_memory()
          2) ask LLM
          3) if TOOL_CALL -> execute tool / final_answer
          4) write tool call/result back to working memory
          5) next round re-retrieve memory (memory decides compression/windowing)
        """
        last_answer = ""
        last_prompt = ""
        for _ in range(self._max_tool_steps):
            user_prompt: str = self._retrieve_working_prompt()
            last_prompt = user_prompt
            self.notify_observers(user_prompt)
            user_content = self._build_user_content(
                user_prompt=user_prompt,
                image_b64=image_b64,
                image_media_type=image_media_type,
            )

            messages = [
                Message("system", self._build_system_prompt(enable_tools=True)),
                Message("user", user_content),
            ]
            last_answer = self._reasoning(messages, self.reasoning_config)
            # 每个子步都记录一条 agent 的 in/out（与 tool 事件解耦）
            self.meta_memory.add_working_memory(
                AgentMessage(
                    agent_name="solver",
                    user_instruction=user_prompt,
                    message=last_answer,
                ),
                upstream_ids=[],
            )
            call = self._tool_executor.parse_tool_call(last_answer)
            if call is None:
                if not self._require_final_answer:
                    return last_answer, last_prompt
                self.meta_memory.add_working_memory(
                    (
                        "SYSTEM_HINT: missing_final_answer_call",
                        (
                            "You must terminate by calling final_answer tool.\n"
                            f"Use: TOOL_CALL\n"
                            f'{{"name":"{FINAL_ANSWER_TOOL_NAME}","args":{{"answer":"..."}}}}'
                        ),
                    ),
                    event_type="system",
                    reward=0.0,
                )
                continue

            if call.name == FINAL_ANSWER_TOOL_NAME:
                answer = call.args.get("answer")
                if isinstance(answer, str):
                    return answer.strip(), last_prompt
                return json.dumps(call.args, ensure_ascii=False), last_prompt

            tool_output = self._tool_executor.execute(call)
            self.notify_observers(
                f"Tool call: {call.name}({call.args})\nTool output:\n{tool_output[:1500]}"
            )

            tool_action = f"TOOL_CALL: {call.name} args={json.dumps(call.args, ensure_ascii=False)}"
            tool_observation = (
                f"TOOL_RESULT ({call.name}):\n{tool_output}\n\n"
                "If more tools are needed, emit another TOOL_CALL; otherwise output final answer."
            )
            self.meta_memory.add_working_memory(
                (tool_action, tool_observation),
                event_type="tool",
                reward=0.0,
            )
        return last_answer, last_prompt

    def _retrieve_working_prompt(self) -> str:
        prompt = self.meta_memory.retrieve_working_memory(
            max_chars=self._max_working_memory_chars
        )
        if self._max_working_memory_chars and len(prompt) > self._max_working_memory_chars:
            return prompt[-self._max_working_memory_chars:]
        return prompt

    def _build_system_prompt(self, enable_tools: bool) -> str:
        if not enable_tools:
            return self._system_prompt
        return (
            f"{self._system_prompt}\n\n"
            f"{self._tool_executor.get_tools_prompt()}\n"
            f"To finish, output exactly:\n"
            f"TOOL_CALL\n"
            f'{{"name":"{FINAL_ANSWER_TOOL_NAME}","args":{{"answer":"<final answer>"}}}}'
        )

    @staticmethod
    def _build_user_content(
        user_prompt: str,
        image_b64: Optional[str] = None,
        image_media_type: str = "image/jpeg",
    ):
        if not image_b64:
            return user_prompt
        return [
            {"type": "input_text", "text": user_prompt},
            {"type": "input_image", "image_url": {
                "url": f"data:{image_media_type};base64,{image_b64}"
            }},
        ]

    # ── Observer ──────────────────────────────────────────────────────────────

    def add_observer(self, observer) -> None:
        self.observers.append(observer)

    def notify_observers(self, message: str) -> None:
        for observer in self.observers:
            observer.log(message)
