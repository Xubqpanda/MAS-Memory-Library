# src/solver/single_agent/single_agent.py
"""
SingleAgentSolver：单 agent solver baseline。

继承 MetaSolver，run_task 主循环与三个 MAS 框架完全对齐：

    for i in range(max_trials):
        answer = reasoning(prompt)
        add_working_memory(AgentMessage)
        observation, reward, done = env.step(answer)
        add_working_memory((answer, observation), reward=reward)
        if done: break

    final_reward, final_done, final_feedback = env.feedback()
    meta_memory.add_experiential_memory(label=final_done, feedback=final_feedback)
    return final_reward, final_done

设计原则：
  - solver 只负责驱动推理和 env 交互，不感知 memory 内部策略。
  - 检索超参（topk、threshold 等）属于各 memory 方法的内部配置，
    在 memory 初始化时设置，solver 只调 retrieve，不传任何检索参数。
  - token 消耗通过 observer 在每题结束后打印快照，方便调试。
"""

from dataclasses import dataclass

from src.envs.base       import Env
from src.solver.base        import MetaSolver
from src.reasoning       import ReasoningBase, ReasoningConfig
from src.memory.base     import SolverMemoryBase
from src.common.message  import AgentMessage
from src.llm             import Message, token_tracker
from src.solver.format      import format_task_prompt_with_insights, format_task_context

SINGLE_AGENT_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your answer}"
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
        solver_memory: SolverMemoryBase,
        env: Env,
        config: dict,
    ) -> None:
        """
        注入推理模块、memory、环境，完成初始化。

        config 目前只支持一个字段：
          system_prompt (str) : 覆盖默认 system prompt（可选）

        检索超参（topk、threshold 等）属于 memory 内部策略，
        在各 memory 方法的初始化中配置，此处不接收也不传递。
        """
        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning must be an instance of ReasoningBase")
        if not isinstance(solver_memory, SolverMemoryBase):
            raise TypeError("solver_memory must be an instance of SolverMemoryBase")
        if not isinstance(env, Env):
            raise TypeError("env must be an instance of Env")

        self._system_prompt: str = config.get("system_prompt", SINGLE_AGENT_SYSTEM_PROMPT)
        self._reasoning          = reasoning
        self.meta_memory         = solver_memory
        self.set_env(env)

    # ── run_task ──────────────────────────────────────────────────────────────

    def run_task(self, task_config: dict) -> tuple[float, bool]:
        """
        执行单个任务，主循环与三个 MAS 框架完全对齐。

        task_config 字段：
          task_main        (str)  : 题目/任务核心内容（必填，memory 检索 key）
          task_description (str)  : 完整任务描述，默认同 task_main
          few_shots        (list) : in-context few-shot，默认空列表
          context_hint     (dict) : 可选任务元信息，透传给 memory
          max_trials       (int)  : 最大交互步数；
                                    未指定时优先读 env.max_trials，
                                    env 也没有则默认 1（QA 场景）
        """
        if task_config.get("task_main") is None:
            raise ValueError("Missing required key 'task_main' in task_config")

        task_main:        str  = task_config["task_main"]
        task_description: str  = task_config.get("task_description", task_main)
        few_shots:        list = task_config.get("few_shots", [])
        context_hint:     dict = task_config.get("context_hint", {})
        max_trials:       int  = task_config.get(
            "max_trials", getattr(self.env, "max_trials", 1)
        )

        env = self.env
        env.reset()

        # ── 初始化 working memory ──────────────────────────────────────────
        self.meta_memory.init_working_memory(
            task_main=task_main,
            task_description=task_description,
            context_hint=context_hint,
        )

        # ── 检索 experiential memory（memory 内部决定返回什么）────────────
        successful_trajs, _, insights = self.meta_memory.retrieve_experiential_memory(
            query_task=task_main,
        )

        memory_few_shots: list[str] = [
            format_task_context(
                traj.task_description,
                traj.task_trajectory,
                traj.get_extra_field("key_steps"),
            )
            for traj in successful_trajs
        ]
        raw_insights: list[str] = list(insights)

        # ── 主循环 ─────────────────────────────────────────────────────────
        for i in range(max_trials):

            user_prompt: str = format_task_prompt_with_insights(
                few_shots=few_shots,
                memory_few_shots=memory_few_shots,
                insights=raw_insights,
                task_description=self.meta_memory.retrieve_working_memory(),
            )
            self.notify_observers(user_prompt)

            messages = [
                Message("system", self._system_prompt),
                Message("user",   user_prompt),
            ]

            answer: str = self._reasoning(messages, self.reasoning_config)
            self.notify_observers(f"Step {i+1} Answer: {answer}")

            self.meta_memory.add_working_memory(
                AgentMessage(
                    agent_name="solver",
                    user_instruction=user_prompt,
                    message=answer,
                ),
                upstream_ids=[],
            )

            observation, reward, done = env.step(answer)
            self.notify_observers(f"Act {i+1}: {answer}\nObs {i+1}: {observation}")

            self.meta_memory.add_working_memory(
                (answer, observation),
                reward=reward,
            )

            if done:
                break

        # ── 结尾（与三个框架对齐）──────────────────────────────────────────
        final_reward, final_done, final_feedback = env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.add_experiential_memory(
            label=final_done,
            feedback=final_feedback,
        )

        # ── Token 快照 ─────────────────────────────────────────────────────
        t = token_tracker.summary()
        self.notify_observers(
            f"Token usage — "
            f"solver: {t['solver']['total']}  "
            f"env: {t['env']['total']}  "
            f"memory: {t['memory']['total']}  "
            f"total: {t['total']['total']}"
        )

        return final_reward, final_done

    # ── Observer ──────────────────────────────────────────────────────────────

    def add_observer(self, observer) -> None:
        self.observers.append(observer)

    def notify_observers(self, message: str) -> None:
        for observer in self.observers:
            observer.log(message)