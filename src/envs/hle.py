# src/envs/hle.py
"""
HLEEnv：HLE 数据集的 Env 实现。

单轮 QA 环境，step() 内部调用 LLM judge 完成评分。
max_trials = 1，所有 MAS 框架的主循环只跑一次。
"""

from src.envs.base import Env
from src.model_caller import ModelCaller

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

class HLEEnv(Env):
    """
    HLE 单轮 QA 环境。

    Attributes:
        judge_caller : ModelCaller，用于调用 judge LLM。
        verbose      : 是否打印 judge 输出，调试用。

    max_trials = 1：所有 MAS 框架主循环只跑一次。
    step(answer)   ：调 judge LLM，缓存结果，返回 (judge_output, reward, done=True)。
    feedback()     ：返回缓存的 (final_reward, True, judge_output)。
    """

    max_trials = 1

    def __init__(self, judge_caller: ModelCaller, verbose: bool = False):
        super().__init__()
        self.judge_caller = judge_caller
        self.verbose = verbose

        self._question:       str   = ""
        self._correct_answer: str   = ""
        self._reward:         float = 0.0
        self._feedback_str:   str   = ""

    def set_task(self, problem: str, reference: str, **kwargs) -> None:
        """每道题开始时由 evaluator 调用，注入题目和参考答案。"""
        self._question       = problem
        self._correct_answer = reference
        self._reward         = 0.0
        self._feedback_str   = ""

    def reset(self) -> None:
        self._reward       = 0.0
        self._feedback_str = ""

    def step(self, action: str) -> tuple[str, float, bool]:
        """
        接收 solver 的答案，调 judge LLM 评分。

        Returns:
            (observation, reward, done)
            observation : judge 的分析文本
            reward      : 1.0 正确，0.0 错误
            done        : 始终 True（单轮 QA）
        """
        prompt = JUDGE_PROMPT.format(
            question=self._question,
            correct_answer=self._correct_answer,
            response=action,
        )

        response = self.judge_caller.call(prompt=prompt)
        content: str = response['content']

        if self.verbose:
            print("\n" + "=" * 60)
            print("HLE JUDGE OUTPUT:")
            print("=" * 60)
            print(content)
            print("=" * 60 + "\n")

        correct = self._parse_correct(content)
        self._reward       = 1.0 if correct else 0.0
        self._feedback_str = content

        return content, self._reward, True

    def feedback(self) -> tuple[float, bool, str]:
        return self._reward, bool(self._reward), self._feedback_str

    def process_action(self, action: str) -> str:
        return action

    def _parse_correct(self, judge_output: str) -> bool:
        for line in judge_output.splitlines():
            line = line.strip().lower()
            if line.startswith("correct:"):
                value = line.split("correct:", 1)[1].strip()
                return value.startswith("yes")
        return False