# experiments/benchmarks/FrontierScience/runner.py
"""
FrontierScience benchmark runner。

约定接口：顶层实现 run(cfg, logger)，由 run_experiment.py 动态加载调用。

包含：
  FrontierScienceDataset  — 数据加载（CSV 格式，olympiad / research 两个 track）
  FrontierScienceEvaluator — 评测逻辑（多 trial，olympiad 多数投票，research rubric 打分）
  run()                   — 组装入口

评测流程：
  Olympiad track : num_trials 次独立采样 → 多数投票决定正确与否
  Research track : num_trials 次独立采样 → 平均 rubric 分数 ≥ success_threshold 则视为成功

judge 均通过 ModelCaller（litellm）调用，支持任意 provider。
"""

import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.model_caller import ModelCaller

REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

logger = logging.getLogger("emams")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FrontierScienceDataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FrontierScienceDataset:
    """CSV 格式数据集加载器。必需列：problem / answer / subject / task_group_id / category。"""

    REQUIRED_COLUMNS = {"problem", "answer", "subject", "task_group_id", "category"}
    VALID_CATEGORIES = {"olympiad", "research"}

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self._validate()

    def _validate(self):
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"缺少必需列: {missing}")
        invalid = set(self.df["category"].unique()) - self.VALID_CATEGORIES
        if invalid:
            raise ValueError(f"无效的 category 值: {invalid}")

    def get_problems(
        self,
        category: Optional[str] = None,
        subject: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        df = self.df.copy()
        if category:
            df = df[df["category"] == category]
        if subject:
            df = df[df["subject"] == subject]
        if limit:
            df = df.head(limit)
        return df.to_dict("records")

    def get_olympiad_problems(self, subject=None, limit=None):
        return self.get_problems(category="olympiad", subject=subject, limit=limit)

    def get_research_problems(self, subject=None, limit=None):
        return self.get_problems(category="research", subject=subject, limit=limit)

    def get_statistics(self) -> Dict:
        return {
            "total_problems":    len(self.df),
            "olympiad_problems": int((self.df["category"] == "olympiad").sum()),
            "research_problems": int((self.df["category"] == "research").sum()),
            "subjects":          self.df["subject"].value_counts().to_dict(),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Judge prompt 模板（原 prompts/ 目录下的文本，内联保存）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 如果 prompts/ 目录下有文件则优先读文件，否则使用内联默认值。
# 这样既保留了外部可配置的灵活性，又不依赖文件必须存在。

_OLYMPIAD_JUDGE_PROMPT_DEFAULT = """\
You are an expert judge evaluating a student's solution to an olympiad problem.

Problem:
{problem}

Reference Answer:
{reference_answer}

Student's Answer:
{answer}

Evaluate whether the student's answer is correct. Consider mathematical equivalence,
not just textual similarity. Minor arithmetic errors are not acceptable.

End your evaluation with exactly one of:
VERDICT: CORRECT
VERDICT: INCORRECT"""

_RESEARCH_JUDGE_PROMPT_DEFAULT = """\
You are an expert evaluating a response to a research-level scientific question.

Problem:
{problem}

Evaluation Rubric:
{rubric}

Response to Evaluate:
{answer}

Score the response from 0 to 10 based on the rubric. Be strict and precise.

End your evaluation with:
VERDICT: <score>

where <score> is a number between 0 and 10 (decimals allowed)."""


def _load_prompt(prompt_path: Path, default: str) -> str:
    """读取外部 prompt 文件；文件不存在时使用内联默认值。"""
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    logger.debug(f"Prompt 文件不存在，使用内联默认值: {prompt_path}")
    return default


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FrontierScienceEvaluator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FrontierScienceEvaluator:
    """
    FrontierScience 评测器。

    Olympiad track : num_trials 次独立采样 → 多数投票
    Research track : num_trials 次独立采样 → 平均 rubric 分 ≥ success_threshold
    """

    def __init__(
        self,
        dataset: FrontierScienceDataset,
        model: str,
        judge_model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        output_dir: str = "results",
        verbose: bool = False,
        max_workers: int = 1,
    ):
        self.dataset          = dataset
        self.model            = model
        self.judge_model      = judge_model or "gpt-4o"
        self.reasoning_effort = reasoning_effort
        self.output_dir       = Path(output_dir)
        self.verbose          = verbose
        self.max_workers      = max_workers
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_caller = ModelCaller(model=model, reasoning_effort=reasoning_effort)
        self.judge_caller = ModelCaller(
            model=self.judge_model,
            reasoning_effort="high"
            if any(k in self.judge_model.lower() for k in ("gpt-5", "o1", "o3"))
            else None,
        )

        prompts_dir = Path(__file__).parent / "prompts"
        self.olympiad_judge_prompt = _load_prompt(
            prompts_dir / "olympiad_judge_prompt.txt", _OLYMPIAD_JUDGE_PROMPT_DEFAULT
        )
        self.research_judge_prompt = _load_prompt(
            prompts_dir / "research_judge_prompt.txt", _RESEARCH_JUDGE_PROMPT_DEFAULT
        )

    # ── Olympiad ──────────────────────────────────────────────────────────────

    def evaluate_olympiad(
        self,
        subject: Optional[str] = None,
        limit: Optional[int] = None,
        num_trials: int = 20,
    ) -> Dict:
        start    = time.time()
        problems = self.dataset.get_olympiad_problems(subject=subject, limit=limit)
        logger.info(f"Evaluating {len(problems)} Olympiad problems × {num_trials} trials ...")

        results = []
        for idx, prob in enumerate(tqdm(problems, desc="Olympiad", disable=self.verbose)):
            results.append(self._eval_olympiad_problem(prob, num_trials))

        total   = len(results)
        correct = sum(r["correct"] for r in results)
        runtime = time.time() - start

        summary = {
            "track": "olympiad", "model": self.model, "judge_model": self.judge_model,
            "num_problems": total, "num_trials": num_trials,
            "accuracy": correct / total if total else 0,
            "correct": correct, "total": total,
            "subject_filter": subject,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_seconds": round(runtime, 2),
            "avg_seconds_per_trial": round(runtime / (total * num_trials), 2) if total else 0,
        }
        self._save(summary, "olympiad")
        return summary

    def _eval_olympiad_problem(self, problem_data: Dict, num_trials: int) -> Dict:
        problem          = problem_data["problem"]
        reference_answer = problem_data["answer"]

        run_trial = lambda trial: self._olympiad_trial(trial, problem, reference_answer)
        trial_results = self._run_trials(num_trials, run_trial)
        trial_results.sort(key=lambda x: x["trial"])

        correct_count = sum(t["correct"] for t in trial_results if "correct" in t)
        return {
            "problem":          problem[:200] + "...",
            "reference_answer": reference_answer,
            "subject":          problem_data.get("subject"),
            "task_group_id":    problem_data.get("task_group_id"),
            "correct":          correct_count > num_trials / 2,   # 多数投票
            "correct_trials":   correct_count,
            "total_trials":     num_trials,
            "trials":           trial_results,
        }

    def _olympiad_trial(self, trial: int, problem: str, reference_answer: str) -> Dict:
        try:
            response       = self.model_caller.call(prompt=problem)
            attempted      = response["content"]
            judge_result   = self._judge_olympiad(problem, reference_answer, attempted)
            return {
                "trial": trial, "attempted_answer": attempted,
                "correct": judge_result["correct"],
                "judge_reasoning": judge_result["reasoning"],
                "usage": response["usage"],
            }
        except Exception as e:
            logger.error(f"Olympiad trial {trial} error: {e}")
            return {"trial": trial, "error": str(e), "correct": False}

    def _judge_olympiad(self, problem: str, reference_answer: str, attempted_answer: str) -> Dict:
        prompt   = self.olympiad_judge_prompt.format(
            problem=problem, reference_answer=reference_answer, answer=attempted_answer
        )
        response = self.judge_caller.call(prompt=prompt)
        content  = response["content"]
        if self.verbose:
            print(f"\n{'='*60}\nOLYMPIAD JUDGE:\n{content}\n{'='*60}")
        return {"correct": "VERDICT: CORRECT" in content, "reasoning": content}

    # ── Research ──────────────────────────────────────────────────────────────

    def evaluate_research(
        self,
        subject: Optional[str] = None,
        limit: Optional[int] = None,
        num_trials: int = 30,
        success_threshold: float = 7.0,
    ) -> Dict:
        start    = time.time()
        problems = self.dataset.get_research_problems(subject=subject, limit=limit)
        logger.info(f"Evaluating {len(problems)} Research problems × {num_trials} trials ...")

        results = []
        for prob in tqdm(problems, desc="Research", disable=self.verbose):
            results.append(self._eval_research_problem(prob, num_trials, success_threshold))

        total         = len(results)
        successful    = sum(r["success"] for r in results)
        avg_rubric    = sum(r["avg_rubric_score"] for r in results) / total if total else 0
        runtime       = time.time() - start

        summary = {
            "track": "research", "model": self.model, "judge_model": self.judge_model,
            "num_problems": total, "num_trials": num_trials,
            "success_threshold": success_threshold,
            "accuracy": successful / total if total else 0,
            "avg_rubric_score": round(avg_rubric, 4),
            "successful": successful, "total": total,
            "subject_filter": subject,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_seconds": round(runtime, 2),
            "avg_seconds_per_trial": round(runtime / (total * num_trials), 2) if total else 0,
        }
        self._save(summary, "research")
        return summary

    def _eval_research_problem(
        self, problem_data: Dict, num_trials: int, success_threshold: float
    ) -> Dict:
        problem = problem_data["problem"]
        rubric  = problem_data["answer"]

        run_trial = lambda trial: self._research_trial(trial, problem, rubric)
        trial_results = self._run_trials(num_trials, run_trial)
        trial_results.sort(key=lambda x: x["trial"])

        avg_score = sum(t.get("rubric_score", 0) for t in trial_results) / len(trial_results)
        return {
            "problem":         problem[:200] + "...",
            "rubric":          rubric[:200] + "...",
            "subject":         problem_data.get("subject"),
            "task_group_id":   problem_data.get("task_group_id"),
            "avg_rubric_score": round(avg_score, 4),
            "success":         avg_score >= success_threshold,
            "num_trials":      num_trials,
            "trials":          trial_results,
        }

    def _research_trial(self, trial: int, problem: str, rubric: str) -> Dict:
        try:
            response     = self.model_caller.call(prompt=problem)
            attempted    = response["content"]
            judge_result = self._judge_research(problem, rubric, attempted)
            return {
                "trial": trial, "attempted_answer": attempted,
                "rubric_score": judge_result["rubric_score"],
                "judge_reasoning": judge_result["reasoning"],
                "usage": response["usage"],
            }
        except Exception as e:
            logger.error(f"Research trial {trial} error: {e}")
            return {"trial": trial, "error": str(e), "rubric_score": 0}

    def _judge_research(self, problem: str, rubric: str, attempted_answer: str) -> Dict:
        prompt   = self.research_judge_prompt.format(
            problem=problem, rubric=rubric, answer=attempted_answer
        )
        response = self.judge_caller.call(prompt=prompt)
        content  = response["content"]
        if self.verbose:
            print(f"\n{'='*60}\nRESEARCH JUDGE:\n{content}\n{'='*60}")

        rubric_score = -1.0
        if "VERDICT:" in content:
            try:
                verdict_line = next(l for l in content.split("\n") if "VERDICT:" in l)
                score_str    = verdict_line.split("VERDICT:")[1].strip()
                match        = re.search(r"[\d.]+", score_str)
                if match:
                    rubric_score = float(match.group())
                    if not 0 <= rubric_score <= 10:
                        logger.warning(f"Rubric score {rubric_score} 超出 [0,10]，置为 -1。")
                        rubric_score = -1.0
            except (StopIteration, ValueError) as e:
                logger.warning(f"Rubric score 解析失败: {e}")
        else:
            logger.warning("Judge 输出中未找到 VERDICT，rubric_score 置为 -1。")

        return {"rubric_score": rubric_score, "reasoning": content}

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _run_trials(self, num_trials: int, trial_fn) -> List[Dict]:
        """顺序或并行执行多个 trial，由 max_workers 决定。"""
        if self.max_workers > 1:
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = {ex.submit(trial_fn, t): t for t in range(num_trials)}
                for fut in tqdm(as_completed(futures), total=num_trials,
                                desc="  Trials", position=1, leave=False, disable=self.verbose):
                    results.append(fut.result())
            return results
        return [trial_fn(t) for t in tqdm(
            range(num_trials), desc="  Trials", position=1, leave=False, disable=self.verbose
        )]

    def _save(self, results: Dict, track: str) -> None:
        ts       = results["timestamp"].replace(" ", "_").replace(":", "-")
        filename = f"{track}_{self.model.replace('/', '_')}_{ts}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存 → {filepath}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run() — 约定入口，由 run_experiment.py 动态调用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(cfg: dict, logger: logging.Logger) -> None:
    """FrontierScience 实验入口。"""

    model_cfg  = cfg["model"]
    out_cfg    = cfg["output"]
    exp_cfg    = cfg["experiment"]
    output_dir = REPO_ROOT / out_cfg["dir"] / cfg["benchmark"]["name"] / exp_cfg["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 数据集 ────────────────────────────────────────────────────────────────
    data_path = REPO_ROOT / cfg["benchmark"]["data_path"]
    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        sys.exit(1)

    logger.info(f"Loading: {data_path}")
    dataset = FrontierScienceDataset(str(data_path))
    stats   = dataset.get_statistics()
    logger.info(f"Dataset — olympiad={stats['olympiad_problems']}  research={stats['research_problems']}")

    # ── Evaluator ─────────────────────────────────────────────────────────────
    evaluator = FrontierScienceEvaluator(
        dataset=dataset,
        model=model_cfg["solver"],
        judge_model=model_cfg["judge"],
        reasoning_effort=model_cfg.get("reasoning_effort"),
        output_dir=str(output_dir),
        verbose=out_cfg.get("verbose", False),
        max_workers=out_cfg.get("max_workers", 1),
    )

    results = {}

    oly = cfg.get("tracks", {}).get("olympiad", {})
    if oly.get("enabled", False):
        logger.info("=" * 55)
        logger.info(f"OLYMPIAD  trials={oly['num_trials']}  "
                    f"limit={oly.get('limit')}  subject={oly.get('subject')}")
        logger.info("=" * 55)
        r = evaluator.evaluate_olympiad(
            subject=oly.get("subject"), limit=oly.get("limit"), num_trials=oly["num_trials"],
        )
        results["olympiad"] = r
        logger.info(f"accuracy : {r['accuracy']:.2%}  ({r['correct']}/{r['total']})")
        logger.info(f"runtime  : {r['runtime_seconds']:.1f}s  ({r['avg_seconds_per_trial']:.1f}s/trial)")

    res = cfg.get("tracks", {}).get("research", {})
    if res.get("enabled", False):
        logger.info("=" * 55)
        logger.info(f"RESEARCH  trials={res['num_trials']}  "
                    f"limit={res.get('limit')}  threshold={res.get('success_threshold', 7.0)}")
        logger.info("=" * 55)
        r = evaluator.evaluate_research(
            subject=res.get("subject"), limit=res.get("limit"),
            num_trials=res["num_trials"], success_threshold=res.get("success_threshold", 7.0),
        )
        results["research"] = r
        logger.info(f"accuracy   : {r['accuracy']:.2%}  ({r['successful']}/{r['total']})")
        logger.info(f"avg rubric : {r['avg_rubric_score']:.2f}/10")
        logger.info(f"runtime    : {r['runtime_seconds']:.1f}s  ({r['avg_seconds_per_trial']:.1f}s/trial)")

    ts           = time.strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": exp_cfg, "model": model_cfg, "timestamp": ts, **results}, f, indent=2)
    logger.info(f"结果保存 → {summary_path}")