# experiments/benchmarks/HLE/runner.py
"""
HLE benchmark runner。

约定接口：顶层实现 run(cfg, logger)，由 run_experiment.py 动态加载调用。

职责：
  1. 数据加载（HLEDataset）
  2. 评测逻辑（HLEEvaluator）
  3. MAS 栈组装（judge → env → solver LLM → memory → solver）
  4. 结果汇总和持久化

实际数据字段（来自 HLE 官方数据集）：
  id            : 题目唯一 ID
  question      : 题目文本
  answer        : 参考答案
  answer_type   : "exactMatch" 等
  category      : 类别，如 "Math" / "Other" / "Science"（首字母大写）
  image         : 始终为 null，不可用于多模态判断
  image_preview : 非 null 时表示该题有图片，是多模态判断的实际依据

cfg 关键字段（benchmark yaml + method yaml 合并后）：
  benchmark.data_path        HLE 数据文件路径（相对仓库根目录）
  model.solver               求解模型名
  model.judge                judge 模型名（用于 HLEEnv 内部打分）
  model.base_url             可选，自定义 API 端点（如 https://gmn.chuangzuoli.com）
  model.api_format           可选，"chat"（默认）或 "responses"
  evaluation.category        类别过滤（null = 全量，注意首字母大写，如 "Math"）
  evaluation.text_only       是否跳过多模态题目（默认 true）
  evaluation.limit           题目数上限（null = 全量，调试时填小数字）
  evaluation.max_workers     并行线程数（默认 1，仅 EmptyMemory 安全并行）
  output.dir                 结果根目录
  output.verbose             逐题打印开关
  experiment.name            实验名（用于目录命名）
  mas_config                 传给 SingleAgentSolver.build_system() 的超参
  memory_config.namespace    Memory 命名空间
  memory_config.working_dir  Memory 持久化目录
"""

import base64
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.envs.hle                    import HLEEnv
from src.solver.single_agent         import SingleAgentSolver
from src.memory.methods.empty        import EmptyMemory
from src.reasoning                   import ReasoningIO
from src.llm.model_caller            import ModelCaller
from src.llm.token_tracker           import token_tracker
from src.llm.llm_io_logger           import llm_io_logger

logger = logging.getLogger("emams")

REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HLEDataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HLEDataset:
    """
    HLE 数据集加载器。

    多模态判断逻辑：
      - `image` 字段在 HLE 中始终为 null，不作为判断依据。
      - `image_preview` 非 null 时才是真正有图片的多模态题目。

    统一输出字段：
      id / question / answer / answer_type / category / is_multimodal / raw

    支持格式：JSON Array 或 JSON Lines。
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"HLE 数据文件不存在: {self.data_path}")
        raw = self._load_file()
        self.problems: List[Dict] = [self._normalize(r) for r in raw]
        self._warn_if_empty()

    def _load_file(self) -> List[dict]:
        text = self.data_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"数据文件为空: {self.data_path}")
        if text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        records = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON Lines 解析失败（第 {lineno} 行）: {e}")
        return records

    def _normalize(self, record: dict) -> dict:
        is_multimodal = record.get("image_preview") is not None
        return {
            "id":            record.get("id", ""),
            "question":      str(record.get("question", "")).strip(),
            "answer":        str(record.get("answer", "")).strip(),
            "answer_type":   record.get("answer_type", "exactMatch"),
            "category":      str(record.get("category", "Unknown")).strip(),
            "is_multimodal": is_multimodal,
            "raw":           record,
        }

    def _warn_if_empty(self):
        empty_q = sum(1 for p in self.problems if not p["question"])
        empty_a = sum(1 for p in self.problems if not p["answer"])
        if empty_q:
            logger.warning(f"[HLEDataset] {empty_q} 条记录的 question 字段为空。")
        if empty_a:
            logger.warning(f"[HLEDataset] {empty_a} 条记录的 answer 字段为空。")

    def get_problems(
        self,
        category: Optional[str] = None,
        text_only: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        problems = self.problems
        if category is not None:
            category_lower = category.lower()
            problems = [p for p in problems if p["category"].lower() == category_lower]
        if text_only:
            problems = [p for p in problems if not p["is_multimodal"]]
        if limit is not None:
            problems = problems[:limit]
        return problems

    def get_statistics(self) -> Dict:
        cats: Dict[str, int] = {}
        multimodal = 0
        for p in self.problems:
            cats[p["category"]] = cats.get(p["category"], 0) + 1
            if p["is_multimodal"]:
                multimodal += 1
        return {
            "total":      len(self.problems),
            "text_only":  len(self.problems) - multimodal,
            "multimodal": multimodal,
            "categories": cats,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HLEEvaluator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HLEEvaluator:
    """
    HLE 评测器。

    并行模式说明：
      max_workers > 1 时使用线程池。EmptyMemory 无跨题状态，并行安全。
      带 cross-trial memory 时必须保持 max_workers=1，否则并发写入会产生竞态。
    """

    def __init__(
        self,
        dataset: HLEDataset,
        solver,
        env: HLEEnv,
        output_dir: str,
        verbose: bool = False,
        max_workers: int = 1,
    ):
        self.dataset     = dataset
        self.solver      = solver
        self.env         = env
        self.output_dir  = Path(output_dir)
        self.verbose     = verbose
        self.max_workers = max_workers
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        category: Optional[str] = None,
        text_only: bool = False,
        limit: Optional[int] = None,
    ) -> Dict:
        start    = time.time()
        problems = self.dataset.get_problems(category=category, text_only=text_only, limit=limit)

        if not problems:
            logger.warning("未找到符合条件的题目，请检查 category / text_only / limit 参数。")
            return {"accuracy": 0.0, "correct": 0, "total": 0, "results": []}

        logger.info(f"共 {len(problems)} 道题目开始评测 ...")
        token_tracker.reset()

        results = (
            self._run_parallel(problems)
            if self.max_workers > 1
            else self._run_sequential(problems)
        )

        total    = len(results)
        correct  = sum(1 for r in results if r.get("correct", False))
        accuracy = correct / total if total else 0.0
        runtime  = time.time() - start
        token_summary = token_tracker.summary()

        summary = {
            "benchmark":            "hle",
            "total":                total,
            "correct":              correct,
            "accuracy":             round(accuracy, 4),
            "category_filter":      category,
            "text_only":            text_only,
            "runtime_seconds":      round(runtime, 2),
            "avg_seconds_per_item": round(runtime / total, 2) if total else 0.0,
            "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
            "token_usage":          token_summary,
            "results":              results,
        }
        self._save(summary)
        logger.info(f"评测完成：accuracy={accuracy:.2%} ({correct}/{total})  耗时={runtime:.1f}s")
        logger.info(
            f"Token 统计 — "
            f"solver={token_summary['solver']['total']}  "
            f"env={token_summary['env']['total']}  "
            f"total={token_summary['total']['total']}"
        )
        return summary

    def _run_sequential(self, problems: List[Dict]) -> List[Dict]:
        results = []
        bar = tqdm(problems, desc="HLE Eval", disable=self.verbose)
        for idx, prob in enumerate(bar):
            r = self._run_single(idx, prob)
            results.append(r)
            bar.set_postfix(acc=f"{sum(x['correct'] for x in results)}/{len(results)}")
        return results

    def _run_parallel(self, problems: List[Dict]) -> List[Dict]:
        results = [None] * len(problems)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._run_single, i, p): i for i, p in enumerate(problems)}
            for fut in tqdm(as_completed(futures), total=len(problems),
                            desc="HLE Eval (parallel)", disable=self.verbose):
                results[futures[fut]] = fut.result()
        return results

    def _run_single(self, idx: int, problem: Dict) -> Dict:
        question    = problem["question"]
        answer      = problem["answer"]
        answer_type = problem.get("answer_type", "exactMatch")
        category    = problem.get("category", "Unknown")
        prob_id     = problem.get("id", str(idx))
        t0          = time.time()

        try:
            # 图片编码（多模态题目）
            image_b64        = None
            image_media_type = "image/jpeg"
            image_preview    = problem.get("raw", {}).get("image_preview")
            if image_preview and Path(image_preview).exists():
                suffix = Path(image_preview).suffix.lower()
                image_media_type = {
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png",  ".gif":  "image/gif",
                    ".webp": "image/webp",
                }.get(suffix, "image/jpeg")
                image_b64 = base64.b64encode(
                    Path(image_preview).read_bytes()
                ).decode("utf-8")
            elif image_preview:
                logger.warning(f"image_preview 路径不存在: {image_preview}")

            self.env.set_task(problem=question, reference=answer)
            reward, done = self.solver.run_task({
                "task_main":        question,
                "task_description": question,
                "context_hint": {
                    "id":               prob_id,
                    "category":         category,
                    "answer_type":      answer_type,
                    "index":            idx,
                    "image_b64":        image_b64,
                    "image_media_type": image_media_type,
                },
            })
        except Exception as e:
            logger.error(f"题目 {idx} ({prob_id}) 执行异常: {e}", exc_info=True)
            return {
                "index": idx, "id": prob_id,
                "question": question[:200], "answer": answer,
                "answer_type": answer_type, "category": category,
                "correct": False, "reward": 0.0,
                "error": str(e), "elapsed": round(time.time() - t0, 2),
            }

        elapsed = time.time() - t0
        if self.verbose:
            print(f"\n[{idx:04d}] {'✓' if done else '✗'}  category={category}  "
                  f"reward={reward:.2f}  elapsed={elapsed:.1f}s\n  Q: {question[:100]}...")
        return {
            "index": idx, "id": prob_id,
            "question": question[:200], "answer": answer,
            "answer_type": answer_type, "category": category,
            "correct": bool(done), "reward": float(reward),
            "elapsed": round(elapsed, 2),
        }

    def _save(self, summary: Dict) -> None:
        ts   = summary["timestamp"].replace(" ", "_").replace(":", "-")
        path = self.output_dir / f"hle_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存 → {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run() — 约定入口，由 run_experiment.py 动态调用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(cfg: dict, logger: logging.Logger) -> None:
    """HLE 实验入口。组装 solver 栈并启动评测。"""

    model_cfg = cfg["model"]
    eval_cfg  = cfg.get("evaluation", {})
    out_cfg   = cfg["output"]
    exp_cfg   = cfg["experiment"]
    mem_cfg   = cfg.get("memory_config", {})
    mas_cfg   = cfg.get("mas_config", {})

    # 自定义 API 端点配置（两个 caller 共用同一套端点）
    base_url   = model_cfg.get("base_url")
    api_format = model_cfg.get("api_format", "chat")

    output_dir = REPO_ROOT / out_cfg["dir"] / cfg["benchmark"]["name"] / exp_cfg["name"]
    ts_dir = output_dir / f"hle_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    ts_dir.mkdir(parents=True, exist_ok=True)
    llm_io_logger.setup(log_dir=str(ts_dir / "llm_io"))
    # ── 数据集 ────────────────────────────────────────────────────────────────
    data_path = REPO_ROOT / cfg["benchmark"]["data_path"]
    if not data_path.exists():
        logger.error(f"HLE 数据文件不存在: {data_path}")
        sys.exit(1)

    logger.info(f"Loading HLE dataset: {data_path}")
    dataset = HLEDataset(str(data_path))
    stats   = dataset.get_statistics()
    logger.info(
        f"Dataset — total={stats['total']}  "
        f"text_only={stats['text_only']}  multimodal={stats['multimodal']}"
    )
    logger.info(f"Categories: {stats['categories']}")

    # ── MAS 栈组装 ────────────────────────────────────────────────────────────
    judge_caller = ModelCaller(
        model=model_cfg["judge"],
        role="env",
        base_url=base_url,
    )
    env = HLEEnv(judge_caller=judge_caller, verbose=out_cfg.get("verbose", False))

    solver_caller = ModelCaller(
        model=model_cfg["solver"],
        role="solver",
        base_url=base_url,
    )
    reasoning = ReasoningIO(llm_model=solver_caller)

    working_dir = mem_cfg.get("working_dir", str(output_dir / "memory_store"))
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    memory = EmptyMemory(
        namespace=mem_cfg.get("namespace", "hle_empty"),
        global_config={"working_dir": str(ts_dir / "memory_store")},
        llm_model=None,
        embedding_func=None,
    )

    solver = SingleAgentSolver()
    solver.build_system(reasoning=reasoning, solver_memory=memory, env=env, config=mas_cfg)

    # ── 评测 ──────────────────────────────────────────────────────────────────
    evaluator = HLEEvaluator(
        dataset=dataset,
        solver=solver,
        env=env,
        output_dir=str(ts_dir),
        verbose=out_cfg.get("verbose", False),
        max_workers=eval_cfg.get("max_workers", 1),
    )

    logger.info("=" * 55)
    logger.info(f"  solver     : {model_cfg['solver']}")
    logger.info(f"  judge      : {model_cfg['judge']}")
    logger.info(f"  api_format : {api_format}")
    logger.info(f"  base_url   : {base_url or 'default'}")
    logger.info(f"  category   : {eval_cfg.get('category') or 'all'}")
    logger.info(f"  text_only  : {eval_cfg.get('text_only', True)}")
    logger.info(f"  limit      : {eval_cfg.get('limit') or 'all'}")
    logger.info("=" * 55)

    results = evaluator.evaluate(
        category=eval_cfg.get("category"),
        text_only=eval_cfg.get("text_only", True),
        limit=eval_cfg.get("limit"),
    )

    logger.info("=" * 55)
    logger.info(
        f"accuracy : {results['accuracy']:.2%}  "
        f"({results['correct']}/{results['total']})"
    )
    logger.info(
        f"runtime  : {results['runtime_seconds']:.1f}s  "
        f"({results['avg_seconds_per_item']:.1f}s/item)"
    )
    logger.info("=" * 55)