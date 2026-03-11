# experiments/run_experiment.py
#!/usr/bin/env python3
"""
experiments/run_experiment.py

统一实验入口。本文件不感知任何具体 benchmark：
    - 解析 CLI 参数
    - 合并 benchmark config + method config
    - 动态加载 experiments/benchmarks/{name}/runner.py 并调用 run(cfg, logger)

新增 benchmark 只需在 experiments/benchmarks/ 下建目录并实现 runner.py，
本文件无需任何改动。

用法：
    # 新版组件化配置（推荐）
    python experiments/run_experiment.py \
        --env    experiments/configs/envs/hle.yaml \
        --solver experiments/configs/solver/single_agent.yaml \
        --tool   experiments/configs/tool/default.yaml \
        --memory experiments/configs/memory/empty.yaml

    # 旧版双文件配置（兼容）
    python experiments/run_experiment.py \
        --benchmark experiments/configs/benchmarks/hle.yaml \
        --method    experiments/configs/methods/single_agent_emptymemory.yaml

    # 调试：临时覆盖参数
    python experiments/run_experiment.py \
        --benchmark experiments/configs/benchmarks/hle.yaml \
        --method    experiments/configs/methods/single_agent_emptymemory.yaml \
        --override  evaluation.limit=10 model.solver=gpt-4o-mini
"""

import argparse 
import importlib.util
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, benchmark_name: str, exp_name: str) -> logging.Logger:
    """输出到控制台（INFO+）和日志文件（DEBUG+）。"""
    ts       = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / benchmark_name / exp_name / f"{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("emams")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log → {log_path}")
    return logger


# ── Config 工具 ───────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, overlay: dict) -> dict:
    """
    递归合并 dict，overlay 覆盖 base 的同名键。
    """
    merged = deepcopy(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_configs(benchmark_cfg: dict, method_cfg: dict) -> dict:
    """
    合并两份 config。优先级：benchmark > method。
    benchmark 中已有的顶层键不会被 method 覆盖；
    method 中的专属键（mas_config、memory_config、experiment）正常写入。
    """
    return {**method_cfg, **benchmark_cfg}


def compose_component_configs(
    env_cfg: dict,
    solver_cfg: dict,
    tool_cfg: dict,
    memory_cfg: dict,
) -> dict:
    """
    新版四段式配置组合：
      env + solver + tool + memory -> runner 兼容配置
    """
    cfg = {}
    for piece in (memory_cfg, tool_cfg, solver_cfg, env_cfg):
        cfg = deep_merge(cfg, piece or {})

    cfg.setdefault("benchmark", {})
    cfg.setdefault("model", {})
    cfg.setdefault("experiment", {})
    cfg.setdefault("memory_config", {})
    cfg.setdefault("mas_config", {})
    cfg.setdefault("tool_config", {})

    # 将 tool_config.enable_tools 同步到 mas_config（若未显式设置）
    if "enable_tools" in cfg["tool_config"] and "enable_tools" not in cfg["mas_config"]:
        cfg["mas_config"]["enable_tools"] = bool(cfg["tool_config"]["enable_tools"])
    if "max_tool_steps" in cfg["tool_config"] and "max_tool_steps" not in cfg["mas_config"]:
        cfg["mas_config"]["max_tool_steps"] = cfg["tool_config"]["max_tool_steps"]
    if "require_final_answer" in cfg["tool_config"] and "require_final_answer" not in cfg["mas_config"]:
        cfg["mas_config"]["require_final_answer"] = bool(cfg["tool_config"]["require_final_answer"])

    # 缺省实验名自动生成，避免必须手写
    exp = cfg["experiment"]
    if "name" not in exp or not exp["name"]:
        env_name = cfg["benchmark"].get("name", "env")
        framework = exp.get("agent_framework", "solver")
        mem = exp.get("memory_method", "memory")
        tool_mode = "tools" if cfg["mas_config"].get("enable_tools") else "notools"
        exp["name"] = f"{env_name}_{framework}_{mem}_{tool_mode}"

    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """key.sub.leaf=value 形式的临时覆盖，value 经 YAML 解析（支持 int/float/bool/null）。"""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override 格式错误（需要 key=value）: {item}")
        key_path, raw = item.split("=", 1)
        val  = yaml.safe_load(raw)
        node = cfg
        for k in key_path.split(".")[:-1]:
            node = node.setdefault(k, {})
        node[key_path.split(".")[-1]] = val
    return cfg


# ── 动态 Runner 加载 ──────────────────────────────────────────────────────────

def load_runner(benchmark_name: str):
    """
    动态加载 experiments/benchmarks/{benchmark_name}/runner.py。

    约定：每个 runner.py 必须实现顶层函数
        def run(cfg: dict, logger: logging.Logger) -> None

    Raises:
        FileNotFoundError : runner.py 不存在。
        AttributeError    : runner.py 未实现 run() 函数。
    """
    runner_path = (
        REPO_ROOT / "experiments" / "benchmarks" / benchmark_name / "runner.py"
    )
    if not runner_path.exists():
        raise FileNotFoundError(
            f"找不到 benchmark runner: {runner_path}\n"
            f"请在 experiments/benchmarks/{benchmark_name}/ 下创建 runner.py，"
            f"并实现 run(cfg, logger) 函数。"
        )

    spec   = importlib.util.spec_from_file_location(f"runner_{benchmark_name}", runner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "run"):
        raise AttributeError(
            f"{runner_path} 中未找到 run(cfg, logger) 函数。\n"
            f"请确认 runner.py 顶层实现了 def run(cfg: dict, logger: logging.Logger) -> None。"
        )
    return module


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agent-Evolving-Library experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--benchmark", help="旧版: benchmark config yaml 路径")
    parser.add_argument("--method",    help="旧版: method config yaml 路径")
    parser.add_argument("--env",       help="新版: env config yaml 路径")
    parser.add_argument("--solver",    help="新版: solver config yaml 路径")
    parser.add_argument("--tool",      help="新版: tool config yaml 路径")
    parser.add_argument("--memory",    help="新版: memory config yaml 路径")
    parser.add_argument(
        "--override", nargs="*", default=[], metavar="key=value",
        help="临时覆盖 config 中的值（支持嵌套路径），如 evaluation.limit=10",
    )
    args = parser.parse_args()

    use_old = bool(args.benchmark or args.method)
    use_new = bool(args.env or args.solver or args.tool or args.memory)

    if use_old and use_new:
        raise ValueError("请二选一：使用旧版 --benchmark/--method 或新版 --env/--solver/--tool/--memory。")

    if use_new:
        required = {"--env": args.env, "--solver": args.solver, "--tool": args.tool, "--memory": args.memory}
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"新版配置缺少参数: {', '.join(missing)}")
        cfg = compose_component_configs(
            env_cfg=load_yaml(args.env),
            solver_cfg=load_yaml(args.solver),
            tool_cfg=load_yaml(args.tool),
            memory_cfg=load_yaml(args.memory),
        )
    else:
        if not args.benchmark or not args.method:
            raise ValueError("旧版配置需要同时提供 --benchmark 和 --method。")
        cfg = merge_configs(load_yaml(args.benchmark), load_yaml(args.method))

    if args.override:
        cfg = apply_overrides(cfg, args.override)

    benchmark_name = cfg["benchmark"]["name"]
    exp_name       = cfg["experiment"]["name"]
    log_dir        = REPO_ROOT / "experiments" / "logs"

    logger = setup_logging(log_dir, benchmark_name, exp_name)

    logger.info("=" * 55)
    logger.info(f"Experiment : {exp_name}")
    logger.info(f"Benchmark  : {benchmark_name}")
    logger.info(f"Memory     : {cfg['experiment']['memory_method']}")
    logger.info(f"Framework  : {cfg['experiment']['agent_framework']}")
    logger.info(f"Solver     : {cfg['model']['solver']}")
    logger.info(f"Judge      : {cfg['model']['judge']}")
    logger.info("=" * 55)

    try:
        runner = load_runner(benchmark_name)
    except (FileNotFoundError, AttributeError) as e:
        logger.error(str(e))
        sys.exit(1)

    runner.run(cfg, logger)


if __name__ == "__main__":
    main()
