#!/usr/bin/env bash
# experiments/scripts/run_HLE.sh
#
# 用法：
#   # 正式跑
#   nohup bash run_HLE.sh > 2026_3_10_test_hle_noagent_emptymemory.log 2>&1 &
#
#   # 调试（只跑 10 道题，输出详细日志）
#   nohup bash run_HLE.sh --debug > 2026_3_10_test_hle_noagent_emptymemory.log 2>&1 &

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# ── API 配置 ──────────────────────────────────────────────────────────────────
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""

# ── 路径 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BENCHMARK_CFG="$REPO_ROOT/experiments/configs/benchmarks/hle.yaml"
METHOD_CFG="$REPO_ROOT/experiments/configs/methods/single_agent_emptymemory.yaml"

# ── 参数解析 ──────────────────────────────────────────────────────────────────
DEBUG=false
for arg in "$@"; do
    case $arg in
        --debug) DEBUG=true ;;
    esac
done

# ── 运行 ──────────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

if [ "$DEBUG" = true ]; then
    echo "[run_HLE] DEBUG mode: limit=10, verbose=true"
    python experiments/run_experiment.py \
        --benchmark "$BENCHMARK_CFG" \
        --method    "$METHOD_CFG" \
        --override  evaluation.limit=10 output.verbose=true \
                    model.base_url=https://gmn.chuangzuoli.com 
else
    echo "[run_HLE] Full run"
    python experiments/run_experiment.py \
        --benchmark "$BENCHMARK_CFG" \
        --method    "$METHOD_CFG" \
        --override  model.base_url=https://gmn.chuangzuoli.com 
fi