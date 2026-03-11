#!/usr/bin/env bash
# experiments/scripts/run_HLE.sh
#
# 用法：
#   # 正式跑
#   nohup bash run_HLE.sh > 2026_3_10_test_hle_noagent_emptymemory.log 2>&1 &
#
#   # 调试（只跑 10 道题，输出详细日志）
#   nohup bash run_HLE.sh --debug > 2026_3_10_test_hle_noagent_emptymemory.log 2>&1 &
#
#   # 切换搜索与网页读取 provider
#   nohup bash run_HLE.sh --search-provider searxng --access-provider crawl4ai > run_hle.log 2>&1 &

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# ── API 配置 ──────────────────────────────────────────────────────────────────
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""

# ── 路径 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_CFG="$REPO_ROOT/experiments/configs/envs/hle.yaml"
SOLVER_CFG="$REPO_ROOT/experiments/configs/solver/single_agent.yaml"
TOOL_CFG="$REPO_ROOT/experiments/configs/tool/default.yaml"
MEMORY_CFG="$REPO_ROOT/experiments/configs/memory/empty.yaml"

# ── 参数解析 ──────────────────────────────────────────────────────────────────
DEBUG=false
SEARCH_PROVIDER=""
ACCESS_PROVIDER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG=true
            shift
            ;;
        --search-provider=*)
            SEARCH_PROVIDER="${1#*=}"
            shift
            ;;
        --access-provider=*)
            ACCESS_PROVIDER="${1#*=}"
            shift
            ;;
        --search-provider)
            SEARCH_PROVIDER="${2:-}"
            shift 2
            ;;
        --access-provider)
            ACCESS_PROVIDER="${2:-}"
            shift 2
            ;;
        *)
            echo "[run_HLE] Unknown arg: $1"
            exit 1
            ;;
    esac
done

# ── 运行 ──────────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

if [ "$DEBUG" = true ]; then
    echo "[run_HLE] DEBUG mode: limit=10, verbose=true"
    OVERRIDES=(
        "evaluation.limit=10"
        "output.verbose=true"
        "model.base_url=https://gmn.chuangzuoli.com"
    )
else
    echo "[run_HLE] Full run"
    OVERRIDES=("model.base_url=https://gmn.chuangzuoli.com")
fi

if [ -n "$SEARCH_PROVIDER" ]; then
    OVERRIDES+=("tool_config.web_search_provider=$SEARCH_PROVIDER")
    echo "[run_HLE] web_search_provider=$SEARCH_PROVIDER"
fi
if [ -n "$ACCESS_PROVIDER" ]; then
    OVERRIDES+=("tool_config.web_access_provider=$ACCESS_PROVIDER")
    echo "[run_HLE] web_access_provider=$ACCESS_PROVIDER"
fi

python experiments/run_experiment.py \
    --env    "$ENV_CFG" \
    --solver "$SOLVER_CFG" \
    --tool   "$TOOL_CFG" \
    --memory "$MEMORY_CFG" \
    --override "${OVERRIDES[@]}"
