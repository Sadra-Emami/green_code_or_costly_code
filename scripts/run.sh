#!/usr/bin/env bash
# Generate initial code completions and run EffiLearner self-optimization.
#
# Prerequisites:
#   1. A running vLLM server:
#        vllm serve <checkpoint> --port <port> [extra vllm flags]
#   2. EffiLearner venv at .venv/ (pip install -r requirements.txt).
#   3. (--with-metrion only) metrion installed at $METRION_ROOT
#      (default: ../metrion, i.e. a sibling of the EffiLearner directory),
#      with passwordless sudo configured for the metrion binary.
#
# Usage:
#   bash scripts/run.sh --model-name NAME --checkpoint HF_PATH [options]
#
# Required:
#   --model-name  NAME    label used for result files and directories
#   --checkpoint  PATH    HF model path/name as it is served by vLLM
#
# Options:
#   --dataset      DATASET   EffiBench|HumanEval|MBPP  (default: EffiBench)
#   --epochs       N         self-optimization epochs   (default: 5)
#   --port         PORT      vLLM server port            (default: 8000)
#   --concurrency  N         parallel requests to vLLM   (default: 32)
#   --max-tokens   N         max output tokens per call  (default: 2048)
#   --temperature  F         sampling temperature        (default: 0.0)
#   --profile-runs N         profiling sweeps averaged   (default: 5)
#   --workers      N         parallel profiling workers  (default: 32)
#   --vllm-timeout N         seconds to wait for vLLM    (default: 1800)
#   --with-metrion           measure inference energy via metrion
#   --metrion-root PATH      override METRION_ROOT env var
#
# Examples:
#   # Basic run:
#   bash scripts/run.sh \
#     --model-name qwen2.5-7b \
#     --checkpoint Qwen/Qwen2.5-Coder-7B-Instruct
#
#   # With energy measurement:
#   bash scripts/run.sh \
#     --model-name qwen2.5-7b \
#     --checkpoint Qwen/Qwen2.5-Coder-7B-Instruct \
#     --with-metrion

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_lib.sh"

MODEL_NAME=""
CHECKPOINT=""
DATASET="EffiBench"
EPOCHS=5
PORT=8000
CONCURRENCY=32
MAX_TOKENS=2048
TEMPERATURE=0.0
PROFILE_RUNS=5
WORKERS=32
VLLM_TIMEOUT=1800
WITH_METRION=0

while (( $# )); do
  case "$1" in
    --model-name)    MODEL_NAME=$2;    shift 2 ;;
    --checkpoint)    CHECKPOINT=$2;    shift 2 ;;
    --dataset)       DATASET=$2;       shift 2 ;;
    --epochs)        EPOCHS=$2;        shift 2 ;;
    --port)          PORT=$2;          shift 2 ;;
    --concurrency)   CONCURRENCY=$2;   shift 2 ;;
    --max-tokens)    MAX_TOKENS=$2;    shift 2 ;;
    --temperature)   TEMPERATURE=$2;   shift 2 ;;
    --profile-runs)  PROFILE_RUNS=$2;  shift 2 ;;
    --workers)       WORKERS=$2;       shift 2 ;;
    --vllm-timeout)  VLLM_TIMEOUT=$2;  shift 2 ;;
    --with-metrion)  WITH_METRION=1;   shift ;;
    --metrion-root)  METRION_ROOT=$2; METRION_DB_DIR="$METRION_ROOT/dbs"; METRION_BASE_CFG="$METRION_ROOT/configs/config.yaml"; shift 2 ;;
    -h|--help)       sed -n '2,43p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL_NAME"  ]] || { echo "Error: --model-name is required"  >&2; exit 2; }
[[ -n "$CHECKPOINT"  ]] || { echo "Error: --checkpoint is required"   >&2; exit 2; }
[[ -x "$EFFI_VENV_PY" ]] || { echo "Error: venv python not found at $EFFI_VENV_PY" >&2; exit 1; }

RESULTS_DIR="$EFFI_ROOT/results/$MODEL_NAME"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  model-name:   $MODEL_NAME"
echo "  checkpoint:   $CHECKPOINT"
echo "  dataset:      $DATASET"
echo "  epochs:       $EPOCHS"
echo "  port:         $PORT"
echo "  with-metrion: $WITH_METRION"
echo "  results →     $RESULTS_DIR"
echo "============================================================"

echo ""
echo "Waiting for vLLM on port $PORT ..."
wait_vllm_ready "$PORT" "$VLLM_TIMEOUT"
echo "vLLM is ready."

METR_SESS="metrion_effi_${MODEL_NAME//[^A-Za-z0-9_]/_}"
METR_CFG=""
DB_PATH=""

if (( WITH_METRION )); then
  [[ -f "$METRION_BASE_CFG" ]] || {
    echo "Error: metrion config not found at $METRION_BASE_CFG" >&2
    echo "  Clone metrion into $(dirname "$EFFI_ROOT")/metrion or set --metrion-root." >&2
    exit 1
  }
  mkdir -p "$METRION_DB_DIR"
  DB_PATH="$METRION_DB_DIR/${MODEL_NAME}.db"
  METR_LOG="$EFFI_LOGS/${MODEL_NAME}.metrion.log"
  METR_CFG=$(write_metrion_config "$DB_PATH")
  METR_CMD="cd $METRION_ROOT && source .venv/bin/activate && METRION_BIN=\$(which metrion) && sudo -n \"\$METRION_BIN\" monitor -c $METR_CFG 2>&1 | tee $METR_LOG"
  screen_start "$METR_SESS" "$METR_CMD"
  sleep 5
  if ! screen_alive "$METR_SESS"; then
    echo "Error: metrion screen died immediately." >&2
    echo "  Check sudo -n permissions and $METR_LOG" >&2
    [[ -n "$METR_CFG" ]] && rm -f "$METR_CFG"
    exit 1
  fi
  echo "metrion started → $DB_PATH"
fi

cleanup() {
  if (( WITH_METRION )); then
    screen_send_ctrlc "$METR_SESS"
    sleep 5
    screen_kill "$METR_SESS"
    [[ -n "$METR_CFG" ]] && rm -f "$METR_CFG"
  fi
}
trap cleanup EXIT

cd "$EFFI_SRC"

echo ""
echo "=== Step 1: code generation ==="
"$EFFI_VENV_PY" vllm_generation.py \
  --checkpoint   "$CHECKPOINT" \
  --model-name   "$MODEL_NAME" \
  --dataset      "$DATASET" \
  --port         "$PORT" \
  --concurrency  "$CONCURRENCY" \
  --max-tokens   "$MAX_TOKENS" \
  --temperature  "$TEMPERATURE"

echo ""
echo "=== Step 2: self-optimization ($EPOCHS epochs) ==="
"$EFFI_VENV_PY" vllm_effilearner.py \
  --checkpoint   "$CHECKPOINT" \
  --model-name   "$MODEL_NAME" \
  --dataset      "$DATASET" \
  --port         "$PORT" \
  --concurrency  "$CONCURRENCY" \
  --max-tokens   "$MAX_TOKENS" \
  --temperature  "$TEMPERATURE" \
  --epoch        "$EPOCHS" \
  --workers      "$WORKERS" \
  --profile-runs "$PROFILE_RUNS"

if (( WITH_METRION )); then
  echo ""
  echo "=== Stopping metrion ==="
  screen_send_ctrlc "$METR_SESS"
  for i in {1..30}; do
    screen_alive "$METR_SESS" || break
    sleep 2
  done
  screen_kill "$METR_SESS"
  trap - EXIT
  [[ -n "$METR_CFG" ]] && rm -f "$METR_CFG"
  if [[ -f "$DB_PATH" ]]; then
    echo "Inference energy DB: $DB_PATH"
  else
    echo "Warning: metrion DB not found at $DB_PATH" >&2
  fi
fi

echo ""
echo "Done. Results written to $RESULTS_DIR"
echo "Next step: bash scripts/profile_and_report.sh --model-name $MODEL_NAME"
