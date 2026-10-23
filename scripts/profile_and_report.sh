#!/usr/bin/env bash
# Validate completions, profile canonical solutions, and generate a CSV report.
#
# Reads from:  results/<model-name>/           (produced by run.sh)
# Writes to:   results/<model-name>-validated/
#              reports/<model-name>/
#
# Prerequisites:
#   EffiLearner venv with memory_profiler, kernprof, and mprof installed.
#   (--with-metrion only) metrion installed at $METRION_ROOT
#   (default: ../metrion, i.e. a sibling of the EffiLearner directory),
#   with passwordless sudo configured for the metrion binary.
#
# Usage:
#   bash scripts/profile_and_report.sh --model-name NAME [options]
#
# Required:
#   --model-name  NAME    same label used in run.sh
#
# Options:
#   --dataset      DATASET   EffiBench|HumanEval|MBPP  (default: EffiBench)
#   --n-runs       N         profiling sweeps to average (default: 5)
#   --workers      N         parallel profiling workers  (default: 32)
#   --with-metrion           also measure code-execution energy via metrion
#   --energy-runs  N         metrion runs per entry      (default: 3)
#   --metrion-root PATH      override METRION_ROOT env var
#
# Examples:
#   # Basic profiling + report:
#   bash scripts/profile_and_report.sh --model-name qwen2.5-7b
#
#   # With code-execution energy measurement:
#   bash scripts/profile_and_report.sh --model-name qwen2.5-7b --with-metrion

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_lib.sh"

MODEL_NAME=""
DATASET="EffiBench"
N_RUNS=5
WORKERS=32
WITH_METRION=0
ENERGY_RUNS=3

while (( $# )); do
  case "$1" in
    --model-name)    MODEL_NAME=$2;   shift 2 ;;
    --dataset)       DATASET=$2;      shift 2 ;;
    --n-runs)        N_RUNS=$2;       shift 2 ;;
    --workers)       WORKERS=$2;      shift 2 ;;
    --with-metrion)  WITH_METRION=1;  shift ;;
    --energy-runs)   ENERGY_RUNS=$2;  shift 2 ;;
    --metrion-root)  METRION_ROOT=$2; METRION_DB_DIR="$METRION_ROOT/dbs"; METRION_BASE_CFG="$METRION_ROOT/configs/config.yaml"; shift 2 ;;
    -h|--help)       sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL_NAME"  ]] || { echo "Error: --model-name is required" >&2; exit 2; }
[[ -x "$EFFI_VENV_PY" ]] || { echo "Error: venv python not found at $EFFI_VENV_PY" >&2; exit 1; }

RESULTS_DIR="$EFFI_ROOT/results/$MODEL_NAME"
VALIDATED_DIR="$EFFI_ROOT/results/${MODEL_NAME}-validated"

[[ -d "$RESULTS_DIR" ]] || {
  echo "Error: results directory not found: $RESULTS_DIR" >&2
  echo "  Run scripts/run.sh --model-name $MODEL_NAME first." >&2
  exit 1
}

echo "============================================================"
echo "  model-name:   $MODEL_NAME"
echo "  dataset:      $DATASET"
echo "  n-runs:       $N_RUNS"
echo "  workers:      $WORKERS"
echo "  with-metrion: $WITH_METRION"
echo "  results ←     $RESULTS_DIR"
echo "  validated →   $VALIDATED_DIR"
echo "============================================================"

cd "$EFFI_SRC"

echo ""
echo "=== Step 1: validate completions ==="
files=()
base="$RESULTS_DIR/${DATASET}_vllm_${MODEL_NAME}.json"
[[ -f "$base" ]] && files+=("$base")
for i in 1 2 3 4 5; do
  f="$RESULTS_DIR/${DATASET}_vllm_${MODEL_NAME}_${i}.json"
  [[ -f "$f" ]] && files+=("$f")
done

(( ${#files[@]} > 0 )) || {
  echo "Error: no result JSON files found in $RESULTS_DIR" >&2
  echo "  Expected: ${DATASET}_vllm_${MODEL_NAME}.json (and epoch variants)" >&2
  exit 1
}

echo "  Validating ${#files[@]} file(s)..."
"$EFFI_VENV_PY" validate_completions.py \
  --save \
  --workers "$WORKERS" \
  "${files[@]}"

echo ""
echo "=== Step 2: profile canonical solutions ==="
"$EFFI_VENV_PY" profile_canonical.py \
  --model-name  "$MODEL_NAME" \
  --dataset     "$DATASET" \
  --results-dir "$VALIDATED_DIR" \
  --workers     "$WORKERS" \
  --n-runs      "$N_RUNS"

echo ""
echo "=== Step 3: generate report ==="
CANONICAL_FILE="$VALIDATED_DIR/canonical_${DATASET}_vllm_${MODEL_NAME}.json"
[[ -f "$CANONICAL_FILE" ]] || CANONICAL_FILE="$RESULTS_DIR/canonical_${DATASET}_vllm_${MODEL_NAME}.json"

"$EFFI_VENV_PY" vllm_report.py \
  --model-name     "$MODEL_NAME" \
  --dataset        "$DATASET" \
  --results-dir    "$VALIDATED_DIR" \
  --canonical-file "$CANONICAL_FILE"

if (( WITH_METRION )); then
  echo ""
  echo "=== Step 4: code-execution energy measurement (metrion) ==="
  [[ -f "$METRION_BASE_CFG" ]] || {
    echo "Error: metrion config not found at $METRION_BASE_CFG" >&2
    echo "  Clone metrion into $(dirname "$EFFI_ROOT")/metrion or set --metrion-root." >&2
    exit 1
  }

  # Collect validated baseline + epoch JSONs; skip canonical/overhead/energy files.
  energy_files=()
  shopt -s nullglob
  for j in "$VALIDATED_DIR"/${DATASET}_vllm_${MODEL_NAME}.json \
            "$VALIDATED_DIR"/${DATASET}_vllm_${MODEL_NAME}_*.json; do
    bn=$(basename "$j")
    suffix="${bn#${DATASET}_vllm_${MODEL_NAME}}"
    suffix="${suffix%.json}"
    # Only accept: no suffix (baseline) or _<digits> (epoch N)
    [[ -z "$suffix" || "$suffix" =~ ^_[0-9]+$ ]] || continue
    energy_files+=("$j")
  done
  shopt -u nullglob

  (( ${#energy_files[@]} > 0 )) || {
    echo "Warning: no validated epoch JSONs found for energy measurement." >&2
    echo "  Skipping metrion step." >&2
  }

  if (( ${#energy_files[@]} > 0 )); then
    echo "  Running energy measurement on ${#energy_files[@]} file(s)..."
    "$EFFI_VENV_PY" run_effibench_energy.py \
      --input-files "${energy_files[@]}" \
      --config      "$METRION_BASE_CFG" \
      --runs        "$ENERGY_RUNS" \
      --log-dir     "$EFFI_LOGS"
    echo "  Energy results written alongside validated JSONs (*_energy_results.json)."
  fi
fi

echo ""
echo "Done."
echo "  Validated results: $VALIDATED_DIR"
echo "  Reports:           $EFFI_ROOT/reports/$MODEL_NAME/"
