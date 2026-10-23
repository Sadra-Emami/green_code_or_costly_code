#!/usr/bin/env bash
# Full pipeline smoke test: generation → optimization → profiling → energy.
# Uses a small slice of EffiBench (see LIMIT below) with a few epochs and
# minimal profiling runs. All outputs land under results/smoke-test/,
# reports/smoke-test/, logs/.
#
# Prerequisites:
#   - vLLM already running and serving $CHECKPOINT, e.g.:
#       vllm serve "$CHECKPOINT" --port 8000 \
#           --tensor-parallel-size 2 \
#           --default-chat-template-kwargs '{"enable_thinking":false}'
#     CHECKPOINT defaults to a local checkpoint under MODEL_ROOT (a sibling
#     "models" directory, see below) but can be any identifier your vLLM
#     instance was launched with — override it with the CHECKPOINT env var.
#   - EffiLearner venv installed (pip install -r requirements.txt)
#   - metrion installed into the venv (pip install -e ../metrion)
#   - passwordless sudo for the metrion binary
#
# All paths below are derived relative to this script's location and the
# repo checkout, so this works out of the box after `git clone`. Override
# any of MODEL_ROOT / CHECKPOINT / METRION_ROOT via the environment if your
# layout differs.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_lib.sh"
source "$HERE/experiment_lib.sh"

# metrion's checkout is sometimes named "metrion", sometimes "metrion-internal"
# depending on how it was cloned; fall back to whichever sibling exists.
if [[ ! -d "$METRION_ROOT" ]]; then
    _alt="$(dirname "$EFFI_ROOT")/metrion-internal"
    if [[ -d "$_alt" ]]; then
        export METRION_ROOT="$_alt"
        export METRION_BASE_CFG="$METRION_ROOT/configs/config.yaml"
    fi
fi

MODEL_NAME="smoke-test"
MODEL_ROOT="${MODEL_ROOT:-$(dirname "$EFFI_ROOT")/models}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/Qwen/Qwen3.5-4B}"
DATASET="EffiBench"
LIMIT=10
EPOCHS=5
PROFILE_RUNS=1
ENERGY_RUNS=1
WORKERS=4
PORT=8000
CONCURRENCY=8

RESULTS_DIR="$EFFI_ROOT/results/$MODEL_NAME"
VALIDATED_DIR="$EFFI_ROOT/results/${MODEL_NAME}-validated"
REPORT_DIR="$EFFI_ROOT/reports/$MODEL_NAME"
mkdir -p "$RESULTS_DIR" "$VALIDATED_DIR" "$REPORT_DIR" "$EFFI_LOGS"

log() { echo "[smoke-test] $*"; }

log "============================================================"
log "  model:        $MODEL_NAME"
log "  checkpoint:   $CHECKPOINT"
log "  dataset:      $DATASET  (limit=$LIMIT)"
log "  epochs:       $EPOCHS"
log "  profile-runs: $PROFILE_RUNS"
log "  energy-runs:  $ENERGY_RUNS"
log "  results  →    $RESULTS_DIR"
log "  validated →   $VALIDATED_DIR"
log "  reports  →    $REPORT_DIR"
log "============================================================"

cd "$EFFI_SRC"

# ── Step 1: generation ────────────────────────────────────────────────────────
log ""
log "=== Step 1: code generation ==="
"$EFFI_VENV_PY" vllm_generation.py \
    --checkpoint  "$CHECKPOINT" \
    --model-name  "$MODEL_NAME" \
    --dataset     "$DATASET" \
    --port        "$PORT" \
    --concurrency "$CONCURRENCY" \
    --limit       "$LIMIT"

# ── Step 2: self-optimization ─────────────────────────────────────────────────
log ""
log "=== Step 2: self-optimization ($EPOCHS epoch) ==="
"$EFFI_VENV_PY" vllm_effilearner.py \
    --checkpoint   "$CHECKPOINT" \
    --model-name   "$MODEL_NAME" \
    --dataset      "$DATASET" \
    --port         "$PORT" \
    --concurrency  "$CONCURRENCY" \
    --epoch        "$EPOCHS" \
    --workers      "$WORKERS" \
    --profile-runs "$PROFILE_RUNS"

# ── Step 3: validate completions ──────────────────────────────────────────────
log ""
log "=== Step 3: validate completions ==="
files=()
base="$RESULTS_DIR/${DATASET}_vllm_${MODEL_NAME}.json"
[[ -f "$base" ]] && files+=("$base")
for i in $(seq 1 "$EPOCHS"); do
    f="$RESULTS_DIR/${DATASET}_vllm_${MODEL_NAME}_${i}.json"
    [[ -f "$f" ]] && files+=("$f")
done
log "  Validating ${#files[@]} file(s)..."
"$EFFI_VENV_PY" validate_completions.py \
    --save \
    --workers "$WORKERS" \
    "${files[@]}"

# ── Step 4: profile canonical solutions ───────────────────────────────────────
log ""
log "=== Step 4: profile canonical solutions ==="
"$EFFI_VENV_PY" profile_canonical.py \
    --model-name  "$MODEL_NAME" \
    --dataset     "$DATASET" \
    --results-dir "$VALIDATED_DIR" \
    --workers     "$WORKERS" \
    --n-runs      "$PROFILE_RUNS"

# ── Step 5: generate report ───────────────────────────────────────────────────
log ""
log "=== Step 5: generate report ==="
CANONICAL_FILE="$VALIDATED_DIR/canonical_${DATASET}_vllm_${MODEL_NAME}.json"
[[ -f "$CANONICAL_FILE" ]] || CANONICAL_FILE="$RESULTS_DIR/canonical_${DATASET}_vllm_${MODEL_NAME}.json"
"$EFFI_VENV_PY" vllm_report.py \
    --model-name     "$MODEL_NAME" \
    --dataset        "$DATASET" \
    --results-dir    "$VALIDATED_DIR" \
    --canonical-file "$CANONICAL_FILE"

# ── Step 6: energy measurement ────────────────────────────────────────────────
log ""
log "=== Step 6: energy measurement (metrion) ==="
energy_files=()
shopt -s nullglob
for j in "$VALIDATED_DIR"/${DATASET}_vllm_${MODEL_NAME}.json \
          "$VALIDATED_DIR"/${DATASET}_vllm_${MODEL_NAME}_*.json; do
    bn=$(basename "$j")
    suffix="${bn#${DATASET}_vllm_${MODEL_NAME}}"
    suffix="${suffix%.json}"
    [[ -z "$suffix" || "$suffix" =~ ^_[0-9]+$ ]] || continue
    energy_files+=("$j")
done
shopt -u nullglob

if [[ ! -f "$METRION_BASE_CFG" ]]; then
    log "WARNING: metrion config not found at $METRION_BASE_CFG — skipping energy step."
    log "  (set METRION_ROOT to point at your metrion checkout if it lives elsewhere)"
elif (( ${#energy_files[@]} == 0 )); then
    log "WARNING: no validated epoch JSONs found — skipping energy step."
else
    log "  Running energy measurement on ${#energy_files[@]} file(s)..."
    "$EFFI_VENV_PY" run_effibench_energy.py \
        --input-files "${energy_files[@]}" \
        --config      "$METRION_BASE_CFG" \
        --runs        "$ENERGY_RUNS" \
        --log-dir     "$EFFI_LOGS" \
        --limit       "$LIMIT"
    log "  Energy results written alongside validated JSONs (*_energy_results.json)."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
log ""
log "============================================================"
log "  SMOKE TEST COMPLETE"
log "  Generation results:  $RESULTS_DIR"
log "  Validated results:   $VALIDATED_DIR"
log "  Report:              $REPORT_DIR"
log "  Logs:                $EFFI_LOGS"
log "  Energy results:      ${VALIDATED_DIR}/*_energy_results.json"
log "============================================================"
