#!/usr/bin/env bash
# run_all_experiments.sh — loops models × run_tags serially.
# Run this *inside* the long-lived `orchestrator` screen so laptop sleep
# doesn't kill it:
#
#   screen -dmS orchestrator bash -lc '/path/to/EffiLearner/scripts/run_all_experiments.sh'
#
# Flags:
#   (no flag)        all 44 runs — 11 models × 4 tags (default, t0.5, ctx65536, mt1024)
#   --repeat N       run the full 44-run set N times; outputs suffixed _r1, _r2, ... _rN
#   --force / -f     overwrite existing results
#
# Or pick explicit lists:
#   SWEEP_MODELS="qwen3.5-4b qwen3.5-9b" SWEEP_TAGS="t0.5"  ./run_all_experiments.sh
#
# Per-experiment logs: EffiLearner/logs/<model><suffix>[_rN].{vllm,metrion,infer}.log
# Sweep log:          EffiLearner/logs/sweep.log

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/experiment_lib.sh"

REPEAT=1
FORCE_ARG=()
METRION_ARG=()
while (( $# )); do
  case "$1" in
    --repeat)      REPEAT=$2; shift 2 ;;
    --repeat=*)    REPEAT="${1#--repeat=}"; shift ;;
    --force|-f)    FORCE_ARG=(--force); shift ;;
    --with-metrion) METRION_ARG=(--with-metrion); shift ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *)             echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# All 44: default tag first, then the 3 parameter variants.
ALL_TAGS=(default "${RUN_TAGS[@]}")

SWEEP_LOG=$EFFI_LOGS/sweep.log
exec > >(tee -a "$SWEEP_LOG") 2>&1

# shellcheck disable=SC2206
MODELS=(${SWEEP_MODELS:-${MODEL_KEYS[@]}})
# shellcheck disable=SC2206
TAGS=(${SWEEP_TAGS:-${ALL_TAGS[@]}})

failures=()

for (( rep=1; rep<=REPEAT; rep++ )); do
  REPEAT_ARG=()
  if (( REPEAT > 1 )); then
    REPEAT_ARG=(--repeat-id "$rep")
  fi

  echo "============================================================"
  echo "Sweep started: $(date -Iseconds)  repeat=$rep/$REPEAT"
  echo "Models: ${MODELS[*]}"
  echo "Tags:   ${TAGS[*]}"
  echo "============================================================"

  for model in "${MODELS[@]}"; do
    for tag in "${TAGS[@]}"; do
      echo ""
      echo "###### $(date -Iseconds)  $model  $tag  rep=$rep/$REPEAT ######"
      if bash "$HERE/run_experiment.sh" "${FORCE_ARG[@]}" "${REPEAT_ARG[@]}" "${METRION_ARG[@]}" "$model" "$tag"; then
        echo "###### OK:   $model $tag rep=$rep ######"
      else
        rc=$?
        echo "###### FAIL($rc): $model $tag rep=$rep ######"
        failures+=("$model/$tag/r${rep}(rc=$rc)")
      fi
      sleep 10  # breathing room between experiments
    done
  done
done

echo ""
echo "============================================================"
echo "Sweep finished: $(date -Iseconds)  repeats=$REPEAT"
if (( ${#failures[@]} == 0 )); then
  echo "All experiments succeeded."
else
  echo "Failures: ${#failures[@]}"
  printf '  %s\n' "${failures[@]}"
fi
echo "============================================================"
