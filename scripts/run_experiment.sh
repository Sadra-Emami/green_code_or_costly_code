#!/usr/bin/env bash
# run_experiment.sh MODEL_KEY RUN_TAG
#
# Runs one (model, run_tag) experiment end-to-end in 3 detached screens:
#   vllm_<tag>       — vLLM OpenAI-compatible server
#   metrion_<tag>    — sudo metrion monitor (writes to metrion/dbs/<name>.db)
#   infer_<tag>      — vllm_generation → vllm_effilearner → profile_canonical → vllm_report
#
# The infer screen touches $EFFI_FLAGS/<name>.done when finished; this orchestrator
# waits on that file, then stops metrion (Ctrl+C), then stops vLLM.
# Intended to be called from run_all_experiments.sh, itself running inside a
# long-lived `orchestrator` screen so laptop sleep doesn't kill the sweep.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/experiment_lib.sh"

FORCE=0
REPEAT_ID=""
TEST_MODE=0
WITH_METRION=0
ARGS=()
while (( $# )); do
  case "$1" in
    --force|-f)      FORCE=1; shift ;;
    --repeat-id)     REPEAT_ID=$2; shift 2 ;;
    --repeat-id=*)   REPEAT_ID="${1#--repeat-id=}"; shift ;;
    --test)          TEST_MODE=1; shift ;;
    --with-metrion)  WITH_METRION=1; shift ;;
    *)               ARGS+=("$1"); shift ;;
  esac
done
if (( ${#ARGS[@]} != 2 )); then
  echo "usage: $0 [--force] [--repeat-id N] [--test] [--with-metrion] MODEL_KEY RUN_TAG" >&2
  exit 2
fi

MODEL_KEY=${ARGS[0]}
RUN_TAG=${ARGS[1]}
CKPT=$(checkpoint_of "$MODEL_KEY")
SERVER_FLAGS=$(server_flags_of "$MODEL_KEY")
SUFFIX=$(suffix_of "$RUN_TAG")
SERVER_EXTRA=$(server_extra_of "$RUN_TAG")
CLIENT_EXTRA=$(client_extra_of "$RUN_TAG")

# _r1/_r2/... suffix appended when running multiple repeats so outputs don't overwrite.
REPEAT_SUFFIX=${REPEAT_ID:+_r${REPEAT_ID}}
if (( TEST_MODE )); then TEST_SUFFIX=_test; NUM_EPOCHS=2; else TEST_SUFFIX=""; NUM_EPOCHS=5; fi

NAME=${MODEL_KEY}${SUFFIX}               # e.g. qwen3.5-4b_t0.5  (no suffix for `default`)
NAME_R=${NAME}${REPEAT_SUFFIX}${REPEAT_SUFFIX:+${SUFFIX}}${TEST_SUFFIX}  # e.g. qwen3.5-4b_t0.5_r2_t0.5 or qwen3.5-4b_test
TAG=${MODEL_KEY//./_}_${RUN_TAG//./_}${REPEAT_SUFFIX//./_}  # screen-safe, e.g. qwen3_5-4b_t0_5_r2
VLLM_SESS=vllm_${TAG}
METR_SESS=metrion_${TAG}
INF_SESS=infer_${TAG}

DB_PATH=$METRION_DB_DIR/${NAME_R}.db
VLLM_LOG=$EFFI_LOGS/${NAME_R}.vllm.log
METR_LOG=$EFFI_LOGS/${NAME_R}.metrion.log
INF_LOG=$EFFI_LOGS/${NAME_R}.infer.log
DONE_FLAG=$EFFI_FLAGS/${NAME_R}.done

# Resume behavior: if the final epoch result already exists, skip — unless --force.
FINAL_RESULT="$EFFI_ROOT/results/${NAME_R}/EffiBench_vllm_${NAME_R}_${NUM_EPOCHS}.json"
if [[ -f "$FINAL_RESULT" && $FORCE -eq 0 ]]; then
  echo "[${NAME_R}] already complete (${FINAL_RESULT}); skipping (use --force to overwrite)."
  exit 0
fi
if [[ $FORCE -eq 1 && -d "$EFFI_ROOT/results/${NAME_R}" ]]; then
  echo "[${NAME_R}] --force: removing existing results dir $EFFI_ROOT/results/${NAME_R}"
  rm -rf "$EFFI_ROOT/results/${NAME_R}"
fi

echo "============================================================"
echo "[${NAME_R}] starting"
echo "  checkpoint:  $CKPT"
echo "  server:      $SERVER_FLAGS $SERVER_EXTRA"
echo "  client:      $CLIENT_EXTRA"
echo "  db:          $DB_PATH"
echo "  logs:        $VLLM_LOG  $METR_LOG  $INF_LOG"
echo "============================================================"

rm -f "$DONE_FLAG"
if (( WITH_METRION )); then rm -f "$DB_PATH"; fi
METR_CFG=""

cleanup() {
  echo "[${NAME_R}] cleanup: stopping vllm + infer screens"
  if (( WITH_METRION )); then
    screen_send_ctrlc "$METR_SESS"
    sleep 5
    screen_kill "$METR_SESS"
  fi
  screen_kill "$VLLM_SESS"
  screen_kill "$INF_SESS"
}
trap cleanup EXIT

# ── 1. vLLM server ───────────────────────────────────────────────────────────
VLLM_CMD="source $EFFI_ROOT/.venv/bin/activate && python3 -m vllm.entrypoints.openai.api_server --model $CKPT $SERVER_FLAGS $SERVER_EXTRA 2>&1 | tee $VLLM_LOG"
screen_start "$VLLM_SESS" "$VLLM_CMD"
echo "[${NAME_R}] waiting for vLLM ready..."
wait_vllm_ready 8000 1800

# ── 2. metrion (opt-in via --with-metrion) ───────────────────────────────────
if (( WITH_METRION )); then
  METR_CFG=$(write_metrion_config "$DB_PATH")
  METR_CMD="cd $METRION_ROOT && source .venv/bin/activate && METRION_BIN=\$(which metrion) && sudo -n \"\$METRION_BIN\" monitor -c $METR_CFG 2>&1 | tee $METR_LOG"
  screen_start "$METR_SESS" "$METR_CMD"
  sleep 5
  if ! screen_alive "$METR_SESS"; then
    echo "[${NAME_R}] ERROR: metrion screen died immediately. Check sudo -n permissions and $METR_LOG" >&2
    exit 1
  fi
  echo "[${NAME_R}] metrion started → $DB_PATH"
fi

# ── 3. inference ─────────────────────────────────────────────────────────────
# Write the infer pipeline into a heredoc script so quoting stays sane.
INF_SCRIPT=$(mktemp -t infer_${TAG}.XXXXXX.sh)
cat >"$INF_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$EFFI_SRC"
export C_INCLUDE_PATH="$C_INCLUDE_PATH"
{
  echo "=== generation ==="
  "$EFFI_VENV_PY" vllm_generation.py \\
    --checkpoint "$CKPT" --model-name "$NAME_R" --dataset EffiBench \\
    --port 8000 --concurrency 64 $CLIENT_EXTRA

  echo "=== effilearner ($NUM_EPOCHS epochs) ==="
  "$EFFI_VENV_PY" vllm_effilearner.py \\
    --checkpoint "$CKPT" --model-name "$NAME_R" --dataset EffiBench \\
    --port 8000 --concurrency 64 --epoch $NUM_EPOCHS $CLIENT_EXTRA

  echo "=== validate ==="
  files=("../results/$NAME_R/EffiBench_vllm_$NAME_R.json")
  for i in \$(seq 1 $NUM_EPOCHS); do
    f="../results/$NAME_R/EffiBench_vllm_${NAME_R}_\${i}.json"
    [[ -f "\$f" ]] && files+=("\$f")
  done
  "$EFFI_VENV_PY" validate_completions.py --save --workers 8 "\${files[@]}"

  echo "=== profile_canonical ==="
  "$EFFI_VENV_PY" profile_canonical.py \\
    --model-name "$NAME_R" --dataset EffiBench \\
    --results-dir "../results/${NAME_R}-validated"

  echo "=== vllm_report ==="
  "$EFFI_VENV_PY" vllm_report.py \\
    --model-name "$NAME_R" --dataset EffiBench \\
    --results-dir "../results/${NAME_R}-validated" \\
    --canonical-file "../results/${NAME_R}-validated/canonical_EffiBench_vllm_$NAME_R.json"
} 2>&1 | tee "$INF_LOG"
echo \$? > "$DONE_FLAG"
EOF
chmod +x "$INF_SCRIPT"
screen_start "$INF_SESS" "$INF_SCRIPT"

# ── 4. wait for inference, then stop metrion + vllm ─────────────────────────
echo "[${NAME_R}] inference running (tail -f $INF_LOG to watch)"
while [[ ! -f "$DONE_FLAG" ]]; do
  sleep 30
  if ! screen_alive "$INF_SESS"; then
    echo "[${NAME_R}] infer screen died without writing done flag" >&2
    echo "1" > "$DONE_FLAG"
    break
  fi
done
RC=$(cat "$DONE_FLAG" 2>/dev/null || echo 1)

if (( WITH_METRION )); then
  echo "[${NAME_R}] inference finished (rc=$RC), stopping metrion..."
  # Graceful Ctrl+C so metrion flushes duckdb cleanly.
  screen_send_ctrlc "$METR_SESS"
  for i in {1..60}; do
    screen_alive "$METR_SESS" || break
    sleep 2
  done
  screen_kill "$METR_SESS"
else
  echo "[${NAME_R}] inference finished (rc=$RC)"
fi
screen_kill "$VLLM_SESS"
screen_kill "$INF_SESS"
trap - EXIT

rm -f "$INF_SCRIPT"
[[ -n "$METR_CFG" ]] && rm -f "$METR_CFG"

if (( WITH_METRION )); then
  if [[ -f "$DB_PATH" ]]; then
    echo "[${NAME_R}] db: $DB_PATH ($(du -h "$DB_PATH" | cut -f1))"
  else
    echo "[${NAME_R}] WARNING: db not found at $DB_PATH" >&2
  fi
fi

echo "[${NAME_R}] done (rc=$RC)"
exit "$RC"
