# experiment_lib.sh — shared config + helpers for the EffiBench parameter sweep.
# Sourced by run_experiment.sh and run_all_experiments.sh.

set -u

# ── paths (auto-detected from this file's location) ──────────────────────────
_ELIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EFFI_ROOT="$(cd "$_ELIB_DIR/.." && pwd)"
export EFFI_SRC=$EFFI_ROOT/src
export EFFI_VENV_PY=$EFFI_ROOT/.venv/bin/python3
export EFFI_LOGS=$EFFI_ROOT/logs
export EFFI_FLAGS=$EFFI_ROOT/scripts/.flags
# METRION_ROOT: override via environment variable if your checkout differs.
export METRION_ROOT="${METRION_ROOT:-$(dirname "$EFFI_ROOT")/metrion}"
export METRION_DB_DIR=$METRION_ROOT/dbs
export METRION_BASE_CFG=$METRION_ROOT/configs/config.yaml

# C_INCLUDE_PATH: set only if the python-headers directory exists alongside EFFI_ROOT.
_PY_HEADERS="$(dirname "$EFFI_ROOT")/.python-headers"
if [[ -d "$_PY_HEADERS" ]]; then
  export C_INCLUDE_PATH="$_PY_HEADERS/usr/include/python3.12:$_PY_HEADERS/usr/include:${C_INCLUDE_PATH:-}"
fi

mkdir -p "$EFFI_LOGS" "$EFFI_FLAGS"

# ── model registry ───────────────────────────────────────────────────────────
MODEL_KEYS=(
  qwen3.5-4b
  qwen3.5-9b
  qwen3.5-27b
  qwen3.5-35b-a3b
  glm-4.7-flash
  glm-4.6v-flash
  ministral-3-3b-instruct-2512
  ministral-3-8b-instruct-2512
  ministral-3-14b-instruct-2512
  nemotron-cascade-2-30b-a3b
  nvidia-nometron-3-nano-30b-a3b-bf16
)

checkpoint_of() {
  case "$1" in
    qwen3.5-4b)                           echo ${MODEL_ROOT}/Qwen/Qwen3.5-4B ;;
    qwen3.5-9b)                           echo ${MODEL_ROOT}/Qwen/Qwen3.5-9B ;;
    qwen3.5-27b)                          echo ${MODEL_ROOT}/Qwen/Qwen3.5-27B ;;
    qwen3.5-35b-a3b)                      echo ${MODEL_ROOT}/Qwen/Qwen3.5-35B-A3B ;;
    glm-4.7-flash)                        echo ${MODEL_ROOT}/zai-org/GLM-4.7-Flash ;;
    glm-4.6v-flash)                       echo ${MODEL_ROOT}/zai-org/GLM-4.6V-Flash ;;
    ministral-3-3b-instruct-2512)         echo ${MODEL_ROOT}/mistralai/Ministral-3-3B-Instruct-2512 ;;
    ministral-3-8b-instruct-2512)         echo ${MODEL_ROOT}/mistralai/Ministral-3-8B-Instruct-2512 ;;
    ministral-3-14b-instruct-2512)        echo ${MODEL_ROOT}/mistralai/Ministral-3-14B-Instruct-2512 ;;
    nemotron-cascade-2-30b-a3b)           echo ${MODEL_ROOT}/nvidia/Nemotron-Cascade-2-30B-A3B ;;
    nvidia-nometron-3-nano-30b-a3b-bf16)  echo ${MODEL_ROOT}/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 ;;
    *) echo "UNKNOWN_MODEL:$1" >&2; return 1 ;;
  esac
}

# Extra server flags common to every run of this model (no --max-model-len here;
# that comes from the run_tag).
server_flags_of() {
  case "$1" in
    qwen3.5-4b|qwen3.5-9b|qwen3.5-27b|qwen3.5-35b-a3b|nemotron-cascade-2-30b-a3b)
      echo "--tensor-parallel-size 2 --default-chat-template-kwargs '{\"enable_thinking\":false}'"
      ;;
    *)
      echo "--tensor-parallel-size 2"
      ;;
  esac
}

# ── run-tag registry ─────────────────────────────────────────────────────────
# suffix_of TAG        → filename suffix, e.g. "_t0.5"  (empty for "default")
# server_extra_of TAG  → extra server flags (may be empty)
# client_extra_of TAG  → extra flags for vllm_generation.py / vllm_effilearner.py
RUN_TAGS=(t0.5 ctx65536 mt1024)

suffix_of() {
  case "$1" in
    default)   echo "" ;;
    t0.5)      echo "_t0.5" ;;
    ctx65536)  echo "_ctx65536" ;;
    mt1024)    echo "_mt1024" ;;
    *) echo "UNKNOWN_TAG:$1" >&2; return 1 ;;
  esac
}

server_extra_of() {
  case "$1" in
    ctx65536)  echo "--max-model-len 65536" ;;
    *)         echo "" ;;
  esac
}

client_extra_of() {
  case "$1" in
    default)   echo "" ;;
    t0.5)      echo "--temperature 0.5" ;;
    ctx65536)  echo "--context-size 65536" ;;
    mt1024)    echo "--max-tokens 1024" ;;
    *) echo "UNKNOWN_TAG:$1" >&2; return 1 ;;
  esac
}

# ── screen helpers ───────────────────────────────────────────────────────────
screen_alive() { screen -list | grep -qE "\.$1\b"; }

screen_kill() {
  if screen_alive "$1"; then
    screen -S "$1" -X quit >/dev/null 2>&1 || true
    sleep 1
  fi
}

screen_send_ctrlc() {
  if screen_alive "$1"; then
    screen -S "$1" -p 0 -X stuff $'\003' >/dev/null 2>&1 || true
  fi
}

# Start a detached screen with `bash -lc "<cmd>"`.
# $1=session name  $2=command string
screen_start() {
  local name=$1 cmd=$2
  screen_kill "$name"
  screen -dmS "$name" bash -lc "$cmd"
}

# ── vLLM readiness check ─────────────────────────────────────────────────────
wait_vllm_ready() {
  local port=${1:-8000} timeout_s=${2:-1800} t=0
  while (( t < timeout_s )); do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
    t=$((t + 5))
  done
  echo "vLLM did not become ready within ${timeout_s}s" >&2
  return 1
}

# ── metrion config generator ─────────────────────────────────────────────────
# Writes a tmp YAML that hardcodes the db path and disables debug timestamping.
# $1=absolute db path  → echoes the config path on stdout
write_metrion_config() {
  local db_path=$1 out
  out=$(mktemp -t metrion_cfg.XXXXXX.yaml)
  awk -v dbp="$db_path" '
    BEGIN { in_general = 0; in_db = 0 }
    /^general:/ { in_general = 1; print; next }
    in_general && /^  database:/ { in_db = 1; print; next }
    in_db && /^    path:/ { print "    path: " dbp; in_db = 0; next }
    in_general && /^  debug:/ { print "  debug: false"; next }
    /^[^ ]/ && !/^general:/ { in_general = 0; in_db = 0 }
    { print }
  ' "$METRION_BASE_CFG" > "$out"
  echo "$out"
}
