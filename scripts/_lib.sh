# _lib.sh — portable shared constants and helpers for EffiLearner scripts.
# Source with: source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EFFI_ROOT="$(cd "$_LIB_DIR/.." && pwd)"
export EFFI_SRC="$EFFI_ROOT/src"
export EFFI_VENV_PY="$EFFI_ROOT/.venv/bin/python3"
export EFFI_LOGS="$EFFI_ROOT/logs"

# Metrion is expected as a sibling directory called 'metrion'.
# Override by setting METRION_ROOT in the environment before sourcing this file.
export METRION_ROOT="${METRION_ROOT:-$(dirname "$EFFI_ROOT")/metrion}"
export METRION_DB_DIR="$METRION_ROOT/dbs"
export METRION_BASE_CFG="$METRION_ROOT/configs/config.yaml"

mkdir -p "$EFFI_LOGS"

# C_INCLUDE_PATH: set only if the python-headers directory exists alongside EFFI_ROOT.
# Required for vLLM (and line_profiler) to compile their C extensions.
_PY_HEADERS="$(dirname "$EFFI_ROOT")/.python-headers"
if [[ -d "$_PY_HEADERS" ]]; then
  export C_INCLUDE_PATH="$_PY_HEADERS/usr/include/python3.12:$_PY_HEADERS/usr/include:${C_INCLUDE_PATH:-}"
fi

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

screen_start() {
  local name=$1 cmd=$2
  screen_kill "$name"
  screen -dmS "$name" bash -lc "$cmd"
}

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

# $1 = absolute db path; echoes the generated tmp config path on stdout.
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
