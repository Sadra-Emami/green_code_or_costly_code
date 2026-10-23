#!/usr/bin/env bash
# Validate every results/<variant>/ that doesn't already have a sibling
# results/<variant>-validated/. Skips canonical_/overhead_ files; runs
# baseline + epochs 1..5 through validate_completions.py --save.
#
# Usage:
#   ./validate_all.sh                    # all unvalidated variants
#   ./validate_all.sh --force            # re-validate even if -validated exists
#   ./validate_all.sh --workers 64       # forwarded to validate_completions.py (default 64)
#   ./validate_all.sh --parallel 4       # variants to validate concurrently (default 4)
#   ./validate_all.sh --variants v1 v2   # only these variants

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$HERE/../src
RESULTS=$HERE/../results
PY=${PY:-$HERE/../.venv/bin/python3}
[[ -x "$PY" ]] || { echo "venv python not found at $PY (set PY=... to override)" >&2; exit 1; }

FORCE=0
WORKERS=64
PARALLEL=4
VARIANTS=()

while (( $# )); do
  case "$1" in
    --force)    FORCE=1; shift ;;
    --workers)  WORKERS=$2; shift 2 ;;
    --parallel) PARALLEL=$2; shift 2 ;;
    --variants) shift; while (( $# )) && [[ "$1" != --* ]]; do VARIANTS+=("$1"); shift; done ;;
    -h|--help)  sed -n '2,10p' "$0"; exit 0 ;;
    *)          echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if (( ${#VARIANTS[@]} == 0 )); then
  for d in "$RESULTS"/*/; do
    v=$(basename "$d")
    [[ "$v" == *-validated || "$v" == *-reprofiled ]] && continue
    if (( FORCE == 0 )) && [[ -d "$RESULTS/${v}-validated" ]]; then
      continue
    fi
    VARIANTS+=("$v")
  done
fi

if (( ${#VARIANTS[@]} == 0 )); then
  echo "Nothing to validate (all variants already have -validated/). Pass --force to redo."
  exit 0
fi

echo "Variants to validate (${#VARIANTS[@]}): ${VARIANTS[*]}"
echo "Workers: $WORKERS  Parallel: $PARALLEL"
echo

cd "$SRC"

fail=0
pids=()
pid_variants=()

wait_one() {
  local new_pids=() new_vars=() i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      :
    else
      echo "[${pid_variants[$i]}] FAILED" >&2
      (( fail++ )) || true
    fi
  done
  pids=()
  pid_variants=()
}

for v in "${VARIANTS[@]}"; do
  files=()
  base="$RESULTS/$v/EffiBench_vllm_$v.json"
  [[ -f "$base" ]] && files+=("$base")
  for i in 1 2 3 4 5; do
    f="$RESULTS/$v/EffiBench_vllm_${v}_${i}.json"
    [[ -f "$f" ]] && files+=("$f")
  done

  if (( ${#files[@]} == 0 )); then
    echo "[$v] no result files; skipping" >&2
    continue
  fi

  echo "============================================================"
  echo "[$v] validating ${#files[@]} files"
  echo "============================================================"
  (
    "$PY" validate_completions.py --save --workers "$WORKERS" "${files[@]}"
    echo "[$v] done"
  ) &
  pids+=($!)
  pid_variants+=("$v")

  if (( ${#pids[@]} >= PARALLEL )); then
    wait_one
  fi
done

wait_one

echo
echo "Done. Validated dirs:"
ls -d "$RESULTS"/*-validated/ 2>/dev/null
(( fail == 0 ))
