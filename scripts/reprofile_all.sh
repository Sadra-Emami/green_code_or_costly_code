#!/usr/bin/env bash
# Re-profile every model-run JSON under results/<variant>-validated/ with
# n_runs=5 (matching the canonical methodology).
#
# Skips:
#   - canonical_*.json   (already at n=5)
#   - overhead_*.json    (aggregate, not per-problem)
#
# Usage:
#   bash reprofile_all.sh                       # all variants
#   bash reprofile_all.sh qwen3.5-4b ...        # only listed variants
#   bash reprofile_all.sh --with-canonical      # also reprofile the clean canonical first
#   N_RUNS=3 bash reprofile_all.sh              # override n_runs
#   WORKERS=16 bash reprofile_all.sh            # override workers
#   COOLDOWN_SEC=60 bash reprofile_all.sh       # pause between JSONs (default 30s)

set -uo pipefail   # NOTE: no -e so one bad JSON doesn't abort the whole sweep

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$HERE/../src
RESULTS=$HERE/../results
PY=${PY:-$HERE/../.venv/bin/python3}
N_RUNS=${N_RUNS:-5}
WORKERS=${WORKERS:-32}
COOLDOWN_SEC=${COOLDOWN_SEC:-30}
[[ -x "$PY" ]] || { echo "venv python not found at $PY" >&2; exit 1; }

WITH_CANONICAL=0
variants=()
while (( $# )); do
  case "$1" in
    --with-canonical) WITH_CANONICAL=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) variants+=("$1"); shift ;;
  esac
done

if (( ${#variants[@]} == 0 )); then
  for d in "$RESULTS"/*-validated/; do
    [[ -d "$d" ]] || continue
    variants+=("$(basename "$d" -validated)")
  done
fi

cd "$SRC"

if (( WITH_CANONICAL )); then
  echo "============================================================"
  echo "[canonical] reprofiling clean canonical (n_runs=$N_RUNS)"
  "$PY" profile_canonical.py \
    --model-name canonical_EffiBench \
    --dataset EffiBench \
    --results-dir "$RESULTS/canonical" \
    --workers "$WORKERS" \
    --n-runs "$N_RUNS" || { echo "[canonical] FAILED — aborting" >&2; exit 1; }
fi

echo "Reprofiling ${#variants[@]} variants  (n_runs=$N_RUNS  workers=$WORKERS  cooldown=${COOLDOWN_SEC}s)"
ok=0; fail=0; first=1
for v in "${variants[@]}"; do
  vdir=$RESULTS/${v}-validated
  if [[ ! -d "$vdir" ]]; then
    echo "[$v] missing $vdir; skip" >&2
    continue
  fi

  # Only match the baseline (EffiBench_vllm_<v>.json) and epoch JSONs
  # (EffiBench_vllm_<v>_<digits>.json). The earlier broad glob picked up
  # *_energy_results*.json and *.pids.json — DO NOT change this regex
  # without testing against the full file list.
  shopt -s nullglob
  for j in "$vdir"/EffiBench_vllm_${v}.json "$vdir"/EffiBench_vllm_${v}_*.json; do
    [[ -f "$j" ]] || continue
    bn=$(basename "$j")
    suffix=${bn#EffiBench_vllm_${v}}
    suffix=${suffix%.json}
    if [[ -n "$suffix" && ! "$suffix" =~ ^_[0-9]+$ ]]; then
      echo "  [skip] $bn (not a baseline/epoch JSON)"
      continue
    fi
    echo "============================================================"
    echo "[$v] $bn"

    # Cooldown between JSONs so the system can settle (avoids load drift
    # that biases ET upward across the sweep). Skipped before the very first
    # JSON of the run.
    if (( first )); then
      first=0
    elif (( COOLDOWN_SEC > 0 )); then
      echo "  cooldown ${COOLDOWN_SEC}s..."
      sleep "$COOLDOWN_SEC"
    fi

    if "$PY" reprofile_results.py \
        --input-file "$j" \
        --dataset EffiBench \
        --n-runs "$N_RUNS" \
        --workers "$WORKERS" \
        --stage "reprofile_n${N_RUNS}"; then
      ok=$((ok+1))
    else
      fail=$((fail+1))
      echo "  [FAIL] $j" >&2
    fi
  done
done

echo
echo "Done. $ok reprofiled, $fail failed."
