#!/usr/bin/env bash
# Generate vllm_report.py CSVs for every results/<variant>-validated/.
# Each report is written to reports/<variant>/report_<variant>_<timestamp>.csv.
#
# Canonical normalization uses ../results/<variant>/canonical_*.json
# (the per-variant canonical from the original session).

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$HERE/../src
RESULTS=$HERE/../results
PY=${PY:-$HERE/../.venv/bin/python3}
[[ -x "$PY" ]] || { echo "venv python not found at $PY" >&2; exit 1; }

SHARED_CANONICAL=0
PIN_K_TO_LAST=0
CLEAN_CANONICAL=0
KNOWN_SUFFIXES=( "_t0.5" "_ctx65536" "_mt1024" )
CLEAN_CANON_PATH=""  # set below once HERE is known

variants=()
while (( $# )); do
  case "$1" in
    --shared-canonical) SHARED_CANONICAL=1; shift ;;
    --pin-k-to-last)    PIN_K_TO_LAST=1;    shift ;;
    --clean-canonical)  CLEAN_CANONICAL=1;  shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) variants+=("$1"); shift ;;
  esac
done

cd "$SRC"

if (( ${#variants[@]} == 0 )); then
  for d in "$RESULTS"/*-validated/; do
    [[ -d "$d" ]] || continue
    variants+=("$(basename "$d" -validated)")
  done
fi

base_of() {
  local v=$1
  # Strip _rN repeat suffix first
  v=$(echo "$v" | sed 's/_r[0-9]\+$//')
  for s in "${KNOWN_SUFFIXES[@]}"; do
    [[ "$v" == *"$s" ]] && { echo "${v%$s}"; return; }
  done
  echo "$v"
}

CLEAN_CANON_PATH="$RESULTS/canonical/canonical_EffiBench_vllm_canonical_EffiBench.json"

echo "Reporting on ${#variants[@]} variants  (shared_canonical=$SHARED_CANONICAL  pin_k_to_last=$PIN_K_TO_LAST  clean_canonical=$CLEAN_CANONICAL)"
for v in "${variants[@]}"; do
  vdir=$RESULTS/${v}-validated
  origdir=$RESULTS/${v}
  if (( CLEAN_CANONICAL )); then
    canon=$CLEAN_CANON_PATH
  elif (( SHARED_CANONICAL )); then
    base=$(base_of "$v")
    canon=$RESULTS/${base}/canonical_EffiBench_vllm_${base}.json
  else
    canon=$origdir/canonical_EffiBench_vllm_${v}.json
    # Repeat runs have their canonical in the -validated dir, not the plain dir
    [[ ! -f "$canon" ]] && canon=$vdir/canonical_EffiBench_vllm_${v}.json
  fi
  overhead=$origdir/overhead_vllm_${v}.json

  if [[ ! -d "$vdir" ]]; then echo "[$v] missing $vdir; skip" >&2; continue; fi
  if [[ ! -f "$canon"  ]]; then echo "[$v] missing canonical $canon; skip" >&2; continue; fi

  # vllm_report.py expects overhead_<v>.json inside results_dir; copy if absent.
  if [[ -f "$overhead" && ! -f "$vdir/$(basename "$overhead")" ]]; then
    cp "$overhead" "$vdir/"
  fi

  echo "============================================================"
  echo "[$v]"
  extra_flags=()
  (( PIN_K_TO_LAST )) && extra_flags+=( --pin-k-to-last )

  "$PY" vllm_report.py \
    --model-name "$v" \
    --dataset EffiBench \
    --results-dir "$vdir" \
    --canonical-file "$canon" \
    "${extra_flags[@]}"
done

echo
echo "Reports landed in: $HERE/../reports/<variant>/"
