#!/usr/bin/env bash
# Benchmark call_variants at different batch sizes to find the optimal
# value for Apple Silicon unified memory + Metal GPU.
#
# Usage: bash scripts/benchmark_batch_sizes.sh

set -euo pipefail

BATCH_SIZES=(512 1024 2048 4096 8192)
BASE_OUTPUT="$HOME/deepvariant-benchmark/batch-size-sweep"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GREEN='\033[0;32m'; BOLD='\033[1m'; NC='\033[0m'

mkdir -p "$BASE_OUTPUT"

echo -e "${BOLD}Batch size sweep: ${BATCH_SIZES[*]}${NC}"
echo "Results will be saved to: $BASE_OUTPUT"
echo ""

for bs in "${BATCH_SIZES[@]}"; do
  echo -e "${BOLD}━━━ Running batch_size=$bs ━━━${NC}"
  bash "$SCRIPT_DIR/benchmark.sh" \
    --batch-size "$bs" \
    --skip-accuracy \
    --output-dir "$BASE_OUTPUT/bs-$bs" \
    --runs 1
  echo ""
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}━━━ Batch Size Comparison ━━━${NC}"
printf "%-12s %-18s %-18s %-18s\n" "Batch Size" "make_examples (s)" "call_variants (s)" "Total (s)"
printf "%-12s %-18s %-18s %-18s\n" "----------" "------------------" "------------------" "----------"

for bs in "${BATCH_SIZES[@]}"; do
  results="$BASE_OUTPUT/bs-$bs/benchmark_results.json"
  if [[ -f "$results" ]]; then
    me=$(python3 -c "import json; d=json.load(open('$results')); print(d['summary']['make_examples']['mean'])")
    cv=$(python3 -c "import json; d=json.load(open('$results')); print(d['summary']['call_variants']['mean'])")
    total=$(python3 -c "import json; d=json.load(open('$results')); print(d['summary']['total']['mean'])")
    printf "%-12s %-18.1f %-18.1f %-18.1f\n" "$bs" "$me" "$cv" "$total"
  else
    printf "%-12s %-18s %-18s %-18s\n" "$bs" "FAILED" "FAILED" "FAILED"
  fi
done

echo -e "\n${GREEN}Results saved to: $BASE_OUTPUT${NC}"
