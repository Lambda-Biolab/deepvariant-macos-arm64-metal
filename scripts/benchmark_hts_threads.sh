#!/usr/bin/env bash
# Sweep hts_num_threads values to find optimal BAM decompression parallelism.
# Only runs make_examples (the bottleneck stage) to save time.
# Results saved to ~/deepvariant-benchmark/hts-threads-sweep/
#
# Usage:
#   bash scripts/benchmark_hts_threads.sh

set -euo pipefail

DV_HOME="${DEEPVARIANT_HOME:-$HOME/.deepvariant}"
OUTPUT_DIR="$HOME/deepvariant-benchmark/hts-threads-sweep"
DATA_DIR="$HOME/deepvariant-benchmark/data"
REF="$DATA_DIR/reference/GRCh38_no_alt_analysis_set.fasta"
BAM="$DATA_DIR/input/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"
SHARDS=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.logicalcpu)
RESULTS="$OUTPUT_DIR/results.jsonl"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()   { echo -e "${BLUE}==>${NC} $*"; }
pass()   { echo -e "${GREEN}✓${NC}  $*"; }
banner() { echo -e "\n${BOLD}$*${NC}"; }

mkdir -p "$OUTPUT_DIR"

# Activate conda env
ENV_INFO=$(cat "$DV_HOME/.env_type" 2>/dev/null || echo "")
if [[ "$ENV_INFO" == conda:* ]]; then
  CONDA_ENV="${ENV_INFO#conda:}"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ "$ENV_INFO" == venv && -f "$DV_HOME/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$DV_HOME/venv/bin/activate"
fi

export TF_CPP_MIN_LOG_LEVEL=3
export TF2_BEHAVIOR=1
export TPU_ML_PLATFORM=Tensorflow

echo "================================================================"
echo "  hts_num_threads sweep — make_examples only"
echo "================================================================"
echo "  Shards:  $SHARDS"
echo "  Output:  $OUTPUT_DIR"
echo ""

# Thread counts to test: 0 = default (single-threaded), plus 1,2,4,8
THREAD_COUNTS=(0 1 2 4 8)

for THREADS in "${THREAD_COUNTS[@]}"; do
  banner "=== hts_num_threads=$THREADS ==="
  run_dir="$OUTPUT_DIR/threads_${THREADS}"
  rm -rf "$run_dir"
  mkdir -p "$run_dir"

  EXAMPLES="$run_dir/make_examples.tfrecord@${SHARDS}.gz"

  info "make_examples with --hts_num_threads=$THREADS ($SHARDS shards)"
  SECONDS=0

  local_flags="--mode calling --ref $REF --reads $BAM --examples $EXAMPLES --checkpoint $DV_HOME/models/wgs --regions chr20"
  if (( THREADS > 0 )); then
    local_flags="$local_flags --hts_num_threads=$THREADS"
  fi

  seq 0 $((SHARDS - 1)) | parallel -q --halt 2 --line-buffer \
    "$DV_HOME/bin/make_examples" \
      --mode calling \
      --ref "$REF" \
      --reads "$BAM" \
      --examples "$EXAMPLES" \
      --checkpoint "$DV_HOME/models/wgs" \
      --regions chr20 \
      --task {} \
      $(if (( THREADS > 0 )); then echo "--hts_num_threads=$THREADS"; fi) \
    2>&1 | tee "$run_dir/make_examples.log"
  me_seconds=$SECONDS

  pass "hts_num_threads=$THREADS: make_examples=${me_seconds}s"

  python3 -c "
import json
entry = {
    'hts_num_threads': $THREADS,
    'make_examples_seconds': $me_seconds,
    'shards': $SHARDS
}
print(json.dumps(entry))
" >> "$RESULTS"
done

banner "=== Results ==="
python3 - "$RESULTS" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
baseline = next(r for r in rows if r['hts_num_threads'] == 0)['make_examples_seconds']
print(f"{'threads':>8}  {'make_examples (s)':>18}  {'vs default':>10}")
print("-" * 42)
for r in rows:
    pct = (r['make_examples_seconds'] - baseline) / baseline * 100
    sign = "+" if pct >= 0 else ""
    print(f"{r['hts_num_threads']:>8}  {r['make_examples_seconds']:>18}  {sign}{pct:>+.1f}%")
PYEOF
