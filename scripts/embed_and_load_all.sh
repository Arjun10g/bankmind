#!/usr/bin/env bash
# Orchestrator: run embed_and_load.py once per (module, strategy) collection.
# Each subprocess gets a fresh MPS state, avoiding the memory thrashing we saw
# when running multiple collections in one long-lived Python process.
#
# Idempotency: collections already at full point count will auto-skip.
# Use FORCE=1 to re-embed everything.

set -e
cd "$(dirname "$0")/.."

source .venv/bin/activate

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then
  FORCE_FLAG="--force"
fi

# (module, strategy) — order: cheaper first so faster feedback
COLLECTIONS=(
  "compliance regulatory_boundary"
  "compliance semantic"
  "compliance hierarchical"
  "credit narrative_section"
  "credit semantic"
  "credit financial_statement"
)

START=$(date +%s)
echo "===== embed_and_load_all start: $(date) ====="

for entry in "${COLLECTIONS[@]}"; do
  read -r module strategy <<< "$entry"
  echo
  echo "===== ${module}/${strategy} ====="
  python -u scripts/embed_and_load.py --module "$module" --strategy "$strategy" $FORCE_FLAG \
    || { echo "FAILED on ${module}/${strategy}"; exit 1; }
done

ELAPSED=$(( $(date +%s) - START ))
echo
echo "===== embed_and_load_all done in ${ELAPSED}s ($(date)) ====="
