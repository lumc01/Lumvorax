#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="$ROOT_DIR/backups/research_cycle_$STAMP_UTC"
mkdir -p "$BACKUP_DIR"
cp -a "$ROOT_DIR/include" "$BACKUP_DIR/"
cp -a "$ROOT_DIR/src" "$BACKUP_DIR/"
cp -a "$ROOT_DIR/Makefile" "$BACKUP_DIR/"
cp -a "$ROOT_DIR/benchmarks" "$BACKUP_DIR/"

make -C "$ROOT_DIR" clean all
"$ROOT_DIR/hubbard_hts_research_runner" "$ROOT_DIR"

LATEST_RUN="$(ls -1 "$ROOT_DIR/results" | rg '^research_' | tail -n 1)"
RUN_DIR="$ROOT_DIR/results/$LATEST_RUN"

{
  echo "timestamp_utc=$STAMP_UTC"
  echo "hostname=$(hostname)"
  echo "uname=$(uname -a)"
  echo "gcc_version=$(gcc --version | head -n 1)"
} > "$RUN_DIR/logs/environment_versions.log"

# Cycle integrations: metadata first, then guard/gates, then physics pack.
python3 "$ROOT_DIR/tools/post_run_metadata_capture.py" "$RUN_DIR"
python3 "$ROOT_DIR/tools/post_run_cycle_guard.py" "$ROOT_DIR" "$RUN_DIR"
python3 "$ROOT_DIR/tools/post_run_physics_readiness_pack.py" "$RUN_DIR"
python3 "$ROOT_DIR/tools/post_run_v4next_integration_status.py" "$RUN_DIR"
python3 "$ROOT_DIR/tools/post_run_authenticity_audit.py" "$ROOT_DIR" "$RUN_DIR"

(
  cd "$RUN_DIR"
  find . -type f | sort | xargs sha256sum > logs/checksums.sha256
)

echo "Research cycle terminé: $RUN_DIR"
