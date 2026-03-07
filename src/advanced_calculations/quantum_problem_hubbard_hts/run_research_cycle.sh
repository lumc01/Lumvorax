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

TOTAL_STEPS=17
CURRENT_STEP=0
print_progress() {
  CURRENT_STEP=$((CURRENT_STEP + 1))
  local label="$1"
  local width=40
  local filled=$((CURRENT_STEP * width / TOTAL_STEPS))
  local empty=$((width - filled))
  local bar
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
  bar+=$(printf '%*s' "$empty" '' | tr ' ' '-')
  printf "\r[%s] %3d%% (%d/%d) %s" "$bar" "$((CURRENT_STEP * 100 / TOTAL_STEPS))" "$CURRENT_STEP" "$TOTAL_STEPS" "$label"
  if [ "$CURRENT_STEP" -eq "$TOTAL_STEPS" ]; then
    printf "\n"
  fi
}

make -C "$ROOT_DIR" clean all
print_progress "build"
"$ROOT_DIR/hubbard_hts_research_runner" "$ROOT_DIR"
print_progress "core simulation"

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
print_progress "metadata capture"
python3 "$ROOT_DIR/tools/post_run_cycle_guard.py" "$ROOT_DIR" "$RUN_DIR"
print_progress "cycle guard"
python3 "$ROOT_DIR/tools/post_run_physics_readiness_pack.py" "$RUN_DIR"
print_progress "physics readiness"
python3 "$ROOT_DIR/tools/post_run_v4next_integration_status.py" "$RUN_DIR"
print_progress "v4next status"
ROLL_MODE="${LUMVORAX_ROLLOUT_MODE:-shadow}"
python3 "$ROOT_DIR/tools/v4next_rollout_controller.py" "$RUN_DIR" "$ROLL_MODE"
print_progress "rollout controller"
python3 "$ROOT_DIR/tools/post_run_v4next_rollout_progress.py" "$RUN_DIR"
print_progress "rollout progress"
python3 "$ROOT_DIR/tools/post_run_v4next_realtime_evolution.py" "$ROOT_DIR" "$RUN_DIR"
print_progress "realtime evolution"
python3 "$ROOT_DIR/tools/post_run_low_level_telemetry.py" "$RUN_DIR"
print_progress "low-level telemetry"
python3 "$ROOT_DIR/tools/post_run_advanced_observables_pack.py" "$RUN_DIR"
print_progress "advanced observables"
python3 "$ROOT_DIR/tools/post_run_chatgpt_critical_tests.py" "$RUN_DIR"
print_progress "critical tests"
python3 "$ROOT_DIR/tools/post_run_problem_solution_progress.py" "$RUN_DIR"
print_progress "solution progress"
python3 "$ROOT_DIR/tools/post_run_authenticity_audit.py" "$ROOT_DIR" "$RUN_DIR"
print_progress "authenticity audit"
python3 "$ROOT_DIR/tools/post_run_cycle35_exhaustive_report.py" "$ROOT_DIR" "$RUN_DIR"
print_progress "cycle35 report"
python3 "$ROOT_DIR/tools/post_run_full_scope_integrator.py" --root "$ROOT_DIR" "$RUN_DIR"
print_progress "full-scope integration"

(
  cd "$RUN_DIR"
  find . -type f | sort | xargs sha256sum > logs/checksums.sha256
)
print_progress "checksums"

echo "Research cycle terminé: $RUN_DIR"
