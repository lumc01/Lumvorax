#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$(realpath "$0")"
STAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="$ROOT_DIR/backups/research_cycle_$STAMP_UTC"
SESSION_LOG_DIR="$ROOT_DIR/logs"
SESSION_LOG="$SESSION_LOG_DIR/research_cycle_session_${STAMP_UTC}.log"

mkdir -p "$BACKUP_DIR"
mkdir -p "$SESSION_LOG_DIR"

# Persistent real-time log (console + file)
exec > >(stdbuf -oL tee -a "$SESSION_LOG") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%S.%N)Z] run_research_cycle start stamp=${STAMP_UTC}"
cp -a "$ROOT_DIR/include" "$BACKUP_DIR/"
cp -a "$ROOT_DIR/src" "$BACKUP_DIR/"
cp -a "$ROOT_DIR/Makefile" "$BACKUP_DIR/"
cp -a "$ROOT_DIR/benchmarks" "$BACKUP_DIR/"

TOTAL_STEPS="$(grep -c '^[[:space:]]*print_progress "' "$SCRIPT_PATH")"
if [ "${TOTAL_STEPS:-0}" -le 0 ]; then
  TOTAL_STEPS=1
fi
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

  printf "\r[%s] %3d%% (%d/%d) %s" \
    "$bar" \
    "$((CURRENT_STEP * 100 / TOTAL_STEPS))" \
    "$CURRENT_STEP" \
    "$TOTAL_STEPS" \
    "$label"

  if [ "$CURRENT_STEP" -eq "$TOTAL_STEPS" ]; then
    printf "\n"
  fi
}

write_checksums() {
  local target_run_dir="$1"
  (
    cd "$target_run_dir"
    rm -f logs/checksums.sha256
    find . -type f ! -path './logs/checksums.sha256' -print0 | sort -z | xargs -0 sha256sum > logs/checksums.sha256
  )
}

verify_checksums() {
  local target_run_dir="$1"
  (
    cd "$target_run_dir"
    sha256sum -c logs/checksums.sha256 >/dev/null
  )
}

make -C "$ROOT_DIR" clean all
print_progress "build"


# Force forensic runtime toggles ON for full traceability contract
export LUMVORAX_FORENSIC_REALTIME="1"
export LUMVORAX_LOG_PERSISTENCE="1"
export LUMVORAX_HFBL360_ENABLED="1"
export LUMVORAX_MEMORY_TRACKER="1"
export LUMVORAX_RUN_GROUP="campaign_${STAMP_UTC}"

export LUMVORAX_SOLVER_VARIANT="fullscale"
"$ROOT_DIR/hubbard_hts_research_runner" "$ROOT_DIR"
print_progress "fullscale simulation"

LATEST_FULLSCALE_RUN="$(ls -1 "$ROOT_DIR/results" | rg '^research_' | tail -n 1)"
FULLSCALE_RUN_DIR="$ROOT_DIR/results/$LATEST_FULLSCALE_RUN"

python3 "$ROOT_DIR/tools/post_run_csv_schema_guard.py" "$FULLSCALE_RUN_DIR"
print_progress "fullscale csv schema guard"

write_checksums "$FULLSCALE_RUN_DIR"
print_progress "fullscale checksums"
verify_checksums "$FULLSCALE_RUN_DIR"
print_progress "fullscale checksum verify"

export LUMVORAX_SOLVER_VARIANT="advanced_parallel"
"$ROOT_DIR/hubbard_hts_research_runner_advanced_parallel" "$ROOT_DIR"
print_progress "advanced parallel simulation"

LATEST_ADV_RUN="$(ls -1 "$ROOT_DIR/results" | rg '^research_' | tail -n 1)"
ADV_RUN_DIR="$ROOT_DIR/results/$LATEST_ADV_RUN"

RUN_DIR="$ADV_RUN_DIR"
LATEST_RUN="$LATEST_ADV_RUN"

python3 "$ROOT_DIR/tools/post_run_csv_schema_guard.py" "$RUN_DIR"
print_progress "advanced csv schema guard"

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

# BC-10 : mise à jour automatique runtime benchmark (R13 — RMSE < 0.05 requis)
python3 "$ROOT_DIR/tools/post_run_update_runtime_benchmark.py" "$RUN_DIR" "$ROOT_DIR/benchmarks"
print_progress "runtime benchmark update"

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

python3 "$ROOT_DIR/tools/run_independent_physics_modules.py" "$RUN_DIR"
print_progress "independent qmc/dmrg/arpes/stm"

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
  rm -f logs/checksums.sha256
)

python3 "$ROOT_DIR/tools/post_run_scientific_report_cycle.py" "$RUN_DIR"
print_progress "scientific report"

python3 "$ROOT_DIR/tools/post_run_independent_log_review.py" "$RUN_DIR"
print_progress "independent review"

python3 "$ROOT_DIR/tools/post_run_3d_modelization_export.py" "$RUN_DIR"
print_progress "3d model export"

python3 "$ROOT_DIR/tools/post_run_remote_depot_independent_analysis.py" "$ROOT_DIR" --run-dir "$RUN_DIR"
print_progress "remote independent analysis"

python3 "$ROOT_DIR/tools/post_run_parallel_calibration_bridge.py" "$RUN_DIR"
print_progress "parallel calibration bridge"

python3 "$ROOT_DIR/tools/post_run_hfbl360_forensic_logger.py" "$RUN_DIR" --standard-names "$ROOT_DIR/../../../STANDARD_NAMES.md"
print_progress "hfbl360 forensic logger"

write_checksums "$RUN_DIR"
print_progress "checksums"


CAMPAIGN_DIR="$ROOT_DIR/results/${LUMVORAX_RUN_GROUP}"
mkdir -p "$CAMPAIGN_DIR"
cat > "$CAMPAIGN_DIR/campaign_manifest.txt" <<MANIFEST
stamp_utc=$STAMP_UTC
run_group=${LUMVORAX_RUN_GROUP}
fullscale_run=$LATEST_FULLSCALE_RUN
advanced_run=$LATEST_ADV_RUN
fullscale_run_dir=$FULLSCALE_RUN_DIR
advanced_run_dir=$ADV_RUN_DIR
MANIFEST
print_progress "campaign manifest"

python3 "$ROOT_DIR/tools/post_run_fullscale_vs_advanced_compare.py" "$FULLSCALE_RUN_DIR" "$ADV_RUN_DIR" --out-dir "$CAMPAIGN_DIR"
print_progress "fullscale vs advanced compare"

python3 "$ROOT_DIR/tools/post_run_fullscale_vs_fullscale_benchmark.py" "$RUN_DIR"
print_progress "fullscale vs fullscale benchmark"

echo "Research cycle terminé (advanced): $RUN_DIR"
echo "Fullscale run: $FULLSCALE_RUN_DIR"
echo "Advanced run: $ADV_RUN_DIR"
echo "Campaign artifacts: $CAMPAIGN_DIR"
echo "Session log: $SESSION_LOG"

if [ "${LUMVORAX_FULLSCALE_STRICT:-1}" = "1" ]; then
  "$ROOT_DIR/run_fullscale_strict_protocol.sh" "$RUN_DIR"
  print_progress "fullscale strict protocol audit"
fi

# Finalize checksums at very end so later post-steps cannot stale the manifest.
write_checksums "$RUN_DIR"
print_progress "final checksums"
verify_checksums "$RUN_DIR"
print_progress "final checksum verify"
