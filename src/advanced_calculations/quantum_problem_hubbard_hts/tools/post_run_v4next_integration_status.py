#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def main():
    if len(sys.argv) != 2:
        print("Usage: post_run_v4next_integration_status.py <run_dir>", file=sys.stderr)
        return 2

    run_dir = Path(sys.argv[1]).resolve()
    tests_dir = run_dir / "tests"

    gate_rows = read_csv(tests_dir / "integration_gate_summary.csv")
    physics_gate_rows = read_csv(tests_dir / "integration_physics_gate_summary.csv")
    matrix_rows = read_csv(tests_dir / "integration_physics_enriched_test_matrix.csv")
    metadata_rows = read_csv(tests_dir / "integration_absent_metadata_fields.csv")
    new_tests_rows = read_csv(tests_dir / "new_tests_results.csv")

    gate_pct = pct(sum(r["status"] == "PASS" for r in gate_rows), len(gate_rows))
    physics_gate_pct = pct(sum(r["status"] == "PASS" for r in physics_gate_rows), len(physics_gate_rows))
    matrix_pct = pct(sum(r["status"] == "PASS" for r in matrix_rows), len(matrix_rows))
    metadata_pct = pct(sum(r["status"] == "PRESENT" for r in metadata_rows), len(metadata_rows))
    test_pass_pct = pct(sum(r["status"] == "PASS" for r in new_tests_rows), len(new_tests_rows))

    benchmark_rows = [r for r in new_tests_rows if r.get("test_family") == "benchmark"]
    benchmark_pass_pct = pct(sum(r["status"] == "PASS" for r in benchmark_rows), len(benchmark_rows)) if benchmark_rows else 0.0

    rollout_rows = []
    rollout_path = tests_dir / "integration_v4next_rollout_progress.csv"
    if rollout_path.exists():
        rollout_rows = read_csv(rollout_path)

    shadow_ready = any(r.get("stage") == "shadow" and r.get("status") == "READY" for r in rollout_rows)
    canary_ready = any(r.get("stage") == "canary" and r.get("status") == "READY" for r in rollout_rows)
    full_ready = any(r.get("stage") == "full" and r.get("status") == "READY" for r in rollout_rows)
    rollout_evidence_pct = pct(sum([shadow_ready, canary_ready, full_ready]), 3)

    # Dynamic and evidence-based scores (no fixed constants)
    proxy_modularity_pct = min(100.0, 0.40 * matrix_pct + 0.30 * metadata_pct + 0.30 * gate_pct)

    v4next_connection_pct = min(
        95.0,
        0.35 * gate_pct + 0.25 * physics_gate_pct + 0.20 * matrix_pct + 0.10 * metadata_pct + 0.10 * rollout_evidence_pct,
    )

    shadow_mode_safety_pct = min(
        95.0,
        0.40 * gate_pct + 0.30 * physics_gate_pct + 0.20 * rollout_evidence_pct + 0.10 * metadata_pct,
    )

    realism_raw = 0.55 * test_pass_pct + 0.25 * benchmark_pass_pct + 0.20 * matrix_pct
    # realism cannot claim >54 while non-proxy crosscheck is still open by design
    realism_cap = 54.0
    realistic_simulation_level_pct = min(realism_cap, realism_raw)

    weighted_global = (
        0.14 * gate_pct
        + 0.14 * physics_gate_pct
        + 0.10 * matrix_pct
        + 0.10 * metadata_pct
        + 0.12 * proxy_modularity_pct
        + 0.16 * v4next_connection_pct
        + 0.14 * shadow_mode_safety_pct
        + 0.10 * realistic_simulation_level_pct
    )

    rows = [
        ["pipeline_gates", f"{gate_pct:.2f}", "PASS ratio in integration_gate_summary.csv", "HIGH"],
        ["physics_gates", f"{physics_gate_pct:.2f}", "PASS ratio in integration_physics_gate_summary.csv", "HIGH"],
        ["physics_matrix", f"{matrix_pct:.2f}", "PASS ratio in integration_physics_enriched_test_matrix.csv", "MEDIUM"],
        ["metadata_completeness", f"{metadata_pct:.2f}", "PRESENT ratio in integration_absent_metadata_fields.csv", "HIGH"],
        ["proxy_modularity", f"{proxy_modularity_pct:.2f}", "Derived from matrix+metadata+gate coverage", "MEDIUM"],
        ["v4next_connection_readiness", f"{v4next_connection_pct:.2f}", "Derived from gates, metadata, rollout evidence", "MEDIUM"],
        ["shadow_mode_safety", f"{shadow_mode_safety_pct:.2f}", "Derived from gate stability and rollout evidence", "MEDIUM"],
        ["realistic_simulation_level", f"{realistic_simulation_level_pct:.2f}", "Capped until non-proxy crosscheck closes", "LOW"],
        ["global_weighted_readiness", f"{weighted_global:.2f}", "Weighted aggregate readiness score", "MEDIUM"],
    ]

    out_status = tests_dir / "integration_v4next_connection_readiness.csv"
    write_csv(out_status, ["dimension", "percent", "interpretation", "confidence"], rows)

    backlog = [
        ["Q_missing_units", "Are physical units explicit and consistent for all observables?", "Open", "Add units schema and unit-consistency gate"],
        ["Q_solver_crosscheck", "Do proxy results match at least one independent non-proxy solver on larger lattice?", "Open", "Add DMRG/QMC cross-check at 8x8 or 10x10"],
        ["Q_dt_real_sweep", "Is dt stability validated by true multi-run dt/2,dt,2dt (not proxy only)?", "Open", "Schedule 3-run sweep in CI night job"],
        ["Q_phase_criteria", "Are phase-transition criteria explicit (order parameter + finite-size scaling)?", "Open", "Add formal criteria and thresholds"],
        ["Q_production_guardrails", "Can V4 NEXT rollback instantly on degraded metrics?", "Open", "Implement shadow-mode rollback contract"],
    ]
    out_backlog = tests_dir / "integration_open_questions_backlog.csv"
    write_csv(out_backlog, ["question_id", "question", "status", "recommended_action"], backlog)

    print(f"[v4next-status] generated: {out_status}")
    print(f"[v4next-status] generated: {out_backlog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
