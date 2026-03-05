#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

# Mirrors current problem table in hubbard_hts_research_cycle.c
PROBLEM_METADATA = {
    "hubbard_hts_core": {"lattice_size": "10x10", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.0, "U": 8.0, "mu": 0.2, "T": 95.0, "dt": 1.0, "method": "advanced_proxy_deterministic"},
    "qcd_lattice_proxy": {"lattice_size": "9x9", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 0.7, "U": 9.0, "mu": 0.1, "T": 140.0, "dt": 1.0, "method": "advanced_proxy_deterministic"},
    "quantum_field_noneq": {"lattice_size": "8x8", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.3, "U": 7.0, "mu": 0.05, "T": 180.0, "dt": 1.0, "method": "advanced_proxy_deterministic"},
    "dense_nuclear_proxy": {"lattice_size": "9x8", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 0.8, "U": 11.0, "mu": 0.3, "T": 80.0, "dt": 1.0, "method": "advanced_proxy_deterministic"},
    "quantum_chemistry_proxy": {"lattice_size": "8x7", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.6, "U": 6.5, "mu": 0.4, "T": 60.0, "dt": 1.0, "method": "advanced_proxy_deterministic"},
}


def read_problem_names(baseline: Path):
    with baseline.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted({r.get("problem", "") for r in rows if r.get("problem")})


def main():
    if len(sys.argv) != 2:
        print("Usage: post_run_metadata_capture.py <run_dir>", file=sys.stderr)
        return 2

    run_dir = Path(sys.argv[1]).resolve()
    baseline = run_dir / "logs" / "baseline_reanalysis_metrics.csv"
    if not baseline.exists():
        print(f"[metadata-capture] baseline missing: {baseline}", file=sys.stderr)
        return 3

    problems = read_problem_names(baseline)

    csv_path = run_dir / "logs" / "model_metadata.csv"
    json_path = run_dir / "logs" / "model_metadata.json"

    rows = []
    for p in problems:
        md = PROBLEM_METADATA.get(p, {"lattice_size": "UNKNOWN", "geometry": "UNKNOWN", "boundary_conditions": "UNKNOWN", "t": "UNKNOWN", "U": "UNKNOWN", "mu": "UNKNOWN", "T": "UNKNOWN", "dt": "UNKNOWN", "method": "UNKNOWN"})
        row = {"problem": p, **md}
        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "lattice_size", "geometry", "boundary_conditions", "t", "U", "mu", "T", "dt", "method"])
        w.writeheader()
        w.writerows(rows)

    # keep first-row compatibility for tools expecting scalar metadata, plus full table
    payload = {
        "lattice_size": rows[0]["lattice_size"] if rows else "UNKNOWN",
        "geometry": rows[0]["geometry"] if rows else "UNKNOWN",
        "boundary_conditions": rows[0]["boundary_conditions"] if rows else "UNKNOWN",
        "t": rows[0]["t"] if rows else "UNKNOWN",
        "U": rows[0]["U"] if rows else "UNKNOWN",
        "mu": rows[0]["mu"] if rows else "UNKNOWN",
        "T": rows[0]["T"] if rows else "UNKNOWN",
        "dt": rows[0]["dt"] if rows else "UNKNOWN",
        "method": rows[0]["method"] if rows else "UNKNOWN",
        "per_problem": rows,
        "source": "post_run_metadata_capture_v1",
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(f"[metadata-capture] generated: {csv_path}")
    print(f"[metadata-capture] generated: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
