#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

# Mirrors current problem table in hubbard_hts_research_cycle.c
PROBLEM_METADATA = {
    "hubbard_hts_core": {"lattice_size": "10x10", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.0, "U": 8.0, "mu": 0.2, "T": 95.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_hubbard_v4next", "hamiltonian_id": "hubbard_2d_single_band", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.0"},
    "qcd_lattice_proxy": {"lattice_size": "9x9", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 0.7, "U": 9.0, "mu": 0.1, "T": 140.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_qcd_v4next", "hamiltonian_id": "lattice_qcd_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.0"},
    "quantum_field_noneq": {"lattice_size": "8x8", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.3, "U": 7.0, "mu": 0.05, "T": 180.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_qft_v4next", "hamiltonian_id": "noneq_field_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.0"},
    "dense_nuclear_proxy": {"lattice_size": "9x8", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 0.8, "U": 11.0, "mu": 0.3, "T": 80.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_nuclear_v4next", "hamiltonian_id": "dense_nuclear_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.0"},
    "quantum_chemistry_proxy": {"lattice_size": "8x7", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.6, "U": 6.5, "mu": 0.4, "T": 60.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_qchem_v4next", "hamiltonian_id": "electronic_structure_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.0"},
    "spin_liquid_exotic": {"lattice_size": "12x10", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 0.9, "U": 10.5, "mu": 0.12, "T": 55.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_spin_liquid_v1", "hamiltonian_id": "frustrated_spin_liquid_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.1"},
    "topological_correlated_materials": {"lattice_size": "11x11", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.1, "U": 7.8, "mu": 0.15, "T": 70.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_topological_corr_v1", "hamiltonian_id": "chern_correlated_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.1"},
    "correlated_fermions_non_hubbard": {"lattice_size": "10x9", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.2, "U": 8.6, "mu": 0.18, "T": 85.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_corr_fermions_v1", "hamiltonian_id": "extended_fermionic_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.1"},
    "multi_state_excited_chemistry": {"lattice_size": "9x9", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.5, "U": 6.8, "mu": 0.22, "T": 48.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_excited_qchem_v1", "hamiltonian_id": "excited_state_electronic_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.1"},
    "bosonic_multimode_systems": {"lattice_size": "10x8", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 0.6, "U": 5.2, "mu": 0.06, "T": 110.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_bosonic_multimode_v1", "hamiltonian_id": "bosonic_multimode_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.1"},
    "multiscale_nonlinear_field_models": {"lattice_size": "12x8", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.4, "U": 9.2, "mu": 0.10, "T": 125.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_multiscale_field_v1", "hamiltonian_id": "rh_like_multiscale_field_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.1"},
    "far_from_equilibrium_kinetic_lattices": {"lattice_size": "11x9", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.0, "U": 8.0, "mu": 0.09, "T": 150.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_far_from_eq_v1", "hamiltonian_id": "noneq_kinetic_lattice_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.1"},
    "multi_correlated_fermion_boson_networks": {"lattice_size": "10x10", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.05, "U": 7.4, "mu": 0.14, "T": 100.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_fermion_boson_network_v1", "hamiltonian_id": "mixed_fermion_boson_network_proxy", "seed": "deterministic_proxy_seed", "solver_version": "hubbard_hts_research_runner_v4next", "schema_version": "2.1"},
}


def parse_lattice_sites(lattice_size: str) -> int:
    try:
        w, h = lattice_size.lower().split("x")
        return int(w) * int(h)
    except Exception:
        return 0


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
        md = PROBLEM_METADATA.get(p, {"lattice_size": "UNKNOWN", "geometry": "UNKNOWN", "boundary_conditions": "UNKNOWN", "t": "UNKNOWN", "U": "UNKNOWN", "mu": "UNKNOWN", "T": "UNKNOWN", "dt": "UNKNOWN", "method": "UNKNOWN", "model_id": "UNKNOWN", "hamiltonian_id": "UNKNOWN", "seed": "UNKNOWN", "solver_version": "UNKNOWN", "schema_version": "2.0"})
        lattice_sites = parse_lattice_sites(str(md.get("lattice_size", "")))
        t_value = md.get("t")
        u_value = md.get("U")
        u_over_t = "UNKNOWN"
        try:
            t_float = float(t_value)
            u_float = float(u_value)
            if t_float != 0.0:
                u_over_t = u_float / t_float
        except Exception:
            pass
        row = {"problem": p, **md}
        row["lattice_sites"] = lattice_sites
        row["u_over_t"] = u_over_t
        row["time_step"] = md.get("dt", "UNKNOWN")
        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "model_id", "hamiltonian_id", "schema_version", "lattice_size", "lattice_sites", "geometry", "boundary_conditions", "t", "U", "u_over_t", "mu", "T", "dt", "time_step", "seed", "solver_version", "method"])
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
        "model_id": rows[0]["model_id"] if rows else "UNKNOWN",
        "hamiltonian_id": rows[0]["hamiltonian_id"] if rows else "UNKNOWN",
        "schema_version": rows[0]["schema_version"] if rows else "2.0",
        "u_over_t": rows[0]["u_over_t"] if rows else "UNKNOWN",
        "time_step": rows[0]["time_step"] if rows else "UNKNOWN",
        "seed": rows[0]["seed"] if rows else "UNKNOWN",
        "solver_version": rows[0]["solver_version"] if rows else "UNKNOWN",
        "per_problem": rows,
        "source": "post_run_metadata_capture_v2",
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(f"[metadata-capture] generated: {csv_path}")
    print(f"[metadata-capture] generated: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
