# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `38488709214`
- total_qubits_simulated_effective: `2410`
- avg_cpu_percent_global: `2.75`
- avg_mem_percent_global: `62.93`

## Architecture (mode FULL V4 NEXT)
```mermaid
flowchart LR
  A[run_research_cycle.sh] --> B[metadata_capture]
  B --> C[cycle_guard]
  C --> D[physics_readiness_pack]
  D --> E[v4next_integration_status]
  E --> F[v4next_rollout_controller full]
  F --> G[v4next_rollout_progress]
  G --> H[v4next_realtime_evolution]
  H --> I[chatgpt_critical_tests]
  I --> J[authenticity_audit]
  J --> K[checksums/audit]
```

## Module runtime share
```mermaid
pie title Module runtime share (%)
    "bosonic_multimode_systems (5.70%)" : 5.70
    "correlated_fermions_non_hubbard (6.25%)" : 6.25
    "dense_nuclear_fullscale (5.40%)" : 5.40
    "ed_validation_2x2 (9.53%)" : 9.53
    "far_from_equilibrium_kinetic_lattices (6.28%)" : 6.28
    "fermionic_sign_problem (8.39%)" : 8.39
    "hubbard_hts_core (7.67%)" : 7.67
    "multi_correlated_fermion_boson_networks (6.21%)" : 6.21
    "multi_state_excited_chemistry (6.68%)" : 6.68
    "multiscale_nonlinear_field_models (6.04%)" : 6.04
    "qcd_lattice_fullscale (5.82%)" : 5.82
    "quantum_chemistry_fullscale (5.60%)" : 5.60
    "quantum_field_noneq (5.69%)" : 5.69
    "spin_liquid_exotic (6.92%)" : 6.92
    "topological_correlated_materials (7.82%)" : 7.82
```
