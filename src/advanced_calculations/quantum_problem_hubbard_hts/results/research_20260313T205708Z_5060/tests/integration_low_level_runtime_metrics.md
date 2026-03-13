# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `268395519`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `16.02`
- avg_mem_percent_global: `55.13`

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
    "bosonic_multimode_systems (6.33%)" : 6.33
    "correlated_fermions_non_hubbard (7.77%)" : 7.77
    "dense_nuclear_fullscale (5.54%)" : 5.54
    "far_from_equilibrium_kinetic_lattices (8.75%)" : 8.75
    "hubbard_hts_core (10.11%)" : 10.11
    "multi_correlated_fermion_boson_networks (8.59%)" : 8.59
    "multi_state_excited_chemistry (6.82%)" : 6.82
    "multiscale_nonlinear_field_models (7.93%)" : 7.93
    "qcd_lattice_fullscale (6.46%)" : 6.46
    "quantum_chemistry_fullscale (4.65%)" : 4.65
    "quantum_field_noneq (5.19%)" : 5.19
    "spin_liquid_exotic (11.13%)" : 11.13
    "topological_correlated_materials (10.72%)" : 10.72
```
