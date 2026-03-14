# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `275328570`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `26.45`
- avg_mem_percent_global: `72.05`

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
    "bosonic_multimode_systems (6.26%)" : 6.26
    "correlated_fermions_non_hubbard (7.56%)" : 7.56
    "dense_nuclear_fullscale (5.92%)" : 5.92
    "far_from_equilibrium_kinetic_lattices (8.44%)" : 8.44
    "hubbard_hts_core (9.95%)" : 9.95
    "multi_correlated_fermion_boson_networks (8.39%)" : 8.39
    "multi_state_excited_chemistry (6.73%)" : 6.73
    "multiscale_nonlinear_field_models (7.82%)" : 7.82
    "qcd_lattice_fullscale (6.30%)" : 6.30
    "quantum_chemistry_fullscale (4.91%)" : 4.91
    "quantum_field_noneq (5.45%)" : 5.45
    "spin_liquid_exotic (11.59%)" : 11.59
    "topological_correlated_materials (10.68%)" : 10.68
```
