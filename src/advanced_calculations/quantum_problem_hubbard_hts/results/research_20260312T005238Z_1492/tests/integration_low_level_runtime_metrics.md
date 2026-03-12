# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `24542662745`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `15.92`
- avg_mem_percent_global: `66.90`

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
    "bosonic_multimode_systems (7.15%)" : 7.15
    "correlated_fermions_non_hubbard (7.88%)" : 7.88
    "dense_nuclear_fullscale (6.75%)" : 6.75
    "far_from_equilibrium_kinetic_lattices (7.97%)" : 7.97
    "hubbard_hts_core (9.21%)" : 9.21
    "multi_correlated_fermion_boson_networks (7.93%)" : 7.93
    "multi_state_excited_chemistry (7.48%)" : 7.48
    "multiscale_nonlinear_field_models (7.48%)" : 7.48
    "qcd_lattice_fullscale (7.12%)" : 7.12
    "quantum_chemistry_fullscale (7.13%)" : 7.13
    "quantum_field_noneq (7.21%)" : 7.21
    "spin_liquid_exotic (8.52%)" : 8.52
    "topological_correlated_materials (8.16%)" : 8.16
```
