# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `264415747`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `11.72`
- avg_mem_percent_global: `54.36`

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
    "bosonic_multimode_systems (6.32%)" : 6.32
    "correlated_fermions_non_hubbard (7.90%)" : 7.90
    "dense_nuclear_fullscale (5.46%)" : 5.46
    "far_from_equilibrium_kinetic_lattices (8.56%)" : 8.56
    "hubbard_hts_core (10.29%)" : 10.29
    "multi_correlated_fermion_boson_networks (8.52%)" : 8.52
    "multi_state_excited_chemistry (6.79%)" : 6.79
    "multiscale_nonlinear_field_models (7.85%)" : 7.85
    "qcd_lattice_fullscale (6.47%)" : 6.47
    "quantum_chemistry_fullscale (4.77%)" : 4.77
    "quantum_field_noneq (5.07%)" : 5.07
    "spin_liquid_exotic (11.21%)" : 11.21
    "topological_correlated_materials (10.79%)" : 10.79
```
