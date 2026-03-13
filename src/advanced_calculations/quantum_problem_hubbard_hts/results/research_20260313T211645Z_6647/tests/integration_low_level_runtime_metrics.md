# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `265901399`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `15.98`
- avg_mem_percent_global: `54.86`

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
    "bosonic_multimode_systems (6.43%)" : 6.43
    "correlated_fermions_non_hubbard (7.94%)" : 7.94
    "dense_nuclear_fullscale (5.47%)" : 5.47
    "far_from_equilibrium_kinetic_lattices (8.53%)" : 8.53
    "hubbard_hts_core (10.12%)" : 10.12
    "multi_correlated_fermion_boson_networks (8.63%)" : 8.63
    "multi_state_excited_chemistry (6.82%)" : 6.82
    "multiscale_nonlinear_field_models (7.99%)" : 7.99
    "qcd_lattice_fullscale (6.47%)" : 6.47
    "quantum_chemistry_fullscale (4.70%)" : 4.70
    "quantum_field_noneq (5.02%)" : 5.02
    "spin_liquid_exotic (11.15%)" : 11.15
    "topological_correlated_materials (10.73%)" : 10.73
```
