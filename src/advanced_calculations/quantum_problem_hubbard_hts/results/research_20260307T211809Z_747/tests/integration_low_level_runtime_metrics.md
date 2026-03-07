# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15101519433`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `15.53`
- avg_mem_percent_global: `70.91`

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
    "bosonic_multimode_systems (7.19%)" : 7.19
    "correlated_fermions_non_hubbard (7.77%)" : 7.77
    "dense_nuclear_proxy (6.83%)" : 6.83
    "far_from_equilibrium_kinetic_lattices (7.80%)" : 7.80
    "hubbard_hts_core (9.26%)" : 9.26
    "multi_correlated_fermion_boson_networks (7.91%)" : 7.91
    "multi_state_excited_chemistry (7.48%)" : 7.48
    "multiscale_nonlinear_field_models (7.60%)" : 7.60
    "qcd_lattice_proxy (7.19%)" : 7.19
    "quantum_chemistry_proxy (7.17%)" : 7.17
    "quantum_field_noneq (6.76%)" : 6.76
    "spin_liquid_exotic (8.71%)" : 8.71
    "topological_correlated_materials (8.31%)" : 8.31
```
