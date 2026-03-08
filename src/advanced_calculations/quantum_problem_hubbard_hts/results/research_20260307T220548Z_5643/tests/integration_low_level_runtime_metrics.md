# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `19780823228`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `16.19`
- avg_mem_percent_global: `3.36`

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
    "bosonic_multimode_systems (7.03%)" : 7.03
    "correlated_fermions_non_hubbard (7.99%)" : 7.99
    "dense_nuclear_proxy (6.65%)" : 6.65
    "far_from_equilibrium_kinetic_lattices (7.87%)" : 7.87
    "hubbard_hts_core (9.17%)" : 9.17
    "multi_correlated_fermion_boson_networks (7.71%)" : 7.71
    "multi_state_excited_chemistry (7.44%)" : 7.44
    "multiscale_nonlinear_field_models (7.44%)" : 7.44
    "qcd_lattice_proxy (7.51%)" : 7.51
    "quantum_chemistry_proxy (7.38%)" : 7.38
    "quantum_field_noneq (6.99%)" : 6.99
    "spin_liquid_exotic (8.65%)" : 8.65
    "topological_correlated_materials (8.16%)" : 8.16
```
