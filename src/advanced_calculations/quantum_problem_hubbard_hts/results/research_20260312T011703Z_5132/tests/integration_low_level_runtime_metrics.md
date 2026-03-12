# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `24355911904`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `16.16`
- avg_mem_percent_global: `71.67`

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
    "correlated_fermions_non_hubbard (7.87%)" : 7.87
    "dense_nuclear_fullscale (6.84%)" : 6.84
    "far_from_equilibrium_kinetic_lattices (7.87%)" : 7.87
    "hubbard_hts_core (9.24%)" : 9.24
    "multi_correlated_fermion_boson_networks (7.90%)" : 7.90
    "multi_state_excited_chemistry (7.53%)" : 7.53
    "multiscale_nonlinear_field_models (7.54%)" : 7.54
    "qcd_lattice_fullscale (7.16%)" : 7.16
    "quantum_chemistry_fullscale (7.23%)" : 7.23
    "quantum_field_noneq (6.82%)" : 6.82
    "spin_liquid_exotic (8.59%)" : 8.59
    "topological_correlated_materials (8.24%)" : 8.24
```
