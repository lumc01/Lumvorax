# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `249570775`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `20.97`
- avg_mem_percent_global: `70.24`

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
    "bosonic_multimode_systems (6.42%)" : 6.42
    "correlated_fermions_non_hubbard (7.83%)" : 7.83
    "dense_nuclear_fullscale (5.60%)" : 5.60
    "far_from_equilibrium_kinetic_lattices (8.40%)" : 8.40
    "hubbard_hts_core (10.09%)" : 10.09
    "multi_correlated_fermion_boson_networks (8.76%)" : 8.76
    "multi_state_excited_chemistry (6.70%)" : 6.70
    "multiscale_nonlinear_field_models (7.99%)" : 7.99
    "qcd_lattice_fullscale (6.36%)" : 6.36
    "quantum_chemistry_fullscale (4.68%)" : 4.68
    "quantum_field_noneq (5.04%)" : 5.04
    "spin_liquid_exotic (11.29%)" : 11.29
    "topological_correlated_materials (10.85%)" : 10.85
```
