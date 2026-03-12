# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `294275822`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `17.88`
- avg_mem_percent_global: `64.19`

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
    "bosonic_multimode_systems (6.08%)" : 6.08
    "correlated_fermions_non_hubbard (10.57%)" : 10.57
    "dense_nuclear_fullscale (5.23%)" : 5.23
    "far_from_equilibrium_kinetic_lattices (8.22%)" : 8.22
    "hubbard_hts_core (9.70%)" : 9.70
    "multi_correlated_fermion_boson_networks (8.14%)" : 8.14
    "multi_state_excited_chemistry (7.70%)" : 7.70
    "multiscale_nonlinear_field_models (7.76%)" : 7.76
    "qcd_lattice_fullscale (6.22%)" : 6.22
    "quantum_chemistry_fullscale (4.57%)" : 4.57
    "quantum_field_noneq (4.76%)" : 4.76
    "spin_liquid_exotic (10.59%)" : 10.59
    "topological_correlated_materials (10.45%)" : 10.45
```
