# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `248196445`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `20.99`
- avg_mem_percent_global: `67.39`

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
    "bosonic_multimode_systems (6.47%)" : 6.47
    "correlated_fermions_non_hubbard (7.76%)" : 7.76
    "dense_nuclear_fullscale (5.47%)" : 5.47
    "far_from_equilibrium_kinetic_lattices (8.61%)" : 8.61
    "hubbard_hts_core (10.33%)" : 10.33
    "multi_correlated_fermion_boson_networks (8.60%)" : 8.60
    "multi_state_excited_chemistry (6.70%)" : 6.70
    "multiscale_nonlinear_field_models (8.02%)" : 8.02
    "qcd_lattice_fullscale (6.43%)" : 6.43
    "quantum_chemistry_fullscale (4.71%)" : 4.71
    "quantum_field_noneq (5.07%)" : 5.07
    "spin_liquid_exotic (11.06%)" : 11.06
    "topological_correlated_materials (10.77%)" : 10.77
```
