# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `36884685994`
- total_qubits_simulated_effective: `2410`
- avg_cpu_percent_global: `2.61`
- avg_mem_percent_global: `54.95`

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
    "bosonic_multimode_systems (5.81%)" : 5.81
    "correlated_fermions_non_hubbard (6.23%)" : 6.23
    "dense_nuclear_fullscale (5.53%)" : 5.53
    "ed_validation_2x2 (9.73%)" : 9.73
    "far_from_equilibrium_kinetic_lattices (6.46%)" : 6.46
    "fermionic_sign_problem (7.94%)" : 7.94
    "hubbard_hts_core (7.46%)" : 7.46
    "multi_correlated_fermion_boson_networks (6.49%)" : 6.49
    "multi_state_excited_chemistry (6.06%)" : 6.06
    "multiscale_nonlinear_field_models (6.62%)" : 6.62
    "qcd_lattice_fullscale (5.78%)" : 5.78
    "quantum_chemistry_fullscale (6.36%)" : 6.36
    "quantum_field_noneq (5.68%)" : 5.68
    "spin_liquid_exotic (7.11%)" : 7.11
    "topological_correlated_materials (6.72%)" : 6.72
```
