# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `33348145560`
- total_qubits_simulated_effective: `2410`
- avg_cpu_percent_global: `19.77`
- avg_mem_percent_global: `75.36`

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
    "bosonic_multimode_systems (5.84%)" : 5.84
    "correlated_fermions_non_hubbard (6.33%)" : 6.33
    "dense_nuclear_fullscale (5.45%)" : 5.45
    "ed_validation_2x2 (10.01%)" : 10.01
    "far_from_equilibrium_kinetic_lattices (6.44%)" : 6.44
    "fermionic_sign_problem (7.84%)" : 7.84
    "hubbard_hts_core (7.66%)" : 7.66
    "multi_correlated_fermion_boson_networks (6.20%)" : 6.20
    "multi_state_excited_chemistry (5.95%)" : 5.95
    "multiscale_nonlinear_field_models (6.06%)" : 6.06
    "qcd_lattice_fullscale (5.63%)" : 5.63
    "quantum_chemistry_fullscale (5.53%)" : 5.53
    "quantum_field_noneq (6.33%)" : 6.33
    "spin_liquid_exotic (6.96%)" : 6.96
    "topological_correlated_materials (7.75%)" : 7.75
```
