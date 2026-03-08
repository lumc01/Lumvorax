# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15037126520`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `15.90`
- avg_mem_percent_global: `50.53`

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
    "bosonic_multimode_systems (7.09%)" : 7.09
    "correlated_fermions_non_hubbard (7.77%)" : 7.77
    "dense_nuclear_proxy (6.67%)" : 6.67
    "far_from_equilibrium_kinetic_lattices (7.92%)" : 7.92
    "hubbard_hts_core (9.07%)" : 9.07
    "multi_correlated_fermion_boson_networks (7.75%)" : 7.75
    "multi_state_excited_chemistry (7.49%)" : 7.49
    "multiscale_nonlinear_field_models (7.62%)" : 7.62
    "qcd_lattice_proxy (7.90%)" : 7.90
    "quantum_chemistry_proxy (7.19%)" : 7.19
    "quantum_field_noneq (6.70%)" : 6.70
    "spin_liquid_exotic (8.60%)" : 8.60
    "topological_correlated_materials (8.22%)" : 8.22
```
