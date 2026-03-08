# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15270237970`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `15.53`
- avg_mem_percent_global: `65.44`

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
    "bosonic_multimode_systems (7.05%)" : 7.05
    "correlated_fermions_non_hubbard (7.72%)" : 7.72
    "dense_nuclear_proxy (7.12%)" : 7.12
    "far_from_equilibrium_kinetic_lattices (7.62%)" : 7.62
    "hubbard_hts_core (9.02%)" : 9.02
    "multi_correlated_fermion_boson_networks (9.16%)" : 9.16
    "multi_state_excited_chemistry (7.42%)" : 7.42
    "multiscale_nonlinear_field_models (7.26%)" : 7.26
    "qcd_lattice_proxy (7.19%)" : 7.19
    "quantum_chemistry_proxy (7.13%)" : 7.13
    "quantum_field_noneq (6.66%)" : 6.66
    "spin_liquid_exotic (8.43%)" : 8.43
    "topological_correlated_materials (8.22%)" : 8.22
```
