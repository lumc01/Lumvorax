# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15013898308`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `17.34`
- avg_mem_percent_global: `65.81`

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
    "correlated_fermions_non_hubbard (7.95%)" : 7.95
    "dense_nuclear_proxy (6.96%)" : 6.96
    "far_from_equilibrium_kinetic_lattices (7.90%)" : 7.90
    "hubbard_hts_core (9.32%)" : 9.32
    "multi_correlated_fermion_boson_networks (7.91%)" : 7.91
    "multi_state_excited_chemistry (7.49%)" : 7.49
    "multiscale_nonlinear_field_models (7.39%)" : 7.39
    "qcd_lattice_proxy (7.21%)" : 7.21
    "quantum_chemistry_proxy (7.06%)" : 7.06
    "quantum_field_noneq (6.83%)" : 6.83
    "spin_liquid_exotic (8.49%)" : 8.49
    "topological_correlated_materials (8.30%)" : 8.30
```
