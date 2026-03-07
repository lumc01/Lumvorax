# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `14924599810`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `15.51`
- avg_mem_percent_global: `61.34`

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
    "bosonic_multimode_systems (7.11%)" : 7.11
    "correlated_fermions_non_hubbard (7.76%)" : 7.76
    "dense_nuclear_proxy (6.78%)" : 6.78
    "far_from_equilibrium_kinetic_lattices (7.86%)" : 7.86
    "hubbard_hts_core (9.34%)" : 9.34
    "multi_correlated_fermion_boson_networks (7.86%)" : 7.86
    "multi_state_excited_chemistry (7.54%)" : 7.54
    "multiscale_nonlinear_field_models (7.54%)" : 7.54
    "qcd_lattice_proxy (7.13%)" : 7.13
    "quantum_chemistry_proxy (7.15%)" : 7.15
    "quantum_field_noneq (6.85%)" : 6.85
    "spin_liquid_exotic (8.59%)" : 8.59
    "topological_correlated_materials (8.50%)" : 8.50
```
