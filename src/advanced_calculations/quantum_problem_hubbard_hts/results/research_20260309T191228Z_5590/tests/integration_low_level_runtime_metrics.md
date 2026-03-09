# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15142989066`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `16.28`
- avg_mem_percent_global: `69.60`

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
    "bosonic_multimode_systems (7.29%)" : 7.29
    "correlated_fermions_non_hubbard (8.05%)" : 8.05
    "dense_nuclear_proxy (6.75%)" : 6.75
    "far_from_equilibrium_kinetic_lattices (7.77%)" : 7.77
    "hubbard_hts_core (9.15%)" : 9.15
    "multi_correlated_fermion_boson_networks (7.92%)" : 7.92
    "multi_state_excited_chemistry (7.60%)" : 7.60
    "multiscale_nonlinear_field_models (7.43%)" : 7.43
    "qcd_lattice_proxy (7.22%)" : 7.22
    "quantum_chemistry_proxy (7.17%)" : 7.17
    "quantum_field_noneq (6.80%)" : 6.80
    "spin_liquid_exotic (8.46%)" : 8.46
    "topological_correlated_materials (8.37%)" : 8.37
```
