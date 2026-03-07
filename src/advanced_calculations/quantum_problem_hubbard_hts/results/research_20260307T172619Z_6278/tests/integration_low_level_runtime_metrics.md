# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `13877555295`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `0.49`
- avg_mem_percent_global: `3.42`

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
    "bosonic_multimode_systems (7.18%)" : 7.18
    "correlated_fermions_non_hubbard (7.95%)" : 7.95
    "dense_nuclear_proxy (6.75%)" : 6.75
    "far_from_equilibrium_kinetic_lattices (7.97%)" : 7.97
    "hubbard_hts_core (9.37%)" : 9.37
    "multi_correlated_fermion_boson_networks (8.21%)" : 8.21
    "multi_state_excited_chemistry (7.55%)" : 7.55
    "multiscale_nonlinear_field_models (7.46%)" : 7.46
    "qcd_lattice_proxy (7.12%)" : 7.12
    "quantum_chemistry_proxy (7.09%)" : 7.09
    "quantum_field_noneq (6.70%)" : 6.70
    "spin_liquid_exotic (8.40%)" : 8.40
    "topological_correlated_materials (8.23%)" : 8.23
```
