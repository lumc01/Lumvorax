# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `24297969271`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `17.93`
- avg_mem_percent_global: `54.16`

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
    "bosonic_multimode_systems (7.17%)" : 7.17
    "correlated_fermions_non_hubbard (7.87%)" : 7.87
    "dense_nuclear_proxy (6.88%)" : 6.88
    "far_from_equilibrium_kinetic_lattices (7.83%)" : 7.83
    "hubbard_hts_core (9.26%)" : 9.26
    "multi_correlated_fermion_boson_networks (7.86%)" : 7.86
    "multi_state_excited_chemistry (7.54%)" : 7.54
    "multiscale_nonlinear_field_models (7.50%)" : 7.50
    "qcd_lattice_proxy (7.23%)" : 7.23
    "quantum_chemistry_proxy (7.29%)" : 7.29
    "quantum_field_noneq (6.84%)" : 6.84
    "spin_liquid_exotic (8.54%)" : 8.54
    "topological_correlated_materials (8.21%)" : 8.21
```
