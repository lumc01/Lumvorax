# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15178374893`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `16.28`
- avg_mem_percent_global: `70.09`

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
    "bosonic_multimode_systems (7.22%)" : 7.22
    "correlated_fermions_non_hubbard (8.03%)" : 8.03
    "dense_nuclear_proxy (6.74%)" : 6.74
    "far_from_equilibrium_kinetic_lattices (7.68%)" : 7.68
    "hubbard_hts_core (9.35%)" : 9.35
    "multi_correlated_fermion_boson_networks (7.62%)" : 7.62
    "multi_state_excited_chemistry (7.47%)" : 7.47
    "multiscale_nonlinear_field_models (7.38%)" : 7.38
    "qcd_lattice_proxy (7.09%)" : 7.09
    "quantum_chemistry_proxy (7.22%)" : 7.22
    "quantum_field_noneq (6.77%)" : 6.77
    "spin_liquid_exotic (8.78%)" : 8.78
    "topological_correlated_materials (8.65%)" : 8.65
```
