# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `281231946`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `19.96`
- avg_mem_percent_global: `85.77`

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
    "bosonic_multimode_systems (6.12%)" : 6.12
    "correlated_fermions_non_hubbard (7.49%)" : 7.49
    "dense_nuclear_fullscale (5.93%)" : 5.93
    "far_from_equilibrium_kinetic_lattices (8.21%)" : 8.21
    "hubbard_hts_core (10.98%)" : 10.98
    "multi_correlated_fermion_boson_networks (8.12%)" : 8.12
    "multi_state_excited_chemistry (6.63%)" : 6.63
    "multiscale_nonlinear_field_models (7.52%)" : 7.52
    "qcd_lattice_fullscale (6.70%)" : 6.70
    "quantum_chemistry_fullscale (5.62%)" : 5.62
    "quantum_field_noneq (5.10%)" : 5.10
    "spin_liquid_exotic (11.31%)" : 11.31
    "topological_correlated_materials (10.28%)" : 10.28
```
