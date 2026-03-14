# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `274048449`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `16.65`
- avg_mem_percent_global: `66.37`

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
    "bosonic_multimode_systems (6.24%)" : 6.24
    "correlated_fermions_non_hubbard (7.98%)" : 7.98
    "dense_nuclear_fullscale (5.58%)" : 5.58
    "far_from_equilibrium_kinetic_lattices (8.37%)" : 8.37
    "hubbard_hts_core (10.92%)" : 10.92
    "multi_correlated_fermion_boson_networks (8.53%)" : 8.53
    "multi_state_excited_chemistry (6.81%)" : 6.81
    "multiscale_nonlinear_field_models (7.88%)" : 7.88
    "qcd_lattice_fullscale (6.34%)" : 6.34
    "quantum_chemistry_fullscale (4.59%)" : 4.59
    "quantum_field_noneq (4.89%)" : 4.89
    "spin_liquid_exotic (11.11%)" : 11.11
    "topological_correlated_materials (10.77%)" : 10.77
```
