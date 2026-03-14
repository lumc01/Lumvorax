# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `800581735`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `22.46`
- avg_mem_percent_global: `61.82`

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
    "bosonic_multimode_systems (7.31%)" : 7.31
    "correlated_fermions_non_hubbard (7.68%)" : 7.68
    "dense_nuclear_fullscale (6.27%)" : 6.27
    "far_from_equilibrium_kinetic_lattices (8.43%)" : 8.43
    "hubbard_hts_core (9.37%)" : 9.37
    "multi_correlated_fermion_boson_networks (8.21%)" : 8.21
    "multi_state_excited_chemistry (7.63%)" : 7.63
    "multiscale_nonlinear_field_models (7.90%)" : 7.90
    "qcd_lattice_fullscale (6.88%)" : 6.88
    "quantum_chemistry_fullscale (6.23%)" : 6.23
    "quantum_field_noneq (6.12%)" : 6.12
    "spin_liquid_exotic (9.27%)" : 9.27
    "topological_correlated_materials (8.70%)" : 8.70
```
