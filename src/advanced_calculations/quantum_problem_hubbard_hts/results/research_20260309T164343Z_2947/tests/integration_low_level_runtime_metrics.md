# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `14962924518`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `16.29`
- avg_mem_percent_global: `70.31`

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
    "correlated_fermions_non_hubbard (7.79%)" : 7.79
    "dense_nuclear_proxy (6.79%)" : 6.79
    "far_from_equilibrium_kinetic_lattices (7.79%)" : 7.79
    "hubbard_hts_core (9.36%)" : 9.36
    "multi_correlated_fermion_boson_networks (7.98%)" : 7.98
    "multi_state_excited_chemistry (7.43%)" : 7.43
    "multiscale_nonlinear_field_models (7.53%)" : 7.53
    "qcd_lattice_proxy (7.19%)" : 7.19
    "quantum_chemistry_proxy (7.16%)" : 7.16
    "quantum_field_noneq (6.72%)" : 6.72
    "spin_liquid_exotic (8.55%)" : 8.55
    "topological_correlated_materials (8.50%)" : 8.50
```
