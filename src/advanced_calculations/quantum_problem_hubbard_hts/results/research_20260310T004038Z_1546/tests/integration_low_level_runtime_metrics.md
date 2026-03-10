# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15301028848`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `17.35`
- avg_mem_percent_global: `50.38`

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
    "correlated_fermions_non_hubbard (7.97%)" : 7.97
    "dense_nuclear_proxy (6.95%)" : 6.95
    "far_from_equilibrium_kinetic_lattices (7.66%)" : 7.66
    "hubbard_hts_core (9.05%)" : 9.05
    "multi_correlated_fermion_boson_networks (7.67%)" : 7.67
    "multi_state_excited_chemistry (8.08%)" : 8.08
    "multiscale_nonlinear_field_models (7.41%)" : 7.41
    "qcd_lattice_proxy (7.07%)" : 7.07
    "quantum_chemistry_proxy (7.68%)" : 7.68
    "quantum_field_noneq (6.68%)" : 6.68
    "spin_liquid_exotic (8.50%)" : 8.50
    "topological_correlated_materials (8.15%)" : 8.15
```
