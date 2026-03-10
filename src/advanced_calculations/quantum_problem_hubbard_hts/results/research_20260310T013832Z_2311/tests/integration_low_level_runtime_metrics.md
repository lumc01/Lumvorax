# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15713315230`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `17.40`
- avg_mem_percent_global: `60.07`

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
    "bosonic_multimode_systems (8.15%)" : 8.15
    "correlated_fermions_non_hubbard (7.98%)" : 7.98
    "dense_nuclear_proxy (6.77%)" : 6.77
    "far_from_equilibrium_kinetic_lattices (7.57%)" : 7.57
    "hubbard_hts_core (9.01%)" : 9.01
    "multi_correlated_fermion_boson_networks (7.60%)" : 7.60
    "multi_state_excited_chemistry (8.03%)" : 8.03
    "multiscale_nonlinear_field_models (7.65%)" : 7.65
    "qcd_lattice_proxy (6.97%)" : 6.97
    "quantum_chemistry_proxy (7.07%)" : 7.07
    "quantum_field_noneq (6.66%)" : 6.66
    "spin_liquid_exotic (8.76%)" : 8.76
    "topological_correlated_materials (7.77%)" : 7.77
```
