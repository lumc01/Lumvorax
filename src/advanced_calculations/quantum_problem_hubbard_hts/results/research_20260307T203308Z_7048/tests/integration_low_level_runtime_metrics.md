# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15618018900`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `15.51`
- avg_mem_percent_global: `66.60`

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
    "bosonic_multimode_systems (6.90%)" : 6.90
    "correlated_fermions_non_hubbard (8.18%)" : 8.18
    "dense_nuclear_proxy (6.49%)" : 6.49
    "far_from_equilibrium_kinetic_lattices (7.81%)" : 7.81
    "hubbard_hts_core (8.89%)" : 8.89
    "multi_correlated_fermion_boson_networks (8.33%)" : 8.33
    "multi_state_excited_chemistry (7.66%)" : 7.66
    "multiscale_nonlinear_field_models (7.78%)" : 7.78
    "qcd_lattice_proxy (7.04%)" : 7.04
    "quantum_chemistry_proxy (6.89%)" : 6.89
    "quantum_field_noneq (6.62%)" : 6.62
    "spin_liquid_exotic (8.55%)" : 8.55
    "topological_correlated_materials (8.86%)" : 8.86
```
