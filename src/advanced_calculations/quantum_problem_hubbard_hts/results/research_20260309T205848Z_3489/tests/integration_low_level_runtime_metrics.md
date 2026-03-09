# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15220157078`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `17.56`
- avg_mem_percent_global: `82.90`

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
    "correlated_fermions_non_hubbard (7.79%)" : 7.79
    "dense_nuclear_proxy (6.93%)" : 6.93
    "far_from_equilibrium_kinetic_lattices (8.02%)" : 8.02
    "hubbard_hts_core (9.41%)" : 9.41
    "multi_correlated_fermion_boson_networks (7.88%)" : 7.88
    "multi_state_excited_chemistry (7.33%)" : 7.33
    "multiscale_nonlinear_field_models (7.58%)" : 7.58
    "qcd_lattice_proxy (7.17%)" : 7.17
    "quantum_chemistry_proxy (7.15%)" : 7.15
    "quantum_field_noneq (6.81%)" : 6.81
    "spin_liquid_exotic (8.64%)" : 8.64
    "topological_correlated_materials (8.11%)" : 8.11
```
