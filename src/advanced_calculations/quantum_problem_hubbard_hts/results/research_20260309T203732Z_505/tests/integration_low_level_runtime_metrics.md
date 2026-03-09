# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15123350749`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `17.49`
- avg_mem_percent_global: `83.26`

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
    "correlated_fermions_non_hubbard (7.83%)" : 7.83
    "dense_nuclear_proxy (6.97%)" : 6.97
    "far_from_equilibrium_kinetic_lattices (7.82%)" : 7.82
    "hubbard_hts_core (9.44%)" : 9.44
    "multi_correlated_fermion_boson_networks (7.49%)" : 7.49
    "multi_state_excited_chemistry (7.44%)" : 7.44
    "multiscale_nonlinear_field_models (7.42%)" : 7.42
    "qcd_lattice_proxy (7.27%)" : 7.27
    "quantum_chemistry_proxy (7.34%)" : 7.34
    "quantum_field_noneq (6.89%)" : 6.89
    "spin_liquid_exotic (8.64%)" : 8.64
    "topological_correlated_materials (8.26%)" : 8.26
```
