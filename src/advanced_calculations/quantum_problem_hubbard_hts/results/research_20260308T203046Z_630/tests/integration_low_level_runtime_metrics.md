# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `14726296458`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `17.85`
- avg_mem_percent_global: `41.49`

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
    "correlated_fermions_non_hubbard (7.85%)" : 7.85
    "dense_nuclear_proxy (6.84%)" : 6.84
    "far_from_equilibrium_kinetic_lattices (7.69%)" : 7.69
    "hubbard_hts_core (9.26%)" : 9.26
    "multi_correlated_fermion_boson_networks (7.78%)" : 7.78
    "multi_state_excited_chemistry (7.50%)" : 7.50
    "multiscale_nonlinear_field_models (7.76%)" : 7.76
    "qcd_lattice_proxy (7.15%)" : 7.15
    "quantum_chemistry_proxy (7.41%)" : 7.41
    "quantum_field_noneq (6.88%)" : 6.88
    "spin_liquid_exotic (8.49%)" : 8.49
    "topological_correlated_materials (8.23%)" : 8.23
```
