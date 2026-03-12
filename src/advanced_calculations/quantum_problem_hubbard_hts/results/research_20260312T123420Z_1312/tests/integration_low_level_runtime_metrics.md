# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `24703264880`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `16.75`
- avg_mem_percent_global: `52.03`

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
    "bosonic_multimode_systems (7.16%)" : 7.16
    "correlated_fermions_non_hubbard (7.83%)" : 7.83
    "dense_nuclear_fullscale (6.82%)" : 6.82
    "far_from_equilibrium_kinetic_lattices (7.87%)" : 7.87
    "hubbard_hts_core (9.27%)" : 9.27
    "multi_correlated_fermion_boson_networks (7.88%)" : 7.88
    "multi_state_excited_chemistry (7.46%)" : 7.46
    "multiscale_nonlinear_field_models (7.49%)" : 7.49
    "qcd_lattice_fullscale (7.41%)" : 7.41
    "quantum_chemistry_fullscale (7.17%)" : 7.17
    "quantum_field_noneq (6.84%)" : 6.84
    "spin_liquid_exotic (8.58%)" : 8.58
    "topological_correlated_materials (8.23%)" : 8.23
```
