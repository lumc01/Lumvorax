# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `268877270`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `24.90`
- avg_mem_percent_global: `69.75`

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
    "bosonic_multimode_systems (6.27%)" : 6.27
    "correlated_fermions_non_hubbard (8.04%)" : 8.04
    "dense_nuclear_fullscale (5.41%)" : 5.41
    "far_from_equilibrium_kinetic_lattices (8.52%)" : 8.52
    "hubbard_hts_core (10.00%)" : 10.00
    "multi_correlated_fermion_boson_networks (8.46%)" : 8.46
    "multi_state_excited_chemistry (6.85%)" : 6.85
    "multiscale_nonlinear_field_models (7.88%)" : 7.88
    "qcd_lattice_fullscale (6.29%)" : 6.29
    "quantum_chemistry_fullscale (4.63%)" : 4.63
    "quantum_field_noneq (4.95%)" : 4.95
    "spin_liquid_exotic (10.86%)" : 10.86
    "topological_correlated_materials (11.84%)" : 11.84
```
