# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `264053096`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `17.10`
- avg_mem_percent_global: `80.38`

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
    "bosonic_multimode_systems (6.30%)" : 6.30
    "correlated_fermions_non_hubbard (7.75%)" : 7.75
    "dense_nuclear_fullscale (5.60%)" : 5.60
    "far_from_equilibrium_kinetic_lattices (8.56%)" : 8.56
    "hubbard_hts_core (10.39%)" : 10.39
    "multi_correlated_fermion_boson_networks (8.52%)" : 8.52
    "multi_state_excited_chemistry (6.71%)" : 6.71
    "multiscale_nonlinear_field_models (7.87%)" : 7.87
    "qcd_lattice_fullscale (6.48%)" : 6.48
    "quantum_chemistry_fullscale (4.64%)" : 4.64
    "quantum_field_noneq (5.12%)" : 5.12
    "spin_liquid_exotic (11.21%)" : 11.21
    "topological_correlated_materials (10.84%)" : 10.84
```
