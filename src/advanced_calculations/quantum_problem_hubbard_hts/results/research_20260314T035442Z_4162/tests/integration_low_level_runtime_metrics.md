# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `269848934`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `16.59`
- avg_mem_percent_global: `63.08`

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
    "bosonic_multimode_systems (6.36%)" : 6.36
    "correlated_fermions_non_hubbard (7.76%)" : 7.76
    "dense_nuclear_fullscale (5.50%)" : 5.50
    "far_from_equilibrium_kinetic_lattices (8.54%)" : 8.54
    "hubbard_hts_core (10.41%)" : 10.41
    "multi_correlated_fermion_boson_networks (8.73%)" : 8.73
    "multi_state_excited_chemistry (6.76%)" : 6.76
    "multiscale_nonlinear_field_models (7.95%)" : 7.95
    "qcd_lattice_fullscale (6.40%)" : 6.40
    "quantum_chemistry_fullscale (4.73%)" : 4.73
    "quantum_field_noneq (5.02%)" : 5.02
    "spin_liquid_exotic (11.07%)" : 11.07
    "topological_correlated_materials (10.78%)" : 10.78
```
