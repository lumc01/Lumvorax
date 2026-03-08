# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `16281631163`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `16.66`
- avg_mem_percent_global: `80.25`

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
    "bosonic_multimode_systems (6.85%)" : 6.85
    "correlated_fermions_non_hubbard (7.17%)" : 7.17
    "dense_nuclear_proxy (7.06%)" : 7.06
    "far_from_equilibrium_kinetic_lattices (7.66%)" : 7.66
    "hubbard_hts_core (10.24%)" : 10.24
    "multi_correlated_fermion_boson_networks (7.54%)" : 7.54
    "multi_state_excited_chemistry (7.30%)" : 7.30
    "multiscale_nonlinear_field_models (7.64%)" : 7.64
    "qcd_lattice_proxy (7.49%)" : 7.49
    "quantum_chemistry_proxy (7.06%)" : 7.06
    "quantum_field_noneq (7.90%)" : 7.90
    "spin_liquid_exotic (8.45%)" : 8.45
    "topological_correlated_materials (7.64%)" : 7.64
```
