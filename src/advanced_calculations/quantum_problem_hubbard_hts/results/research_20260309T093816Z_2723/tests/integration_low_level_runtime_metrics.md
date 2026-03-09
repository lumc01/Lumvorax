# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15503508230`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `21.20`
- avg_mem_percent_global: `68.53`

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
    "bosonic_multimode_systems (7.01%)" : 7.01
    "correlated_fermions_non_hubbard (7.78%)" : 7.78
    "dense_nuclear_proxy (7.13%)" : 7.13
    "far_from_equilibrium_kinetic_lattices (7.72%)" : 7.72
    "hubbard_hts_core (9.11%)" : 9.11
    "multi_correlated_fermion_boson_networks (8.45%)" : 8.45
    "multi_state_excited_chemistry (7.35%)" : 7.35
    "multiscale_nonlinear_field_models (7.38%)" : 7.38
    "qcd_lattice_proxy (6.95%)" : 6.95
    "quantum_chemistry_proxy (7.26%)" : 7.26
    "quantum_field_noneq (6.96%)" : 6.96
    "spin_liquid_exotic (8.75%)" : 8.75
    "topological_correlated_materials (8.15%)" : 8.15
```
