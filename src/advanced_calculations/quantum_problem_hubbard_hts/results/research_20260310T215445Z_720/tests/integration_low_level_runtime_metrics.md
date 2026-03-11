# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `24716986046`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `17.25`
- avg_mem_percent_global: `53.88`

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
    "bosonic_multimode_systems (7.11%)" : 7.11
    "correlated_fermions_non_hubbard (7.90%)" : 7.90
    "dense_nuclear_proxy (6.81%)" : 6.81
    "far_from_equilibrium_kinetic_lattices (7.85%)" : 7.85
    "hubbard_hts_core (9.24%)" : 9.24
    "multi_correlated_fermion_boson_networks (7.80%)" : 7.80
    "multi_state_excited_chemistry (7.60%)" : 7.60
    "multiscale_nonlinear_field_models (7.47%)" : 7.47
    "qcd_lattice_proxy (7.19%)" : 7.19
    "quantum_chemistry_proxy (7.32%)" : 7.32
    "quantum_field_noneq (6.98%)" : 6.98
    "spin_liquid_exotic (8.52%)" : 8.52
    "topological_correlated_materials (8.21%)" : 8.21
```
