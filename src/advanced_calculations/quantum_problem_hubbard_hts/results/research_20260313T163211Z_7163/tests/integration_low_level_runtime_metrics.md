# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `266388106`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `21.64`
- avg_mem_percent_global: `69.52`

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
    "bosonic_multimode_systems (6.32%)" : 6.32
    "correlated_fermions_non_hubbard (7.79%)" : 7.79
    "dense_nuclear_fullscale (5.44%)" : 5.44
    "far_from_equilibrium_kinetic_lattices (8.57%)" : 8.57
    "hubbard_hts_core (10.24%)" : 10.24
    "multi_correlated_fermion_boson_networks (8.51%)" : 8.51
    "multi_state_excited_chemistry (7.13%)" : 7.13
    "multiscale_nonlinear_field_models (7.98%)" : 7.98
    "qcd_lattice_fullscale (6.47%)" : 6.47
    "quantum_chemistry_fullscale (4.67%)" : 4.67
    "quantum_field_noneq (4.98%)" : 4.98
    "spin_liquid_exotic (11.21%)" : 11.21
    "topological_correlated_materials (10.69%)" : 10.69
```
