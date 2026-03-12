# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `282160724`
- total_qubits_simulated_effective: `1160`
- avg_cpu_percent_global: `17.71`
- avg_mem_percent_global: `67.18`

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
    "bosonic_multimode_systems (6.37%)" : 6.37
    "correlated_fermions_non_hubbard (7.75%)" : 7.75
    "dense_nuclear_fullscale (5.64%)" : 5.64
    "far_from_equilibrium_kinetic_lattices (8.48%)" : 8.48
    "hubbard_hts_core (10.12%)" : 10.12
    "multi_correlated_fermion_boson_networks (8.63%)" : 8.63
    "multi_state_excited_chemistry (6.81%)" : 6.81
    "multiscale_nonlinear_field_models (7.93%)" : 7.93
    "qcd_lattice_fullscale (6.69%)" : 6.69
    "quantum_chemistry_fullscale (4.76%)" : 4.76
    "quantum_field_noneq (4.99%)" : 4.99
    "spin_liquid_exotic (11.06%)" : 11.06
    "topological_correlated_materials (10.78%)" : 10.78
```
