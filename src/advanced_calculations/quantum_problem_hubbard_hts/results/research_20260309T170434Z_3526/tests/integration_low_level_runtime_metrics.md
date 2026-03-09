# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15199887922`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `16.30`
- avg_mem_percent_global: `64.42`

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
    "bosonic_multimode_systems (7.25%)" : 7.25
    "correlated_fermions_non_hubbard (7.88%)" : 7.88
    "dense_nuclear_proxy (6.87%)" : 6.87
    "far_from_equilibrium_kinetic_lattices (7.81%)" : 7.81
    "hubbard_hts_core (9.22%)" : 9.22
    "multi_correlated_fermion_boson_networks (7.69%)" : 7.69
    "multi_state_excited_chemistry (7.50%)" : 7.50
    "multiscale_nonlinear_field_models (7.41%)" : 7.41
    "qcd_lattice_proxy (7.32%)" : 7.32
    "quantum_chemistry_proxy (7.26%)" : 7.26
    "quantum_field_noneq (6.83%)" : 6.83
    "spin_liquid_exotic (8.59%)" : 8.59
    "topological_correlated_materials (8.39%)" : 8.39
```
