# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `24333716841`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `23.10`
- avg_mem_percent_global: `51.37`

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
    "bosonic_multimode_systems (7.18%)" : 7.18
    "correlated_fermions_non_hubbard (7.88%)" : 7.88
    "dense_nuclear_proxy (6.80%)" : 6.80
    "far_from_equilibrium_kinetic_lattices (7.85%)" : 7.85
    "hubbard_hts_core (9.18%)" : 9.18
    "multi_correlated_fermion_boson_networks (7.89%)" : 7.89
    "multi_state_excited_chemistry (7.54%)" : 7.54
    "multiscale_nonlinear_field_models (7.60%)" : 7.60
    "qcd_lattice_proxy (7.16%)" : 7.16
    "quantum_chemistry_proxy (7.17%)" : 7.17
    "quantum_field_noneq (6.98%)" : 6.98
    "spin_liquid_exotic (8.55%)" : 8.55
    "topological_correlated_materials (8.22%)" : 8.22
```
