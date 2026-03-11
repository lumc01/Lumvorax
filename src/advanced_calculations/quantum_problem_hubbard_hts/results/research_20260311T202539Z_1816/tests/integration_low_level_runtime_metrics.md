# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `24460717611`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `17.56`
- avg_mem_percent_global: `63.82`

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
    "bosonic_multimode_systems (7.13%)" : 7.13
    "correlated_fermions_non_hubbard (7.84%)" : 7.84
    "dense_nuclear_proxy (6.87%)" : 6.87
    "far_from_equilibrium_kinetic_lattices (7.90%)" : 7.90
    "hubbard_hts_core (9.30%)" : 9.30
    "multi_correlated_fermion_boson_networks (7.93%)" : 7.93
    "multi_state_excited_chemistry (7.52%)" : 7.52
    "multiscale_nonlinear_field_models (7.52%)" : 7.52
    "qcd_lattice_proxy (7.17%)" : 7.17
    "quantum_chemistry_proxy (7.18%)" : 7.18
    "quantum_field_noneq (6.83%)" : 6.83
    "spin_liquid_exotic (8.63%)" : 8.63
    "topological_correlated_materials (8.20%)" : 8.20
```
