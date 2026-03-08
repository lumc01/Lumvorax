# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15269877969`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `16.21`
- avg_mem_percent_global: `68.09`

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
    "bosonic_multimode_systems (7.16%)" : 7.16
    "correlated_fermions_non_hubbard (8.00%)" : 8.00
    "dense_nuclear_proxy (6.70%)" : 6.70
    "far_from_equilibrium_kinetic_lattices (7.78%)" : 7.78
    "hubbard_hts_core (9.17%)" : 9.17
    "multi_correlated_fermion_boson_networks (7.78%)" : 7.78
    "multi_state_excited_chemistry (7.35%)" : 7.35
    "multiscale_nonlinear_field_models (7.45%)" : 7.45
    "qcd_lattice_proxy (7.02%)" : 7.02
    "quantum_chemistry_proxy (7.18%)" : 7.18
    "quantum_field_noneq (6.61%)" : 6.61
    "spin_liquid_exotic (8.97%)" : 8.97
    "topological_correlated_materials (8.84%)" : 8.84
```
