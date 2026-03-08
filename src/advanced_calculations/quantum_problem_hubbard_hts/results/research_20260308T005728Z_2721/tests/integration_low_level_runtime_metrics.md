# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15578799560`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `22.03`
- avg_mem_percent_global: `41.68`

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
    "bosonic_multimode_systems (7.39%)" : 7.39
    "correlated_fermions_non_hubbard (7.44%)" : 7.44
    "dense_nuclear_proxy (6.56%)" : 6.56
    "far_from_equilibrium_kinetic_lattices (9.28%)" : 9.28
    "hubbard_hts_core (9.17%)" : 9.17
    "multi_correlated_fermion_boson_networks (8.73%)" : 8.73
    "multi_state_excited_chemistry (7.03%)" : 7.03
    "multiscale_nonlinear_field_models (7.97%)" : 7.97
    "qcd_lattice_proxy (7.06%)" : 7.06
    "quantum_chemistry_proxy (6.78%)" : 6.78
    "quantum_field_noneq (6.50%)" : 6.50
    "spin_liquid_exotic (8.35%)" : 8.35
    "topological_correlated_materials (7.73%)" : 7.73
```
