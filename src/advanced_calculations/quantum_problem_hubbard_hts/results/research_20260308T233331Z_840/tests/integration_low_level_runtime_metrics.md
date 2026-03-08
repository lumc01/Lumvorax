# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15340862302`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `17.78`
- avg_mem_percent_global: `62.81`

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
    "bosonic_multimode_systems (6.94%)" : 6.94
    "correlated_fermions_non_hubbard (7.94%)" : 7.94
    "dense_nuclear_proxy (7.40%)" : 7.40
    "far_from_equilibrium_kinetic_lattices (7.71%)" : 7.71
    "hubbard_hts_core (9.04%)" : 9.04
    "multi_correlated_fermion_boson_networks (7.62%)" : 7.62
    "multi_state_excited_chemistry (7.52%)" : 7.52
    "multiscale_nonlinear_field_models (7.39%)" : 7.39
    "qcd_lattice_proxy (7.05%)" : 7.05
    "quantum_chemistry_proxy (7.36%)" : 7.36
    "quantum_field_noneq (7.38%)" : 7.38
    "spin_liquid_exotic (8.58%)" : 8.58
    "topological_correlated_materials (8.06%)" : 8.06
```
