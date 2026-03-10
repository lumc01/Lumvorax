# Low-level Telemetry (module/hardware/interoperability)

- total_runtime_ns: `15208329862`
- total_qubits_simulated_proxy: `1160`
- avg_cpu_percent_global: `16.54`
- avg_mem_percent_global: `61.64`

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
    "bosonic_multimode_systems (7.24%)" : 7.24
    "correlated_fermions_non_hubbard (7.75%)" : 7.75
    "dense_nuclear_proxy (6.96%)" : 6.96
    "far_from_equilibrium_kinetic_lattices (7.64%)" : 7.64
    "hubbard_hts_core (9.52%)" : 9.52
    "multi_correlated_fermion_boson_networks (7.56%)" : 7.56
    "multi_state_excited_chemistry (7.37%)" : 7.37
    "multiscale_nonlinear_field_models (7.33%)" : 7.33
    "qcd_lattice_proxy (7.49%)" : 7.49
    "quantum_chemistry_proxy (7.09%)" : 7.09
    "quantum_field_noneq (7.01%)" : 7.01
    "spin_liquid_exotic (8.48%)" : 8.48
    "topological_correlated_materials (8.57%)" : 8.57
```
