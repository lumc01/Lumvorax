# Warp feasibility forensic summary

- run_id: case_aggressive_20260303_225941
- model: warp_feasibility_proxy_v2_gr1p1d_gr3p1d
- metric_profile: aggressive
- scenario_count: 3
- rows_total: 660
- calc_rows_per_second: 400.846
- mean_abs_negative_energy_j: 1.097883e+47
- max_abs_negative_energy_j: 1.536561e+47
- feasible_rate: 0.000000
- forensic_chain_final_hash: 7df6591d29626e4ae1a4e7d53baeffda753f9aec42d52160bcdc3881f9936501
- max_memory_mb: 19.855
- qubits_simulated: 0

Interpretation:
- If feasible_rate is near 0, current-physics feasibility is not supported by this proxy model.
- 1+1D and 3+1D are included simultaneously to tighten bounds compared to a single-scaling estimate.
- Multi-shell scenarios test whether geometry layering reduces exotic energy demands (it does not here).
