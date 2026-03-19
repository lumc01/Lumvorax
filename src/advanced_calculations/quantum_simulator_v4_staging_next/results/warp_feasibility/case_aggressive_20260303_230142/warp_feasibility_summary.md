# Warp feasibility forensic summary

- run_id: case_aggressive_20260303_230142
- model: warp_feasibility_proxy_v2_gr1p1d_gr3p1d
- metric_profile: aggressive
- scenario_count: 3
- rows_total: 660
- calc_rows_per_second: 394.303
- mean_abs_negative_energy_j: 1.097883e+47
- max_abs_negative_energy_j: 1.536561e+47
- feasible_rate: 0.000000
- forensic_chain_final_hash: 1f286635519fab8b0c7b3c1ff20e62036f97d8ed8539afac9777a7ba5ed71de0
- max_memory_mb: 19.754
- qubits_simulated: 0

Interpretation:
- If feasible_rate is near 0, current-physics feasibility is not supported by this proxy model.
- 1+1D and 3+1D are included simultaneously to tighten bounds compared to a single-scaling estimate.
- Multi-shell scenarios test whether geometry layering reduces exotic energy demands (it does not here).
