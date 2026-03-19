# Warp feasibility forensic summary

- run_id: case_conservative_20260303_225938
- model: warp_feasibility_proxy_v2_gr1p1d_gr3p1d
- metric_profile: conservative
- scenario_count: 3
- rows_total: 660
- calc_rows_per_second: 399.121
- mean_abs_negative_energy_j: 3.903621e+45
- max_abs_negative_energy_j: 5.463487e+45
- feasible_rate: 0.000000
- forensic_chain_final_hash: 473bd76f5cfe037431125a5656c8c867d798a2ab4e80213917315ec7c3fb4cbf
- max_memory_mb: 18.855
- qubits_simulated: 0

Interpretation:
- If feasible_rate is near 0, current-physics feasibility is not supported by this proxy model.
- 1+1D and 3+1D are included simultaneously to tighten bounds compared to a single-scaling estimate.
- Multi-shell scenarios test whether geometry layering reduces exotic energy demands (it does not here).
