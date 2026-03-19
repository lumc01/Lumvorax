# Warp feasibility forensic summary

- run_id: 20260303_224828
- model: warp_feasibility_proxy_v2_gr1p1d_gr3p1d
- metric_profile: balanced
- scenario_count: 3
- rows_total: 1260
- calc_rows_per_second: 400.123
- mean_abs_negative_energy_j: 2.372290e+46
- max_abs_negative_energy_j: 3.319908e+46
- feasible_rate: 0.000000
- forensic_chain_final_hash: 01381ac6ecdee56e37e7f7b48bda12bc473a237177030240bb2ff791cdda27f4
- max_memory_mb: 21.262
- qubits_simulated: 0

Interpretation:
- If feasible_rate is near 0, current-physics feasibility is not supported by this proxy model.
- 1+1D and 3+1D are included simultaneously to tighten bounds compared to a single-scaling estimate.
- Multi-shell scenarios test whether geometry layering reduces exotic energy demands (it does not here).
