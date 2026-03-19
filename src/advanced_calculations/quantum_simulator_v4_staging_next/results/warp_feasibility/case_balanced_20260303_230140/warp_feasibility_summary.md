# Warp feasibility forensic summary

- run_id: case_balanced_20260303_230140
- model: warp_feasibility_proxy_v2_gr1p1d_gr3p1d
- metric_profile: balanced
- scenario_count: 3
- rows_total: 660
- calc_rows_per_second: 389.742
- mean_abs_negative_energy_j: 2.371979e+46
- max_abs_negative_energy_j: 3.319772e+46
- feasible_rate: 0.000000
- forensic_chain_final_hash: c251402fc01a936d123bea2034f02374d808efca99410e85fcb593fcc9be91fa
- max_memory_mb: 19.629
- qubits_simulated: 0

Interpretation:
- If feasible_rate is near 0, current-physics feasibility is not supported by this proxy model.
- 1+1D and 3+1D are included simultaneously to tighten bounds compared to a single-scaling estimate.
- Multi-shell scenarios test whether geometry layering reduces exotic energy demands (it does not here).
