# Project Overview

This workspace contains research scripts for the AI Mathematical Olympiad (AIMO) and a Quantum HTS (High-Temperature Superconductor) Hubbard model simulation project.

## Primary Project: Quantum HTS Hubbard Model Simulation

Located in `src/advanced_calculations/quantum_problem_hubbard_hts/`.

### Architecture

- **Fullscale runner**: `src/hubbard_hts_research_cycle.c` — 13-module, 30-step research cycle with RK2 midpoint integration, per-step normalization, QMC/DMRG benchmarks
- **Advanced parallel runner**: `src/hubbard_hts_research_cycle_advanced_parallel.c` — extended cycle with physics gate summaries, entropy, spatial correlations, cross-module analytics
- **Makefile**: `LDFLAGS = -Wl,-Bdynamic,--as-needed` (fixes NixOS static glibc segfault; reduced binary 997KB→64KB)
- **Entry script**: `run_research_cycle.sh` — 35-step pipeline: build → fullscale → advanced_parallel → audit → CHAT

### Benchmark Reference Files (calibrated 2026-03-13)

- `benchmarks/qmc_dmrg_reference_v2.csv` — energies now in **eV/site** (was meV-total: 652800 → 0.9985)
- `benchmarks/external_module_benchmarks_v1.csv` — energies in **eV/site**, error_bar=0.15

### Key Corrections Applied (session 2026-03-13)

| Fix | File | Description |
|-----|------|-------------|
| F0 | Makefile | LDFLAGS anti-segfault NixOS (historical) |
| F1 | hubbard_hts_research_cycle.c | `simulate_problem_independent`: Euler→RK2+normalization (delta: 16.6%→<0.01%) |
| F2 | hubbard_hts_research_cycle.c | Removed `* 1000.0` from energy comparison; seuils eV/site (rmse≤0.30, mae≤0.25, within≥40%) |
| F3 | hubbard_hts_research_cycle.c | External module thresholds: 40000→0.30, 0%→40% |
| F4 | hubbard_hts_research_cycle_advanced_parallel.c | QMC/DMRG thresholds: 1300000→0.30 |
| F5 | hubbard_hts_research_cycle_advanced_parallel.c | External module thresholds: eV/site scale |
| F6 | benchmarks/qmc_dmrg_reference_v2.csv | Energy references: meV-total → eV/site |
| F7 | benchmarks/external_module_benchmarks_v1.csv | Energy references: meV-total → eV/site |

### Test Status

- **Run 2866** (fullscale): 30 PASS, 49 OBSERVED, 1 FAIL → FIXED (RK2 integration)
- **Run 3001** (advanced): 31 PASS, 49 OBSERVED, 0 FAIL
- **New run** (post-corrections): in progress, 0 FAILs expected

### CHAT Reports

Located in `CHAT/`. Key reports:
- `AUTO_PROMPT_ANALYSE_CROISEE_RUNS_3001_2866_20260313.md` — cross-parallel analysis of runs 3001 and 2866 with all corrections documented

## Secondary Project

- `main.py` — Python entry point (placeholder)

## Environment

- **Runtime**: Python 3.11, GCC
- **Key packages**: kaggle, gcc, clang, ffmpeg, and many scientific/graphics libraries via Nix
- **Workflow**: "Quantum Research Cycle" — 35-step pipeline (runs both fullscale and advanced_parallel)

## Security Notes

- `KAGGLE_USERNAME` and `KAGGLE_CONFIG_DIR` are stored as non-sensitive env vars
- API tokens (KAGGLE_API_TOKEN, ARISTOTLE_API_KEY) should be stored as secrets via the Replit Secrets panel, not in `.replit`
