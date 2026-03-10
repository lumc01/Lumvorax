# Project Overview

This is a Python 3.11 workspace containing various research and analysis scripts, primarily related to the AI Mathematical Olympiad (AIMO) competition and related mathematical/algorithmic work.

## Structure

- `main.py` - Entry point (simple hello world placeholder)
- `src/advanced_calculations/quantum_problem_hubbard_hts/src/hubbard_hts_research_cycle.c` - Core simulation logic with 5 stability corrections: spectral radius (1e-6), damping (0.015), local dissipation, energy normalization by sites, drift reduction (1e-10).
- `src/advanced_calculations/quantum_problem_hubbard_hts/results/` - Research artifacts, logs, and comparative reports (research_20260310T013238Z_2300, research_20260310T013832Z_2311, research_20260310T012023Z_1673).
- `RAPPORT_FINAL_AVANT_APRES_COMPLET.md` - Comprehensive before/after analysis: removed artificial 1e6 factor, validated unit consistency, confirmed 3-run determinism.

## Running

The project runs via the "Run Python" workflow which executes `main.py`.

## Environment

- **Runtime**: Python 3.11
- **Key packages**: kaggle, gcc, clang, ffmpeg, and many scientific/graphics libraries via Nix

## Security Notes

- `KAGGLE_USERNAME` and `KAGGLE_CONFIG_DIR` are stored as non-sensitive env vars
- API tokens (KAGGLE_API_TOKEN, ARISTOTLE_API_KEY) should be stored as secrets via the Replit Secrets panel, not in `.replit`
