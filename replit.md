# Project Overview

This is a research and computation project focused on mathematical optimization, physics simulations, and puzzle-solving algorithms (primarily for Kaggle competitions and related research).

## Entry Point

- `main.py` — simple Python entry point (`python main.py`)

## Key Scripts

- `nx47_vesu_kernel_*.py` — various versions of the NX47/Vesuvius kernel for Kaggle
- `aimo3_lum_*.py` — AIMO3 LUM computation kernels
- `tools/` — utility scripts for log parsing and analysis

## Environment Variables

Secrets and API keys are stored securely in Replit's environment secrets (not in code):
- `KAGGLE_USERNAME` — Kaggle account username
- `KAGGLE_CONFIG_DIR` — Kaggle config directory path
- `KAGGLE_API_TOKEN` — Kaggle API token (secret)
- `ARISTOTLE_API_KEY` — Aristotle API key (secret)

## Workflow

- **Run Python**: `python main.py` (console output)

## Security Notes

- API keys and tokens are stored in Replit secrets, not hardcoded in `.replit` or source files
