# Project Overview

This is a Python 3.11 workspace containing various research and analysis scripts, primarily related to the AI Mathematical Olympiad (AIMO) competition and related mathematical/algorithmic work.

## Structure

- `main.py` - Entry point (simple hello world placeholder)
- `aimo3_*.py` - AIMO competition kernel scripts
- `*.md` - Analysis and research documentation files
- Various Python scripts for report generation, PDF creation, and Kaggle integration

## Running

The project runs via the "Run Python" workflow which executes `main.py`.

## Environment

- **Runtime**: Python 3.11
- **Key packages**: kaggle, gcc, clang, ffmpeg, and many scientific/graphics libraries via Nix

## Security Notes

- `KAGGLE_USERNAME` and `KAGGLE_CONFIG_DIR` are stored as non-sensitive env vars
- API tokens (KAGGLE_API_TOKEN, ARISTOTLE_API_KEY) should be stored as secrets via the Replit Secrets panel, not in `.replit`
