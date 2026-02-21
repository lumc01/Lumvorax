# LUM/VORAX System

## Overview
Scientific computing project combining C and Python for mathematical optimization, black hole simulation, cryptographic validation, and AI/ML competitions (AIMO, Vesuvius).

## Project Architecture
- **Language**: C (gcc, compiled via Makefile) + Python 3.12 (managed via uv/pyproject.toml)
- **Build System**: GNU Make (`make all` to build)
- **C Modules (39)**: Located in `src/` subdirectories:
  - `src/lum/` - Core LUM engine
  - `src/vorax/` - VORAX operations
  - `src/binary/` - Binary converter
  - `src/parser/` - VORAX parser
  - `src/logger/` - Logging system
  - `src/debug/` - Memory tracker, forensic logging
  - `src/crypto/` - Cryptographic validator
  - `src/persistence/` - Data persistence, WAL, recovery
  - `src/optimization/` - Memory, Pareto, SIMD, zero-copy
  - `src/parallel/` - Parallel processor
  - `src/metrics/` - Performance metrics
  - `src/advanced_calculations/` - Neural networks, audio/image, TSP, quantum simulator
  - `src/complex_modules/` - Real-time analytics, distributed computing, AI
  - `src/file_formats/` - Serialization, native file formats
  - `src/spatial/` - Instant displacement
  - `src/network/` - Resource limiter
- **Python Dependencies**: numpy, scipy, torch (CPU), scikit-image, pandas, matplotlib, etc. (see pyproject.toml)
- **Output**: Binaries in `bin/`, logs in `logs/`

## Known Issues
- `src/main.c` references `physics/kerr_metric.h` which does not exist in the project (pre-existing issue from before import)

## Recent Changes
- 2026-02-21: Migrated to Replit environment. Installed Python deps via uv. Verified C build. Cleaned up excess workflows.

## User Preferences
- Project language: French documentation
- Scientific/mathematical computing focus
