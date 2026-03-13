# LUM/VORAX Quantum Research System

## Project Overview
A scientific computing project focused on quantum physics research, black hole simulation (Kerr metric/Gargantua), and advanced numerical methods. The system includes:

- **LUM/VORAX engine**: Core C-based computation framework with 39+ modules
- **Quantum simulations**: Hubbard model, high-temperature superconductors (HTS)
- **Forensic logging**: Bit-level traceability and hardware timestamping (nanosecond precision)
- **Advanced algorithms**: Neural networks, SIMD optimization, parallel processing, TSP, Pareto optimization

## Architecture
- **Language**: C (primary), Python (utilities), Bash (orchestration)
- **Build system**: GNU Make
- **Binaries**: Pre-compiled in `bin/` directory
  - `bin/lum_vorax_complete` — Main computation engine
  - `bin/test_forensic_complete_system` — Forensic test suite
  - `bin/test_integration_complete_39_modules` — Integration tests
  - `bin/test_quantum` — Quantum module tests

## Key Directories
- `src/` — C source code (core, lum, vorax, crypto, debug, advanced_calculations, etc.)
- `bin/` — Compiled executables
- `dataset/`, `DATASET/` — Research data
- `evidence/` — Scientific evidence/results
- `docs/` — Documentation

## Workflows
- **Run Python**: Runs `main.py` (Python utility entry point)
- **Quantum Research Cycle**: Runs the full quantum research bash script

## Dependencies (Nix packages)
arrow-cpp, cairo, clang, ffmpeg-full, gcc, gdb, ghostscript, glib, glibc, glibcLocales, gnumake, gobject-introspection, gtk3, kaggle, libxcrypt, nano, openssh, pkg-config, qhull, tcl, tk, tree, vim-full, xsimd, zlib

## Kaggle Integration
- Username: ndarray2000
- Config stored in `$KAGGLE_CONFIG_DIR/kaggle.json`
