# LUM/VORAX System

## Overview
LUM/VORAX is a high-performance C-based computation system with 39 integrated modules. It includes advanced features like SIMD optimization, parallel processing, memory tracking, forensic logging, and more.

## Project Structure
- `src/` - Source code for all modules
  - `lum/` - Core LUM functionality
  - `vorax/` - VORAX operations
  - `optimization/` - SIMD, memory, and Pareto optimizers
  - `parallel/` - Parallel processing
  - `persistence/` - Data persistence and transaction handling
  - `debug/` - Memory tracking and forensic logging
  - `crypto/` - Cryptographic validation
  - `advanced_calculations/` - Neural networks, image/audio processing
  - `complex_modules/` - Real-time analytics, distributed computing, AI
  - `file_formats/` - File serialization and handling
  - `spatial/` - Spatial displacement algorithms
  - `network/` - Network resource limiting
- `bin/` - Compiled binaries
- `logs/` - Log output directories
- `reports/` - Generated reports
- `tests/` - Test files

## Building
```bash
make all        # Build all targets
make clean      # Clean build artifacts
make test       # Run all tests
```

## Running
```bash
./bin/lum_vorax_complete                      # Default test
./bin/lum_vorax_complete --basic-test         # Basic LUM core test
./bin/lum_vorax_complete --progressive-stress-all  # Stress test with all modules
```

## Dependencies
- C compiler (clang or gcc)
- GNU Make
- pthreads, math library, rt library

## Recent Changes
- January 20, 2026: Initial import to Replit environment, installed C toolchain and make
