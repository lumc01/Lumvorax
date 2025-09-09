# LUM/VORAX Project - Replit Environment Setup

## Overview
LUM/VORAX is a presence-based computing system implemented in C that introduces a novel computational paradigm where information is represented as "presence" units (LUMs) rather than traditional binary data. The system uses spatial transformations and natural operations to manipulate information.

## Project Architecture
This is a **backend-only C application** with no web frontend components. The system includes:

- **Core LUM System**: Basic presence units with spatial coordinates
- **VORAX Operations**: Fusion, split, cycle, move, store/retrieve operations 
- **Binary Conversion**: Bidirectional conversion between traditional data and LUMs
- **Parser**: Custom VORAX language parser for scripting operations
- **Optimization Modules**: Pareto optimization, SIMD acceleration, memory management
- **Crypto Validation**: SHA-256 implementation conforming to RFC 6234
- **Performance Metrics**: Comprehensive benchmarking and analysis tools

## Current State
- ✅ **C toolchain installed** and working (Clang compiler)
- ✅ **Project builds successfully** with make system
- ✅ **Basic system validation** passing (structure sizes, crypto tests)
- ✅ **Core functionality** operational (LUM operations, VORAX operations)
- ⚠️ **Memory management** has some issues in complex scenarios (being addressed)

## Usage Instructions

### Building the Project
```bash
make clean && make all
```

### Running Basic Validation
```bash
# Check structure ABI compliance
./bin/lum_vorax --sizeof-checks

# Validate cryptographic implementation
./bin/lum_vorax --crypto-validation
```

### Running System Demo
```bash
# Full system demonstration
./bin/lum_vorax
```

### Stress Testing (1+ Million LUMs)
```bash
# MANDATORY stress test per project requirements
./bin/lum_vorax --stress-test-million
```

## Project Structure
```
src/
├── lum/              # Core LUM data structures and operations
├── vorax/            # VORAX operation implementations  
├── parser/           # VORAX language parser
├── binary/           # Binary <-> LUM conversion utilities
├── logger/           # Logging and tracing system
├── crypto/           # Cryptographic validation (SHA-256)
├── metrics/          # Performance measurement tools
├── optimization/     # Pareto optimization, SIMD, zero-copy allocation
├── parallel/         # Multi-threading support
├── persistence/      # Data storage and retrieval
├── debug/            # Memory tracking and debugging tools
└── tests/            # Unit and stress testing modules

obj/                  # Compiled object files (auto-generated)
bin/                  # Executable files (auto-generated)  
logs/                 # Runtime logs (auto-generated)
evidence/             # Forensic validation data
reports/              # Performance and analysis reports
```

## Key Features

### LUM (Presence Units)
- **Presence**: 0 or 1 (fundamental state)
- **Spatial coordinates**: X, Y positioning
- **Structure types**: Linear, circular, group, node
- **Traceability**: Unique ID and timestamp

### VORAX Operations
- **⧉ Fusion**: Combine multiple groups
- **⇅ Split**: Distribute evenly across targets
- **⟲ Cycle**: Modular transformation
- **→ Move**: Transfer between zones
- **Store/Retrieve**: Memory management
- **Compress/Expand**: Ω compression

### Advanced Capabilities
- **Pareto Optimization**: Multi-criteria optimization with inverse scoring
- **SIMD Acceleration**: Vectorized operations using AVX2/AVX-512
- **Zero-Copy Allocation**: Memory-mapped high-performance allocation
- **Parallel Processing**: Multi-threaded POSIX operations
- **Forensic Validation**: Complete audit trail with checksums

## Development Environment

### Dependencies
- **Clang/GCC**: C99 compliant compiler
- **Make**: Build system
- **POSIX threads**: Parallel processing support
- **Math library**: Mathematical operations

### Compilation Flags
- `-std=c99`: C99 standard compliance
- `-Wall -Wextra`: Comprehensive warnings
- `-O2`: Optimization level 2
- `-g`: Debug symbols
- `-D_GNU_SOURCE`: GNU extensions
- `-lpthread -lm`: Threading and math libraries

## User Preferences
- **Forensic Compliance**: All operations must maintain complete audit trails
- **Performance Focus**: Million+ LUM stress testing is mandatory
- **Standards Compliance**: RFC 6234, POSIX.1-2017, ISO/IEC 27037
- **Memory Safety**: AddressSanitizer integration for debugging
- **Timestamp Precision**: Nanosecond-level timing for metrics

## Recent Changes
- **2025-09-09**: Project imported to Replit environment
- **2025-09-09**: C toolchain installation and build system validation
- **2025-09-09**: Basic system validation tests confirmed working
- **2025-09-09**: Workflow configuration for continuous testing
- **2025-09-09**: Added .gitignore for C development artifacts

## Technical Notes
This system implements a post-digital computational paradigm where:
1. Information is perceived as **presence** rather than numbers
2. Transformations are **spatial** and **natural**
3. Computation becomes **intuitive** and **visual**  
4. Resource conservation is **mathematically guaranteed**

The system demonstrates capabilities unique to presence-based computing that are not achievable with traditional binary architectures, validated through comprehensive benchmarking and forensic analysis.