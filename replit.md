# LUM/VORAX Project - Replit Environment Setup

## Overview
LUM/VORAX is a presence-based computing system implemented in C that introduces a novel computational paradigm where information is represented as "presence" units (LUMs) rather than traditional binary data. The system uses spatial transformations and natural operations to manipulate information. Its purpose is to demonstrate a post-digital computational paradigm where information is presence, transformations are spatial, computation is intuitive, and resource conservation is mathematically guaranteed. The project aims to achieve unique capabilities not possible with traditional binary architectures.

## User Preferences
- **Forensic Compliance**: All operations must maintain complete audit trails
- **Performance Focus**: Million+ LUM stress testing is mandatory requirement
- **Standards Compliance**: RFC 6234, POSIX.1-2017, ISO/IEC 27037
- **Memory Safety**: AddressSanitizer integration for debugging
- **Timestamp Precision**: Nanosecond-level timing for metrics

## System Architecture
This is a **backend-only C application** with no web frontend components. The system includes:

- **Core LUM System**: Basic presence units with spatial coordinates (X, Y) and fundamental state (0 or 1). Supports linear, circular, group, and node structures with unique ID and timestamp traceability.
- **VORAX Operations**:
    - **⧉ Fusion**: Combine multiple groups
    - **⇅ Split**: Distribute evenly across targets
    - **⟲ Cycle**: Modular transformation
    - **→ Move**: Transfer between zones
    - **Store/Retrieve**: Memory management
    - **Compress/Expand**: Ω compression
- **Binary Conversion**: Bidirectional conversion between traditional data and LUMs.
- **Parser**: Custom VORAX language parser for scripting operations.
- **Optimization Modules**: Pareto optimization (multi-criteria with inverse scoring), SIMD acceleration (AVX2/AVX-512), and Zero-Copy Allocation (memory-mapped high-performance allocation).
- **Parallel Processing**: Multi-threaded POSIX operations.
- **Crypto Validation**: SHA-256 implementation conforming to RFC 6234, with forensic validation for complete audit trails.
- **Memory Management System**: Advanced, generational anti-collision memory tracking system with real-time leak detection and secure double-free protection. All critical modules use tracked allocations.
- **Performance Metrics**: Comprehensive benchmarking, timing, and measurement tools, capable of stress testing millions of LUMs.
- **Build System**: Uses a Makefile supporting standard, debug (with AddressSanitizer), and release builds, as well as specific test targets.
- **Coding Standards**: C99 standard compliance with extensive warnings (`-Wall -Wextra`).

## External Dependencies
- **Clang/GCC**: C99 compliant compiler.
- **Make**: Build system.
- **POSIX threads**: For parallel processing support.
- **Math library**: For mathematical operations.