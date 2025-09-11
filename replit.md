# LUM/VORAX Project - Replit Environment Setup

## Overview
LUM/VORAX is a presence-based computing system implemented in C that introduces a novel computational paradigm where information is represented as "presence" units (LUMs) rather than traditional binary data. The system uses spatial transformations and natural operations to manipulate information. Its purpose is to demonstrate a post-digital computational paradigm where information is presence, transformations are spatial, computation is intuitive, and resource conservation is mathematically guaranteed. The project aims to achieve unique capabilities not possible with traditional binary architectures, pushing towards a "practically inviolable" platform for critical applications.

## User Preferences
- **Forensic Compliance**: All operations must maintain complete audit trails
- **Performance Focus**: Million+ LUM stress testing is mandatory requirement
- **Standards Compliance**: RFC 6234, POSIX.1-2017, ISO/IEC 27037
- **Memory Safety**: AddressSanitizer integration for debugging
- **Timestamp Precision**: Nanosecond-level timing for metrics

## System Architecture
This is a **backend-only C application** with no web frontend components. The system is designed for high resilience and security, focusing on C-centric implementations and formal verification.

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
- **Coding Standards**: C99 standard compliance with extensive warnings (`-Wall -Wextra`), with critical modules aiming for MISRA C/CERT C compliance and potential use of CompCert for formal verification.
- **Security Enhancements**: Focus on secure C coding practices, static analysis (Frama-C, CBMC), fuzzing (libFuzzer, AFL++), integration with hardware security modules (TPM/HSM), secure boot, and mutual TLS for network communications (via C libraries).
- **Resilience**: Aims for a formally verified microkernel (seL4) for sensitive components, reproducible builds, immutable logging, and robust key management.

## External Dependencies
- **Clang/GCC**: C99 compliant compiler.
- **Make**: Build system.
- **POSIX threads**: For parallel processing support.
- **Math library**: For mathematical operations.
- **libsodium/BoringSSL/OpenSSL (or HACL\*)**: For cryptographic operations.
- **TPM 2.0 / HSM API**: For secure boot, key storage, and attestation.
- **seL4 (or similar microkernel)**: For critical component isolation (if integrated).