# LUM/VORAX System - Architecture V32

## Overview
LUM/VORAX est un système de calcul haute performance avec 45+ modules intégrés, optimisé pour AVX2 et le multi-threading massif.

## Project Structure (Updated V32)
- `src/` - Source code for all modules
  - `optimization/` - **[V32]** Async Logging, Slab Allocator, SIMD Batch, Lock-free Queue, LZ4, MMap.
  - `security/` - **[V32]** Audit & Hardening.
  - `monitoring/` - **[V32]** Resource Monitoring & Alerting.
  - `distributed/` - **[V32]** Cluster Computing Node.
  - `wasm/` - **[V32]** WASM Export Module.
  - `versioning/` - **[V32]** Version Manager & API Contract.
  - `cicd/` - **[V32]** Benchmark Runner & Regression Detector.
  - `lum/` - Core LUM functionality.
  - `vorax/` - VORAX operations.
- `reports/` - Generated reports (including `FINAL_VALIDATION_REPORT_V32.md`).

## Building & Testing
```bash
make all
./v32_test
```

## Recent Changes
- **January 27, 2026**: Phase NX-11 finalisée selon la norme **NX-11-HFBL-360**. Instrumentation forensique bit-à-bit, horodatage nanoseconde et traçabilité causale complète.
- **NX-11-HFBL-360**: High-Frequency Bit-Level 360 Forensic Logging implémenté.
- **Validation**: Index Merkle et logs multi-flux (ATOM, DISS, MEM, COMM, CTRL) opérationnels.
