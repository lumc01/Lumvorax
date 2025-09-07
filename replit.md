# LUM/VORAX System

## Overview

LUM/VORAX is an experimental computing paradigm that replaces traditional bit-based computation with "presence units" (LUM) and spatial transformations. The system implements a new approach to computation based on visual/spatial representation of information rather than binary logic. LUM units have presence states (0 or 1), spatial coordinates (X, Y), and structural types (linear, circular, group, node). The VORAX language provides operations for manipulating these LUM structures through fusion, splitting, cycling, flow, and memory operations. This is a C-based implementation with a focus on demonstrating the core concepts of presence-based computing.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core LUM System
The foundation is built around LUM (Light/Presence Units) as the fundamental computing element instead of bits. Each LUM contains:
- Presence state (0 or 1)
- Spatial coordinates (X, Y position)
- Structure type classification
- Unique ID and timestamp for traceability

This design choice moves away from traditional binary computation toward a spatial-visual computing model where information is represented as positioned presence units that can be manipulated through geometric operations.

### VORAX Operations Engine
The system implements eight core operations that define how LUM units can be transformed:
- **⧉ Fusion**: Combines two LUM groups
- **⇅ Split**: Distributes LUM units evenly between zones
- **⟲ Cycle**: Applies modular transformations
- **→ Flux**: Moves LUM units between spatial zones
- **Store/Retrieve**: Memory management operations
- **Compress/Expand**: Ω compression for space optimization

These operations maintain conservation principles - LUM units cannot be arbitrarily created or destroyed, providing computational determinism and traceability.

### Language Parser and Interpreter
A custom parser processes VORAX language syntax, converting high-level presence-based operations into executable transformations. The parser handles zone declarations, memory variable definitions, and operation sequences while maintaining type safety and resource conservation rules.

### Binary Conversion Layer
The system includes bidirectional conversion between traditional binary representations and LUM structures. This allows integration with existing computing systems while maintaining the presence-based computational model internally.

### Logging and Traceability System
Comprehensive logging tracks every LUM manipulation with unique identifiers and timestamps. This provides complete audit trails of computational processes and enables debugging of presence-based algorithms.

## Recent Changes

**September 7, 2025**
- ✅ **Replit Environment Setup Complete**: Fixed compilation issues and set up proper workflow
- 🔧 **Logger Module Fixed**: Corrected `lum_logger_t` struct definition to match implementation
- 🚀 **Workflow Configured**: Set up console workflow that builds and demonstrates the system
- ✅ **Build System Working**: Makefile compiles successfully with clang in Nix environment
- 🎯 **Demo Functional**: All core features working (basic LUM ops, VORAX ops, binary conversion, parser, crypto validation)
- ⚠️ **Known Issue**: Memory cleanup issue at end of full demo (handled with timeout in workflow)

## External Dependencies

### Build System
- **Make**: Build automation and compilation management
- **Clang/GCC**: C compiler toolchain for native code generation

### Development Tools
- Standard C libraries for core functionality
- No external runtime dependencies - designed as a self-contained system

### Optional Integration Points
- Binary conversion interfaces for integration with traditional computing systems
- Logging output can be exported to external analysis tools
- Generated code templates can interface with existing C/C++ codebases

The system is architected to be minimally dependent on external systems, focusing on demonstrating the core presence-based computing concepts through a clean C implementation.