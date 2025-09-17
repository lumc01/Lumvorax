# LUM/VORAX System - Replit Configuration

## Project Overview
This is a sophisticated C-based LUM/VORAX computational system that has been successfully imported and configured for the Replit environment. The system implements a complex mathematical framework for processing "LUM" units (Lumière/Light units) with advanced persistence, recovery, and optimization capabilities.

## Current State
- **Status**: ✅ Successfully imported and running
- **Language**: C with Clang compiler
- **Build System**: Makefile-based with 77+ modules
- **Architecture**: Modular system with advanced logging, persistence, and recovery

## Project Structure
- `src/` - Main source code with modular architecture
  - `lum/` - Core LUM data structures and operations
  - `vorax/` - VORAX operations and algorithms
  - `parser/` - Language parser for VORAX syntax
  - `persistence/` - Data persistence with WAL (Write-Ahead Logging)
  - `debug/` - Memory tracking and forensic logging
  - `optimization/` - SIMD and performance optimizations
- `Makefile` - Build configuration
- `bin/lum_vorax` - Compiled executable

## Recent Changes Made
1. **Fixed compilation issues**: Resolved duplicate function definitions in persistence layer
2. **Added missing function declarations**: Fixed `wal_extension_verify_integrity_complete` declaration
3. **Corrected threading flags**: Changed from `-lpthread` to `-pthread` for better compatibility
4. **Set up workflows**: Configured console-based workflow for development
5. **Deployment configuration**: Set up VM deployment with proper directory creation

## Replit Environment Setup

### Development Workflow
- **Name**: "LUM/VORAX System"
- **Command**: `./bin/lum_vorax`
- **Type**: Console application
- **Auto-starts**: Yes, configured for immediate execution

### Production Deployment
- **Target**: VM (always-running)
- **Build**: `make clean && make all`
- **Run**: `bash -c "mkdir -p logs && ./bin/lum_vorax"`
- **Considerations**: 
  - Logs directory automatically created
  - Suitable for long-running computational tasks
  - Memory tracking enabled for diagnostics

## User Preferences
- **Language**: Technical system with French documentation/output
- **Focus**: High-performance computational framework
- **Monitoring**: Memory tracking and forensic logging enabled

## Architecture Decisions
- **Memory Management**: Custom tracked allocation system
- **Persistence**: File-based with WAL extension for ACID compliance
- **Threading**: Parallel processing with proper pthread configuration
- **Optimization**: SIMD optimizations and performance metrics

## Known Considerations for Production
- Path sanitization for persistence keys should be implemented for untrusted input
- Persistent storage volumes recommended for production deployments
- Performance monitoring through built-in metrics system

## Last Updated
September 17, 2025 - Initial Replit import and configuration completed