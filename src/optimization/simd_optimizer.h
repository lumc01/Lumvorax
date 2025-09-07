
#ifndef SIMD_OPTIMIZER_H
#define SIMD_OPTIMIZER_H

#include "../lum/lum_core.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __AVX512F__
#include <immintrin.h>
#define SIMD_VECTOR_SIZE 16  // 512 bits / 32 bits per int
#elif __AVX2__
#include <immintrin.h>
#define SIMD_VECTOR_SIZE 8   // 256 bits / 32 bits per int
#else
#define SIMD_VECTOR_SIZE 1   // Fallback to scalar
#endif

// SIMD optimization configuration
typedef struct {
    bool avx512_available;
    bool avx2_available;
    bool sse_available;
    int vector_width;
    char cpu_features[256];
} simd_capabilities_t;

// SIMD operation results
typedef struct {
    size_t processed_elements;
    double execution_time;
    double throughput_ops_per_sec;
    bool used_vectorization;
    char optimization_used[64];
} simd_result_t;

// Function declarations
simd_capabilities_t* simd_detect_capabilities(void);
void simd_capabilities_destroy(simd_capabilities_t* caps);

// Vectorized LUM operations
simd_result_t* simd_process_lum_array_bulk(lum_t* lums, size_t count);
simd_result_t* simd_fuse_operations_bulk(lum_group_t** groups, size_t group_count);
simd_result_t* simd_binary_conversion_bulk(uint8_t* data, size_t data_size);

// AVX2 specific implementations
#ifdef __AVX2__
void simd_avx2_process_presence_bits(uint32_t* presence_array, size_t count);
void simd_avx2_parallel_coordinate_transform(float* x_coords, float* y_coords, size_t count);
#endif

// AVX-512 specific implementations  
#ifdef __AVX512F__
void simd_avx512_mass_lum_operations(lum_t* lums, size_t count);
void simd_avx512_vectorized_conservation_check(uint64_t* conservation_data, size_t count);
#endif

// Performance benchmarking
simd_result_t* simd_benchmark_vectorization(size_t test_size);
void simd_result_destroy(simd_result_t* result);
void simd_print_performance_comparison(simd_result_t* scalar, simd_result_t* vectorized);

#endif // SIMD_OPTIMIZER_H
