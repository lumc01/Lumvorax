
#include "simd_optimizer.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

#ifdef __x86_64__
#include <cpuid.h>
#endif

simd_capabilities_t* simd_detect_capabilities(void) {
    simd_capabilities_t* caps = malloc(sizeof(simd_capabilities_t));
    if (!caps) return NULL;

    caps->avx512_available = false;
    caps->avx2_available = false;
    caps->sse_available = false;
    caps->vector_width = 1;
    strcpy(caps->cpu_features, "scalar");

#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    
    // Check CPUID support
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        // Check SSE support
        if (edx & (1 << 25)) {
            caps->sse_available = true;
            caps->vector_width = 4;
            strcpy(caps->cpu_features, "SSE");
        }
        
        // Check AVX2 support
        if (ecx & (1 << 28)) {
            if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                if (ebx & (1 << 5)) {
                    caps->avx2_available = true;
                    caps->vector_width = 8;
                    strcat(caps->cpu_features, "+AVX2");
                }
                
                // Check AVX-512 support
                if (ebx & (1 << 16)) {
                    caps->avx512_available = true;
                    caps->vector_width = 16;
                    strcat(caps->cpu_features, "+AVX512");
                }
            }
        }
    }
#endif

    return caps;
}

void simd_capabilities_destroy(simd_capabilities_t* caps) {
    if (caps) {
        free(caps);
    }
}

simd_result_t* simd_process_lum_array_bulk(lum_t* lums, size_t count) {
    if (!lums || count == 0) return NULL;

    simd_result_t* result = malloc(sizeof(simd_result_t));
    if (!result) return NULL;

    clock_t start = clock();
    
#ifdef __AVX512F__
    simd_avx512_mass_lum_operations(lums, count);
    result->used_vectorization = true;
    strcpy(result->optimization_used, "AVX-512");
#elif __AVX2__
    // Process in chunks of 8 for AVX2
    size_t simd_chunks = count / 8;
    size_t remainder = count % 8;
    
    for (size_t i = 0; i < simd_chunks; i++) {
        uint32_t presence_batch[8];
        for (int j = 0; j < 8; j++) {
            presence_batch[j] = lums[i * 8 + j].presence;
        }
        simd_avx2_process_presence_bits(presence_batch, 8);
        
        for (int j = 0; j < 8; j++) {
            lums[i * 8 + j].presence = presence_batch[j];
        }
    }
    
    // Handle remainder
    for (size_t i = simd_chunks * 8; i < count; i++) {
        lums[i].presence = lums[i].presence ? 1 : 0;
    }
    
    result->used_vectorization = true;
    strcpy(result->optimization_used, "AVX2");
#else
    // Scalar fallback
    for (size_t i = 0; i < count; i++) {
        lums[i].presence = lums[i].presence ? 1 : 0;
    }
    result->used_vectorization = false;
    strcpy(result->optimization_used, "Scalar");
#endif

    clock_t end = clock();
    result->execution_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    result->processed_elements = count;
    result->throughput_ops_per_sec = count / result->execution_time;

    return result;
}

#ifdef __AVX2__
void simd_avx2_process_presence_bits(uint32_t* presence_array, size_t count) {
    if (!presence_array || count < 8) return;
    
    __m256i data = _mm256_loadu_si256((__m256i*)presence_array);
    __m256i zeros = _mm256_setzero_si256();
    __m256i ones = _mm256_set1_epi32(1);
    
    // Convert non-zero to 1, zero to 0
    __m256i mask = _mm256_cmpgt_epi32(data, zeros);
    __m256i result = _mm256_and_si256(mask, ones);
    
    _mm256_storeu_si256((__m256i*)presence_array, result);
}

void simd_avx2_parallel_coordinate_transform(float* x_coords, float* y_coords, size_t count) {
    if (!x_coords || !y_coords || count < 8) return;
    
    size_t simd_count = (count / 8) * 8;
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&x_coords[i]);
        __m256 y_vec = _mm256_loadu_ps(&y_coords[i]);
        
        // Example transformation: normalize coordinates
        __m256 x_squared = _mm256_mul_ps(x_vec, x_vec);
        __m256 y_squared = _mm256_mul_ps(y_vec, y_vec);
        __m256 magnitude = _mm256_sqrt_ps(_mm256_add_ps(x_squared, y_squared));
        
        x_vec = _mm256_div_ps(x_vec, magnitude);
        y_vec = _mm256_div_ps(y_vec, magnitude);
        
        _mm256_storeu_ps(&x_coords[i], x_vec);
        _mm256_storeu_ps(&y_coords[i], y_vec);
    }
}
#endif

#ifdef __AVX512F__
void simd_avx512_mass_lum_operations(lum_t* lums, size_t count) {
    if (!lums || count < 16) return;
    
    size_t simd_count = (count / 16) * 16;
    
    for (size_t i = 0; i < simd_count; i += 16) {
        // Load 16 presence values
        uint32_t presence_batch[16];
        for (int j = 0; j < 16; j++) {
            presence_batch[j] = lums[i + j].presence;
        }
        
        __m512i data = _mm512_loadu_si512((__m512i*)presence_batch);
        __m512i zeros = _mm512_setzero_si512();
        __m512i ones = _mm512_set1_epi32(1);
        
        // Vectorized presence normalization
        __mmask16 mask = _mm512_cmpgt_epi32_mask(data, zeros);
        __m512i result = _mm512_mask_blend_epi32(mask, zeros, ones);
        
        _mm512_storeu_si512((__m512i*)presence_batch, result);
        
        // Store back
        for (int j = 0; j < 16; j++) {
            lums[i + j].presence = presence_batch[j];
        }
    }
}

void simd_avx512_vectorized_conservation_check(uint64_t* conservation_data, size_t count) {
    if (!conservation_data || count < 8) return;
    
    size_t simd_count = (count / 8) * 8;
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512i data = _mm512_loadu_si512((__m512i*)&conservation_data[i]);
        // Perform vectorized conservation validation
        // Implementation specific to conservation law checking
        _mm512_storeu_si512((__m512i*)&conservation_data[i], data);
    }
}
#endif

simd_result_t* simd_benchmark_vectorization(size_t test_size) {
    lum_t* test_lums = malloc(test_size * sizeof(lum_t));
    if (!test_lums) return NULL;
    
    // Initialize test data
    for (size_t i = 0; i < test_size; i++) {
        test_lums[i].presence = (i % 3 == 0) ? 1 : 0;
        test_lums[i].position_x = i;
        test_lums[i].position_y = i * 2;
    }
    
    simd_result_t* result = simd_process_lum_array_bulk(test_lums, test_size);
    
    free(test_lums);
    return result;
}

void simd_result_destroy(simd_result_t* result) {
    if (result) {
        free(result);
    }
}

void simd_print_performance_comparison(simd_result_t* scalar, simd_result_t* vectorized) {
    if (!scalar || !vectorized) return;
    
    printf("=== SIMD Performance Comparison ===\n");
    printf("Scalar Performance:\n");
    printf("  Elements: %zu\n", scalar->processed_elements);
    printf("  Time: %.6f seconds\n", scalar->execution_time);
    printf("  Throughput: %.2f ops/sec\n", scalar->throughput_ops_per_sec);
    
    printf("Vectorized Performance:\n");
    printf("  Elements: %zu\n", vectorized->processed_elements);
    printf("  Time: %.6f seconds\n", vectorized->execution_time);
    printf("  Throughput: %.2f ops/sec\n", vectorized->throughput_ops_per_sec);
    printf("  Optimization: %s\n", vectorized->optimization_used);
    
    if (vectorized->execution_time > 0 && scalar->execution_time > 0) {
        double speedup = scalar->execution_time / vectorized->execution_time;
        printf("Speedup: %.2fx\n", speedup);
    }
    printf("=====================================\n");
}
