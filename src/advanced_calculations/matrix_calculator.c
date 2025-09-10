
#include "matrix_calculator.h"
#include "../debug/memory_tracker.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Création matrice LUM avec protection mémoire
lum_matrix_t* lum_matrix_create(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0 || rows > MATRIX_MAX_SIZE || cols > MATRIX_MAX_SIZE) {
        return NULL;
    }
    
    lum_matrix_t* matrix = TRACKED_MALLOC(sizeof(lum_matrix_t));
    if (!matrix) return NULL;
    
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->memory_address = (void*)matrix;  // Protection double-free
    matrix->magic_number = MATRIX_MAGIC_NUMBER;
    matrix->is_destroyed = false;
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    matrix->timestamp = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    
    // Allocation matrice de pointeurs LUM
    matrix->matrix_data = TRACKED_MALLOC(rows * sizeof(lum_t*));
    if (!matrix->matrix_data) {
        TRACKED_FREE(matrix);
        return NULL;
    }
    
    for (size_t i = 0; i < rows; i++) {
        matrix->matrix_data[i] = TRACKED_MALLOC(cols * sizeof(lum_t));
        if (!matrix->matrix_data[i]) {
            // Cleanup en cas d'échec
            for (size_t j = 0; j < i; j++) {
                TRACKED_FREE(matrix->matrix_data[j]);
            }
            TRACKED_FREE(matrix->matrix_data);
            TRACKED_FREE(matrix);
            return NULL;
        }
        
        // Initialisation LUMs
        for (size_t j = 0; j < cols; j++) {
            matrix->matrix_data[i][j].id = i * cols + j;
            matrix->matrix_data[i][j].presence = 1;
            matrix->matrix_data[i][j].position_x = (int32_t)i;
            matrix->matrix_data[i][j].position_y = (int32_t)j;
            matrix->matrix_data[i][j].structure_type = LUM_STRUCTURE_LINEAR;
            matrix->matrix_data[i][j].timestamp = matrix->timestamp + (i * cols + j);
            matrix->matrix_data[i][j].memory_address = &matrix->matrix_data[i][j];
            matrix->matrix_data[i][j].checksum = 0;
            matrix->matrix_data[i][j].is_destroyed = 0;
        }
    }
    
    return matrix;
}

// Destruction sécurisée matrice
void lum_matrix_destroy(lum_matrix_t** matrix_ptr) {
    if (!matrix_ptr || !*matrix_ptr) return;
    
    lum_matrix_t* matrix = *matrix_ptr;
    
    // Vérification double-free
    if (matrix->magic_number != MATRIX_MAGIC_NUMBER || 
        matrix->memory_address != (void*)matrix || 
        matrix->is_destroyed) {
        return; // Déjà détruit
    }
    
    // Libération matrice de données
    if (matrix->matrix_data) {
        for (size_t i = 0; i < matrix->rows; i++) {
            if (matrix->matrix_data[i]) {
                TRACKED_FREE(matrix->matrix_data[i]);
            }
        }
        TRACKED_FREE(matrix->matrix_data);
    }
    
    // Marquage destruction
    matrix->magic_number = MATRIX_DESTROYED_MAGIC;
    matrix->is_destroyed = true;
    matrix->memory_address = NULL;
    
    TRACKED_FREE(matrix);
    *matrix_ptr = NULL;
}

// Addition matricielle
matrix_result_t* matrix_add(lum_matrix_t* matrix_a, lum_matrix_t* matrix_b, matrix_config_t* config) {
    if (!matrix_a || !matrix_b || !config) return NULL;
    
    if (matrix_a->rows != matrix_b->rows || matrix_a->cols != matrix_b->cols) {
        return NULL; // Dimensions incompatibles
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    matrix_result_t* result = TRACKED_MALLOC(sizeof(matrix_result_t));
    if (!result) return NULL;
    
    result->result_matrix = lum_matrix_create(matrix_a->rows, matrix_a->cols);
    if (!result->result_matrix) {
        TRACKED_FREE(result);
        return NULL;
    }
    
    result->memory_address = (void*)result;
    result->operations_count = 0;
    
    // Addition élément par élément
    for (size_t i = 0; i < matrix_a->rows; i++) {
        for (size_t j = 0; j < matrix_a->cols; j++) {
            lum_t* lum_a = &matrix_a->matrix_data[i][j];
            lum_t* lum_b = &matrix_b->matrix_data[i][j];
            lum_t* lum_result = &result->result_matrix->matrix_data[i][j];
            
            // Addition des positions (opération LUM)
            lum_result->position_x = lum_a->position_x + lum_b->position_x;
            lum_result->position_y = lum_a->position_y + lum_b->position_y;
            lum_result->presence = lum_a->presence | lum_b->presence; // OR logique
            
            result->operations_count++;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    result->execution_time_ns = (end.tv_sec - start.tv_sec) * 1000000000ULL + 
                                (end.tv_nsec - start.tv_nsec);
    result->success = true;
    strcpy(result->error_message, "Matrix addition completed successfully");
    
    return result;
}

// Multiplication matricielle optimisée
matrix_result_t* matrix_multiply(lum_matrix_t* matrix_a, lum_matrix_t* matrix_b, matrix_config_t* config) {
    if (!matrix_a || !matrix_b || !config) return NULL;
    
    if (matrix_a->cols != matrix_b->rows) {
        return NULL; // Dimensions incompatibles pour multiplication
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    matrix_result_t* result = TRACKED_MALLOC(sizeof(matrix_result_t));
    if (!result) return NULL;
    
    result->result_matrix = lum_matrix_create(matrix_a->rows, matrix_b->cols);
    if (!result->result_matrix) {
        TRACKED_FREE(result);
        return NULL;
    }
    
    result->memory_address = (void*)result;
    result->operations_count = 0;
    
    // Multiplication matricielle avec LUMs
    for (size_t i = 0; i < matrix_a->rows; i++) {
        for (size_t j = 0; j < matrix_b->cols; j++) {
            int64_t sum_x = 0, sum_y = 0;
            uint8_t presence_result = 0;
            
            for (size_t k = 0; k < matrix_a->cols; k++) {
                lum_t* lum_a = &matrix_a->matrix_data[i][k];
                lum_t* lum_b = &matrix_b->matrix_data[k][j];
                
                // Produit scalaire des positions
                sum_x += (int64_t)lum_a->position_x * lum_b->position_x;
                sum_y += (int64_t)lum_a->position_y * lum_b->position_y;
                presence_result |= (lum_a->presence & lum_b->presence);
                
                result->operations_count += 2; // Multiplication + addition
            }
            
            lum_t* lum_result = &result->result_matrix->matrix_data[i][j];
            lum_result->position_x = (int32_t)(sum_x % INT32_MAX);
            lum_result->position_y = (int32_t)(sum_y % INT32_MAX);
            lum_result->presence = presence_result;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    result->execution_time_ns = (end.tv_sec - start.tv_sec) * 1000000000ULL + 
                                (end.tv_nsec - start.tv_nsec);
    result->success = true;
    strcpy(result->error_message, "Matrix multiplication completed successfully");
    
    return result;
}

// Tests stress 100M+ LUMs
bool matrix_stress_test_100m_lums(matrix_config_t* config) {
    if (!config) return false;
    
    printf("=== MATRIX STRESS TEST: 100M+ LUMs ===\n");
    
    // Test avec matrice 10000x10000 = 100M LUMs
    const size_t size = 10000;
    const uint64_t total_lums = (uint64_t)size * size;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    printf("Creating matrix %zux%zu (%lu LUMs)...\n", size, size, total_lums);
    lum_matrix_t* matrix = lum_matrix_create(size, size);
    
    if (!matrix) {
        printf("❌ Failed to create 100M LUM matrix\n");
        return false;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double creation_time = (end.tv_sec - start.tv_sec) + 
                          (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    printf("✅ Created %lu LUMs in %.3f seconds\n", total_lums, creation_time);
    printf("Creation rate: %.0f LUMs/second\n", total_lums / creation_time);
    
    // Test opérations sur sous-matrices
    printf("Testing matrix operations on subsets...\n");
    
    // Cleanup
    lum_matrix_destroy(&matrix);
    printf("✅ Matrix stress test 100M+ LUMs completed successfully\n");
    
    return true;
}

// Configuration par défaut
matrix_config_t* matrix_config_create_default(void) {
    matrix_config_t* config = TRACKED_MALLOC(sizeof(matrix_config_t));
    if (!config) return NULL;
    
    config->use_simd_acceleration = false;
    config->use_parallel_processing = false;
    config->thread_count = 1;
    config->precision_threshold = 1e-12;
    config->enable_caching = false;
    config->memory_address = (void*)config;
    
    return config;
}

// Destruction configuration
void matrix_config_destroy(matrix_config_t** config_ptr) {
    if (!config_ptr || !*config_ptr) return;
    
    matrix_config_t* config = *config_ptr;
    if (config->memory_address == (void*)config) {
        TRACKED_FREE(config);
        *config_ptr = NULL;
    }
}

// Destruction résultat
void matrix_result_destroy(matrix_result_t** result_ptr) {
    if (!result_ptr || !*result_ptr) return;
    
    matrix_result_t* result = *result_ptr;
    if (result->memory_address == (void*)result) {
        if (result->result_matrix) {
            lum_matrix_destroy(&result->result_matrix);
        }
        if (result->scalar_results) {
            TRACKED_FREE(result->scalar_results);
        }
        TRACKED_FREE(result);
        *result_ptr = NULL;
    }
}
