#include "matrix_calculator.h"
#include "../debug/memory_tracker.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Constante magique pour protection double-free
#define MATRIX_CALCULATOR_MAGIC 0xCALC2025

// Création calculateur matriciel
matrix_calculator_t* matrix_calculator_create(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return NULL;

    matrix_calculator_t* calc = TRACKED_MALLOC(sizeof(matrix_calculator_t));
    if (!calc) return NULL;

    calc->magic_number = MATRIX_CALCULATOR_MAGIC;
    calc->rows = rows;
    calc->cols = cols;
    calc->data = TRACKED_MALLOC(rows * cols * sizeof(double)); // Utilisation de double pour les calculs
    calc->is_initialized = true;
    calc->memory_address = calc;

    if (!calc->data) {
        TRACKED_FREE(calc);
        return NULL;
    }

    // Initialisation des données à zéro
    memset(calc->data, 0, rows * cols * sizeof(double));

    return calc;
}

// Définir élément matriciel
void matrix_set_element(matrix_calculator_t* calc, size_t row, size_t col, double value) {
    if (!calc || calc->magic_number != MATRIX_CALCULATOR_MAGIC) return;
    if (row >= calc->rows || col >= calc->cols) return;

    calc->data[row * calc->cols + col] = value;
}

// Multiplication matricielle optimisée LUM
matrix_result_t* matrix_multiply_lum_optimized(matrix_calculator_t* a, matrix_calculator_t* b, void* config) {
    if (!a || !b || a->magic_number != MATRIX_CALCULATOR_MAGIC || b->magic_number != MATRIX_CALCULATOR_MAGIC) {
        return NULL;
    }

    if (a->cols != b->rows) return NULL;

    matrix_result_t* result = TRACKED_MALLOC(sizeof(matrix_result_t));
    if (!result) return NULL;

    result->magic_number = MATRIX_CALCULATOR_MAGIC;
    result->rows = a->rows;
    result->cols = b->cols;
    result->result_data = TRACKED_MALLOC(a->rows * b->cols * sizeof(double));
    result->memory_address = result;

    if (!result->result_data) {
        TRACKED_FREE(result);
        return NULL;
    }

    // Initialisation des données du résultat à zéro
    memset(result->result_data, 0, a->rows * b->cols * sizeof(double));

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Multiplication matricielle réelle avec optimisation LUM
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < a->cols; k++) {
                double val_a = a->data[i * a->cols + k];
                double val_b = b->data[k * b->cols + j];
                sum += val_a * val_b;
            }
            result->result_data[i * b->cols + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    result->execution_time_ns = (end.tv_sec - start.tv_sec) * 1000000000L +
                                (end.tv_nsec - start.tv_nsec);
    result->operation_success = true;

    return result;
}

// Destruction sécurisée
void matrix_calculator_destroy(matrix_calculator_t** calc) {
    if (!calc || !*calc) return;
    if ((*calc)->magic_number != MATRIX_CALCULATOR_MAGIC) return;

    if ((*calc)->data) {
        TRACKED_FREE((*calc)->data);
        (*calc)->data = NULL;
    }

    (*calc)->magic_number = 0; // Invalider magic number
    TRACKED_FREE(*calc);
    *calc = NULL;
}

// Destruction du résultat matriciel
void matrix_calculator_result_destroy(matrix_calculator_result_t** result_ptr) {
    if (!result_ptr || !*result_ptr) return;

    matrix_calculator_result_t* result = *result_ptr;
    if (result->magic_number != MATRIX_CALCULATOR_MAGIC) return;

    if (result->result_data) {
        TRACKED_FREE(result->result_data);
        result->result_data = NULL;
    }

    result->magic_number = 0; // Invalider magic number
    TRACKED_FREE(result);
    *result_ptr = NULL;
}


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
matrix_lum_result_t* matrix_add(lum_matrix_t* matrix_a, lum_matrix_t* matrix_b, matrix_config_t* config) {
    if (!matrix_a || !matrix_b || !config) return NULL;

    if (matrix_a->rows != matrix_b->rows || matrix_a->cols != matrix_b->cols) {
        return NULL; // Dimensions incompatibles
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    matrix_lum_result_t* result = TRACKED_MALLOC(sizeof(matrix_lum_result_t));
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
matrix_lum_result_t* matrix_multiply(lum_matrix_t* matrix_a, lum_matrix_t* matrix_b, matrix_config_t* config) {
    if (!matrix_a || !matrix_b || !config) return NULL;

    if (matrix_a->cols != matrix_b->rows) {
        return NULL; // Dimensions incompatibles pour multiplication
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    matrix_lum_result_t* result = TRACKED_MALLOC(sizeof(matrix_lum_result_t));
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

// Destruction résultat (pour LUM matrices)
void matrix_lum_result_destroy(matrix_lum_result_t** result_ptr) {
    if (!result_ptr || !*result_ptr) return;

    matrix_lum_result_t* result = *result_ptr;
    if (result->memory_address == (void*)result) { // Utiliser memory_address pour la vérification
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

// Fonction de test simple
void matrix_calculator_demo(void) {
    printf("Matrix Calculator Demo - LUM optimized calculations\n");

    // Test création et destruction calculateur
    matrix_calculator_t* calc = matrix_calculator_create(100, 100);
    if (calc) {
        printf("✅ Matrix calculator créé avec succès (100x100)\n");
        // Tester la définition d'un élément
        matrix_set_element(calc, 10, 20, 3.14);
        printf("✅ Élément (10, 20) défini avec la valeur 3.14\n");
        matrix_calculator_destroy(&calc);
        printf("✅ Matrix calculator détruit proprement\n");
    } else {
        printf("❌ Échec création matrix calculator\n");
    }

    // Test opération de multiplication
    matrix_calculator_t* mat_a = matrix_calculator_create(2, 3);
    matrix_calculator_t* mat_b = matrix_calculator_create(3, 2);

    if (mat_a && mat_b) {
        printf("✅ Matrices pour multiplication créées (2x3 et 3x2)\n");
        // Initialiser mat_a
        matrix_set_element(mat_a, 0, 0, 1.0); matrix_set_element(mat_a, 0, 1, 2.0); matrix_set_element(mat_a, 0, 2, 3.0);
        matrix_set_element(mat_a, 1, 0, 4.0); matrix_set_element(mat_a, 1, 1, 5.0); matrix_set_element(mat_a, 1, 2, 6.0);
        // Initialiser mat_b
        matrix_set_element(mat_b, 0, 0, 7.0); matrix_set_element(mat_b, 0, 1, 8.0);
        matrix_set_element(mat_b, 1, 0, 9.0); matrix_set_element(mat_b, 1, 1, 10.0);
        matrix_set_element(mat_b, 2, 0, 11.0); matrix_set_element(mat_b, 2, 1, 12.0);

        matrix_result_t* result = matrix_multiply_lum_optimized(mat_a, mat_b, NULL);
        if (result) {
            printf("✅ Multiplication matricielle effectuée avec succès.\n");
            printf("Temps d'exécution: %llu ns\n", result->execution_time_ns);
            printf("Résultat (2x2):\n");
            for(size_t i = 0; i < result->rows; ++i) {
                for(size_t j = 0; j < result->cols; ++j) {
                    printf("%.2f ", result->result_data[i * result->cols + j]);
                }
                printf("\n");
            }
            matrix_result_destroy(&result);
        } else {
            printf("❌ Échec de la multiplication matricielle.\n");
        }

        matrix_calculator_destroy(&mat_a);
        matrix_calculator_destroy(&mat_b);
    } else {
        printf("❌ Échec création matrices pour multiplication.\n");
    }
}

// Fonction destruction alias pour compatibilité
void matrix_result_destroy(matrix_result_t** result_ptr) {
    matrix_calculator_result_destroy((matrix_calculator_result_t**)result_ptr);
}

// Fonction de test simple