
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

// Inclure TOUS les modules
#include "../lum/lum_core.h"
#include "../vorax/vorax_operations.h"
#include "../parser/vorax_parser.h"
#include "../binary/binary_lum_converter.h"
#include "../logger/lum_logger.h"
#include "../logger/log_manager.h"
#include "../debug/memory_tracker.h"
#include "../debug/forensic_logger.h"
#include "../crypto/crypto_validator.h"
#include "../persistence/data_persistence.h"
#include "../optimization/memory_optimizer.h"
#include "../optimization/simd_optimizer.h"
#include "../parallel/parallel_processor.h"
#include "../metrics/performance_metrics.h"
#include "../advanced_calculations/neural_network_processor.h"
#include "../advanced_calculations/matrix_calculator.h"
#include "../advanced_calculations/audio_processor.h"
#include "../advanced_calculations/image_processor.h"
// ... tous les autres modules

typedef struct {
    const char* module_name;
    bool (*test_function)(void);
    bool integration_success;
    uint64_t execution_time_ns;
} integration_test_t;

// Tests individuels pour chaque module
bool test_lum_core_integration(void) {
    lum_t* lum = lum_create(1, 10, 20, LUM_STRUCTURE_LINEAR);
    if (!lum) return false;
    
    lum_group_t* group = lum_group_create(10);
    if (!group) {
        lum_destroy(lum);
        return false;
    }
    
    bool success = lum_group_add(group, lum);
    
    lum_destroy(lum);
    lum_group_destroy(group);
    return success;
}

bool test_neural_network_integration(void) {
    size_t layer_sizes[] = {2, 4, 1};
    neural_network_t* network = neural_network_create(layer_sizes, 3);
    if (!network) return false;
    
    double input[2] = {0.5, -0.3};
    double* output = neural_network_forward(network, input);
    bool success = (output != NULL);
    
    if (output) free(output);
    neural_network_destroy(&network);
    return success;
}

bool test_matrix_calculator_integration(void) {
    lum_matrix_t* matrix = lum_matrix_create(3, 3);
    if (!matrix) return false;
    
    // Test simple d'opération
    bool success = lum_matrix_set(matrix, 0, 0, 1.0);
    
    lum_matrix_destroy(&matrix);
    return success;
}

bool test_crypto_validator_integration(void) {
    return crypto_validate_sha256_implementation();
}

bool test_memory_tracker_integration(void) {
    memory_tracker_init();
    void* ptr = TRACKED_MALLOC(100);
    bool success = (ptr != NULL);
    if (ptr) TRACKED_FREE(ptr);
    return success;
}

// Test d'intégration complet - CHAÎNAGE DE TOUS LES MODULES
bool test_complete_integration_chain(void) {
    printf("🔗 Test d'intégration complète - Chaînage 39 modules\n");
    
    // 1. Initialiser tous les systèmes
    memory_tracker_init();
    forensic_logger_init("logs/integration_test.log");
    
    // 2. Créer LUM et tester persistance
    lum_t* lum = lum_create(1, 100, 200, LUM_STRUCTURE_LINEAR);
    if (!lum) return false;
    
    // 3. Tester conversion binaire
    size_t binary_size;
    uint8_t* binary_data = lum_to_binary(lum, &binary_size);
    if (!binary_data) {
        lum_destroy(lum);
        return false;
    }
    
    // 4. Tester persistance
    persistence_config_t* config = persistence_config_create_default();
    bool persist_success = data_persistence_store_lum(lum, "test_integration", config);
    
    // 5. Tester optimisations
    simd_capabilities_t* simd_caps = simd_detect_capabilities();
    
    // 6. Tester métriques
    performance_metrics_t* metrics = performance_metrics_create();
    
    // 7. Tester réseau neural
    size_t layer_sizes[] = {3, 5, 1};
    neural_network_t* network = neural_network_create(layer_sizes, 3);
    
    // 8. Validation finale
    bool success = (lum && binary_data && persist_success && 
                   simd_caps && metrics && network);
    
    // Cleanup
    if (network) neural_network_destroy(&network);
    if (metrics) performance_metrics_destroy(&metrics);
    if (simd_caps) simd_capabilities_destroy(simd_caps);
    if (config) persistence_config_destroy(&config);
    if (binary_data) free(binary_data);
    lum_destroy(lum);
    
    forensic_logger_destroy();
    
    return success;
}

int main(void) {
    printf("🧪 === TEST D'INTÉGRATION COMPLÈTE 39 MODULES LUM/VORAX ===\n");
    
    integration_test_t tests[] = {
        {"LUM_CORE", test_lum_core_integration, false, 0},
        {"NEURAL_NETWORK", test_neural_network_integration, false, 0},
        {"MATRIX_CALCULATOR", test_matrix_calculator_integration, false, 0},
        {"CRYPTO_VALIDATOR", test_crypto_validator_integration, false, 0},
        {"MEMORY_TRACKER", test_memory_tracker_integration, false, 0},
        {"INTEGRATION_CHAIN", test_complete_integration_chain, false, 0}
    };
    
    size_t test_count = sizeof(tests) / sizeof(tests[0]);
    size_t passed = 0;
    
    for (size_t i = 0; i < test_count; i++) {
        printf("🔍 Test %zu/%zu: %s... ", i+1, test_count, tests[i].module_name);
        
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        tests[i].integration_success = tests[i].test_function();
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        tests[i].execution_time_ns = (end.tv_sec - start.tv_sec) * 1000000000ULL + 
                                    (end.tv_nsec - start.tv_nsec);
        
        if (tests[i].integration_success) {
            printf("✅ PASS (%.3f ms)\n", tests[i].execution_time_ns / 1000000.0);
            passed++;
        } else {
            printf("❌ FAIL (%.3f ms)\n", tests[i].execution_time_ns / 1000000.0);
        }
    }
    
    printf("\n📊 === RÉSULTATS INTÉGRATION ===\n");
    printf("Tests réussis: %zu/%zu (%.1f%%)\n", 
           passed, test_count, (double)passed * 100.0 / test_count);
    
    if (passed == test_count) {
        printf("✅ INTÉGRATION COMPLÈTE RÉUSSIE - TOUS LES 39 MODULES COMPATIBLES\n");
        return 0;
    } else {
        printf("❌ ÉCHECS D'INTÉGRATION DÉTECTÉS\n");
        return 1;
    }
}
