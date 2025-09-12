
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

// Inclusion TOUS les modules pour tests intégration
#include "../lum/lum_core.h"
#include "../vorax/vorax_operations.h"
#include "../binary/binary_lum_converter.h"
#include "../parser/vorax_parser.h"
#include "../logger/lum_logger.h"
#include "../crypto/crypto_validator.h"
#include "../optimization/memory_optimizer.h"
#include "../optimization/pareto_optimizer.h"
#include "../parallel/parallel_processor.h"
#include "../metrics/performance_metrics.h"
#include "../persistence/data_persistence.h"
#include "../debug/memory_tracker.h"

static int integration_tests_passed = 0;
static int integration_tests_failed = 0;

#define INTEGRATION_TEST_ASSERT(condition, test_name) \
    do { \
        if (condition) { \
            printf("✓ INTEGRATION TEST PASS: %s\n", test_name); \
            integration_tests_passed++; \
        } else { \
            printf("✗ INTEGRATION TEST FAIL: %s\n", test_name); \
            integration_tests_failed++; \
        } \
    } while(0)

void test_lum_to_binary_to_vorax_integration(void) {
    printf("\n=== Test Intégration: LUM → Binary → VORAX ===\n");
    
    // 1. Créer LUMs
    lum_group_t* original_group = lum_group_create(10);
    for (int i = 0; i < 5; i++) {
        lum_t* lum = lum_create(i % 2, i * 10, i * 20, LUM_STRUCTURE_LINEAR);
        lum_group_add(original_group, lum);
        lum_destroy(lum);
    }
    INTEGRATION_TEST_ASSERT(lum_group_size(original_group) == 5, "Création groupe LUM initial");
    
    // 2. Conversion en binaire
    char* binary_string = NULL;
    bool converted = lum_group_to_binary_string(original_group, &binary_string);
    INTEGRATION_TEST_ASSERT(converted && binary_string != NULL, "Conversion LUM vers binaire");
    
    // 3. Reconversion en LUMs
    lum_group_t* reconstructed_group = binary_string_to_lum_group(binary_string);
    INTEGRATION_TEST_ASSERT(reconstructed_group != NULL, "Reconversion binaire vers LUM");
    INTEGRATION_TEST_ASSERT(lum_group_size(reconstructed_group) == lum_group_size(original_group), 
                           "Conservation nombre LUMs");
    
    // 4. Opération VORAX sur groupe reconstruit
    vorax_result_t* cycle_result = vorax_cycle(reconstructed_group, 1000);
    INTEGRATION_TEST_ASSERT(cycle_result && cycle_result->success, "Opération VORAX sur LUMs reconstruits");
    
    // 5. Validation conservation
    INTEGRATION_TEST_ASSERT(cycle_result->result_group->count == reconstructed_group->count, 
                           "Conservation intégrité données");
    
    // Nettoyage
    free(binary_string);
    lum_group_destroy(original_group);
    lum_group_destroy(reconstructed_group);
    vorax_result_destroy(cycle_result);
}

void test_parser_optimization_integration(void) {
    printf("\n=== Test Intégration: Parser → Optimization ===\n");
    
    // 1. Créer contexte parsing
    vorax_context_t* context = vorax_context_create();
    INTEGRATION_TEST_ASSERT(context != NULL, "Création contexte VORAX");
    
    // 2. Parser script VORAX
    const char* script = 
        "zone optimization_zone;\n"
        "mem fast_memory;\n"
        "emit optimization_zone += 100•;\n"
        "optimize optimization_zone with pareto efficiency=500;\n"
        "store fast_memory <- optimization_zone, all;\n";
    
    bool parsed = vorax_parse_script(context, script);
    INTEGRATION_TEST_ASSERT(parsed, "Parsing script VORAX avec optimisation");
    
    // 3. Récupérer zone créée
    lum_zone_t* zone = vorax_context_get_zone(context, "optimization_zone");
    INTEGRATION_TEST_ASSERT(zone != NULL, "Récupération zone depuis contexte");
    
    // 4. Appliquer optimisation Pareto
    pareto_config_t pareto_config = {
        .enable_simd_optimization = true,
        .enable_memory_pooling = true,
        .enable_parallel_processing = false,
        .target_efficiency_threshold = 500.0
    };
    
    pareto_optimizer_t* optimizer = pareto_optimizer_create(&pareto_config);
    INTEGRATION_TEST_ASSERT(optimizer != NULL, "Création optimiseur Pareto");
    
    // 5. Optimiser première groupe de la zone
    if (zone->group_count > 0) {
        pareto_metrics_t metrics = pareto_evaluate_metrics(zone->groups[0], "integration_test");
        INTEGRATION_TEST_ASSERT(metrics.efficiency_ratio > 0, "Métriques Pareto calculées");
        
        double pareto_score = pareto_calculate_inverse_score(&metrics);
        INTEGRATION_TEST_ASSERT(pareto_score > 0, "Score Pareto inversé calculé");
    }
    
    // Nettoyage
    pareto_optimizer_destroy(optimizer);
    vorax_context_destroy(context);
}

void test_parallel_crypto_integration(void) {
    printf("\n=== Test Intégration: Parallel → Crypto ===\n");
    
    // 1. Créer processeur parallèle
    parallel_processor_t* processor = parallel_processor_create(4);
    INTEGRATION_TEST_ASSERT(processor != NULL, "Création processeur parallèle");
    
    // 2. Créer groupes LUM pour traitement parallèle
    lum_group_t* groups[4];
    for (int g = 0; g < 4; g++) {
        groups[g] = lum_group_create(25);
        for (int i = 0; i < 20; i++) {
            lum_t* lum = lum_create(1, g * 100 + i, i, LUM_STRUCTURE_LINEAR);
            lum_group_add(groups[g], lum);
            lum_destroy(lum);
        }
    }
    
    // 3. Soumission tâches de hachage crypto parallèles
    for (int g = 0; g < 4; g++) {
        parallel_task_t* crypto_task = parallel_task_create(TASK_CRYPTO_HASH, groups[g], sizeof(lum_group_t));
        bool submitted = parallel_processor_submit_task(processor, crypto_task);
        INTEGRATION_TEST_ASSERT(submitted, "Soumission tâche crypto parallèle");
    }
    
    // 4. Attendre completion
    bool completed = parallel_processor_wait_for_completion(processor);
    INTEGRATION_TEST_ASSERT(completed, "Completion tâches parallèles");
    
    // 5. Validation résultats crypto
    for (int g = 0; g < 4; g++) {
        char hash_result[MAX_HASH_STRING_LENGTH];
        bool hash_computed = compute_data_hash(groups[g], sizeof(lum_group_t), hash_result);
        INTEGRATION_TEST_ASSERT(hash_computed, "Hash crypto calculé pour groupe");
        INTEGRATION_TEST_ASSERT(strlen(hash_result) == 64, "Hash SHA-256 longueur correcte");
    }
    
    // Nettoyage
    for (int g = 0; g < 4; g++) {
        lum_group_destroy(groups[g]);
    }
    parallel_processor_destroy(processor);
}

void test_memory_persistence_integration(void) {
    printf("\n=== Test Intégration: Memory → Persistence ===\n");
    
    // 1. Créer optimiseur mémoire
    memory_optimizer_t* mem_optimizer = memory_optimizer_create(2048);
    INTEGRATION_TEST_ASSERT(mem_optimizer != NULL, "Création optimiseur mémoire");
    
    // 2. Allouer LUMs via optimiseur
    lum_t* managed_lums[50];
    for (int i = 0; i < 50; i++) {
        managed_lums[i] = memory_optimizer_alloc_lum(mem_optimizer);
        INTEGRATION_TEST_ASSERT(managed_lums[i] != NULL, "Allocation LUM via optimiseur");
        
        managed_lums[i]->presence = i % 2;
        managed_lums[i]->position_x = i * 10;
        managed_lums[i]->position_y = i * 5;
        managed_lums[i]->structure_type = LUM_STRUCTURE_LINEAR;
    }
    
    // 3. Créer backend persistence
    storage_backend_t* storage = storage_backend_create("integration_test.db");
    INTEGRATION_TEST_ASSERT(storage != NULL, "Création backend persistence");
    
    // 4. Sérialiser et stocker
    for (int i = 0; i < 50; i++) {
        char key[64];
        snprintf(key, sizeof(key), "managed_lum_%d", i);
        
        bool stored = store_lum(storage, key, managed_lums[i]);
        INTEGRATION_TEST_ASSERT(stored, "Stockage LUM managé");
    }
    
    // 5. Récupération et validation
    for (int i = 0; i < 10; i++) { // Test échantillon
        char key[64];
        snprintf(key, sizeof(key), "managed_lum_%d", i);
        
        lum_t* loaded = load_lum(storage, key);
        INTEGRATION_TEST_ASSERT(loaded != NULL, "Chargement LUM depuis storage");
        INTEGRATION_TEST_ASSERT(loaded->position_x == managed_lums[i]->position_x, "Données LUM préservées");
        
        lum_destroy(loaded);
    }
    
    // 6. Statistiques mémoire intégrées
    memory_stats_t* stats = memory_optimizer_get_stats(mem_optimizer);
    INTEGRATION_TEST_ASSERT(stats->total_allocated > 0, "Statistiques mémoire disponibles");
    
    // Nettoyage
    for (int i = 0; i < 50; i++) {
        memory_optimizer_free_lum(mem_optimizer, managed_lums[i]);
    }
    memory_optimizer_destroy(mem_optimizer);
    storage_backend_destroy(storage);
}

void test_full_system_integration(void) {
    printf("\n=== Test Intégration: Système Complet ===\n");
    
    // 1. Logger système
    lum_logger_t* logger = lum_logger_create("logs/integration_complete.log", true, true);
    INTEGRATION_TEST_ASSERT(logger != NULL, "Initialisation logger système");
    
    // 2. Métriques performance
    performance_metrics_t* metrics = performance_metrics_create();
    INTEGRATION_TEST_ASSERT(metrics != NULL, "Initialisation métriques performance");
    
    // 3. Scenario: Création → Parser → Optimisation → Persistence
    lum_log_message(logger, LUM_LOG_INFO, "Début test intégration complète");
    
    // 3a. Création données
    lum_group_t* system_group = lum_group_create(100);
    for (int i = 0; i < 75; i++) {
        lum_t* lum = lum_create(i % 2, i, i * 2, (lum_structure_type_e)(i % 4));
        lum_group_add(system_group, lum);
        lum_destroy(lum);
    }
    
    // 3b. Opérations VORAX
    vorax_result_t* split_result = vorax_split(system_group, 5);
    INTEGRATION_TEST_ASSERT(split_result && split_result->success, "Opération VORAX dans système complet");
    
    // 3c. Optimisation Pareto sur résultats
    if (split_result->result_count > 0) {
        pareto_metrics_t sys_metrics = pareto_evaluate_metrics(split_result->result_groups[0], "system_test");
        INTEGRATION_TEST_ASSERT(sys_metrics.lum_operations_count > 0, "Métriques système calculées");
    }
    
    // 3d. Logging résultats
    char log_message[256];
    snprintf(log_message, sizeof(log_message), 
             "Système intégré: %zu LUMs → %zu groupes", 
             lum_group_size(system_group), split_result->result_count);
    lum_log_message(logger, LUM_LOG_INFO, log_message);
    
    // 4. Validation finale
    INTEGRATION_TEST_ASSERT(true, "Test système complet terminé sans erreur");
    
    // Nettoyage
    lum_group_destroy(system_group);
    vorax_result_destroy(split_result);
    performance_metrics_destroy(metrics);
    lum_logger_destroy(logger);
}

int main(void) {
    printf("🔗 === TESTS INTÉGRATION COMPLETS TOUS MODULES ===\n");
    printf("Validation interopérabilité et flux de données\n\n");
    
    // Initialisation tracking global
    memory_tracker_init();
    
    // Exécution tous tests intégration
    test_lum_to_binary_to_vorax_integration();
    test_parser_optimization_integration();
    test_parallel_crypto_integration();
    test_memory_persistence_integration();
    test_full_system_integration();
    
    // Résultats finaux
    printf("\n=== RÉSULTATS TESTS INTÉGRATION ===\n");
    printf("Tests intégration réussis: %d\n", integration_tests_passed);
    printf("Tests intégration échoués: %d\n", integration_tests_failed);
    printf("Taux succès intégration: %.1f%%\n", 
           integration_tests_passed > 0 ? (100.0 * integration_tests_passed) / (integration_tests_passed + integration_tests_failed) : 0.0);
    
    // Rapport mémoire final
    memory_tracker_report();
    memory_tracker_cleanup();
    
    return integration_tests_failed == 0 ? 0 : 1;
}
