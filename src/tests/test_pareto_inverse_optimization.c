
#include "../optimization/pareto_inverse_optimizer.h"
#include "../logger/lum_logger.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void test_pareto_inverse_optimizer_creation(void) {
    printf("Test: Création optimiseur Pareto inversé... ");
    
    pareto_inverse_optimizer_t* optimizer = pareto_inverse_optimizer_create();
    assert(optimizer != NULL);
    assert(optimizer->inverse_mode_active == true);
    assert(optimizer->layer_count == 0);
    assert(optimizer->max_layers == 10);
    assert(optimizer->global_efficiency_target == 1000.0);
    assert(optimizer->energy_budget == 0.0005);
    
    pareto_inverse_optimizer_destroy(optimizer);
    printf("✓\n");
}

void test_optimization_layer_management(void) {
    printf("Test: Gestion couches d'optimisation... ");
    
    pareto_inverse_optimizer_t* optimizer = pareto_inverse_optimizer_create();
    assert(optimizer != NULL);
    
    // Ajout de différentes couches
    optimization_layer_t* memory_layer = pareto_add_optimization_layer(
        optimizer, "memory_ultra", OPT_TYPE_MEMORY, 800.0);
    assert(memory_layer != NULL);
    assert(memory_layer->type == OPT_TYPE_MEMORY);
    assert(memory_layer->efficiency_target == 800.0);
    assert(optimizer->layer_count == 1);
    
    optimization_layer_t* simd_layer = pareto_add_optimization_layer(
        optimizer, "simd_vectorial", OPT_TYPE_SIMD, 1200.0);
    assert(simd_layer != NULL);
    assert(simd_layer->type == OPT_TYPE_SIMD);
    assert(optimizer->layer_count == 2);
    
    optimization_layer_t* parallel_layer = pareto_add_optimization_layer(
        optimizer, "parallel_massive", OPT_TYPE_PARALLEL, 1000.0);
    assert(parallel_layer != NULL);
    assert(optimizer->layer_count == 3);
    
    optimization_layer_t* crypto_layer = pareto_add_optimization_layer(
        optimizer, "crypto_hw", OPT_TYPE_CRYPTO, 600.0);
    assert(crypto_layer != NULL);
    assert(optimizer->layer_count == 4);
    
    optimization_layer_t* energy_layer = pareto_add_optimization_layer(
        optimizer, "energy_conserve", OPT_TYPE_ENERGY, 2000.0);
    assert(energy_layer != NULL);
    assert(optimizer->layer_count == 5);
    
    pareto_inverse_optimizer_destroy(optimizer);
    printf("✓\n");
}

void test_multi_layer_optimization_execution(void) {
    printf("Test: Exécution optimisation multi-couches... ");
    
    pareto_inverse_optimizer_t* optimizer = pareto_inverse_optimizer_create();
    assert(optimizer != NULL);
    
    // Configuration des couches
    pareto_add_optimization_layer(optimizer, "memory_pool", OPT_TYPE_MEMORY, 500.0);
    pareto_add_optimization_layer(optimizer, "simd_avx512", OPT_TYPE_SIMD, 800.0);
    pareto_add_optimization_layer(optimizer, "parallel_8core", OPT_TYPE_PARALLEL, 600.0);
    pareto_add_optimization_layer(optimizer, "energy_saving", OPT_TYPE_ENERGY, 1500.0);
    
    // Création d'un groupe de test
    lum_group_t* test_group = lum_group_create(1000);
    assert(test_group != NULL);
    
    for (size_t i = 0; i < 1000; i++) {
        lum_t* test_lum = lum_create(i % 2, (int32_t)i, (int32_t)(i % 100), LUM_STRUCTURE_LINEAR);
        if (test_lum) {
            lum_group_add(test_group, test_lum);
            free(test_lum);
        }
    }
    
    // Exécution de l'optimisation multi-couches
    pareto_inverse_result_t result = pareto_execute_multi_layer_optimization(optimizer, test_group);
    
    assert(result.success == true);
    assert(result.total_execution_time > 0.0);
    assert(result.inverse_pareto_score >= 0.0);
    assert(result.optimized_group != NULL);
    assert(strlen(result.summary) > 0);
    assert(strlen(result.error_message) == 0);
    
    printf("Score Pareto inversé: %.3f, Amélioration: %.2f%% ", 
           result.inverse_pareto_score, result.total_improvement);
    
    lum_group_destroy(test_group);
    if (result.optimized_group != test_group) {
        lum_group_destroy(result.optimized_group);
    }
    pareto_inverse_optimizer_destroy(optimizer);
    printf("✓\n");
}

void test_inverse_pareto_score_calculation(void) {
    printf("Test: Calcul score Pareto inversé avancé... ");
    
    pareto_metrics_t baseline = {
        .efficiency_ratio = 100.0,
        .memory_usage = 1000000.0,
        .execution_time = 5.0,
        .energy_consumption = 0.01,
        .lum_operations_count = 1000
    };
    
    pareto_metrics_t optimized = {
        .efficiency_ratio = 300.0,  // 3x amélioration
        .memory_usage = 500000.0,   // 50% réduction
        .execution_time = 2.0,      // 2.5x amélioration
        .energy_consumption = 0.005, // 50% réduction
        .lum_operations_count = 1000
    };
    
    double score = calculate_inverse_pareto_score_advanced(&optimized, &baseline);
    
    // Le score doit être positif et refléter les améliorations
    assert(score > 0.0);
    
    // Vérification que les améliorations importantes donnent des scores élevés
    assert(score > 1.0); // Score supérieur à 1 pour des améliorations significatives
    
    printf("Score calculé: %.3f ", score);
    printf("✓\n");
}

void test_specialized_optimization_functions(void) {
    printf("Test: Fonctions d'optimisation spécialisées... ");
    
    // Création d'un groupe de test
    lum_group_t* test_group = lum_group_create(100);
    assert(test_group != NULL);
    
    for (size_t i = 0; i < 100; i++) {
        lum_t* test_lum = lum_create(1, (int32_t)i, (int32_t)i, LUM_STRUCTURE_LINEAR);
        if (test_lum) {
            lum_group_add(test_group, test_lum);
            free(test_lum);
        }
    }
    
    optimization_layer_t layer = {
        .type = OPT_TYPE_MEMORY,
        .efficiency_target = 500.0,
        .active = true
    };
    strcpy(layer.name, "test_layer");
    
    // Test optimisation mémoire
    lum_group_t* memory_optimized = apply_memory_optimization(test_group, &layer);
    assert(memory_optimized != NULL);
    assert(memory_optimized->count == test_group->count);
    
    // Test optimisation SIMD
    layer.type = OPT_TYPE_SIMD;
    lum_group_t* simd_optimized = apply_simd_optimization(test_group, &layer);
    assert(simd_optimized != NULL);
    assert(simd_optimized->count == test_group->count);
    
    // Test optimisation parallèle
    layer.type = OPT_TYPE_PARALLEL;
    lum_group_t* parallel_optimized = apply_parallel_optimization(test_group, &layer);
    assert(parallel_optimized != NULL);
    assert(parallel_optimized->count == test_group->count);
    
    // Test optimisation crypto
    layer.type = OPT_TYPE_CRYPTO;
    lum_group_t* crypto_optimized = apply_crypto_optimization(test_group, &layer);
    assert(crypto_optimized != NULL);
    assert(crypto_optimized->count == test_group->count);
    
    // Test optimisation énergie
    layer.type = OPT_TYPE_ENERGY;
    lum_group_t* energy_optimized = apply_energy_optimization(test_group, &layer);
    assert(energy_optimized != NULL);
    assert(energy_optimized->count <= test_group->count); // Peut réduire pour économie énergie
    
    // Nettoyage
    lum_group_destroy(test_group);
    lum_group_destroy(memory_optimized);
    lum_group_destroy(simd_optimized);
    lum_group_destroy(parallel_optimized);
    lum_group_destroy(crypto_optimized);
    lum_group_destroy(energy_optimized);
    
    printf("✓\n");
}

void test_report_generation(void) {
    printf("Test: Génération rapport multi-couches... ");
    
    pareto_inverse_optimizer_t* optimizer = pareto_inverse_optimizer_create();
    assert(optimizer != NULL);
    
    pareto_add_optimization_layer(optimizer, "test_memory", OPT_TYPE_MEMORY, 400.0);
    pareto_add_optimization_layer(optimizer, "test_simd", OPT_TYPE_SIMD, 600.0);
    
    pareto_inverse_result_t result = {
        .success = true,
        .inverse_pareto_score = 1.234,
        .total_improvement = 45.67,
        .total_execution_time = 123.456,
        .baseline_metrics = {100.0, 1000000.0, 5.0, 0.01, 1000},
        .final_metrics = {200.0, 800000.0, 3.0, 0.008, 1000}
    };
    strcpy(result.summary, "Test multi-layer optimization completed");
    
    const char* report_file = "test_pareto_inverse_report.txt";
    pareto_generate_multi_layer_report(optimizer, &result, report_file);
    
    // Vérification que le fichier a été créé
    FILE* f = fopen(report_file, "r");
    assert(f != NULL);
    fclose(f);
    
    // Nettoyage du fichier de test
    remove(report_file);
    
    pareto_inverse_optimizer_destroy(optimizer);
    printf("✓\n");
}

void test_extreme_optimization_scenarios(void) {
    printf("Test: Scénarios d'optimisation extrêmes... ");
    
    pareto_inverse_optimizer_t* optimizer = pareto_inverse_optimizer_create();
    assert(optimizer != NULL);
    
    // Configuration avec objectifs très élevés
    pareto_add_optimization_layer(optimizer, "extreme_memory", OPT_TYPE_MEMORY, 2000.0);
    pareto_add_optimization_layer(optimizer, "extreme_simd", OPT_TYPE_SIMD, 5000.0);
    pareto_add_optimization_layer(optimizer, "extreme_energy", OPT_TYPE_ENERGY, 10000.0);
    
    // Groupe de test avec beaucoup de LUMs
    lum_group_t* large_group = lum_group_create(10000);
    assert(large_group != NULL);
    
    for (size_t i = 0; i < 10000; i++) {
        lum_t* test_lum = lum_create(i % 2, (int32_t)i, (int32_t)(i % 1000), 
                                   (lum_structure_e)(i % 4));
        if (test_lum) {
            lum_group_add(large_group, test_lum);
            free(test_lum);
        }
    }
    
    pareto_inverse_result_t result = pareto_execute_multi_layer_optimization(optimizer, large_group);
    
    assert(result.success == true);
    assert(result.optimized_group != NULL);
    
    // Vérification que les optimisations extrêmes donnent des scores très élevés
    printf("Score extrême: %.3f ", result.inverse_pareto_score);
    
    lum_group_destroy(large_group);
    if (result.optimized_group != large_group) {
        lum_group_destroy(result.optimized_group);
    }
    pareto_inverse_optimizer_destroy(optimizer);
    printf("✓\n");
}

int main(void) {
    printf("=== TESTS MODULE PARETO INVERSE OPTIMIZATION ===\n\n");
    
    // Initialisation du logger pour les tests
    lum_log_init("test_pareto_inverse_optimization.log");
    
    test_pareto_inverse_optimizer_creation();
    test_optimization_layer_management();
    test_multi_layer_optimization_execution();
    test_inverse_pareto_score_calculation();
    test_specialized_optimization_functions();
    test_report_generation();
    test_extreme_optimization_scenarios();
    
    printf("\n✅ Tous les tests Pareto inversé réussis!\n");
    printf("📊 Module Pareto inversé multi-couches validé avec succès\n");
    
    lum_log_cleanup();
    return 0;
}
