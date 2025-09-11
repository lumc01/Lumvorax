
#include "../optimization/pareto_optimizer.h"
#include "../logger/lum_logger.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void test_pareto_optimizer_creation(void) {
    printf("Test: Création optimiseur Pareto... ");
    
    pareto_config_t config = {
        .enable_simd_optimization = true,
        .enable_memory_pooling = true,
        .enable_parallel_processing = true,
        .enable_crypto_acceleration = false,
        .enable_logging_optimization = true,
        .target_efficiency_threshold = 500.0
    };
    
    pareto_optimizer_t* optimizer = pareto_optimizer_create(&config);
    assert(optimizer != NULL);
    assert(optimizer->inverse_pareto_mode == true);
    assert(optimizer->point_count == 0);
    assert(optimizer->point_capacity == 1000);
    
    pareto_optimizer_destroy(optimizer);
    printf("✓\n");
}

void test_pareto_metrics_evaluation(void) {
    printf("Test: Évaluation métriques Pareto... ");
    
    lum_group_t* test_group = lum_group_create(100);
    assert(test_group != NULL);
    
    // Ajout de quelques LUMs de test
    for (size_t i = 0; i < 50; i++) {
        lum_t* test_lum = lum_create(i % 2, (int32_t)i, 0, LUM_STRUCTURE_LINEAR);
        if (test_lum) {
            lum_group_add(test_group, test_lum);
            lum_destroy(test_lum);
        }
    }
    
    pareto_metrics_t metrics = pareto_evaluate_metrics(test_group, "test_operation");
    
    assert(metrics.efficiency_ratio > 0.0);
    assert(metrics.memory_usage > 0.0);
    assert(metrics.execution_time >= 0.0);
    assert(metrics.energy_consumption >= 0.0);
    assert(metrics.lum_operations_count == test_group->count);
    
    lum_group_destroy(test_group);
    printf("✓\n");
}

void test_pareto_dominance(void) {
    printf("Test: Test de dominance Pareto... ");
    
    pareto_metrics_t a = {
        .efficiency_ratio = 100.0,
        .memory_usage = 1000.0,
        .execution_time = 2.0,
        .energy_consumption = 0.001,
        .lum_operations_count = 100
    };
    
    pareto_metrics_t b = {
        .efficiency_ratio = 150.0,  // Meilleur
        .memory_usage = 800.0,      // Meilleur
        .execution_time = 1.5,      // Meilleur
        .energy_consumption = 0.0008, // Meilleur
        .lum_operations_count = 100
    };
    
    pareto_metrics_t c = {
        .efficiency_ratio = 120.0,  // Meilleur que a
        .memory_usage = 1200.0,     // Pire que a
        .execution_time = 1.8,      // Meilleur que a
        .energy_consumption = 0.0009, // Meilleur que a
        .lum_operations_count = 100
    };
    
    // b domine a (meilleur sur tous les critères)
    assert(pareto_is_dominated(&a, &b) == true);
    
    // a n'est pas dominé par c (trade-off)
    assert(pareto_is_dominated(&a, &c) == false);
    
    // c n'est pas dominé par a (trade-off)
    assert(pareto_is_dominated(&c, &a) == false);
    
    printf("✓\n");
}

void test_pareto_inverse_score(void) {
    printf("Test: Calcul score Pareto inversé... ");
    
    pareto_metrics_t good_metrics = {
        .efficiency_ratio = 500.0,
        .memory_usage = 5000.0,
        .execution_time = 1.0,
        .energy_consumption = 0.0005,
        .lum_operations_count = 1000
    };
    
    pareto_metrics_t bad_metrics = {
        .efficiency_ratio = 50.0,
        .memory_usage = 50000.0,
        .execution_time = 10.0,
        .energy_consumption = 0.01,
        .lum_operations_count = 100
    };
    
    double good_score = pareto_calculate_inverse_score(&good_metrics);
    double bad_score = pareto_calculate_inverse_score(&bad_metrics);
    
    assert(good_score > bad_score);
    assert(good_score > 0.0);
    assert(bad_score > 0.0);
    
    printf("✓\n");
}

void test_pareto_optimization_operations(void) {
    printf("Test: Opérations optimisées Pareto... ");
    
    // Création de groupes de test
    lum_group_t* group1 = lum_group_create(50);
    lum_group_t* group2 = lum_group_create(30);
    
    for (size_t i = 0; i < 20; i++) {
        lum_t* lum1 = lum_create(i % 2, (int32_t)i, 0, LUM_STRUCTURE_LINEAR);
        lum_t* lum2 = lum_create((i+1) % 2, (int32_t)i, 1, LUM_STRUCTURE_CIRCULAR);
        
        if (lum1) {
            lum_group_add(group1, lum1);
            free(lum1);
        }
        if (lum2) {
            lum_group_add(group2, lum2);
            free(lum2);
        }
    }
    
    // Test FUSE optimisé
    vorax_result_t* fuse_result = pareto_optimize_fuse_operation(group1, group2);
    assert(fuse_result != NULL);
    assert(fuse_result->success == true);
    assert(fuse_result->result_group != NULL);
    assert(fuse_result->result_group->count == group1->count + group2->count);
    
    // Test SPLIT optimisé
    vorax_result_t* split_result = pareto_optimize_split_operation(group1, 3);
    assert(split_result != NULL);
    assert(split_result->success == true);
    assert(split_result->result_count > 0);
    
    // Test CYCLE optimisé
    vorax_result_t* cycle_result = pareto_optimize_cycle_operation(group1, 7);
    assert(cycle_result != NULL);
    assert(cycle_result->success == true);
    assert(cycle_result->result_group != NULL);
    
    // Nettoyage
    vorax_result_destroy(fuse_result);
    vorax_result_destroy(split_result);
    vorax_result_destroy(cycle_result);
    lum_group_destroy(group1);
    lum_group_destroy(group2);
    
    printf("✓\n");
}

void test_pareto_vorax_optimization(void) {
    printf("Test: Optimisation DSL VORAX... ");
    
    pareto_config_t config = {
        .enable_simd_optimization = true,
        .enable_memory_pooling = true,
        .enable_parallel_processing = true,
        .enable_crypto_acceleration = false,
        .enable_logging_optimization = true,
        .target_efficiency_threshold = 300.0
    };
    
    pareto_optimizer_t* optimizer = pareto_optimizer_create(&config);
    assert(optimizer != NULL);
    
    const char* test_script = 
        "zone test_zone;\n"
        "mem test_mem;\n"
        "emit test_zone += 10•;\n"
        "store test_mem <- test_zone, all;\n";
    
    bool result = pareto_execute_vorax_optimization(optimizer, test_script);
    assert(result == true);
    
    // Vérification qu'au moins un point a été ajouté
    assert(optimizer->point_count > 0);
    
    pareto_optimizer_destroy(optimizer);
    printf("✓\n");
}

void test_pareto_script_generation(void) {
    printf("Test: Génération script VORAX adaptatif... ");
    
    pareto_metrics_t target = {
        .efficiency_ratio = 750.0,
        .memory_usage = 8000.0,
        .execution_time = 1.5,
        .energy_consumption = 0.0008,
        .lum_operations_count = 1000
    };
    
    char* script = pareto_generate_optimization_script(&target);
    assert(script != NULL);
    assert(strlen(script) > 0);
    assert(strstr(script, "zone") != NULL);
    assert(strstr(script, "mem") != NULL);
    assert(strstr(script, "efficiency") != NULL);
    
    printf("✓\n");
}

void test_pareto_point_management(void) {
    printf("Test: Gestion points Pareto... ");
    
    pareto_config_t config = {
        .enable_simd_optimization = true,
        .enable_memory_pooling = false,
        .enable_parallel_processing = false,
        .enable_crypto_acceleration = false,
        .enable_logging_optimization = true,
        .target_efficiency_threshold = 200.0
    };
    
    pareto_optimizer_t* optimizer = pareto_optimizer_create(&config);
    assert(optimizer != NULL);
    
    // Ajout de plusieurs points
    pareto_metrics_t metrics1 = {100.0, 1000.0, 2.0, 0.001, 100};
    pareto_metrics_t metrics2 = {200.0, 800.0, 1.5, 0.0008, 150};
    pareto_metrics_t metrics3 = {150.0, 1200.0, 1.8, 0.0009, 120};
    
    pareto_add_point(optimizer, &metrics1, "test_path_1");
    pareto_add_point(optimizer, &metrics2, "test_path_2");
    pareto_add_point(optimizer, &metrics3, "test_path_3");
    
    assert(optimizer->point_count == 3);
    
    // Trouver le meilleur point
    pareto_point_t* best = pareto_find_best_point(optimizer);
    assert(best != NULL);
    assert(best->is_dominated == false);
    
    pareto_optimizer_destroy(optimizer);
    printf("✓\n");
}

void test_pareto_benchmark(void) {
    printf("Test: Benchmark contre baseline... ");
    
    pareto_config_t config = {
        .enable_simd_optimization = false,
        .enable_memory_pooling = false,
        .enable_parallel_processing = false,
        .enable_crypto_acceleration = false,
        .enable_logging_optimization = false,
        .target_efficiency_threshold = 100.0
    };
    
    pareto_optimizer_t* optimizer = pareto_optimizer_create(&config);
    assert(optimizer != NULL);
    
    pareto_benchmark_against_baseline(optimizer, "test_baseline");
    
    assert(optimizer->point_count > 0);
    
    pareto_optimizer_destroy(optimizer);
    printf("✓\n");
}

int main(void) {
    printf("=== TESTS MODULE PARETO OPTIMIZATION ===\n\n");
    
    // Initialisation du logger pour les tests
    lum_log_init("test_pareto_optimization.log");
    
    test_pareto_optimizer_creation();
    test_pareto_metrics_evaluation();
    test_pareto_dominance();
    test_pareto_inverse_score();
    test_pareto_optimization_operations();
    test_pareto_vorax_optimization();
    test_pareto_script_generation();
    test_pareto_point_management();
    test_pareto_benchmark();
    
    printf("\n✅ Tous les tests Pareto réussis!\n");
    
    lum_log_destroy();
    return 0;
}
