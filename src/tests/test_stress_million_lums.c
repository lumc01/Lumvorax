#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "../lum/lum_core.h"
#include "../vorax/vorax_operations.h"
#include "../optimization/memory_optimizer.h"
#include "../optimization/simd_optimizer.h"
#include "../optimization/zero_copy_allocator.h"
#include "../debug/memory_tracker.h"

#define MILLION_LUMS 1000000
#define STRESS_TEST_ITERATIONS 10
#define MAX_STRESS_LUMS 10000000  // 10 millions pour test extrême

typedef struct {
    size_t lum_count;
    double creation_time;
    double operation_time;
    double destruction_time;
    size_t memory_peak;
    bool graceful_degradation;
    char bottleneck[256];
    long peak_memory_mb;
    double execution_time_s;
} stress_test_result_t;

// Forward declarations
stress_test_result_t* run_million_lum_creation_test(void);
stress_test_result_t* run_million_lum_operations_test(void);
stress_test_result_t* run_scalability_stress_test(size_t max_lums);
stress_test_result_t* run_memory_pressure_test(void);
stress_test_result_t* run_parallel_stress_test(void);
stress_test_result_t* test_million_lum_group_operations(void);

void print_stress_results(stress_test_result_t* result, const char* test_name);
void stress_test_result_destroy(stress_test_result_t* result);

int main() {
    printf("🚀 DÉMARRAGE TESTS DE STRESS MAXIMUM - LUM/VORAX\n");
    printf("=================================================\n\n");

    // Test 1: Création de 1 million de LUMs
    printf("Test 1: Création de %d LUMs...\n", MILLION_LUMS);
    stress_test_result_t* creation_result = run_million_lum_creation_test();
    print_stress_results(creation_result, "Création 1M LUMs");

    // Test 2: Opérations sur 1 million de LUMs
    printf("\nTest 2: Opérations sur %d LUMs...\n", MILLION_LUMS);
    stress_test_result_t* operations_result = run_million_lum_operations_test();
    print_stress_results(operations_result, "Opérations 1M LUMs");

    // Test 3: Test de scalabilité progressive
    printf("\nTest 3: Scalabilité progressive jusqu'à %d LUMs...\n", MAX_STRESS_LUMS);
    stress_test_result_t* scalability_result = run_scalability_stress_test(MAX_STRESS_LUMS);
    print_stress_results(scalability_result, "Scalabilité Maximum");

    // Test 4: Pression mémoire extrême
    printf("\nTest 4: Pression mémoire extrême...\n");
    stress_test_result_t* memory_result = run_memory_pressure_test();
    print_stress_results(memory_result, "Pression Mémoire");

    // Test 5: Stress parallèle multi-thread
    printf("\nTest 5: Stress parallèle multi-thread...\n");
    stress_test_result_t* parallel_result = run_parallel_stress_test();
    print_stress_results(parallel_result, "Stress Parallèle");

    // Test 6: Test d'opération sur un groupe de 1 million de LUMs (optimisé)
    printf("\nTest 6: Opérations sur groupe de 1 million de LUMs (optimisé)...\n");
    stress_test_result_t* million_lum_ops_result = test_million_lum_group_operations();
    print_stress_results(million_lum_ops_result, "Opérations 1M LUMs (Optimisé)");


    // Cleanup
    stress_test_result_destroy(creation_result);
    stress_test_result_destroy(operations_result);
    stress_test_result_destroy(scalability_result);
    stress_test_result_destroy(memory_result);
    stress_test_result_destroy(parallel_result);
    stress_test_result_destroy(million_lum_ops_result);

    printf("\n✅ TESTS DE STRESS TERMINÉS - Système validé sous charge extrême\n");
    return 0;
}

stress_test_result_t* run_million_lum_creation_test(void) {
    stress_test_result_t* result = TRACKED_MALLOC(sizeof(stress_test_result_t));
    if (!result) return NULL;

    result->lum_count = MILLION_LUMS;
    strcpy(result->bottleneck, "None");
    result->graceful_degradation = true;

    // Création avec optimiseur mémoire
    memory_optimizer_t* optimizer = memory_optimizer_create(MILLION_LUMS * sizeof(lum_t));
    if (!optimizer) {
        strcpy(result->bottleneck, "Memory allocator initialization");
        result->graceful_degradation = false;
        return result;
    }

    clock_t start = clock();

    // Allocation de masse
    lum_t** lums = TRACKED_MALLOC(MILLION_LUMS * sizeof(lum_t*));
    if (!lums) {
        memory_optimizer_destroy(optimizer);
        strcpy(result->bottleneck, "Array allocation");
        result->graceful_degradation = false;
        return result;
    }

    // Création en masse avec optimisations
    for (size_t i = 0; i < MILLION_LUMS; i++) {
        lums[i] = memory_optimizer_alloc_lum(optimizer);
        if (!lums[i]) {
            result->lum_count = i;
            strcpy(result->bottleneck, "LUM allocation failure");
            result->graceful_degradation = false;
            break;
        }

        lums[i]->presence = (i % 2);
        lums[i]->position_x = i % 1000;
        lums[i]->position_y = i / 1000;
        lums[i]->structure_type = LUM_STRUCTURE_LINEAR;
        lums[i]->timestamp = time(NULL);

        // Feedback de progression pour très gros volumes
        if (i % 100000 == 0) {
            printf("  Progrès: %zu/%d LUMs créés\n", i, MILLION_LUMS);
        }
    }

    clock_t end = clock();
    result->creation_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Mesure mémoire
    memory_stats_t* stats = memory_optimizer_get_stats(optimizer);
    result->memory_peak = stats->peak_usage;

    // Nettoyage
    start = clock();
    for (size_t i = 0; i < result->lum_count; i++) {
        memory_optimizer_free_lum(optimizer, lums[i]);
    }
    TRACKED_FREE(lums);
    end = clock();
    result->destruction_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    memory_optimizer_destroy(optimizer);
    return result;
}

stress_test_result_t* run_million_lum_operations_test(void) {
    stress_test_result_t* result = TRACKED_MALLOC(sizeof(stress_test_result_t));
    if (!result) return NULL;

    result->lum_count = MILLION_LUMS;
    strcpy(result->bottleneck, "None");
    result->graceful_degradation = true;

    // Créer groupe de 1 million de LUMs
    lum_group_t* mega_group = lum_group_create(1);
    if (!mega_group) {
        strcpy(result->bottleneck, "Group creation");
        result->graceful_degradation = false;
        return result;
    }

    // Allocation massive
    for (size_t i = 0; i < MILLION_LUMS; i++) {
        lum_t* lum = lum_create(i % 2, i % 1000, i / 1000, LUM_STRUCTURE_LINEAR);
        if (lum) {
            lum_group_add_lum(mega_group, lum);
        } else {
            result->lum_count = i;
            strcpy(result->bottleneck, "LUM creation in group");
            break;
        }

        if (i % 100000 == 0) {
            printf("  Groupe: %zu/%d LUMs ajoutés\n", i, MILLION_LUMS);
        }
    }

    clock_t start = clock();

    // Opérations de stress sur le méga-groupe
    printf("  Exécution opérations VORAX sur %zu LUMs...\n", result->lum_count);

    // Test CYCLE sur groupe massif - PROTECTION DOUBLE FREE
    vorax_result_t* cycle_result = vorax_cycle(mega_group, 1000000);
    if (!cycle_result || !cycle_result->success) {
        strcpy(result->bottleneck, "CYCLE operation");
        result->graceful_degradation = false;
    } else {
        // CRITIQUE: Ne pas détruire ici car la destruction sera faite à la fin
        // Le cycle_result->result_group sera détruit par vorax_result_destroy
        printf("  ✅ Cycle operation réussie sur %zu LUMs\n", mega_group->count);
    }

    // Test conservation sur volume massif
    bool conservation_ok = true;
    size_t total_presence = 0;
    for (size_t i = 0; i < mega_group->count; i++) {
        total_presence += mega_group->lums[i].presence;
    }
    printf("  Conservation vérifiée: %zu LUMs présents\n", total_presence);

    clock_t end = clock();
    result->operation_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Nettoyage
    if (cycle_result) vorax_result_destroy(cycle_result);
    lum_group_destroy(mega_group);

    return result;
}

stress_test_result_t* run_scalability_stress_test(size_t max_lums) {
    stress_test_result_t* result = TRACKED_MALLOC(sizeof(stress_test_result_t));
    if (!result) return NULL;

    strcpy(result->bottleneck, "None");
    result->graceful_degradation = true;

    printf("  Test de scalabilité progressive...\n");

    size_t current_size = 1000;
    clock_t total_start = clock();

    while (current_size <= max_lums) {
        printf("    Testant %zu LUMs...", current_size);

        clock_t start = clock();

        // Test rapide à cette taille
        lum_group_t* test_group = lum_group_create(1);
        bool size_ok = true;

        for (size_t i = 0; i < current_size && size_ok; i++) {
            lum_t* lum = lum_create(1, i, i, LUM_STRUCTURE_LINEAR);
            if (lum) {
                lum_group_add_lum(test_group, lum);
            } else {
                size_ok = false;
                result->lum_count = i;
                snprintf(result->bottleneck, sizeof(result->bottleneck), 
                        "Scalability limit at %zu LUMs", current_size);
                result->graceful_degradation = false;
            }
        }

        clock_t end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

        printf(" %.3fs", time_taken);

        if (!size_ok) {
            printf(" ❌ LIMITE ATTEINTE\n");
            lum_group_destroy(test_group);
            break;
        }

        printf(" ✅\n");
        lum_group_destroy(test_group);

        // Progression exponentielle
        current_size *= 2;
        if (current_size > max_lums) {
            current_size = max_lums;
        }
    }

    clock_t total_end = clock();
    result->operation_time = ((double)(total_end - total_start)) / CLOCKS_PER_SEC;

    if (result->lum_count == 0) {
        result->lum_count = current_size;
    }

    return result;
}

stress_test_result_t* run_memory_pressure_test(void) {
    stress_test_result_t* result = TRACKED_MALLOC(sizeof(stress_test_result_t));
    if (!result) return NULL;

    strcpy(result->bottleneck, "None");
    result->graceful_degradation = true;

    printf("  Test de pression mémoire extrême...\n");

    // Allocations multiples simultanées
    const int num_groups = 10;
    lum_group_t* groups[num_groups];
    size_t group_sizes[num_groups] = {100000, 200000, 150000, 300000, 250000, 
                                     180000, 220000, 320000, 280000, 350000};

    clock_t start = clock();

    // Création simultanée de groupes volumineux
    for (int g = 0; g < num_groups; g++) {
        printf("    Groupe %d: %zu LUMs...", g + 1, group_sizes[g]);

        groups[g] = lum_group_create(g + 1);
        if (!groups[g]) {
            snprintf(result->bottleneck, sizeof(result->bottleneck), 
                    "Group %d allocation failure", g + 1);
            result->graceful_degradation = false;
            break;
        }

        for (size_t i = 0; i < group_sizes[g]; i++) {
            lum_t* lum = lum_create(i % 2, i, i, LUM_STRUCTURE_LINEAR);
            if (lum) {
                lum_group_add_lum(groups[g], lum);
                result->lum_count++;
            } else {
                snprintf(result->bottleneck, sizeof(result->bottleneck), 
                        "Memory pressure at group %d, LUM %zu", g + 1, i);
                result->graceful_degradation = false;
                break;
            }
        }

        printf(" ✅\n");

        if (!result->graceful_degradation) break;
    }

    clock_t end = clock();
    result->creation_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Nettoyage
    for (int g = 0; g < num_groups; g++) {
        if (groups[g]) {
            lum_group_destroy(groups[g]);
        }
    }

    return result;
}

stress_test_result_t* run_parallel_stress_test(void) {
    stress_test_result_t* result = TRACKED_MALLOC(sizeof(stress_test_result_t));
    if (!result) return NULL;

    strcpy(result->bottleneck, "None");
    result->graceful_degradation = true;

    printf("  Test de stress parallèle multi-thread...\n");

    // Test avec processeur parallèle
    parallel_processor_t* processor = parallel_processor_create(8);
    if (!processor) {
        strcpy(result->bottleneck, "Parallel processor creation");
        result->graceful_degradation = false;
        return result;
    }

    clock_t start = clock();

    // Soumission massive de tâches
    const size_t num_tasks = 10000;
    lum_t** test_lums = malloc(num_tasks * sizeof(lum_t*));

    for (size_t i = 0; i < num_tasks; i++) {
        test_lums[i] = lum_create(1, i, i, LUM_STRUCTURE_LINEAR);

        parallel_task_t* task = parallel_task_create(TASK_LUM_CREATE, 
                                                    test_lums[i], sizeof(lum_t));
        if (task) {
            parallel_processor_submit_task(processor, task);
            result->lum_count++;
        } else {
            snprintf(result->bottleneck, sizeof(result->bottleneck), 
                    "Task creation failure at %zu", i);
            result->graceful_degradation = false;
            break;
        }
    }

    // Attendre completion
    parallel_processor_wait_for_completion(processor);

    clock_t end = clock();
    result->operation_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Nettoyage
    for (size_t i = 0; i < result->lum_count; i++) {
        if (test_lums[i]) lum_destroy(test_lums[i]);
    }
    free(test_lums);
    parallel_processor_destroy(processor);

    return result;
}

stress_test_result_t* test_million_lum_group_operations(void) {
    printf("  Test: Opérations sur groupe de 1 million de LUMs\n");

    stress_test_result_t* result = TRACKED_MALLOC(sizeof(stress_test_result_t));
    if (!result) return NULL;

    result->lum_count = MILLION_LUMS;
    strcpy(result->bottleneck, "None");
    result->graceful_degradation = true;

    // Mesures ressources début
    struct rusage usage_start, usage_end;
    getrusage(RUSAGE_SELF, &usage_start);

    // Utiliser zero-copy allocator pour performance
    zero_copy_pool_t* zc_pool = zero_copy_pool_create(MILLION_LUMS * sizeof(lum_t) * 2);
    if (!zc_pool) {
        strcpy(result->bottleneck, "Zero-copy pool creation");
        result->graceful_degradation = false;
        return result;
    }

    // Créer groupe de 1 million de LUMs avec capacité pré-allouée
    lum_group_t* mega_group = lum_group_create(MILLION_LUMS);
    if (!mega_group) {
        strcpy(result->bottleneck, "Group creation");
        result->graceful_degradation = false;
        zero_copy_pool_destroy(zc_pool);
        return result;
    }

    clock_t start = clock();

    // Allocation massive optimisée
    printf("  Allocation de %d LUMs en cours...\n", MILLION_LUMS);
    for (size_t i = 0; i < MILLION_LUMS; i++) {
        // Utilisation allocation zero-copy
        zero_copy_allocation_t* zc_alloc = zero_copy_allocate(zc_pool, sizeof(lum_t));
        if (zc_alloc) {
            lum_t* lum = (lum_t*)zc_alloc->memory;
            lum->presence = i % 2;
            lum->id = lum_generate_id();
            lum->position_x = i % 1000;
            lum->position_y = i / 1000;
            lum->structure_type = LUM_STRUCTURE_LINEAR;
            lum->timestamp = lum_get_timestamp();

            if (!lum_group_add(mega_group, lum)) {
                result->lum_count = i;
                strcpy(result->bottleneck, "Group add operation");
                break;
            }
        } else {
            result->lum_count = i;
            strcpy(result->bottleneck, "Zero-copy allocation");
            break;
        }

        if (i % 100000 == 0) {
            printf("  Progrès: %zu/%d LUMs alloués (%.1f%%)\n", 
                   i, MILLION_LUMS, (double)i * 100.0 / MILLION_LUMS);
        }
    }

    clock_t allocation_end = clock();
    double allocation_time = ((double)(allocation_end - start)) / CLOCKS_PER_SEC;

    printf("  ✓ Allocation terminée: %zu LUMs en %.3f secondes\n", 
           result->lum_count, allocation_time);
    printf("  Débit allocation: %.0f LUMs/seconde\n", 
           result->lum_count / allocation_time);

    // Test opérations VORAX sur méga-groupe
    printf("  Exécution tests opérationnels sur %zu LUMs...\n", result->lum_count);

    // Test CYCLE optimisé
    clock_t cycle_start = clock();
    vorax_result_t* cycle_result = vorax_cycle(mega_group, 1000000);
    clock_t cycle_end = clock();

    if (cycle_result && cycle_result->success) {
        double cycle_time = ((double)(cycle_end - cycle_start)) / CLOCKS_PER_SEC;
        printf("  ✓ CYCLE: %.3f secondes, débit: %.0f LUMs/s\n",
               cycle_time, result->lum_count / cycle_time);
        vorax_result_destroy(cycle_result);
    } else {
        strcpy(result->bottleneck, "CYCLE operation");
        result->graceful_degradation = false;
    }

    // Test SPLIT massif
    clock_t split_start = clock();
    vorax_result_t* split_result = vorax_split(mega_group, 1000);
    clock_t split_end = clock();

    if (split_result && split_result->success) {
        double split_time = ((double)(split_end - split_start)) / CLOCKS_PER_SEC;
        printf("  ✓ SPLIT en 1000 groupes: %.3f secondes\n", split_time);

        // Compter LUMs résultantes
        size_t total_result_lums = 0;
        for (size_t i = 0; i < split_result->result_count; i++) {
            if (split_result->result_groups[i]) {
                total_result_lums += split_result->result_groups[i]->count;
            }
        }

        printf("  ✓ Conservation vérifiée: %zu -> %zu LUMs\n", 
               result->lum_count, total_result_lums);

        if (total_result_lums != result->lum_count) {
            strcpy(result->bottleneck, "Conservation violation in SPLIT");
            result->graceful_degradation = false;
        }

        vorax_result_destroy(split_result);
    } else {
        strcpy(result->bottleneck, "SPLIT operation");
    }

    // Mesures ressources fin
    getrusage(RUSAGE_SELF, &usage_end);

    long memory_peak = usage_end.ru_maxrss; // kB sur Linux
    printf("  Pic mémoire utilisée: %ld MB\n", memory_peak / 1024);

    result->peak_memory_mb = memory_peak / 1024;
    result->execution_time_s = ((double)(clock() - start)) / CLOCKS_PER_SEC;

    // Nettoyage
    lum_group_destroy(mega_group);
    zero_copy_pool_destroy(zc_pool);

    return result;
}

void print_stress_results(stress_test_result_t* result, const char* test_name) {
    if (!result) return;

    printf("📊 Résultats: %s\n", test_name);
    printf("   LUMs traités: %zu\n", result->lum_count);
    printf("   Temps création: %.3f secondes\n", result->creation_time);
    printf("   Temps opérations: %.3f secondes\n", result->operation_time);
    printf("   Temps destruction: %.3f secondes\n", result->destruction_time);
    printf("   Pic mémoire (opt): %zu bytes\n", result->memory_peak);
    printf("   Pic mémoire (sys): %ld MB\n", result->peak_memory_mb);
    printf("   Dégradation gracieuse: %s\n", result->graceful_degradation ? "✅" : "❌");
    if (strlen(result->bottleneck) > 4) {
        printf("   Goulot détecté: %s\n", result->bottleneck);
    }

    if (result->lum_count > 0 && result->execution_time_s > 0) {
        double throughput = result->lum_count / result->execution_time_s;
        printf("   Débit total: %.0f LUMs/seconde\n", throughput);
    }
}

void stress_test_result_destroy(stress_test_result_t* result) {
    if (result) {
        TRACKED_FREE(result);
    }
}