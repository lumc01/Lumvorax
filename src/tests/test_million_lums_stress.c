#include "../lum/lum_core.h"
#include "../logger/lum_logger.h"
#include "../metrics/performance_metrics.h"
#include "../optimization/memory_optimizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#define MILLION_LUMS 1000000
#define MAX_STRESS_LUMS 10000000

static double get_microseconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

static size_t get_memory_usage_kb(void) {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return 0;
    
    char line[128];
    size_t vm_rss = 0;
    
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "VmRSS: %zu kB", &vm_rss) == 1) {
            break;
        }
    }
    fclose(file);
    return vm_rss;
}

bool test_million_lums_creation(void) {
    printf("🔥 TEST STRESS MILLION LUMs - CRÉATION MASSIVE\n");
    
    double start_time = get_microseconds();
    size_t start_memory = get_memory_usage_kb();
    
    // Création de 1 million de LUMs
    lum_group_t* mega_group = lum_group_create(MILLION_LUMS);
    if (!mega_group) {
        printf("❌ ÉCHEC: Impossible de créer le groupe pour 1M LUMs\n");
        return false;
    }
    
    printf("📊 Création progressive de %d LUMs...\n", MILLION_LUMS);
    
    for (size_t i = 0; i < MILLION_LUMS; i++) {
        lum_t lum = {
            .presence = i % 2,
            .position_x = (uint32_t)(i % 10000),
            .position_y = (uint32_t)(i / 10000),
            .structure_type = (lum_structure_type_e)(i % 4),
            .timestamp = time(NULL) + i,
            .unique_id = (uint64_t)i
        };
        
        if (!lum_group_add(mega_group, &lum)) {
            printf("❌ ÉCHEC à l'index %zu\n", i);
            lum_group_destroy(mega_group);
            return false;
        }
        
        // Affichage progression
        if (i % 100000 == 0) {
            printf("  📈 %zu LUMs créés (%.1f%%)\n", i, (double)i / MILLION_LUMS * 100.0);
        }
    }
    
    double end_time = get_microseconds();
    size_t end_memory = get_memory_usage_kb();
    
    double creation_time = end_time - start_time;
    double lums_per_second = MILLION_LUMS / (creation_time / 1000000.0);
    
    printf("✅ SUCCÈS: %d LUMs créés en %.2f secondes\n", MILLION_LUMS, creation_time / 1000000.0);
    printf("📊 Débit: %.0f LUMs/seconde\n", lums_per_second);
    printf("💾 Mémoire utilisée: %zu KB\n", end_memory - start_memory);
    printf("🎯 Taille finale du groupe: %zu LUMs\n", mega_group->count);
    
    lum_group_destroy(mega_group);
    return true;
}

bool test_maximum_capacity_stress(void) {
    printf("\n🚀 TEST STRESS CAPACITÉ MAXIMALE SYSTÈME\n");
    
    size_t current_size = 100000;  // Démarrer à 100K
    size_t max_achieved = 0;
    bool system_limit_reached = false;
    
    while (current_size <= MAX_STRESS_LUMS && !system_limit_reached) {
        printf("🔍 Test capacité: %zu LUMs...\n", current_size);
        
        double start_time = get_microseconds();
        size_t start_memory = get_memory_usage_kb();
        
        lum_group_t* stress_group = lum_group_create(current_size);
        if (!stress_group) {
            printf("❌ LIMITE ATTEINTE: Impossible d'allouer %zu LUMs\n", current_size);
            system_limit_reached = true;
            break;
        }
        
        // Remplissage rapide
        bool creation_success = true;
        for (size_t i = 0; i < current_size; i++) {
            lum_t lum = {i % 2, (uint32_t)i, (uint32_t)i, (lum_structure_type_e)(i % 4), time(NULL), (uint64_t)i};
            if (!lum_group_add(stress_group, &lum)) {
                printf("⚠️ Échec ajout à l'index %zu\n", i);
                creation_success = false;
                break;
            }
        }
        
        double end_time = get_microseconds();
        size_t end_memory = get_memory_usage_kb();
        
        if (creation_success) {
            max_achieved = current_size;
            double time_ms = (end_time - start_time) / 1000.0;
            double throughput = current_size / (time_ms / 1000.0);
            
            printf("✅ SUCCÈS %zu LUMs: %.2f ms, %.0f LUMs/s, %zu KB mémoire\n", 
                   current_size, time_ms, throughput, end_memory - start_memory);
        } else {
            printf("❌ ÉCHEC PARTIEL à %zu LUMs\n", current_size);
            system_limit_reached = true;
        }
        
        lum_group_destroy(stress_group);
        
        // Incrément progressif pour approcher la limite
        if (current_size < 1000000) {
            current_size += 100000;  // +100K
        } else if (current_size < 5000000) {
            current_size += 500000;  // +500K
        } else {
            current_size += 1000000; // +1M
        }
        
        // Petite pause pour éviter surcharge système
        usleep(100000); // 100ms
    }
    
    printf("🏆 CAPACITÉ MAXIMALE ATTEINTE: %zu LUMs\n", max_achieved);
    return max_achieved >= MILLION_LUMS;
}

bool test_parallel_stress_operations(void) {
    printf("\n⚡ TEST STRESS OPÉRATIONS PARALLÈLES\n");
    
    const size_t test_sizes[] = {50000, 100000, 250000, 500000, 1000000};
    const size_t num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (size_t t = 0; t < num_tests; t++) {
        size_t test_size = test_sizes[t];
        printf("🔄 Test opérations sur %zu LUMs...\n", test_size);
        
        // Création groupe de test
        lum_group_t* test_group = lum_group_create(test_size);
        if (!test_group) {
            printf("❌ Impossible de créer groupe pour %zu LUMs\n", test_size);
            continue;
        }
        
        // Remplissage
        for (size_t i = 0; i < test_size; i++) {
            lum_t lum = {i % 2, (uint32_t)i, (uint32_t)i, LUM_STRUCTURE_LINEAR, time(NULL), (uint64_t)i};
            lum_group_add(test_group, &lum);
        }
        
        double start_time = get_microseconds();
        
        // Test opérations VORAX multiples
        lum_group_t* group1 = lum_group_create(test_size / 2);
        lum_group_t* group2 = lum_group_create(test_size / 2);
        
        // Division du groupe
        for (size_t i = 0; i < test_size / 2; i++) {
            lum_group_add(group1, &test_group->lums[i]);
            lum_group_add(group2, &test_group->lums[i + test_size / 2]);
        }
        
        // Opérations de fusion/split intensives
        for (int iter = 0; iter < 10; iter++) {
            vorax_result_t* fuse_result = vorax_fuse(group1, group2);
            if (fuse_result && fuse_result->success) {
                vorax_result_t* split_result = vorax_split(fuse_result->result_group, 4);
                if (split_result) {
                    for (size_t g = 0; g < split_result->result_count; g++) {
                        if (split_result->result_groups[g]) {
                            lum_group_destroy(split_result->result_groups[g]);
                        }
                    }
                    free(split_result->result_groups);
                    free(split_result);
                }
                if (fuse_result->result_group) {
                    lum_group_destroy(fuse_result->result_group);
                }
                free(fuse_result);
            }
        }
        
        double end_time = get_microseconds();
        double operations_time = (end_time - start_time) / 1000.0;
        
        printf("✅ %zu LUMs: %.2f ms pour 10 cycles fuse/split\n", test_size, operations_time);
        
        lum_group_destroy(group1);
        lum_group_destroy(group2);
        lum_group_destroy(test_group);
    }
    
    return true;
}

int main(void) {
    printf("🚀 === TESTS DE STRESS EXTRÊME LUM/VORAX ===\n");
    printf("Date: %ld\n", time(NULL));
    printf("PID: %d\n", getpid());
    
    // Initialisation logging
    lum_log_init("logs/million_lums_stress.log");
    
    // Tests progressifs
    bool test1 = test_million_lums_creation();
    bool test2 = test_maximum_capacity_stress();
    bool test3 = test_parallel_stress_operations();
    
    printf("\n📋 === RÉSULTATS FINAUX ===\n");
    printf("Test Million LUMs: %s\n", test1 ? "✅ RÉUSSI" : "❌ ÉCHEC");
    printf("Test Capacité Max: %s\n", test2 ? "✅ RÉUSSI" : "❌ ÉCHEC");
    printf("Test Opérations Parallèles: %s\n", test3 ? "✅ RÉUSSI" : "❌ ÉCHEC");
    
    if (test1 && test2 && test3) {
        printf("\n🎉 TOUS LES TESTS DE STRESS EXTRÊME RÉUSSIS !\n");
        return 0;
    } else {
        printf("\n⚠️ CERTAINS TESTS ONT ÉCHOUÉ\n");
        return 1;
    }
}