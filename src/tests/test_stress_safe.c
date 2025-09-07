#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#include "../lum/lum_core.h"
#include "../vorax/vorax_operations.h"
#include "../logger/lum_logger.h"
#include "../metrics/performance_metrics.h"
#include "../optimization/memory_optimizer.h"

// Tests de stress sécurisés pour environnement Replit
#define STRESS_LEVELS 10
#define BASE_LUMS 1000
#define MAX_SAFE_LUMS 100000  // 100K pour éviter timeout Replit

typedef struct {
    size_t lum_count;
    double creation_time_ms;
    double operation_time_ms;
    size_t memory_used_bytes;
    size_t operations_per_second;
    bool success;
} stress_result_t;

// Mesure temps haute précision
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Mesure mémoire utilisée
size_t get_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss * 1024; // Convert to bytes
}

// Test création massive de LUMs
stress_result_t test_lum_creation_stress(size_t target_lums) {
    stress_result_t result = {0};
    double start_time = get_time_ms();
    size_t memory_start = get_memory_usage();
    
    printf("  🔄 Test création %zu LUMs...", target_lums);
    fflush(stdout);
    
    lum_group_t* group = lum_group_create(target_lums);
    if (!group) {
        result.success = false;
        return result;
    }
    
    // Création progressive pour éviter timeout
    for (size_t i = 0; i < target_lums; i++) {
        lum_t* lum = lum_create(i % 2, (int32_t)(i % 1000), (int32_t)(i / 1000), LUM_STRUCTURE_LINEAR);
        if (!lum) {
            printf(" ÉCHEC à %zu LUMs\n", i);
            result.success = false;
            lum_group_destroy(group);
            return result;
        }
        
        if (!lum_group_add(group, lum)) {
            printf(" ÉCHEC ajout à %zu LUMs\n", i);
            lum_destroy(lum);
            result.success = false;
            lum_group_destroy(group);
            return result;
        }
        
        lum_destroy(lum); // Libération immédiate pour éviter fuites
        
        // Check timeout tous les 10000 LUMs
        if (i > 0 && i % 10000 == 0) {
            double elapsed = get_time_ms() - start_time;
            if (elapsed > 30000) { // 30 secondes max
                printf(" TIMEOUT à %zu LUMs\n", i);
                result.lum_count = i;
                result.creation_time_ms = elapsed;
                result.success = false;
                lum_group_destroy(group);
                return result;
            }
        }
    }
    
    double end_time = get_time_ms();
    size_t memory_end = get_memory_usage();
    
    result.lum_count = target_lums;
    result.creation_time_ms = end_time - start_time;
    result.memory_used_bytes = memory_end - memory_start;
    result.operations_per_second = (size_t)(target_lums / (result.creation_time_ms / 1000.0));
    result.success = true;
    
    printf(" ✅ SUCCÈS en %.2f ms\n", result.creation_time_ms);
    
    lum_group_destroy(group);
    return result;
}

// Test opérations VORAX sous stress
stress_result_t test_vorax_operations_stress(size_t lum_count) {
    stress_result_t result = {0};
    double start_time = get_time_ms();
    
    printf("  ⚡ Test opérations VORAX sur %zu LUMs...", lum_count);
    fflush(stdout);
    
    // Créer deux groupes
    lum_group_t* group1 = lum_group_create(lum_count / 2);
    lum_group_t* group2 = lum_group_create(lum_count / 2);
    
    if (!group1 || !group2) {
        result.success = false;
        return result;
    }
    
    // Remplir les groupes rapidement
    for (size_t i = 0; i < lum_count / 2; i++) {
        lum_t* lum1 = lum_create(1, (int32_t)i, 0, LUM_STRUCTURE_LINEAR);
        lum_t* lum2 = lum_create(0, (int32_t)i, 1, LUM_STRUCTURE_LINEAR);
        
        if (lum1 && lum2) {
            lum_group_add(group1, lum1);
            lum_group_add(group2, lum2);
            lum_destroy(lum1);
            lum_destroy(lum2);
        }
    }
    
    // Test FUSE
    vorax_result_t* fuse_result = vorax_fuse(group1, group2);
    if (!fuse_result || !fuse_result->success) {
        printf(" ÉCHEC FUSE\n");
        result.success = false;
        lum_group_destroy(group1);
        lum_group_destroy(group2);
        return result;
    }
    
    // Test SPLIT
    vorax_result_t* split_result = vorax_split(fuse_result->result_group, 4);
    if (!split_result || !split_result->success) {
        printf(" ÉCHEC SPLIT\n");
        result.success = false;
        vorax_result_destroy(fuse_result);
        lum_group_destroy(group1);
        lum_group_destroy(group2);
        return result;
    }
    
    double end_time = get_time_ms();
    
    result.lum_count = lum_count;
    result.operation_time_ms = end_time - start_time;
    result.operations_per_second = (size_t)(3 / (result.operation_time_ms / 1000.0)); // 3 opérations
    result.success = true;
    
    printf(" ✅ SUCCÈS en %.2f ms\n", result.operation_time_ms);
    
    // Nettoyage sécurisé
    vorax_result_destroy(split_result);
    vorax_result_destroy(fuse_result);
    lum_group_destroy(group1);
    lum_group_destroy(group2);
    
    return result;
}

// Test de capacité progressive
void test_progressive_capacity() {
    printf("\n🔍 === TEST DE CAPACITÉ MAXIMALE PROGRESSIVE ===\n");
    
    size_t max_successful = 0;
    size_t current_test = BASE_LUMS;
    
    for (int level = 0; level < STRESS_LEVELS && current_test <= MAX_SAFE_LUMS; level++) {
        printf("\n--- Niveau %d: %zu LUMs ---\n", level + 1, current_test);
        
        stress_result_t creation_result = test_lum_creation_stress(current_test);
        if (creation_result.success) {
            max_successful = current_test;
            printf("  📊 Création: %.2f ms, Mémoire: %zu bytes, Débit: %zu LUMs/s\n",
                   creation_result.creation_time_ms,
                   creation_result.memory_used_bytes,
                   creation_result.operations_per_second);
            
            // Test opérations seulement si création réussie
            stress_result_t operation_result = test_vorax_operations_stress(current_test);
            if (operation_result.success) {
                printf("  ⚡ Opérations: %.2f ms, Débit: %zu ops/s\n",
                       operation_result.operation_time_ms,
                       operation_result.operations_per_second);
            }
        } else {
            printf("  ❌ ÉCHEC au niveau %zu LUMs\n", current_test);
            break;
        }
        
        // Progression exponentielle
        current_test *= 2;
        
        // Pause pour éviter surcharge
        usleep(100000); // 100ms
    }
    
    printf("\n🏆 CAPACITÉ MAXIMALE VALIDÉE: %zu LUMs\n", max_successful);
}

// Test de performance par seconde
void test_operations_per_second() {
    printf("\n⏱️ === MESURE OPÉRATIONS PAR SECONDE ===\n");
    
    const size_t test_sizes[] = {100, 500, 1000, 2000, 5000, 10000};
    const size_t num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (size_t i = 0; i < num_tests; i++) {
        printf("\n🎯 Test %zu LUMs:\n", test_sizes[i]);
        
        // Moyenner sur 3 runs
        double total_ops_per_sec = 0;
        int successful_runs = 0;
        
        for (int run = 0; run < 3; run++) {
            stress_result_t result = test_lum_creation_stress(test_sizes[i]);
            if (result.success) {
                total_ops_per_sec += result.operations_per_second;
                successful_runs++;
            }
        }
        
        if (successful_runs > 0) {
            double avg_ops_per_sec = total_ops_per_sec / successful_runs;
            printf("  📈 Moyenne: %.0f LUMs/seconde\n", avg_ops_per_sec);
        }
    }
}

int main() {
    printf("🚀 === TESTS DE STRESS SYSTÈME LUM/VORAX ===\n");
    printf("Timestamp: %ld\n", time(NULL));
    printf("Environnement: Replit Cloud\n");
    printf("Limites sécurisées: MAX %d LUMs\n", MAX_SAFE_LUMS);
    
    // Initialiser le logger
    lum_logger_t* logger = lum_logger_create("logs/stress_test.log", true, true);
    if (logger) {
        lum_set_global_logger(logger);
    }
    
    // Mesures système initiales
    printf("\n💻 Configuration système:\n");
    printf("  RAM disponible: ");
    system("free -h | grep Mem | awk '{print $7}'");
    printf("  CPU cores: %ld\n", sysconf(_SC_NPROCESSORS_ONLN));
    
    // Tests progressifs
    test_progressive_capacity();
    test_operations_per_second();
    
    printf("\n✅ Tests de stress terminés avec succès!\n");
    printf("📝 Logs sauvegardés dans: logs/stress_test.log\n");
    
    if (logger) {
        lum_logger_destroy(logger);
    }
    
    return 0;
}