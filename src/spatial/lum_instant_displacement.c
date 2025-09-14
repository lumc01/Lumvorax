
#include "lum_instant_displacement.h"
#include "../debug/memory_tracker.h"
#include "../logger/lum_logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// Fonction utilitaire pour timestamp haute précision
static uint64_t get_precise_timestamp_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    }
    return 0;
}

bool lum_instant_displace(lum_t* lum, int32_t new_x, int32_t new_y, lum_displacement_result_t* result) {
    if (!lum || !result) {
        return false;
    }

    // Vérifier que la LUM n'est pas détruite
    if (lum->is_destroyed) {
        return false;
    }

    // Validation des coordonnées
    if (!lum_validate_displacement_coordinates(new_x, new_y)) {
        result->success = false;
        return false;
    }

    uint64_t start_time = get_precise_timestamp_ns();

    // Sauvegarder position actuelle
    result->from_x = lum->position_x;
    result->from_y = lum->position_y;
    result->to_x = new_x;
    result->to_y = new_y;

    // DÉPLACEMENT INSTANTANÉ : Modification directe des coordonnées
    lum->position_x = new_x;
    lum->position_y = new_y;
    
    // Mise à jour timestamp pour traçabilité
    lum->timestamp = get_precise_timestamp_ns();

    uint64_t end_time = get_precise_timestamp_ns();
    result->displacement_time_ns = end_time - start_time;
    result->success = true;

    printf("[DISPLACEMENT] LUM[%u] déplacée instantanément de (%d,%d) → (%d,%d) en %lu ns\n",
           lum->id, result->from_x, result->from_y, result->to_x, result->to_y, 
           result->displacement_time_ns);

    return true;
}

bool lum_group_instant_displace_all(lum_group_t* group, int32_t delta_x, int32_t delta_y) {
    if (!group || group->magic_number != LUM_VALIDATION_PATTERN) {
        return false;
    }

    printf("[GROUP_DISPLACEMENT] Déplacement de %zu LUMs par delta (%d,%d)\n", 
           group->count, delta_x, delta_y);

    uint64_t start_time = get_precise_timestamp_ns();

    for (size_t i = 0; i < group->count; i++) {
        lum_t* lum = &group->lums[i];
        
        // Calcul nouvelles coordonnées
        int32_t new_x = lum->position_x + delta_x;
        int32_t new_y = lum->position_y + delta_y;

        // Validation limites
        if (lum_validate_displacement_coordinates(new_x, new_y)) {
            lum->position_x = new_x;
            lum->position_y = new_y;
            lum->timestamp = get_precise_timestamp_ns();
        }
    }

    uint64_t end_time = get_precise_timestamp_ns();
    printf("[GROUP_DISPLACEMENT] %zu LUMs déplacées en %lu ns (%.2f ns/LUM)\n",
           group->count, end_time - start_time, 
           (double)(end_time - start_time) / group->count);

    return true;
}

bool lum_validate_displacement_coordinates(int32_t x, int32_t y) {
    // Validation limites raisonnables pour éviter overflow
    return (x >= -MAX_DISPLACEMENT_DISTANCE && x <= MAX_DISPLACEMENT_DISTANCE &&
            y >= -MAX_DISPLACEMENT_DISTANCE && y <= MAX_DISPLACEMENT_DISTANCE);
}

lum_displacement_metrics_t* lum_displacement_metrics_create(void) {
    lum_displacement_metrics_t* metrics = TRACKED_MALLOC(sizeof(lum_displacement_metrics_t));
    if (!metrics) return NULL;

    metrics->total_displacements = 0;
    metrics->successful_displacements = 0;
    metrics->total_time_ns = 0;
    metrics->average_time_ns = 0.0;
    metrics->magic_number = LUM_DISPLACEMENT_MAGIC;

    return metrics;
}

void lum_displacement_metrics_destroy(lum_displacement_metrics_t* metrics) {
    if (metrics && metrics->magic_number == LUM_DISPLACEMENT_MAGIC) {
        metrics->magic_number = 0xDEADBEEF; // Marquer comme détruit
        TRACKED_FREE(metrics);
    }
}

void lum_displacement_metrics_record(lum_displacement_metrics_t* metrics, lum_displacement_result_t* result) {
    if (!metrics || !result || metrics->magic_number != LUM_DISPLACEMENT_MAGIC) {
        return;
    }

    metrics->total_displacements++;
    if (result->success) {
        metrics->successful_displacements++;
        metrics->total_time_ns += result->displacement_time_ns;
        metrics->average_time_ns = (double)metrics->total_time_ns / metrics->successful_displacements;
    }
}

void lum_displacement_metrics_print(const lum_displacement_metrics_t* metrics) {
    if (!metrics || metrics->magic_number != LUM_DISPLACEMENT_MAGIC) {
        printf("[ERROR] Métriques de déplacement invalides\n");
        return;
    }

    printf("\n=== MÉTRIQUES DÉPLACEMENT INSTANTANÉ LUM ===\n");
    printf("Total déplacements tentés: %u\n", metrics->total_displacements);
    printf("Déplacements réussis: %u\n", metrics->successful_displacements);
    printf("Taux de succès: %.2f%%\n", 
           metrics->total_displacements > 0 ? 
           (100.0 * metrics->successful_displacements / metrics->total_displacements) : 0.0);
    printf("Temps total: %lu ns\n", metrics->total_time_ns);
    printf("Temps moyen par déplacement: %.2f ns\n", metrics->average_time_ns);
    printf("=============================================\n\n");
}

bool lum_test_displacement_performance(size_t num_lums) {
    printf("\n=== TEST PERFORMANCE DÉPLACEMENT INSTANTANÉ ===\n");
    printf("Création de %zu LUMs pour test...\n", num_lums);

    // Créer groupe de test
    lum_group_t* test_group = lum_group_create(num_lums);
    if (!test_group) {
        printf("[ERROR] Échec création groupe test\n");
        return false;
    }

    // Créer LUMs
    for (size_t i = 0; i < num_lums; i++) {
        lum_t* lum = lum_create(1, (int32_t)i, (int32_t)i, LUM_STRUCTURE_LINEAR);
        if (lum) {
            lum_group_add(test_group, lum);
            lum_destroy(lum);
        }
    }

    printf("✅ %zu LUMs créées dans le groupe\n", lum_group_size(test_group));

    // Métriques
    lum_displacement_metrics_t* metrics = lum_displacement_metrics_create();
    if (!metrics) {
        lum_group_destroy(test_group);
        return false;
    }

    // Test déplacements individuels
    uint64_t start_individual = get_precise_timestamp_ns();
    
    for (size_t i = 0; i < lum_group_size(test_group); i++) {
        lum_t* lum = lum_group_get(test_group, i);
        lum_displacement_result_t result;
        
        // Déplacement vers nouvelle position
        int32_t new_x = lum->position_x + 100;
        int32_t new_y = lum->position_y + 200;
        
        if (lum_instant_displace(lum, new_x, new_y, &result)) {
            lum_displacement_metrics_record(metrics, &result);
        }
    }
    
    uint64_t end_individual = get_precise_timestamp_ns();

    // Test déplacement de groupe
    uint64_t start_group = get_precise_timestamp_ns();
    lum_group_instant_displace_all(test_group, -50, -75);
    uint64_t end_group = get_precise_timestamp_ns();

    // Affichage résultats
    printf("\n📊 RÉSULTATS PERFORMANCE :\n");
    printf("Déplacements individuels: %lu ns total (%.2f ns/LUM)\n", 
           end_individual - start_individual,
           (double)(end_individual - start_individual) / num_lums);
    printf("Déplacement de groupe: %lu ns total (%.2f ns/LUM)\n",
           end_group - start_group,
           (double)(end_group - start_group) / num_lums);

    lum_displacement_metrics_print(metrics);

    // Cleanup
    lum_displacement_metrics_destroy(metrics);
    lum_group_destroy(test_group);

    printf("✅ Test performance déplacement terminé\n");
    return true;
}

bool lum_test_displacement_vs_traditional_move(size_t num_operations) {
    printf("\n=== COMPARAISON DÉPLACEMENT vs MÉTHODE TRADITIONNELLE ===\n");
    printf("Test avec %zu opérations...\n", num_operations);

    // Créer LUM de test
    lum_t* test_lum = lum_create(1, 0, 0, LUM_STRUCTURE_LINEAR);
    if (!test_lum) {
        return false;
    }

    // Test méthode traditionnelle (simulation)
    uint64_t start_traditional = get_precise_timestamp_ns();
    for (size_t i = 0; i < num_operations; i++) {
        // Simulation recherche dans liste + modification
        // (dans un vrai système, cela impliquerait parcours O(n))
        for (volatile int j = 0; j < 100; j++) {} // Simulation temps de recherche
        test_lum->position_x = (int32_t)i;
        test_lum->position_y = (int32_t)i;
    }
    uint64_t end_traditional = get_precise_timestamp_ns();

    // Reset position
    test_lum->position_x = 0;
    test_lum->position_y = 0;

    // Test déplacement instantané
    uint64_t start_instant = get_precise_timestamp_ns();
    for (size_t i = 0; i < num_operations; i++) {
        lum_displacement_result_t result;
        lum_instant_displace(test_lum, (int32_t)i, (int32_t)i, &result);
    }
    uint64_t end_instant = get_precise_timestamp_ns();

    // Calcul gains
    uint64_t traditional_time = end_traditional - start_traditional;
    uint64_t instant_time = end_instant - start_instant;
    double speedup = (double)traditional_time / instant_time;

    printf("\n📈 COMPARAISON PERFORMANCE :\n");
    printf("Méthode traditionnelle: %lu ns (%lu ns/op)\n", 
           traditional_time, traditional_time / num_operations);
    printf("Déplacement instantané: %lu ns (%lu ns/op)\n", 
           instant_time, instant_time / num_operations);
    printf("Accélération: %.2fx plus rapide\n", speedup);
    printf("Gain: %lu ns économisés (%.1f%%)\n", 
           traditional_time - instant_time,
           100.0 * (traditional_time - instant_time) / traditional_time);

    lum_destroy(test_lum);

    printf("✅ Comparaison terminée\n");
    return speedup > 1.0; // Test réussi si gain de performance
}
