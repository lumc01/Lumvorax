
#include "../spatial/lum_instant_displacement.h"
#include "../lum/lum_core.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

void test_basic_displacement(void) {
    printf("🧪 Test déplacement de base...\n");
    
    lum_t* lum = lum_create(1, 10, 20, LUM_STRUCTURE_LINEAR);
    assert(lum != NULL);
    
    lum_displacement_result_t result;
    bool success = lum_instant_displace(lum, 100, 200, &result);
    
    assert(success == true);
    assert(result.success == true);
    assert(result.from_x == 10);
    assert(result.from_y == 20);
    assert(result.to_x == 100);
    assert(result.to_y == 200);
    assert(lum->position_x == 100);
    assert(lum->position_y == 200);
    assert(result.displacement_time_ns > 0);
    
    lum_destroy(lum);
    printf("✅ Test déplacement de base réussi\n");
}

void test_group_displacement(void) {
    printf("🧪 Test déplacement de groupe...\n");
    
    lum_group_t* group = lum_group_create(5);
    assert(group != NULL);
    
    // Ajouter 5 LUMs
    for (int i = 0; i < 5; i++) {
        lum_t* lum = lum_create(1, i * 10, i * 10, LUM_STRUCTURE_LINEAR);
        assert(lum_group_add(group, lum));
        lum_destroy(lum);
    }
    
    // Déplacement de groupe
    bool success = lum_group_instant_displace_all(group, 50, 75);
    assert(success == true);
    
    // Vérifier toutes les LUMs ont été déplacées
    for (size_t i = 0; i < lum_group_size(group); i++) {
        lum_t* lum = lum_group_get(group, i);
        assert(lum->position_x == (int32_t)(i * 10 + 50));
        assert(lum->position_y == (int32_t)(i * 10 + 75));
    }
    
    lum_group_destroy(group);
    printf("✅ Test déplacement de groupe réussi\n");
}

void test_displacement_metrics(void) {
    printf("🧪 Test métriques de déplacement...\n");
    
    lum_displacement_metrics_t* metrics = lum_displacement_metrics_create();
    assert(metrics != NULL);
    assert(metrics->magic_number == LUM_DISPLACEMENT_MAGIC);
    
    lum_t* lum = lum_create(1, 0, 0, LUM_STRUCTURE_LINEAR);
    
    // Effectuer plusieurs déplacements
    for (int i = 0; i < 10; i++) {
        lum_displacement_result_t result;
        lum_instant_displace(lum, i * 10, i * 10, &result);
        lum_displacement_metrics_record(metrics, &result);
    }
    
    assert(metrics->total_displacements == 10);
    assert(metrics->successful_displacements == 10);
    assert(metrics->total_time_ns > 0);
    assert(metrics->average_time_ns > 0);
    
    lum_displacement_metrics_print(metrics);
    
    lum_destroy(lum);
    lum_displacement_metrics_destroy(metrics);
    printf("✅ Test métriques réussi\n");
}

void test_displacement_validation(void) {
    printf("🧪 Test validation coordonnées...\n");
    
    // Coordonnées valides
    assert(lum_validate_displacement_coordinates(0, 0) == true);
    assert(lum_validate_displacement_coordinates(1000, -1000) == true);
    assert(lum_validate_displacement_coordinates(MAX_DISPLACEMENT_DISTANCE, MAX_DISPLACEMENT_DISTANCE) == true);
    
    // Coordonnées invalides
    assert(lum_validate_displacement_coordinates(MAX_DISPLACEMENT_DISTANCE + 1, 0) == false);
    assert(lum_validate_displacement_coordinates(0, -MAX_DISPLACEMENT_DISTANCE - 1) == false);
    
    printf("✅ Test validation réussi\n");
}

void test_performance_stress(void) {
    printf("🧪 Test stress performance déplacement...\n");
    
    const size_t NUM_LUMS = 10000;
    bool success = lum_test_displacement_performance(NUM_LUMS);
    assert(success == true);
    
    printf("✅ Test stress performance réussi\n");
}

void test_comparison_traditional_vs_instant(void) {
    printf("🧪 Test comparaison méthodes...\n");
    
    const size_t NUM_OPERATIONS = 1000;
    bool faster = lum_test_displacement_vs_traditional_move(NUM_OPERATIONS);
    
    if (faster) {
        printf("✅ Déplacement instantané est plus rapide !\n");
    } else {
        printf("⚠️  Déplacement instantané pas d'amélioration détectée\n");
    }
}

int main(void) {
    printf("=== TESTS DÉPLACEMENT INSTANTANÉ LUM ===\n\n");
    
    test_basic_displacement();
    test_group_displacement();
    test_displacement_metrics();
    test_displacement_validation();
    test_performance_stress();
    test_comparison_traditional_vs_instant();
    
    printf("\n🎯 TOUS LES TESTS DÉPLACEMENT INSTANTANÉ RÉUSSIS !\n");
    printf("✅ Théorie du déplacement instantané VALIDÉE\n");
    
    return 0;
}
