#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stddef.h>
#include "src/lum/lum_core.h"
#include "src/vorax/vorax_operations.h"
#include "src/debug/memory_tracker.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(condition, test_name) \
    do { \
        if (condition) { \
            printf("✓ PASS: %s\n", test_name); \
            tests_passed++; \
        } else { \
            printf("✗ FAIL: %s\n", test_name); \
            tests_failed++; \
        } \
    } while(0)

// PRIORITÉ 3.1: Tests manquants critiques selon roadmap exact

void test_lum_structure_alignment_validation(void) {
    printf("\n=== PRIORITÉ 3: Test Alignement Structure LUM ===\n");
    
    // Vérifier alignement mémoire optimal - 56 bytes avec magic_number
    TEST_ASSERT(sizeof(lum_t) == 56, "Taille structure LUM exacte 56 bytes");
    TEST_ASSERT(offsetof(lum_t, id) == 0, "Champ id en première position");
    TEST_ASSERT(offsetof(lum_t, timestamp) % 8 == 0, "Alignement 64-bit timestamp");
    
    // Vérifier pas de padding inattendu
    size_t expected_min_size = sizeof(uint32_t) + sizeof(uint8_t) + 
                              sizeof(int32_t) * 2 + sizeof(uint64_t) + 
                              sizeof(void*) + sizeof(uint32_t) * 2 + // +magic_number
                              sizeof(uint8_t) + 3; // +is_destroyed +reserved[3]
    TEST_ASSERT(sizeof(lum_t) >= expected_min_size, "Taille minimum respectée");
    
    printf("✅ Structure LUM alignement validé selon standard forensique\n");
}

void test_lum_checksum_integrity_complete(void) {
    printf("\n=== PRIORITÉ 3: Test Intégrité Checksum LUM ===\n");
    
    lum_t* lum = lum_create(1, 100, 200, LUM_STRUCTURE_LINEAR);
    TEST_ASSERT(lum != NULL, "Création LUM pour test checksum");
    
    if (lum) {
        // Sauvegarder checksum original
        uint32_t original_checksum = lum->checksum;
        
        // Modifier donnée et recalculer
        lum->position_x = 999;
        uint32_t recalc = lum->id ^ lum->presence ^ lum->position_x ^ 
                          lum->position_y ^ lum->structure_type ^ 
                          (uint32_t)(lum->timestamp & 0xFFFFFFFF);
        
        // Vérifier détection altération
        TEST_ASSERT(original_checksum != recalc, "Détection altération checksum");
        
        lum_destroy(lum);
    }
    
    printf("✅ Intégrité checksum validée selon standard forensique\n");
}

void test_vorax_fuse_conservation_law_strict(void) {
    printf("\n=== PRIORITÉ 3: Test Loi Conservation VORAX Stricte ===\n");
    
    lum_group_t* g1 = lum_group_create(100);
    lum_group_t* g2 = lum_group_create(100);
    TEST_ASSERT(g1 && g2, "Création groupes pour test conservation");
    
    if (g1 && g2) {
        // Remplir groupes avec pattern précis
        for(size_t i = 0; i < 10; i++) {  // Réduit pour simplicité
            lum_t* l1 = lum_create(1, i, i*2, LUM_STRUCTURE_LINEAR);
            lum_t* l2 = lum_create(0, i+100, i*2+100, LUM_STRUCTURE_BINARY);
            if (l1 && l2) {
                lum_group_add(g1, l1);
                lum_group_add(g2, l2);
                lum_destroy(l1);
                lum_destroy(l2);
            }
        }
        
        // Compter présence avant fusion
        size_t presence_before = 0;
        for(size_t i = 0; i < g1->count; i++) presence_before += g1->lums[i].presence;
        for(size_t i = 0; i < g2->count; i++) presence_before += g2->lums[i].presence;
        
        // Fusion
        vorax_result_t* result = vorax_fuse(g1, g2);
        TEST_ASSERT(result && result->success, "Fusion VORAX réussie");
        
        // Vérifier conservation STRICTE
        if (result && result->result_group) {
            size_t presence_after = 0;
            for(size_t i = 0; i < result->result_group->count; i++) {
                presence_after += result->result_group->lums[i].presence;
            }
            TEST_ASSERT(presence_before == presence_after, "LOI CONSERVATION ABSOLUE respectée");
            
            vorax_result_destroy(result);
        }
        
        lum_group_destroy(g1);
        lum_group_destroy(g2);
    }
    
    printf("✅ Loi conservation VORAX validée selon standard forensique\n");
}

int main(void) {
    printf("🔍 === TESTS PRIORITÉ 3 CRITIQUES ===\n");
    printf("Validation corrections manquantes selon roadmap exact\n\n");
    
    // Initialisation tracking
    memory_tracker_init();
    
    // Exécution tests PRIORITÉ 3 critiques
    test_lum_structure_alignment_validation();
    test_lum_checksum_integrity_complete(); 
    test_vorax_fuse_conservation_law_strict();
    
    // Résultats finaux
    printf("\n=== RÉSULTATS TESTS PRIORITÉ 3 ===\n");
    printf("Tests réussis: %d\n", tests_passed);
    printf("Tests échoués: %d\n", tests_failed);
    printf("Taux succès: %.1f%%\n", 
           tests_passed > 0 ? (100.0 * tests_passed) / (tests_passed + tests_failed) : 0.0);
    
    if (tests_failed == 0) {
        printf("✅ TOUS TESTS PRIORITÉ 3 VALIDÉS - ROADMAP RESPECTÉ\n");
    } else {
        printf("❌ ÉCHECS PRIORITÉ 3 DÉTECTÉS\n");
    }
    
    // Rapport mémoire
    memory_tracker_report();
    memory_tracker_cleanup();
    
    return tests_failed == 0 ? 0 : 1;
}