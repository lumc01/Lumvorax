
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "../lum/lum_core.h"
#include "../debug/memory_tracker.h"

// Tests unitaires complets LUM Core
static int tests_passed = 0;
static int tests_failed = 0;

#define UNIT_TEST_ASSERT(condition, test_name) \
    do { \
        if (condition) { \
            printf("✓ UNIT TEST PASS: %s\n", test_name); \
            tests_passed++; \
        } else { \
            printf("✗ UNIT TEST FAIL: %s\n", test_name); \
            tests_failed++; \
        } \
    } while(0)

void test_lum_creation_edge_cases(void) {
    printf("\n=== Tests Unitaires LUM Création - Cas Limites ===\n");
    
    // Test 1: Création LUM valide
    lum_t* lum1 = lum_create(1, 0, 0, LUM_STRUCTURE_LINEAR);
    UNIT_TEST_ASSERT(lum1 != NULL, "Création LUM valide basique");
    UNIT_TEST_ASSERT(lum1->presence == 1, "Présence correcte");
    UNIT_TEST_ASSERT(lum1->position_x == 0, "Position X correcte");
    UNIT_TEST_ASSERT(lum1->position_y == 0, "Position Y correcte");
    UNIT_TEST_ASSERT(lum1->structure_type == LUM_STRUCTURE_LINEAR, "Type structure correct");
    
    // Test 2: Création avec coordonnées négatives
    lum_t* lum2 = lum_create(0, -100, -200, LUM_STRUCTURE_CIRCULAR);
    UNIT_TEST_ASSERT(lum2 != NULL, "Création avec coordonnées négatives");
    UNIT_TEST_ASSERT(lum2->position_x == -100, "Position X négative");
    UNIT_TEST_ASSERT(lum2->position_y == -200, "Position Y négative");
    
    // Test 3: Création avec coordonnées maximales
    lum_t* lum3 = lum_create(1, INT32_MAX, INT32_MAX, LUM_STRUCTURE_GROUP);
    UNIT_TEST_ASSERT(lum3 != NULL, "Création coordonnées maximales");
    UNIT_TEST_ASSERT(lum3->position_x == INT32_MAX, "Position X maximale");
    UNIT_TEST_ASSERT(lum3->position_y == INT32_MAX, "Position Y maximale");
    
    // Test 4: Validation timestamp
    time_t before = time(NULL);
    lum_t* lum4 = lum_create(1, 10, 20, LUM_STRUCTURE_LINEAR);
    time_t after = time(NULL);
    UNIT_TEST_ASSERT(lum4 != NULL, "Création pour test timestamp");
    UNIT_TEST_ASSERT(lum4->timestamp >= before && lum4->timestamp <= after, "Timestamp dans intervalle valide");
    
    // Test 5: Types de structure valides
    for (int type = 0; type <= 3; type++) {
        lum_t* lum_type = lum_create(1, 0, 0, (lum_structure_type_e)type);
        UNIT_TEST_ASSERT(lum_type != NULL, "Création type structure valide");
        UNIT_TEST_ASSERT(lum_type->structure_type == type, "Type structure assigné correctement");
        lum_destroy(lum_type);
    }
    
    // Nettoyage
    lum_destroy(lum1);
    lum_destroy(lum2);
    lum_destroy(lum3);
    lum_destroy(lum4);
}

void test_lum_group_operations_exhaustive(void) {
    printf("\n=== Tests Unitaires LUM Group - Opérations Exhaustives ===\n");
    
    // Test 1: Création groupe vide
    lum_group_t* group1 = lum_group_create(0);
    UNIT_TEST_ASSERT(group1 != NULL, "Création groupe capacité 0");
    UNIT_TEST_ASSERT(lum_group_size(group1) == 0, "Taille groupe vide");
    UNIT_TEST_ASSERT(lum_group_is_empty(group1), "Groupe vide détecté");
    
    // Test 2: Création groupe capacité normale
    lum_group_t* group2 = lum_group_create(100);
    UNIT_TEST_ASSERT(group2 != NULL, "Création groupe capacité 100");
    UNIT_TEST_ASSERT(group2->capacity >= 100, "Capacité respectée");
    
    // Test 3: Ajout séquentiel de LUMs
    for (int i = 0; i < 50; i++) {
        lum_t* lum = lum_create(i % 2, i, i * 2, LUM_STRUCTURE_LINEAR);
        bool added = lum_group_add(group2, lum);
        UNIT_TEST_ASSERT(added, "Ajout LUM séquentiel réussi");
        UNIT_TEST_ASSERT(lum_group_size(group2) == (size_t)(i + 1), "Taille groupe mise à jour");
        lum_destroy(lum);
    }
    
    // Test 4: Recherche LUM par ID
    lum_t* target_lum = lum_create(1, 999, 888, LUM_STRUCTURE_CIRCULAR);
    lum_group_add(group2, target_lum);
    lum_t* found = lum_group_find_by_id(group2, target_lum->id);
    UNIT_TEST_ASSERT(found != NULL, "Recherche LUM par ID réussie");
    UNIT_TEST_ASSERT(found->id == target_lum->id, "ID LUM trouvé correct");
    lum_destroy(target_lum);
    
    // Test 5: Suppression LUM
    size_t size_before = lum_group_size(group2);
    bool removed = lum_group_remove_by_index(group2, 0);
    UNIT_TEST_ASSERT(removed, "Suppression LUM par index");
    UNIT_TEST_ASSERT(lum_group_size(group2) == size_before - 1, "Taille réduite après suppression");
    
    // Test 6: Itération sur groupe
    size_t count = 0;
    for (size_t i = 0; i < lum_group_size(group2); i++) {
        lum_t* lum = lum_group_get_at(group2, i);
        if (lum != NULL) {
            count++;
        }
    }
    UNIT_TEST_ASSERT(count == lum_group_size(group2), "Itération complète sur groupe");
    
    // Nettoyage
    lum_group_destroy(group1);
    lum_group_destroy(group2);
}

void test_lum_zone_management_complete(void) {
    printf("\n=== Tests Unitaires LUM Zone - Gestion Complète ===\n");
    
    // Test 1: Création zone
    lum_zone_t* zone = lum_zone_create("TestZone_Complete");
    UNIT_TEST_ASSERT(zone != NULL, "Création zone réussie");
    UNIT_TEST_ASSERT(strcmp(zone->name, "TestZone_Complete") == 0, "Nom zone correct");
    UNIT_TEST_ASSERT(lum_zone_is_empty(zone), "Zone initialement vide");
    
    // Test 2: Ajout multiple groupes
    for (int g = 0; g < 5; g++) {
        lum_group_t* group = lum_group_create(20);
        for (int i = 0; i < 10; i++) {
            lum_t* lum = lum_create(i % 2, g * 10 + i, g * 5 + i, LUM_STRUCTURE_LINEAR);
            lum_group_add(group, lum);
            lum_destroy(lum);
        }
        bool added = lum_zone_add_group(zone, group);
        UNIT_TEST_ASSERT(added, "Ajout groupe à zone réussi");
    }
    
    // Test 3: Statistiques zone
    size_t total_lums = lum_zone_get_total_lums(zone);
    UNIT_TEST_ASSERT(total_lums == 50, "Compte total LUMs dans zone");
    UNIT_TEST_ASSERT(!lum_zone_is_empty(zone), "Zone non vide après ajouts");
    
    // Test 4: Recherche dans zone
    lum_t* found_in_zone = lum_zone_find_lum_by_position(zone, 15, 10);
    UNIT_TEST_ASSERT(found_in_zone != NULL, "Recherche LUM par position dans zone");
    
    // Nettoyage
    lum_zone_destroy(zone);
}

void test_lum_memory_operations(void) {
    printf("\n=== Tests Unitaires LUM Memory - Opérations ===\n");
    
    // Test 1: Création mémoire
    lum_memory_t* memory = lum_memory_create("TestMemory", 1000);
    UNIT_TEST_ASSERT(memory != NULL, "Création mémoire réussie");
    UNIT_TEST_ASSERT(strcmp(memory->name, "TestMemory") == 0, "Nom mémoire correct");
    UNIT_TEST_ASSERT(memory->capacity == 1000, "Capacité mémoire correcte");
    
    // Test 2: Stockage LUMs en mémoire
    lum_group_t* group = lum_group_create(100);
    for (int i = 0; i < 50; i++) {
        lum_t* lum = lum_create(1, i, i, LUM_STRUCTURE_LINEAR);
        lum_group_add(group, lum);
        lum_destroy(lum);
    }
    
    bool stored = lum_memory_store_group(memory, "stored_group", group);
    UNIT_TEST_ASSERT(stored, "Stockage groupe en mémoire");
    
    // Test 3: Récupération depuis mémoire
    lum_group_t* retrieved = lum_memory_retrieve_group(memory, "stored_group");
    UNIT_TEST_ASSERT(retrieved != NULL, "Récupération groupe depuis mémoire");
    UNIT_TEST_ASSERT(lum_group_size(retrieved) == lum_group_size(group), "Taille groupe récupéré correcte");
    
    // Test 4: Liste contenu mémoire
    size_t stored_count = lum_memory_list_stored_groups(memory);
    UNIT_TEST_ASSERT(stored_count == 1, "Compte groupes stockés");
    
    // Nettoyage
    lum_group_destroy(group);
    lum_memory_destroy(memory);
}

int main(void) {
    printf("🧪 === TESTS UNITAIRES COMPLETS LUM CORE ===\n");
    printf("Validation exhaustive de tous les composants\n\n");
    
    // Initialisation tracking mémoire
    memory_tracker_init();
    
    // Exécution tous les tests unitaires
    test_lum_creation_edge_cases();
    test_lum_group_operations_exhaustive();
    test_lum_zone_management_complete();
    test_lum_memory_operations();
    
    // Résultats finaux
    printf("\n=== RÉSULTATS TESTS UNITAIRES ===\n");
    printf("Tests réussis: %d\n", tests_passed);
    printf("Tests échoués: %d\n", tests_failed);
    printf("Taux de succès: %.1f%%\n", 
           tests_passed > 0 ? (100.0 * tests_passed) / (tests_passed + tests_failed) : 0.0);
    
    // Rapport mémoire
    memory_tracker_report();
    memory_tracker_cleanup();
    
    return tests_failed == 0 ? 0 : 1;
}
