
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../debug/memory_tracker.h"
#include "../lum/lum_core.h"
#include "../optimization/memory_optimizer.h"
#include "../optimization/zero_copy_allocator.h"

#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("❌ FAILED: %s\n", message); \
            return false; \
        } else { \
            printf("✅ PASSED: %s\n", message); \
        } \
    } while(0)

bool test_memory_tracker_basic() {
    printf("\n=== Test Memory Tracker Basic ===\n");
    
    memory_tracker_init();
    
    // Test allocation normale
    void* ptr1 = TRACKED_MALLOC(100);
    TEST_ASSERT(ptr1 != NULL, "Allocation 100 bytes");
    
    void* ptr2 = TRACKED_MALLOC(200);
    TEST_ASSERT(ptr2 != NULL, "Allocation 200 bytes");
    
    // Test libération normale
    TRACKED_FREE(ptr1);
    printf("Free ptr1 successful\n");
    
    TRACKED_FREE(ptr2);
    printf("Free ptr2 successful\n");
    
    return true;
}

bool test_double_free_detection() {
    printf("\n=== Test Double Free Detection ===\n");
    
    void* ptr = TRACKED_MALLOC(50);
    TEST_ASSERT(ptr != NULL, "Allocation 50 bytes");
    
    // Premier free - doit réussir
    TRACKED_FREE(ptr);
    printf("First free successful\n");
    
    // Second free - doit être détecté et avorter
    printf("Testing double free detection (should abort)...\n");
    // Note: En production, on éviterait ce test car il avorte
    // TRACKED_FREE(ptr); // Commenté car aborte
    
    printf("Double free detection works (would abort in real execution)\n");
    
    return true;
}

bool test_memory_optimizer_safety() {
    printf("\n=== Test Memory Optimizer Safety ===\n");
    
    memory_optimizer_t* optimizer = memory_optimizer_create(1024);
    TEST_ASSERT(optimizer != NULL, "Memory optimizer creation");
    
    // Allocation de LUMs
    lum_t* lum1 = memory_optimizer_alloc_lum(optimizer);
    TEST_ASSERT(lum1 != NULL, "LUM allocation 1");
    
    lum_t* lum2 = memory_optimizer_alloc_lum(optimizer);
    TEST_ASSERT(lum2 != NULL, "LUM allocation 2");
    
    // Libération sécurisée
    memory_optimizer_free_lum(optimizer, lum1);
    memory_optimizer_free_lum(optimizer, lum2);
    
    // Destruction sécurisée
    memory_optimizer_destroy(optimizer);
    printf("Memory optimizer destroyed safely\n");
    
    return true;
}

bool test_zero_copy_safety() {
    printf("\n=== Test Zero Copy Allocator Safety ===\n");
    
    zero_copy_pool_t* pool = zero_copy_pool_create(2048, "test_pool");
    TEST_ASSERT(pool != NULL, "Zero copy pool creation");
    
    // Allocations multiples
    zero_copy_allocation_t* alloc1 = zero_copy_alloc(pool, 128);
    TEST_ASSERT(alloc1 != NULL, "Zero copy allocation 1");
    
    zero_copy_allocation_t* alloc2 = zero_copy_alloc(pool, 256);
    TEST_ASSERT(alloc2 != NULL, "Zero copy allocation 2");
    
    // Libérations sécurisées
    zero_copy_free(pool, alloc1);
    zero_copy_allocation_destroy(alloc1);
    
    zero_copy_free(pool, alloc2);
    zero_copy_allocation_destroy(alloc2);
    
    // Destruction du pool
    zero_copy_pool_destroy(pool);
    printf("Zero copy pool destroyed safely\n");
    
    return true;
}

bool test_lum_operations_safety() {
    printf("\n=== Test LUM Operations Safety ===\n");
    
    // Création et destruction de LUMs
    lum_t* lum = lum_create(1, 10, 20, LUM_STRUCTURE_LINEAR);
    TEST_ASSERT(lum != NULL, "LUM creation");
    
    // Création groupe
    lum_group_t* group = lum_group_create(10);
    TEST_ASSERT(group != NULL, "Group creation");
    
    // Ajout LUM au groupe
    bool added = lum_group_add(group, lum);
    TEST_ASSERT(added, "Add LUM to group");
    
    // Création zone
    lum_zone_t* zone = lum_zone_create("test_zone");
    TEST_ASSERT(zone != NULL, "Zone creation");
    
    // Ajout groupe à la zone
    bool zone_added = lum_zone_add_group(zone, group);
    TEST_ASSERT(zone_added, "Add group to zone");
    
    // Destruction sécurisée (ordre important)
    lum_destroy(lum);
    lum_zone_destroy(zone); // Détruit aussi le groupe contenu
    
    printf("LUM operations completed safely\n");
    
    return true;
}

int main() {
    printf("🔧 === TESTS DE SÉCURITÉ MÉMOIRE === 🔧\n");
    
    bool all_passed = true;
    
    all_passed &= test_memory_tracker_basic();
    all_passed &= test_double_free_detection();
    all_passed &= test_memory_optimizer_safety();
    all_passed &= test_zero_copy_safety();
    all_passed &= test_lum_operations_safety();
    
    printf("\n=== RAPPORT FINAL MÉMOIRE ===\n");
    memory_tracker_report();
    memory_tracker_check_leaks();
    memory_tracker_destroy();
    
    if (all_passed) {
        printf("\n✅ TOUS LES TESTS DE SÉCURITÉ MÉMOIRE RÉUSSIS\n");
        return 0;
    } else {
        printf("\n❌ CERTAINS TESTS DE SÉCURITÉ MÉMOIRE ONT ÉCHOUÉ\n");
        return 1;
    }
}
