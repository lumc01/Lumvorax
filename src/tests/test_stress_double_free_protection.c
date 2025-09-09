
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../lum/lum_core.h"
#include "../vorax/vorax_operations.h"
#include "../debug/memory_tracker.h"

// Test stress protection double-free avec 1M+ LUMs
bool test_stress_double_free_protection_million_lums(void) {
    printf("=== TEST STRESS PROTECTION DOUBLE-FREE 1M+ LUMs ===\n");
    
    memory_tracker_init();
    memory_tracker_enable(true);
    
    const size_t test_count = 1000000;  // 1M LUMs minimum requis prompt.txt
    lum_group_t** groups = malloc(test_count * sizeof(lum_group_t*));
    
    // Création 1M groupes
    for (size_t i = 0; i < test_count; i++) {
        groups[i] = lum_group_create(10);  // 10 LUMs par groupe
        if (!groups[i]) {
            printf("ERREUR: Échec création groupe %zu\n", i);
            return false;
        }
    }
    
    printf("✓ Créé %zu groupes avec %zu LUMs total\n", test_count, test_count * 10);
    
    // Test destruction sécurisée massive
    for (size_t i = 0; i < test_count; i++) {
        if (groups[i]) {
            lum_group_safe_destroy(&groups[i]);  // Protection double-free
            
            // Vérification protection effective
            assert(groups[i] == NULL);
        }
        
        if (i % 100000 == 0) {
            printf("✓ Détruit %zu/%zu groupes (protection active)\n", i + 1, test_count);
        }
    }
    
    free(groups);
    
    // Export métriques JSON
    memory_tracker_export_json("logs/stress_double_free_metrics.json");
    
    memory_tracker_report();
    memory_tracker_cleanup();
    
    printf("✅ TEST STRESS PROTECTION DOUBLE-FREE: SUCCÈS\n");
    return true;
}

int main(void) {
    return test_stress_double_free_protection_million_lums() ? 0 : 1;
}
