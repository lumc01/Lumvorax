#include <stdio.h>
#include <stdlib.h>
#include "lum/lum_core.h"
#include "vorax/vorax_operations.h"
#include "debug/memory_tracker.h"
#include "debug/forensic_logger.h"

int main(void) {
    printf("=== LUM/VORAX Core System ===\n");
    
    // Initialize tracking systems
    memory_tracker_init();
    forensic_logger_init("logs/execution/forensic_simple.log");
    
    // Test basic LUM operations
    printf("Testing basic LUM operations...\n");
    lum_group_t* group = lum_group_create(10);
    if (group) {
        printf("[OK] Group created with capacity 10\n");
        
        // Add some LUMs
        for (int i = 0; i < 5; i++) {
            lum_t* lum = lum_create(i % 2, i * 10, i * 20, LUM_STRUCTURE_LINEAR);
            if (lum) {
                lum_group_add(group, lum);
                lum_destroy(lum);
            }
        }
        printf("[OK] 5 LUMs added to group. Size: %zu\n", lum_group_size(group));
        
        lum_group_destroy(group);
        printf("[OK] Group destroyed successfully\n");
    }
    
    printf("=== LUM/VORAX Core Test Complete ===\n");
    
    // Cleanup
    memory_tracker_report();
    forensic_logger_destroy();
    memory_tracker_destroy();
    
    return 0;
}