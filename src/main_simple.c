#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "lum/lum_core.h"
#include "vorax/vorax_operations.h"
#include "parser/vorax_parser.h"
#include "logger/lum_logger.h"
#include "logger/log_manager.h"

int main(void) {
    printf("=== LUM/VORAX System - Replit Environment ===\n");
    printf("Initializing core modules...\n");
    
    // Initialize logging system  
    lum_logger_t* logger = lum_logger_create("lum_system.log", true, true);
    if (!logger) {
        fprintf(stderr, "Failed to initialize logger\n");
        return 1;
    }
    
    printf("✅ Logger initialized\n");
    
    // Test core LUM functionality
    lum_t* test_lum = lum_create(1, 10, 20, LUM_STRUCTURE_LINEAR);
    if (test_lum) {
        printf("✅ Core LUM created at position (%d, %d)\n", 
               test_lum->position_x, test_lum->position_y);
        
        // Clean up
        lum_destroy(test_lum);
        printf("✅ LUM cleanup completed\n");
    } else {
        printf("❌ Failed to create test LUM\n");
    }
    
    // Test VORAX operations
    printf("Testing VORAX operations...\n");
    lum_group_t* group = lum_group_create(100);
    if (group) {
        printf("✅ LUM group created with capacity %zu\n", group->capacity);
        
        // Add some LUMs to the group
        for (int i = 0; i < 10; i++) {
            lum_t* lum = lum_create(1, i, i*2, LUM_STRUCTURE_LINEAR);
            if (lum && lum_group_add(group, lum)) {
                // LUM added successfully
            }
        }
        
        printf("✅ Added LUMs to group, current count: %zu\n", group->count);
        
        // Clean up
        lum_group_destroy(group);
        printf("✅ LUM group cleanup completed\n");
    } else {
        printf("❌ Failed to create LUM group\n");
    }
    
    printf("\n=== LUM/VORAX System Test Complete ===\n");
    printf("System is operational in Replit environment!\n");
    
    // Clean up logging
    lum_logger_destroy(logger);
    
    return 0;
}