
#include "memory_optimizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

// Create memory optimizer
memory_optimizer_t* memory_optimizer_create(size_t initial_pool_size) {
    memory_optimizer_t* optimizer = malloc(sizeof(memory_optimizer_t));
    if (!optimizer) return NULL;
    
    // Initialize pools
    if (!memory_pool_init(&optimizer->lum_pool, initial_pool_size / 4, sizeof(lum_t))) {
        free(optimizer);
        return NULL;
    }
    
    if (!memory_pool_init(&optimizer->group_pool, initial_pool_size / 4, sizeof(lum_group_t))) {
        free(optimizer);
        return NULL;
    }
    
    if (!memory_pool_init(&optimizer->zone_pool, initial_pool_size / 2, sizeof(lum_zone_t))) {
        free(optimizer);
        return NULL;
    }
    
    // Initialize statistics
    memset(&optimizer->stats, 0, sizeof(memory_stats_t));
    optimizer->auto_defrag_enabled = false;
    optimizer->defrag_threshold = initial_pool_size / 10;
    
    return optimizer;
}

void memory_optimizer_destroy(memory_optimizer_t* optimizer) {
    if (!optimizer) return;
    
    if (optimizer->lum_pool.pool_start) free(optimizer->lum_pool.pool_start);
    if (optimizer->group_pool.pool_start) free(optimizer->group_pool.pool_start);
    if (optimizer->zone_pool.pool_start) free(optimizer->zone_pool.pool_start);
    
    free(optimizer);
}

// Pool management implementation
bool memory_pool_init(memory_pool_t* pool, size_t size, size_t alignment) {
    if (!pool || size == 0) return false;
    
    pool->pool_start = aligned_alloc(alignment, size);
    if (!pool->pool_start) return false;
    
    pool->current_ptr = pool->pool_start;
    pool->pool_size = size;
    pool->used_size = 0;
    pool->alignment = alignment;
    pool->is_initialized = true;
    
    return true;
}

void* memory_pool_alloc(memory_pool_t* pool, size_t size) {
    if (!pool || !pool->is_initialized || size == 0) return NULL;
    
    // Align size to boundary
    size_t aligned_size = (size + pool->alignment - 1) & ~(pool->alignment - 1);
    
    if (pool->used_size + aligned_size > pool->pool_size) {
        return NULL; // Pool exhausted
    }
    
    void* result = pool->current_ptr;
    pool->current_ptr = (char*)pool->current_ptr + aligned_size;
    pool->used_size += aligned_size;
    
    return result;
}

bool memory_pool_free(memory_pool_t* pool, void* ptr, size_t size) {
    if (!pool || !ptr) return false;
    
    // Simple implementation - mark as available for defragmentation
    // In a real implementation, this would maintain a free list
    return true;
}

void memory_pool_reset(memory_pool_t* pool) {
    if (!pool || !pool->is_initialized) return;
    
    pool->current_ptr = pool->pool_start;
    pool->used_size = 0;
}

void memory_pool_defragment(memory_pool_t* pool) {
    if (!pool || !pool->is_initialized) return;
    
    // Defragmentation implementation would compact allocated blocks
    // For this implementation, we reset the pool
    memory_pool_reset(pool);
}

// Optimized allocations
lum_t* memory_optimizer_alloc_lum(memory_optimizer_t* optimizer) {
    if (!optimizer) return NULL;
    
    lum_t* lum = (lum_t*)memory_pool_alloc(&optimizer->lum_pool, sizeof(lum_t));
    if (lum) {
        optimizer->stats.total_allocated += sizeof(lum_t);
        optimizer->stats.current_usage += sizeof(lum_t);
        optimizer->stats.allocation_count++;
        
        if (optimizer->stats.current_usage > optimizer->stats.peak_usage) {
            optimizer->stats.peak_usage = optimizer->stats.current_usage;
        }
    }
    
    return lum;
}

lum_group_t* memory_optimizer_alloc_group(memory_optimizer_t* optimizer, size_t capacity) {
    if (!optimizer) return NULL;
    
    lum_group_t* group = (lum_group_t*)memory_pool_alloc(&optimizer->group_pool, sizeof(lum_group_t));
    if (group) {
        size_t lum_array_size = capacity * sizeof(lum_t);
        group->lums = (lum_t*)memory_pool_alloc(&optimizer->lum_pool, lum_array_size);
        
        if (group->lums) {
            group->capacity = capacity;
            group->count = 0;
            
            optimizer->stats.total_allocated += sizeof(lum_group_t) + lum_array_size;
            optimizer->stats.current_usage += sizeof(lum_group_t) + lum_array_size;
            optimizer->stats.allocation_count++;
            
            if (optimizer->stats.current_usage > optimizer->stats.peak_usage) {
                optimizer->stats.peak_usage = optimizer->stats.current_usage;
            }
        }
    }
    
    return group;
}

lum_zone_t* memory_optimizer_alloc_zone(memory_optimizer_t* optimizer, const char* name) {
    if (!optimizer) return NULL;
    
    lum_zone_t* zone = (lum_zone_t*)memory_pool_alloc(&optimizer->zone_pool, sizeof(lum_zone_t));
    if (zone && name) {
        strncpy(zone->name, name, sizeof(zone->name) - 1);
        zone->name[sizeof(zone->name) - 1] = '\0';
        
        optimizer->stats.total_allocated += sizeof(lum_zone_t);
        optimizer->stats.current_usage += sizeof(lum_zone_t);
        optimizer->stats.allocation_count++;
        
        if (optimizer->stats.current_usage > optimizer->stats.peak_usage) {
            optimizer->stats.peak_usage = optimizer->stats.current_usage;
        }
    }
    
    return zone;
}

void memory_optimizer_free_lum(memory_optimizer_t* optimizer, lum_t* lum) {
    if (!optimizer || !lum) return;
    
    memory_pool_free(&optimizer->lum_pool, lum, sizeof(lum_t));
    optimizer->stats.total_freed += sizeof(lum_t);
    optimizer->stats.current_usage -= sizeof(lum_t);
    optimizer->stats.free_count++;
}

void memory_optimizer_free_group(memory_optimizer_t* optimizer, lum_group_t* group) {
    if (!optimizer || !group) return;
    
    size_t group_size = sizeof(lum_group_t) + (group->capacity * sizeof(lum_t));
    
    memory_pool_free(&optimizer->group_pool, group, sizeof(lum_group_t));
    if (group->lums) {
        memory_pool_free(&optimizer->lum_pool, group->lums, group->capacity * sizeof(lum_t));
    }
    
    optimizer->stats.total_freed += group_size;
    optimizer->stats.current_usage -= group_size;
    optimizer->stats.free_count++;
}

void memory_optimizer_free_zone(memory_optimizer_t* optimizer, lum_zone_t* zone) {
    if (!optimizer || !zone) return;
    
    memory_pool_free(&optimizer->zone_pool, zone, sizeof(lum_zone_t));
    optimizer->stats.total_freed += sizeof(lum_zone_t);
    optimizer->stats.current_usage -= sizeof(lum_zone_t);
    optimizer->stats.free_count++;
}

// Statistics
memory_stats_t* memory_optimizer_get_stats(memory_optimizer_t* optimizer) {
    if (!optimizer) return NULL;
    
    // Update fragmentation statistics
    optimizer->stats.fragmentation_bytes = optimizer->stats.total_allocated - optimizer->stats.current_usage;
    if (optimizer->stats.total_allocated > 0) {
        optimizer->stats.fragmentation_ratio = 
            (double)optimizer->stats.fragmentation_bytes / optimizer->stats.total_allocated;
    }
    
    return &optimizer->stats;
}

void memory_optimizer_print_stats(memory_optimizer_t* optimizer) {
    if (!optimizer) return;
    
    memory_stats_t* stats = memory_optimizer_get_stats(optimizer);
    
    printf("=== Memory Optimizer Statistics ===\n");
    printf("Total Allocated: %zu bytes\n", stats->total_allocated);
    printf("Total Freed: %zu bytes\n", stats->total_freed);
    printf("Current Usage: %zu bytes\n", stats->current_usage);
    printf("Peak Usage: %zu bytes\n", stats->peak_usage);
    printf("Allocations: %zu\n", stats->allocation_count);
    printf("Frees: %zu\n", stats->free_count);
    printf("Fragmentation: %zu bytes (%.2f%%)\n", 
           stats->fragmentation_bytes, stats->fragmentation_ratio * 100.0);
    printf("=====================================\n");
}

bool memory_optimizer_analyze_fragmentation(memory_optimizer_t* optimizer) {
    if (!optimizer) return false;
    
    memory_optimizer_get_stats(optimizer);
    return optimizer->stats.fragmentation_bytes > optimizer->defrag_threshold;
}

bool memory_optimizer_auto_defrag(memory_optimizer_t* optimizer) {
    if (!optimizer || !optimizer->auto_defrag_enabled) return false;
    
    if (memory_optimizer_analyze_fragmentation(optimizer)) {
        memory_pool_defragment(&optimizer->lum_pool);
        memory_pool_defragment(&optimizer->group_pool);
        memory_pool_defragment(&optimizer->zone_pool);
        return true;
    }
    
    return false;
}

void memory_optimizer_set_auto_defrag(memory_optimizer_t* optimizer, bool enabled, size_t threshold) {
    if (!optimizer) return;
    
    optimizer->auto_defrag_enabled = enabled;
    optimizer->defrag_threshold = threshold;
}
