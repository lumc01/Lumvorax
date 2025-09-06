#ifndef MEMORY_OPTIMIZER_H
#define MEMORY_OPTIMIZER_H

#include "../lum/lum_core.h"
#include <stdint.h>
#include <stdbool.h>

// Memory pool structure for efficient allocation
typedef struct {
    void* pool_start;
    void* current_ptr;
    size_t size;
    size_t alignment;
    bool is_initialized;
} memory_pool_t;

// Memory statistics structure
typedef struct {
    size_t allocated_bytes;
    size_t free_bytes;
    size_t allocation_count;
} memory_stats_t;

// Memory optimizer context
typedef struct {
    memory_pool_t lum_pool;
    memory_pool_t group_pool;
    memory_pool_t zone_pool;
    memory_stats_t stats;
    bool auto_defrag_enabled;
    size_t defrag_threshold;
} memory_optimizer_t;

// Function declarations
memory_pool_t* memory_pool_create(size_t size, size_t alignment);
void memory_pool_destroy(memory_pool_t* pool);
bool memory_pool_init(memory_pool_t* pool, size_t size, size_t alignment);
void* memory_pool_alloc(memory_pool_t* pool, size_t size);
bool memory_pool_free(memory_pool_t* pool, void* ptr, size_t size);
void memory_pool_reset(memory_pool_t* pool);
size_t memory_pool_get_used_size(memory_pool_t* pool);
size_t memory_pool_get_free_size(memory_pool_t* pool);
void memory_pool_get_stats(memory_pool_t* pool, memory_stats_t* stats);

// Optimized LUM allocations
lum_t* memory_optimizer_alloc_lum(memory_optimizer_t* optimizer);
lum_group_t* memory_optimizer_alloc_group(memory_optimizer_t* optimizer, size_t capacity);
lum_zone_t* memory_optimizer_alloc_zone(memory_optimizer_t* optimizer, const char* name);

void memory_optimizer_free_lum(memory_optimizer_t* optimizer, lum_t* lum);
void memory_optimizer_free_group(memory_optimizer_t* optimizer, lum_group_t* group);
void memory_optimizer_free_zone(memory_optimizer_t* optimizer, lum_zone_t* zone);

// Statistics and analysis
memory_stats_t* memory_optimizer_get_stats(memory_optimizer_t* optimizer);
void memory_optimizer_print_stats(memory_optimizer_t* optimizer);
bool memory_optimizer_analyze_fragmentation(memory_optimizer_t* optimizer);

// Auto-optimization
bool memory_optimizer_auto_defrag(memory_optimizer_t* optimizer);
void memory_optimizer_set_auto_defrag(memory_optimizer_t* optimizer, bool enabled, size_t threshold);

#endif // MEMORY_OPTIMIZER_H