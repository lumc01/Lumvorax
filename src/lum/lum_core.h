#ifndef LUM_CORE_H
#define LUM_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <assert.h>
#include <pthread.h>

// Vérification de l'ABI - la structure doit faire exactement 32 bytes avec padding
_Static_assert(sizeof(struct { uint8_t a; uint32_t b; int32_t c; int32_t d; uint8_t e; uint64_t f; }) == 32,
               "Basic lum_t structure should be 32 bytes on this platform");

// Note: avec padding d'alignement sur 8 bytes, la structure complète fait 32 bytes

// Core LUM structure - a single presence unit
typedef struct {
    uint8_t presence;           // 0 or 1 - the fundamental presence
    uint32_t id;               // Unique identifier for traceability
    int32_t position_x;        // Spatial position X
    int32_t position_y;        // Spatial position Y
    uint8_t structure_type;    // LINEAR, CIRCULAR, GROUP, NODE
    uint64_t timestamp;        // Creation/modification timestamp
} lum_t;

// LUM structure types
typedef enum {
    LUM_STRUCTURE_LINEAR = 0,
    LUM_STRUCTURE_CIRCULAR = 1,
    LUM_STRUCTURE_BINARY = 2,
    LUM_STRUCTURE_GROUP = 3,
    LUM_STRUCTURE_COMPRESSED = 4,
    LUM_STRUCTURE_NODE = 5,
    LUM_STRUCTURE_MAX = 6
} lum_structure_type_e;

// LUM Group - collection of LUMs
typedef struct {
    lum_t* lums;              // Array of LUMs
    size_t count;             // Number of LUMs
    size_t capacity;          // Allocated capacity
    uint32_t group_id;        // Group identifier
    lum_structure_type_e type; // Group structure type
} lum_group_t;

// Zone - spatial container for LUMs
typedef struct {
    char name[32];            // Zone name (A, B, C, etc.)
    lum_group_t** groups;     // Array of pointers to LUM groups
    size_t group_count;       // Number of groups
    size_t group_capacity;    // Allocated capacity for groups
    bool is_empty;            // Quick empty check
} lum_zone_t;

// Memory storage for LUMs
typedef struct {
    char name[32];            // Memory variable name (#alpha, #beta, etc.)
    lum_group_t stored_group; // Stored group
    bool is_occupied;         // Whether memory contains data
} lum_memory_t;

// Core functions
lum_t* lum_create(uint8_t presence, int32_t x, int32_t y, lum_structure_type_e type);
void lum_destroy(lum_t* lum);

lum_group_t* lum_group_create(size_t initial_capacity);
void lum_group_destroy(lum_group_t* group);
void lum_group_safe_destroy(lum_group_t** group);
void lum_group_safe_destroy(lum_group_t** group_ptr);
bool lum_group_add(lum_group_t* group, lum_t* lum);
lum_t* lum_group_get(lum_group_t* group, size_t index);
size_t lum_group_size(lum_group_t* group);

lum_zone_t* lum_zone_create(const char* name);
void lum_zone_destroy(lum_zone_t* zone);
bool lum_zone_add_group(lum_zone_t* zone, lum_group_t* group);
bool lum_zone_is_empty(lum_zone_t* zone);

lum_memory_t* lum_memory_create(const char* name);
void lum_memory_destroy(lum_memory_t* memory);
bool lum_memory_store(lum_memory_t* memory, lum_group_t* group);
lum_group_t* lum_memory_retrieve(lum_memory_t* memory);

// Utility functions
uint32_t lum_generate_id(void);
uint64_t lum_get_timestamp(void);
void lum_print(const lum_t* lum);
void lum_group_print(const lum_group_t* group);

#endif // LUM_CORE_H