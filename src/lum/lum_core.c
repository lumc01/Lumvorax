#include "lum_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

static uint32_t lum_id_counter = 1;
static pthread_mutex_t id_counter_mutex = PTHREAD_MUTEX_INITIALIZER;

// Core LUM functions
lum_t* lum_create(uint8_t presence, int32_t x, int32_t y, lum_structure_type_e type) {
    lum_t* lum = malloc(sizeof(lum_t));
    if (!lum) return NULL;
    
    lum->presence = (presence > 0) ? 1 : 0;
    lum->id = lum_generate_id();
    lum->position_x = x;
    lum->position_y = y;
    lum->structure_type = type;
    lum->timestamp = lum_get_timestamp();
    
    return lum;
}

void lum_destroy(lum_t* lum) {
    if (lum) {
        free(lum);
    }
}

// LUM Group functions
lum_group_t* lum_group_create(size_t initial_capacity) {
    lum_group_t* group = malloc(sizeof(lum_group_t));
    if (!group) return NULL;
    
    group->lums = malloc(sizeof(lum_t) * initial_capacity);
    if (!group->lums) {
        free(group);
        return NULL;
    }
    
    group->count = 0;
    group->capacity = initial_capacity;
    group->group_id = lum_generate_id();
    group->type = LUM_STRUCTURE_GROUP;
    
    return group;
}

void lum_group_destroy(lum_group_t* group) {
    if (group) {
        if (group->lums) {
            free(group->lums);
        }
        free(group);
    }
}

bool lum_group_add(lum_group_t* group, lum_t* lum) {
    if (!group || !lum) return false;
    
    if (group->count >= group->capacity) {
        // Resize array
        size_t new_capacity = group->capacity * 2;
        lum_t* new_lums = realloc(group->lums, sizeof(lum_t) * new_capacity);
        if (!new_lums) return false;
        
        group->lums = new_lums;
        group->capacity = new_capacity;
    }
    
    group->lums[group->count] = *lum;
    group->count++;
    
    return true;
}

lum_t* lum_group_get(lum_group_t* group, size_t index) {
    if (!group || index >= group->count) return NULL;
    return &group->lums[index];
}

size_t lum_group_size(lum_group_t* group) {
    return group ? group->count : 0;
}

// Zone functions
lum_zone_t* lum_zone_create(const char* name) {
    lum_zone_t* zone = malloc(sizeof(lum_zone_t));
    if (!zone) return NULL;
    
    strncpy(zone->name, name, sizeof(zone->name) - 1);
    zone->name[sizeof(zone->name) - 1] = '\0';
    
    zone->groups = malloc(sizeof(lum_group_t*) * 10);
    if (!zone->groups) {
        free(zone);
        return NULL;
    }
    
    zone->group_count = 0;
    zone->group_capacity = 10;
    zone->is_empty = true;
    
    return zone;
}

void lum_zone_destroy(lum_zone_t* zone) {
    if (zone) {
        if (zone->groups) {
            for (size_t i = 0; i < zone->group_count; i++) {
                lum_group_destroy(zone->groups[i]);
            }
            free(zone->groups);
        }
        free(zone);
    }
}

bool lum_zone_add_group(lum_zone_t* zone, lum_group_t* group) {
    if (!zone || !group) return false;
    
    if (zone->group_count >= zone->group_capacity) {
        size_t new_capacity = zone->group_capacity * 2;
        lum_group_t** new_groups = realloc(zone->groups, sizeof(lum_group_t*) * new_capacity);
        if (!new_groups) return false;
        
        zone->groups = new_groups;
        zone->group_capacity = new_capacity;
    }
    
    zone->groups[zone->group_count] = group;
    zone->group_count++;
    zone->is_empty = false;
    
    return true;
}

bool lum_zone_is_empty(lum_zone_t* zone) {
    if (!zone) return true;
    
    if (zone->group_count == 0) {
        zone->is_empty = true;
        return true;
    }
    
    // Check if all groups are empty
    for (size_t i = 0; i < zone->group_count; i++) {
        if (zone->groups[i]->count > 0) {
            zone->is_empty = false;
            return false;
        }
    }
    
    zone->is_empty = true;
    return true;
}

// Memory functions
lum_memory_t* lum_memory_create(const char* name) {
    lum_memory_t* memory = malloc(sizeof(lum_memory_t));
    if (!memory) return NULL;
    
    strncpy(memory->name, name, sizeof(memory->name) - 1);
    memory->name[sizeof(memory->name) - 1] = '\0';
    
    memory->stored_group.lums = NULL;
    memory->stored_group.count = 0;
    memory->stored_group.capacity = 0;
    memory->stored_group.group_id = 0;
    memory->stored_group.type = LUM_STRUCTURE_GROUP;
    memory->is_occupied = false;
    
    return memory;
}

void lum_memory_destroy(lum_memory_t* memory) {
    if (memory) {
        if (memory->stored_group.lums) {
            free(memory->stored_group.lums);
        }
        free(memory);
    }
}

bool lum_memory_store(lum_memory_t* memory, lum_group_t* group) {
    if (!memory || !group) return false;
    
    // Free existing data
    if (memory->stored_group.lums) {
        free(memory->stored_group.lums);
    }
    
    // Deep copy the group
    memory->stored_group.lums = malloc(sizeof(lum_t) * group->capacity);
    if (!memory->stored_group.lums) return false;
    
    memcpy(memory->stored_group.lums, group->lums, sizeof(lum_t) * group->count);
    memory->stored_group.count = group->count;
    memory->stored_group.capacity = group->capacity;
    memory->stored_group.group_id = group->group_id;
    memory->stored_group.type = group->type;
    memory->is_occupied = true;
    
    return true;
}

lum_group_t* lum_memory_retrieve(lum_memory_t* memory) {
    if (!memory || !memory->is_occupied) return NULL;
    
    lum_group_t* group = lum_group_create(memory->stored_group.capacity);
    if (!group) return NULL;
    
    memcpy(group->lums, memory->stored_group.lums, sizeof(lum_t) * memory->stored_group.count);
    group->count = memory->stored_group.count;
    group->group_id = memory->stored_group.group_id;
    group->type = memory->stored_group.type;
    
    return group;
}

// Utility functions
uint32_t lum_generate_id(void) {
    pthread_mutex_lock(&id_counter_mutex);
    uint32_t id = lum_id_counter++;
    pthread_mutex_unlock(&id_counter_mutex);
    return id;
}

uint64_t lum_get_timestamp(void) {
    return (uint64_t)time(NULL);
}

void lum_print(const lum_t* lum) {
    if (lum) {
        printf("LUM[%u]: presence=%u, pos=(%d,%d), type=%u, ts=%lu\n",
               lum->id, lum->presence, lum->position_x, lum->position_y,
               lum->structure_type, lum->timestamp);
    }
}

void lum_group_print(const lum_group_t* group) {
    if (group) {
        printf("Group[%u]: %zu LUMs\n", group->group_id, group->count);
        for (size_t i = 0; i < group->count; i++) {
            printf("  ");
            lum_print(&group->lums[i]);
        }
    }
}