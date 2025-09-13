#include "vorax_operations.h"
#include "../logger/lum_logger.h"
#include "../debug/memory_tracker.h"  // CORRECTION: Include pour TRACKED_MALLOC/FREE
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// lum_log est définie dans le module logger

// Fuse operation: ⧉ - Combines two groups into one
vorax_result_t* vorax_fuse(lum_group_t* group1, lum_group_t* group2) {
    vorax_result_t* result = vorax_result_create();
    if (!result || !group1 || !group2) {
        if (result) vorax_result_set_error(result, "Invalid input groups for fusion");
        return result;
    }

    size_t total_count = group1->count + group2->count;
    lum_group_t* fused = lum_group_create(total_count);
    if (!fused) {
        vorax_result_set_error(result, "Failed to create fused group");
        return result;
    }

    // Copy all LUMs from both groups
    for (size_t i = 0; i < group1->count; i++) {
        lum_group_add(fused, &group1->lums[i]);
    }
    for (size_t i = 0; i < group2->count; i++) {
        lum_group_add(fused, &group2->lums[i]);
    }

    result->result_group = fused;
    vorax_result_set_success(result, "Fusion completed successfully");
    return result;
}

// Split operation: ⇅ - Distributes group into multiple parts
vorax_result_t* vorax_split(lum_group_t* group, size_t parts) {
    vorax_result_t* result = vorax_result_create();
    if (!result || !group || parts == 0) {
        if (result) vorax_result_set_error(result, "Invalid input for split operation");
        return result;
    }

    if (group->count == 0) {
        vorax_result_set_error(result, "Cannot split empty group");
        return result;
    }

    // Create result groups
    result->result_groups = TRACKED_MALLOC(sizeof(lum_group_t*) * parts);
    if (!result->result_groups) {
        vorax_result_set_error(result, "Memory allocation failed for split result");
        return result;
    }

    result->result_count = parts;

    // Initialize all groups
    for (size_t i = 0; i < parts; i++) {
        result->result_groups[i] = lum_group_create(group->count / parts + 1);
        if (!result->result_groups[i]) {
            vorax_result_set_error(result, "Failed to create split group");
            // Clean up already allocated groups before returning
            for (size_t j = 0; j < i; j++) {
                if (result->result_groups[j]) {
                    lum_group_destroy(result->result_groups[j]);
                }
            }
            TRACKED_FREE(result->result_groups);
            result->result_groups = NULL;
            result->result_count = 0;
            return result;
        }
    }

    // Distribute LUMs evenly
    for (size_t i = 0; i < group->count; i++) {
        size_t target_group = i % parts;
        lum_group_add(result->result_groups[target_group], &group->lums[i]);
    }

    vorax_result_set_success(result, "Split completed successfully");
    return result;
}

// Cycle operation: ⟲ - Applies modulo transformation
vorax_result_t* vorax_cycle(lum_group_t* group, size_t modulo) {
    vorax_result_t* result = vorax_result_create();
    if (!result || !group || modulo == 0) {
        if (result) vorax_result_set_error(result, "Invalid input for cycle operation");
        return result;
    }

    size_t new_count = group->count % modulo;
    if (new_count == 0 && group->count > 0) {
        new_count = modulo;
    }

    // CORRECTION CRITIQUE: Utiliser capacité plus grande pour éviter confusion pointeurs
    size_t safe_capacity = (new_count > 0) ? new_count + 8 : 9; // +8 pour éviter collision
    lum_group_t* cycled = lum_group_create(safe_capacity);
    if (!cycled) {
        vorax_result_set_error(result, "Failed to create cycled group");
        return result;
    }

    // Copy first new_count LUMs
    for (size_t i = 0; i < new_count && i < group->count; i++) {
        lum_group_add(cycled, &group->lums[i]);
    }

    result->result_group = cycled;
    vorax_result_set_success(result, "Cycle completed successfully");
    return result;
}

// Move operation: → - Transfer LUMs between zones
vorax_result_t* vorax_move(lum_zone_t* from_zone, lum_zone_t* to_zone, size_t lum_count) {
    vorax_result_t* result = vorax_result_create();
    if (!result || !from_zone || !to_zone) {
        if (result) vorax_result_set_error(result, "Invalid zones for move operation");
        return result;
    }

    if (lum_zone_is_empty(from_zone)) {
        vorax_result_set_error(result, "Source zone is empty");
        return result;
    }

    // Find LUMs to move
    size_t moved = 0;
    lum_group_t* move_group = lum_group_create(lum_count);
    if (!move_group) {
        vorax_result_set_error(result, "Failed to create move group");
        return result;
    }

    // Move LUMs from first available group in source zone
    for (size_t g = 0; g < from_zone->group_count && moved < lum_count; g++) {
        lum_group_t* source_group = from_zone->groups[g];
        size_t moved_from_this_group = 0;  // Track per-group moved count
        
        for (size_t i = 0; i < source_group->count && moved < lum_count; i++) {
            lum_group_add(move_group, &source_group->lums[i]);
            moved++;
            moved_from_this_group++;
        }

        // Remove moved LUMs from source group using per-group count
        if (moved_from_this_group <= source_group->count) {
            source_group->count -= moved_from_this_group;
        } else {
            source_group->count = 0;
        }
    }

    // Add to destination zone
    lum_zone_add_group(to_zone, move_group);

    char msg[256];
    snprintf(msg, sizeof(msg), "Moved %zu LUMs from %s to %s", moved, from_zone->name, to_zone->name);
    vorax_result_set_success(result, msg);
    return result;
}

// Store operation: Store LUMs in memory
vorax_result_t* vorax_store(lum_memory_t* memory, lum_zone_t* zone, size_t lum_count) {
    vorax_result_t* result = vorax_result_create();
    if (!result || !memory || !zone) {
        if (result) vorax_result_set_error(result, "Invalid parameters for store operation");
        return result;
    }

    if (lum_zone_is_empty(zone)) {
        vorax_result_set_error(result, "Cannot store from empty zone");
        return result;
    }

    // Create group with LUMs to store
    lum_group_t* store_group = lum_group_create(lum_count);
    if (!store_group) {
        vorax_result_set_error(result, "Failed to create store group");
        return result;
    }

    size_t stored = 0;
    for (size_t g = 0; g < zone->group_count && stored < lum_count; g++) {
        lum_group_t* source_group = zone->groups[g];
        for (size_t i = 0; i < source_group->count && stored < lum_count; i++) {
            lum_group_add(store_group, &source_group->lums[i]);
            stored++;
        }
    }

    if (!lum_memory_store(memory, store_group)) {
        lum_group_destroy(store_group);
        vorax_result_set_error(result, "Failed to store in memory");
        return result;
    }

    lum_group_destroy(store_group);
    char msg[256];
    snprintf(msg, sizeof(msg), "Stored %zu LUMs in memory %s", stored, memory->name);
    vorax_result_set_success(result, msg);
    return result;
}

// Retrieve operation: Retrieve LUMs from memory
vorax_result_t* vorax_retrieve(lum_memory_t* memory, lum_zone_t* zone) {
    vorax_result_t* result = vorax_result_create();
    if (!result || !memory || !zone) {
        if (result) vorax_result_set_error(result, "Invalid parameters for retrieve operation");
        return result;
    }

    if (!memory->is_occupied) {
        vorax_result_set_error(result, "Memory is empty");
        return result;
    }

    lum_group_t* retrieved = lum_memory_retrieve(memory);
    if (!retrieved) {
        vorax_result_set_error(result, "Failed to retrieve from memory");
        return result;
    }

    lum_zone_add_group(zone, retrieved);

    char msg[256];
    snprintf(msg, sizeof(msg), "Retrieved %zu LUMs from memory %s to zone %s", 
             retrieved->count, memory->name, zone->name);
    vorax_result_set_success(result, msg);
    return result;
}

// Compress operation: Convert group to Ω (compressed form)
vorax_result_t* vorax_compress(lum_group_t* group) {
    vorax_result_t* result = vorax_result_create();
    if (!result || !group) {
        if (result) vorax_result_set_error(result, "Invalid group for compression");
        return result;
    }

    // Create a single LUM representing the compressed state
    lum_group_t* compressed = lum_group_create(1);
    if (!compressed) {
        vorax_result_set_error(result, "Failed to create compressed group");
        return result;
    }

    lum_t* omega_lum = lum_create(1, 0, 0, LUM_STRUCTURE_COMPRESSED);
    if (!omega_lum) {
        lum_group_destroy(compressed);
        vorax_result_set_error(result, "Failed to create Ω LUM");
        return result;
    }

    lum_group_add(compressed, omega_lum);
    lum_destroy(omega_lum); // CORRECTION: Utiliser lum_destroy au lieu de free

    result->result_group = compressed;
    char msg[256];
    snprintf(msg, sizeof(msg), "Compressed %zu LUMs to Ω", group->count);
    vorax_result_set_success(result, msg);
    return result;
}

// Expand operation: Expand Ω back to multiple LUMs
vorax_result_t* vorax_expand(lum_group_t* compressed_group, size_t parts) {
    vorax_result_t* result = vorax_result_create();
    if (!result || !compressed_group || parts == 0) {
        if (result) vorax_result_set_error(result, "Invalid parameters for expand operation");
        return result;
    }

    // Check if group contains compressed LUM
    if (compressed_group->count != 1 || 
        compressed_group->lums[0].structure_type != LUM_STRUCTURE_COMPRESSED) {
        vorax_result_set_error(result, "Group is not in compressed Ω form");
        return result;
    }

    lum_group_t* expanded = lum_group_create(parts);
    if (!expanded) {
        vorax_result_set_error(result, "Failed to create expanded group");
        return result;
    }

    // Create parts number of LUMs
    for (size_t i = 0; i < parts; i++) {
        lum_t* new_lum = lum_create(1, (int32_t)i, 0, LUM_STRUCTURE_LINEAR);
        if (new_lum) {
            lum_group_add(expanded, new_lum);
            lum_destroy(new_lum); // CORRECTION: Utiliser lum_destroy au lieu de free
        }
    }

    result->result_group = expanded;
    char msg[256];
    snprintf(msg, sizeof(msg), "Expanded Ω to %zu LUMs", parts);
    vorax_result_set_success(result, msg);
    return result;
}

// Utility functions
vorax_result_t* vorax_result_create(void) {
    vorax_result_t* result = TRACKED_MALLOC(sizeof(vorax_result_t));
    if (result) {
        result->success = false;
        result->message[0] = '\0';
        result->result_group = NULL;
        result->result_count = 0;
        result->result_groups = NULL;
        result->magic_number = 0;  // Initialisé à 0, sera mis à DEADBEEF lors destruction
    }
    return result;
}

void vorax_result_destroy(vorax_result_t* result) {
    if (!result) return;

    // Vérifier magic number pour détecter corruption
    static const uint32_t VORAX_RESULT_MAGIC = 0xDEADBEEF;
    if (result->magic_number == VORAX_RESULT_MAGIC) {
        lum_log(LUM_LOG_WARN, "Tentative double destruction vorax_result_t détectée et évitée");
        return;
    }

    // Marquer immédiatement comme en cours de destruction
    result->magic_number = VORAX_RESULT_MAGIC;

    // Destruction sécurisée du groupe principal
    if (result->result_group) {
        lum_group_destroy(result->result_group);
        result->result_group = NULL;
    }

    // Destruction sécurisée des groupes multiples
    if (result->result_groups && result->result_count > 0) {
        for (size_t i = 0; i < result->result_count; i++) {
            if (result->result_groups[i]) {
                lum_group_destroy(result->result_groups[i]);
                result->result_groups[i] = NULL;
            }
        }
        TRACKED_FREE(result->result_groups);
        result->result_groups = NULL;
        result->result_count = 0;
    }

    // Effacer la structure avant libération
    memset(result, 0xFF, sizeof(vorax_result_t));
    TRACKED_FREE(result);
}

void vorax_result_set_success(vorax_result_t* result, const char* message) {
    if (result) {
        result->success = true;
        strncpy(result->message, message, sizeof(result->message) - 1);
        result->message[sizeof(result->message) - 1] = '\0';
    }
}

void vorax_result_set_error(vorax_result_t* result, const char* message) {
    if (result) {
        result->success = false;
        strncpy(result->message, message, sizeof(result->message) - 1);
        result->message[sizeof(result->message) - 1] = '\0';
    }
}

// Conservation checking
bool vorax_check_conservation(lum_group_t** input_groups, size_t input_count,
                              lum_group_t** output_groups, size_t output_count) {
    size_t input_total = vorax_count_total_lums(input_groups, input_count);
    size_t output_total = vorax_count_total_lums(output_groups, output_count);
    return input_total == output_total;
}

size_t vorax_count_total_lums(lum_group_t** groups, size_t group_count) {
    size_t total = 0;
    for (size_t i = 0; i < group_count; i++) {
        if (groups[i]) {
            total += groups[i]->count;
        }
    }
    return total;
}

// Advanced operations
vorax_result_t* vorax_create_node(lum_group_t* group1, lum_group_t* group2) {
    vorax_result_t* result = vorax_result_create();
    if (!result || !group1 || !group2) {
        if (result) vorax_result_set_error(result, "Invalid groups for node creation");
        return result;
    }

    lum_group_t* node = lum_group_create(2);
    if (!node) {
        vorax_result_set_error(result, "Failed to create node");
        return result;
    }

    // Create two node LUMs representing the connection
    lum_t* node_lum1 = lum_create(1, 0, 0, LUM_STRUCTURE_NODE);
    lum_t* node_lum2 = lum_create(1, 1, 0, LUM_STRUCTURE_NODE);

    if (node_lum1 && node_lum2) {
        lum_group_add(node, node_lum1);
        lum_group_add(node, node_lum2);
        lum_destroy(node_lum1);
        lum_destroy(node_lum2);

        result->result_group = node;
        vorax_result_set_success(result, "Node created successfully");
    } else {
        lum_group_destroy(node);
        if (node_lum1) lum_destroy(node_lum1);
        if (node_lum2) lum_destroy(node_lum2);
        vorax_result_set_error(result, "Failed to create node LUMs");
    }

    return result;
}

vorax_result_t* vorax_emit_lums(lum_zone_t* zone, size_t count) {
    vorax_result_t* result = vorax_result_create();
    if (!result || !zone) {
        if (result) vorax_result_set_error(result, "Invalid zone for emit operation");
        return result;
    }

    lum_group_t* emitted = lum_group_create(count);
    if (!emitted) {
        vorax_result_set_error(result, "Failed to create emitted group");
        return result;
    }

    // Create new LUMs
    for (size_t i = 0; i < count; i++) {
        lum_t* new_lum = lum_create(1, (int32_t)i, 0, LUM_STRUCTURE_LINEAR);
        if (new_lum) {
            lum_group_add(emitted, new_lum);
            lum_destroy(new_lum); // CORRECTION: Utiliser lum_destroy
        }
    }

    lum_zone_add_group(zone, emitted);

    char msg[256];
    snprintf(msg, sizeof(msg), "Emitted %zu LUMs to zone %s", count, zone->name);
    vorax_result_set_success(result, msg);
    return result;
}