#include "memory_tracker.h"
#include <pthread.h>
#include <string.h> // Added for strncpy
#include <stdlib.h> // Added for abort() and free()
#include <time.h>   // Added for time()
#include <stdio.h>  // Added for printf()

#include "../logger/lum_logger.h" // Include for lum_log function

// Global tracking variables - removing unused duplicates
static size_t g_count = 0; // Current number of active allocations
static size_t g_total_allocated = 0; // Total bytes ever allocated
static size_t g_total_freed = 0; // Total bytes ever freed
static bool g_tracking_enabled = true; // Flag to enable/disable tracking
static bool g_release_mode = false; // Flag for release mode

void memory_tracker_enable(bool enable) {
    g_tracking_enabled = enable;
}

bool memory_tracker_is_enabled(void) {
    return g_tracking_enabled && !g_release_mode;
}

void memory_tracker_set_release_mode(bool mode) {
    g_release_mode = mode;
    if (mode) {
        g_tracking_enabled = false;  // Désactive en mode release
    }
}

void memory_tracker_export_json(const char* filename) {
    if (!memory_tracker_is_enabled()) return;

    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "[MEMORY_TRACKER] ERROR: Could not open file %s for writing.\n", filename);
        return;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"total_allocated\": %zu,\n", g_total_allocated);
    fprintf(fp, "  \"total_freed\": %zu,\n", g_total_freed);
    fprintf(fp, "  \"current_allocations\": %zu,\n", g_count);
    fprintf(fp, "  \"leak_detection\": %s\n", (g_total_allocated > g_total_freed) ? "true" : "false");
    fprintf(fp, "}\n");

    fclose(fp);
}

static memory_tracker_t g_tracker = {0};
static pthread_mutex_t g_tracker_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_tracker_initialized = 0;

void memory_tracker_init(void) {
    pthread_mutex_lock(&g_tracker_mutex);
    if (!g_tracker_initialized) {
        memset(&g_tracker, 0, sizeof(memory_tracker_t));
        g_tracker_initialized = 1;
        printf("[MEMORY_TRACKER] Initialized - tracking enabled\n");
    }
    pthread_mutex_unlock(&g_tracker_mutex);
}

static int find_entry(void* ptr) {
    for (size_t i = 0; i < g_tracker.count; i++) {
        if (g_tracker.entries[i].ptr == ptr && !g_tracker.entries[i].is_freed) {
            return (int)i;
        }
    }
    return -1;
}

static void add_entry(void* ptr, size_t size, const char* file, int line, const char* func) {
    if (g_tracker.count >= MAX_MEMORY_ENTRIES) {
        printf("[MEMORY_TRACKER] WARNING: Max entries reached!\n");
        return;
    }

    memory_entry_t* entry = &g_tracker.entries[g_tracker.count];
    entry->ptr = ptr;
    entry->size = size;
    entry->file = file;
    entry->line = line;
    entry->function = func;
    entry->allocated_time = time(NULL);
    entry->is_freed = 0;
    entry->freed_time = 0;
    entry->freed_file = NULL;
    entry->freed_line = 0;
    entry->freed_function = NULL;

    g_tracker.count++;
    g_tracker.total_allocated += size;
    g_tracker.current_usage += size;

    if (g_tracker.current_usage > g_tracker.peak_usage) {
        g_tracker.peak_usage = g_tracker.current_usage;
    }

    printf("[MEMORY_TRACKER] ALLOC: %p (%zu bytes) at %s:%d in %s()\n",
           ptr, size, file, line, func);
}

void* tracked_malloc(size_t size, const char* file, int line, const char* func) {
    if (!g_tracker_initialized) memory_tracker_init();
    if (!memory_tracker_is_enabled()) return malloc(size);

    void* ptr = malloc(size);
    if (ptr) {
        pthread_mutex_lock(&g_tracker_mutex);
        
        // CORRECTION CRITIQUE: Vérifier réutilisation d'adresse - mais seulement pour adresses vraiment actives
        for (size_t i = 0; i < g_tracker.count; i++) {
            if (g_tracker.entries[i].ptr == ptr && !g_tracker.entries[i].is_freed) {
                // Vérifier si c'est vraiment une réutilisation problématique
                // ou juste une adresse recyclée par l'allocateur système
                printf("[MEMORY_TRACKER] WARNING: Address %p potentially reused\n", ptr);
                printf("[MEMORY_TRACKER] Previous allocation at %s:%d in %s()\n",
                       g_tracker.entries[i].file, g_tracker.entries[i].line, g_tracker.entries[i].function);
                
                // Marquer l'ancienne entrée comme corrompue au lieu d'abort
                g_tracker.entries[i].is_freed = 1;
                g_tracker.entries[i].freed_time = time(NULL);
                g_tracker.entries[i].freed_file = "SYSTEM_RECYCLED";
                g_tracker.entries[i].freed_line = 0;
                g_tracker.entries[i].freed_function = "auto_recycled";
                break;
            }
        }
        
        add_entry(ptr, size, file, line, func);
        pthread_mutex_unlock(&g_tracker_mutex);
    }
    return ptr;
}

// Modified tracked_free function with double-free protection
void tracked_free(void* ptr, const char* file, int line, const char* func) {
    if (!ptr) return;
    if (!g_tracker_initialized) {
        printf("[MEMORY_TRACKER] ERROR: Free called before init at %s:%d\n", file, line);
        return;
    }
    if (!memory_tracker_is_enabled()) {
        free(ptr);
        return;
    }

    pthread_mutex_lock(&g_tracker_mutex);

    // CORRECTION CRITIQUE: Validation intégrité avant libération
    int found_entry_idx = -1;
    for (size_t i = 0; i < g_tracker.count; i++) {
        if (g_tracker.entries[i].ptr == ptr) {
            found_entry_idx = (int)i;
            break;
        }
    }

    if (found_entry_idx == -1) {
        printf("[MEMORY_TRACKER] CRITICAL ERROR: Free of untracked pointer %p at %s:%d in %s()\n",
               ptr, file, line, func);
        printf("[MEMORY_TRACKER] This indicates memory corruption or double-free!\n");
        pthread_mutex_unlock(&g_tracker_mutex);
        abort(); // Arrêt immédiat sur pointeur non suivi
    }

    memory_entry_t* entry = &g_tracker.entries[found_entry_idx];

    // PROTECTION ABSOLUE: Vérifier double-free
    if (entry->is_freed) {
        printf("[MEMORY_TRACKER] CRITICAL ERROR: DOUBLE FREE DETECTED!\n");
        printf("[MEMORY_TRACKER] Pointer %p at %s:%d in %s()\n", ptr, file, line, func);
        printf("[MEMORY_TRACKER] Previously freed at %s:%d in %s() at time %ld\n",
               entry->freed_file ? entry->freed_file : "N/A",
               entry->freed_line,
               entry->freed_function ? entry->freed_function : "N/A",
               entry->freed_time);
        printf("[MEMORY_TRACKER] SYSTEM HALTED TO PREVENT CORRUPTION\n");
        pthread_mutex_unlock(&g_tracker_mutex);
        abort(); // Arrêt immédiat sur double-free
    }

    // Marquer comme libéré avec validation
    entry->is_freed = 1;
    entry->freed_time = time(NULL);
    entry->freed_file = file;
    entry->freed_line = line;
    entry->freed_function = func;

    g_tracker.total_freed += entry->size;
    g_tracker.current_usage -= entry->size;

    printf("[MEMORY_TRACKER] FREE: %p (%zu bytes) at %s:%d in %s() - originally allocated at %s:%d\n",
           ptr, entry->size, file, line, func, entry->file, entry->line);

    pthread_mutex_unlock(&g_tracker_mutex);

    // LIBÉRATION SÉCURISÉE
    free(ptr);
    
    // INVALIDATION du pointeur (note: ne peut pas modifier ptr directement)
    // Le code appelant doit mettre ptr = NULL après cette fonction
}

void* tracked_calloc(size_t nmemb, size_t size, const char* file, int line, const char* func) {
    if (!g_tracker_initialized) memory_tracker_init();
    if (!memory_tracker_is_enabled()) return calloc(nmemb, size);

    void* ptr = calloc(nmemb, size);
    if (ptr) {
        pthread_mutex_lock(&g_tracker_mutex);
        add_entry(ptr, nmemb * size, file, line, func);
        pthread_mutex_unlock(&g_tracker_mutex);
    }
    return ptr;
}

void* tracked_realloc(void* ptr, size_t size, const char* file, int line, const char* func) {
    if (!g_tracker_initialized) memory_tracker_init();
    if (!memory_tracker_is_enabled()) return realloc(ptr, size);

    // If ptr is NULL, equivalent to malloc
    if (!ptr) {
        return tracked_malloc(size, file, line, func);
    }

    pthread_mutex_lock(&g_tracker_mutex);
    int entry_idx = find_entry(ptr); // Find active entry
    int found_any_entry_idx = -1; // Declare here for wider scope
    size_t old_size = 0;

    if (entry_idx != -1) {
        old_size = g_tracker.entries[entry_idx].size;
        // Mark the old entry as freed
        g_tracker.entries[entry_idx].is_freed = 1;
        g_tracker.entries[entry_idx].freed_time = time(NULL);
        g_tracker.entries[entry_idx].freed_file = file;
        g_tracker.entries[entry_idx].freed_line = line;
        g_tracker.entries[entry_idx].freed_function = func;
        g_tracker.current_usage -= old_size;
    } else {
        // If the pointer was not found as active, it might be a double realloc or an untracked pointer.
        // For realloc, we still need to proceed if it's a valid pointer, but we should log a warning.
        // We search for any entry to get the old size if available.
        for (size_t i = 0; i < g_tracker.count; i++) {
            if (g_tracker.entries[i].ptr == ptr) {
                found_any_entry_idx = (int)i;
                break;
            }
        }
        if (found_any_entry_idx != -1) {
            old_size = g_tracker.entries[found_any_entry_idx].size;
            // If it was already freed, this is problematic.
            if (g_tracker.entries[found_any_entry_idx].is_freed) {
                 printf("[MEMORY_TRACKER] WARNING: Realloc called on a freed pointer %p at %s:%d in %s()\n",
                       ptr, file, line, func);
                 // Depending on policy, might want to abort or just proceed. Proceeding for now.
            }
             // We don't update the freed status here again if it was already marked, to avoid confusion.
             // But we do adjust current_usage if it was active.
             if (!g_tracker.entries[found_any_entry_idx].is_freed) {
                 g_tracker.current_usage -= old_size;
             }
        } else {
             printf("[MEMORY_TRACKER] WARNING: Realloc called on an untracked pointer %p at %s:%d in %s()\n",
                   ptr, file, line, func);
        }
    }
    pthread_mutex_unlock(&g_tracker_mutex);

    void* new_ptr = realloc(ptr, size);
    if (new_ptr) {
        pthread_mutex_lock(&g_tracker_mutex);
        add_entry(new_ptr, size, file, line, func);
        pthread_mutex_unlock(&g_tracker_mutex);
    } else {
        // If realloc fails, the original pointer is still valid and should be considered active again.
        if (entry_idx != -1) { // If it was a tracked active pointer before realloc
            pthread_mutex_lock(&g_tracker_mutex);
            g_tracker.entries[entry_idx].is_freed = 0; // Revert the freed status
            g_tracker.entries[entry_idx].freed_time = 0;
            g_tracker.entries[entry_idx].freed_file = NULL;
            g_tracker.entries[entry_idx].freed_line = 0;
            g_tracker.entries[entry_idx].freed_function = NULL;
            g_tracker.current_usage += old_size; // Restore usage
            pthread_mutex_unlock(&g_tracker_mutex);
        } else if (found_any_entry_idx != -1 && !g_tracker.entries[found_any_entry_idx].is_freed) {
            // If it was an untracked but valid pointer and realloc failed, restore usage if it was active.
            pthread_mutex_lock(&g_tracker_mutex);
            g_tracker.current_usage += old_size;
            pthread_mutex_unlock(&g_tracker_mutex);
        }
    }

    return new_ptr;
}

void memory_tracker_report(void) {
    if (!g_tracker_initialized) return;
    if (!memory_tracker_is_enabled()) {
        printf("[MEMORY_TRACKER] Tracking is disabled.\n");
        return;
    }

    pthread_mutex_lock(&g_tracker_mutex);

    printf("\n=== MEMORY TRACKER REPORT ===\n");
    printf("Total allocations: %zu bytes\n", g_tracker.total_allocated);
    printf("Total freed: %zu bytes\n", g_tracker.total_freed);
    printf("Current usage: %zu bytes\n", g_tracker.current_usage);
    printf("Peak usage: %zu bytes\n", g_tracker.peak_usage);
    printf("Active entries: ");

    size_t active_count = 0;
    for (size_t i = 0; i < g_tracker.count; i++) {
        if (!g_tracker.entries[i].is_freed) {
            active_count++;
        }
    }
    printf("%zu\n", active_count);

    if (active_count > 0) {
        printf("\nACTIVE ALLOCATIONS (potential leaks):\n");
        for (size_t i = 0; i < g_tracker.count; i++) {
            if (!g_tracker.entries[i].is_freed) {
                printf("  %p (%zu bytes) - allocated at %s:%d in %s()\n",
                       g_tracker.entries[i].ptr,
                       g_tracker.entries[i].size,
                       g_tracker.entries[i].file,
                       g_tracker.entries[i].line,
                       g_tracker.entries[i].function);
            }
        }
    }

    printf("==============================\n\n");

    pthread_mutex_unlock(&g_tracker_mutex);
}

void memory_tracker_check_leaks(void) {
    if (!g_tracker_initialized) return;
     if (!memory_tracker_is_enabled()) {
        printf("[MEMORY_TRACKER] Leak check skipped: tracking is disabled.\n");
        return;
    }

    pthread_mutex_lock(&g_tracker_mutex);

    size_t leak_count = 0;
    size_t leak_size = 0;

    for (size_t i = 0; i < g_tracker.count; i++) {
        if (!g_tracker.entries[i].is_freed) {
            leak_count++;
            leak_size += g_tracker.entries[i].size;
        }
    }

    if (leak_count > 0) {
        printf("[MEMORY_TRACKER] LEAK DETECTION: %zu leaks (%zu bytes total)\n",
               leak_count, leak_size);
    } else {
        printf("[MEMORY_TRACKER] No memory leaks detected\n");
    }

    pthread_mutex_unlock(&g_tracker_mutex);
}

void memory_tracker_destroy(void) {
    if (!g_tracker_initialized) return;

    printf("[MEMORY_TRACKER] Final report before shutdown:\n");
    memory_tracker_report();
    memory_tracker_check_leaks();

    pthread_mutex_lock(&g_tracker_mutex);
    g_tracker_initialized = 0;
    pthread_mutex_unlock(&g_tracker_mutex);
}

// lum_log is defined in the logger mod

void memory_tracker_cleanup(void) {
    if (!g_tracker_initialized) return;

    printf("[MEMORY_TRACKER] Cleanup initiated\n");
    memory_tracker_report();
    memory_tracker_check_leaks();

    pthread_mutex_lock(&g_tracker_mutex);

    // Reset all entries
    memset(&g_tracker, 0, sizeof(g_tracker));
    g_tracker_initialized = false;
    g_tracking_enabled = true; // Reset to default

    pthread_mutex_unlock(&g_tracker_mutex);
    printf("[MEMORY_TRACKER] Cleanup completed\n");
}