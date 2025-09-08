#include "memory_tracker.h"
#include <pthread.h>
#include <string.h> // Added for strncpy
#include <stdlib.h> // Added for abort() and free()
#include <time.h>   // Added for time()
#include <stdio.h>  // Added for printf()

// Forward declaration of the lum_log function if it's not defined in memory_tracker.h
// Assuming a basic implementation for demonstration if not provided.
#ifndef LUM_LOG_DEBUG
#define LUM_LOG_DEBUG 0
#define LUM_LOG_ERROR 1
#define LUM_LOG_WARNING 2
#endif

void lum_log(int level, const char* format, ...); // Declaration assuming it exists elsewhere

#define MAX_MEMORY_ENTRIES 1024 // Define a maximum for entries

// Define a structure for memory entries
typedef struct {
    void* ptr;
    size_t size;
    const char* file;
    int line;
    const char* function;
    time_t allocated_time;
    int is_freed;
    time_t freed_time;
    const char* freed_file;
    int freed_line;
    const char* freed_function;
} memory_entry_t;

// Define the memory tracker structure
typedef struct {
    memory_entry_t entries[MAX_MEMORY_ENTRIES];
    size_t count;
    size_t total_allocated;
    size_t total_freed;
    size_t current_usage;
    size_t peak_usage;
} memory_tracker_t;

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

    void* ptr = malloc(size);
    if (ptr) {
        pthread_mutex_lock(&g_tracker_mutex);
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

    pthread_mutex_lock(&g_tracker_mutex);

    int entry_idx = find_entry(ptr); // find_entry only finds active entries
    
    // Need to search for any entry with the pointer, regardless of is_freed status, to detect double free.
    int found_any_entry_idx = -1;
    for (size_t i = 0; i < g_tracker.count; i++) {
        if (g_tracker.entries[i].ptr == ptr) {
            found_any_entry_idx = (int)i;
            break;
        }
    }

    if (found_any_entry_idx == -1) {
        // Pointer was never tracked or already processed in a way that it's not found.
        printf("[MEMORY_TRACKER] ERROR: Free of untracked pointer %p at %s:%d in %s()\n",
               ptr, file, line, func);
        pthread_mutex_unlock(&g_tracker_mutex);
        // In a real scenario, you might want to decide whether to abort or just log and continue.
        // For safety, we can abort if it's a critical error.
        abort();
        return;
    }

    memory_entry_t* entry = &g_tracker.entries[found_any_entry_idx];

    if (entry->is_freed) {
        // Double free detected
        printf("[MEMORY_TRACKER] ERROR: Double free detected for pointer %p at %s:%d in %s()\n",
               ptr, file, line, func);
        printf("[MEMORY_TRACKER] Previously freed at %s:%d in %s()\n",
               entry->freed_file ? entry->freed_file : "N/A",
               entry->freed_line,
               entry->freed_function ? entry->freed_function : "N/A");
        pthread_mutex_unlock(&g_tracker_mutex);
        abort(); // Abort on double free
        return;
    }

    // Mark as freed
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

    free(ptr);
}

void* tracked_calloc(size_t nmemb, size_t size, const char* file, int line, const char* func) {
    if (!g_tracker_initialized) memory_tracker_init();

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

    // If ptr is NULL, equivalent to malloc
    if (!ptr) {
        return tracked_malloc(size, file, line, func);
    }

    pthread_mutex_lock(&g_tracker_mutex);
    int entry_idx = find_entry(ptr); // Find active entry
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
        int found_any_entry_idx = -1;
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

// Dummy implementation of lum_log if not provided elsewhere.
// In a real scenario, this would be defined in a logging utility header/source.
#ifndef LUM_LOG_IMPLEMENTATION
#define LUM_LOG_IMPLEMENTATION
void lum_log(int level, const char* format, ...) {
    va_list args;
    va_start(args, format);
    // Basic logging to stderr. A real implementation would be more sophisticated.
    fprintf(stderr, "[LUM_LOG] ");
    switch (level) {
        case LUM_LOG_DEBUG: fprintf(stderr, "DEBUG: "); break;
        case LUM_LOG_ERROR: fprintf(stderr, "ERROR: "); break;
        case LUM_LOG_WARNING: fprintf(stderr, "WARNING: "); break;
        default: fprintf(stderr, "INFO: "); break;
    }
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}
#endif