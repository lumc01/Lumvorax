
#include "memory_tracker.h"
#include <pthread.h>

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

void tracked_free(void* ptr, const char* file, int line, const char* func) {
    if (!ptr) return;
    if (!g_tracker_initialized) {
        printf("[MEMORY_TRACKER] ERROR: Free called before init at %s:%d\n", file, line);
        return;
    }
    
    pthread_mutex_lock(&g_tracker_mutex);
    
    int entry_idx = find_entry(ptr);
    if (entry_idx == -1) {
        printf("[MEMORY_TRACKER] ERROR: Double free or invalid pointer %p at %s:%d in %s()\n", 
               ptr, file, line, func);
        printf("[MEMORY_TRACKER] Searching for any previous reference...\n");
        
        // Recherche exhaustive pour debug
        for (size_t i = 0; i < g_tracker.count; i++) {
            if (g_tracker.entries[i].ptr == ptr) {
                printf("[MEMORY_TRACKER] Found previous entry: allocated at %s:%d, freed at %s:%d\n",
                       g_tracker.entries[i].file, g_tracker.entries[i].line,
                       g_tracker.entries[i].freed_file ? g_tracker.entries[i].freed_file : "NEVER",
                       g_tracker.entries[i].freed_line);
                break;
            }
        }
        
        pthread_mutex_unlock(&g_tracker_mutex);
        abort(); // Arrêt immédiat pour diagnostic
        return;
    }
    
    memory_entry_t* entry = &g_tracker.entries[entry_idx];
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
    
    // Si ptr est NULL, équivalent à malloc
    if (!ptr) {
        return tracked_malloc(size, file, line, func);
    }
    
    pthread_mutex_lock(&g_tracker_mutex);
    int entry_idx = find_entry(ptr);
    size_t old_size = 0;
    
    if (entry_idx != -1) {
        old_size = g_tracker.entries[entry_idx].size;
        // Marquer l'ancienne entrée comme libérée
        g_tracker.entries[entry_idx].is_freed = 1;
        g_tracker.entries[entry_idx].freed_time = time(NULL);
        g_tracker.entries[entry_idx].freed_file = file;
        g_tracker.entries[entry_idx].freed_line = line;
        g_tracker.entries[entry_idx].freed_function = func;
        g_tracker.current_usage -= old_size;
    }
    pthread_mutex_unlock(&g_tracker_mutex);
    
    void* new_ptr = realloc(ptr, size);
    if (new_ptr) {
        pthread_mutex_lock(&g_tracker_mutex);
        add_entry(new_ptr, size, file, line, func);
        pthread_mutex_unlock(&g_tracker_mutex);
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
