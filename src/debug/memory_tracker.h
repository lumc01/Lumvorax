
#ifndef MEMORY_TRACKER_H
#define MEMORY_TRACKER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Configuration du debugging mémoire
#define MEMORY_DEBUG_ENABLED 1
#define MAX_MEMORY_ENTRIES 10000

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

typedef struct {
    memory_entry_t entries[MAX_MEMORY_ENTRIES];
    size_t count;
    size_t total_allocated;
    size_t total_freed;
    size_t peak_usage;
    size_t current_usage;
} memory_tracker_t;

// Macros pour traçage automatique
#define TRACKED_MALLOC(size) tracked_malloc(size, __FILE__, __LINE__, __func__)
#define TRACKED_FREE(ptr) tracked_free(ptr, __FILE__, __LINE__, __func__)
#define TRACKED_CALLOC(nmemb, size) tracked_calloc(nmemb, size, __FILE__, __LINE__, __func__)
#define TRACKED_REALLOC(ptr, size) tracked_realloc(ptr, size, __FILE__, __LINE__, __func__)

// Fonctions publiques
void memory_tracker_init(void);
void* tracked_malloc(size_t size, const char* file, int line, const char* func);
void tracked_free(void* ptr, const char* file, int line, const char* func);
void* tracked_calloc(size_t nmemb, size_t size, const char* file, int line, const char* func);
void* tracked_realloc(void* ptr, size_t size, const char* file, int line, const char* func);
void memory_tracker_report(void);
void memory_tracker_check_leaks(void);
void memory_tracker_destroy(void);

#endif // MEMORY_TRACKER_H
