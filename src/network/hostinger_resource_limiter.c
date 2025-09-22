
#include "hostinger_resource_limiter.h"
#include "../debug/memory_tracker.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Limites serveur Hostinger exactes
#define HOSTINGER_MAX_CPU_CORES 2
#define HOSTINGER_MAX_RAM_GB 6
#define HOSTINGER_MAX_STORAGE_GB 90
#define HOSTINGER_MAX_CONCURRENT_LUMS 1000000  // 1M max sur serveur

// Structure hostinger_resource_monitor_t déjà définie dans hostinger_resource_limiter.h

static hostinger_resource_monitor_t* global_monitor = NULL;

bool hostinger_check_cpu_availability(void) {
    if (!global_monitor) return false;
    
    if (global_monitor->active_threads >= HOSTINGER_MAX_CPU_CORES) {
        printf("[HOSTINGER_LIMITER] ❌ CPU limité: %u/%d threads actifs\n",
               (uint32_t)global_monitor->active_threads, HOSTINGER_MAX_CPU_CORES);
        return false;
    }
    
    printf("[HOSTINGER_LIMITER] ✅ CPU disponible: %u/%d threads\n",
           (uint32_t)global_monitor->active_threads, HOSTINGER_MAX_CPU_CORES);
    return true;
}

bool hostinger_check_ram_availability(size_t required_mb) {
    if (!global_monitor) return false;
    
    size_t max_ram_mb = HOSTINGER_MAX_RAM_GB * 1024;
    size_t total_needed = global_monitor->current_ram_usage_mb + required_mb;
    
    if (total_needed > max_ram_mb) {
        printf("[HOSTINGER_LIMITER] ❌ RAM insuffisante: %zu MB + %zu MB > %zu MB max\n",
               global_monitor->current_ram_usage_mb, required_mb, max_ram_mb);
        return false;
    }
    
    printf("[HOSTINGER_LIMITER] ✅ RAM disponible: %zu MB libres\n",
           max_ram_mb - total_needed);
    return true;
}

bool hostinger_check_lum_processing_limit(size_t lum_count) {
    if (!global_monitor) return false;
    
    if (lum_count > HOSTINGER_MAX_CONCURRENT_LUMS) {
        printf("[HOSTINGER_LIMITER] ❌ Trop de LUMs: %zu > %d max autorisés\n",
               lum_count, HOSTINGER_MAX_CONCURRENT_LUMS);
        printf("[HOSTINGER_LIMITER] Limitation serveur 2CPU/6GB RAM\n");
        return false;
    }
    
    printf("[HOSTINGER_LIMITER] ✅ LUMs autorisés: %zu/%d\n",
           lum_count, HOSTINGER_MAX_CONCURRENT_LUMS);
    return true;
}

hostinger_resource_monitor_t* hostinger_resource_monitor_create(void) {
    if (global_monitor) return global_monitor;
    
    global_monitor = TRACKED_MALLOC(sizeof(hostinger_resource_monitor_t));
    if (!global_monitor) return NULL;
    
    memset(global_monitor, 0, sizeof(hostinger_resource_monitor_t));
    global_monitor->memory_address = global_monitor;
    global_monitor->magic_number = 0x484F5354;  // "HOST"
    global_monitor->resource_check_enabled = true;
    
    printf("[HOSTINGER_LIMITER] Monitor créé - Serveur 72.60.185.90\n");
    printf("[HOSTINGER_LIMITER] Limites: CPU=2cores, RAM=6GB, LUMs=1M max\n");
    
    return global_monitor;
}

void hostinger_resource_monitor_destroy(void) {
    if (global_monitor && global_monitor->memory_address == global_monitor) {
        TRACKED_FREE(global_monitor);
        global_monitor = NULL;
        printf("[HOSTINGER_LIMITER] Monitor détruit\n");
    }
}
