
#include "forensic_logger.h"
#include <stdio.h>
#include <time.h>
#include <string.h>

static FILE* forensic_log_file = NULL;

bool forensic_logger_init(const char* filename) {
    forensic_log_file = fopen(filename, "w");
    if (!forensic_log_file) return false;
    
    uint64_t timestamp = lum_get_timestamp();
    fprintf(forensic_log_file, "=== FORENSIC LOG STARTED (timestamp: %lu ns) ===\n", timestamp);
    fflush(forensic_log_file);
    return true;
}

void forensic_log_memory_operation(const char* operation, void* ptr, size_t size) {
    if (!forensic_log_file) return;
    
    uint64_t timestamp = lum_get_timestamp();
    fprintf(forensic_log_file, "[%lu] MEMORY_%s: ptr=%p, size=%zu\n", 
            timestamp, operation, ptr, size);
    fflush(forensic_log_file);
}

void forensic_log_lum_operation(const char* operation, uint64_t lum_count, double duration_ns) {
    if (!forensic_log_file) return;
    
    uint64_t timestamp = lum_get_timestamp();
    fprintf(forensic_log_file, "[%lu] LUM_%s: count=%lu, duration=%.3f ns\n",
            timestamp, operation, lum_count, duration_ns);
    fflush(forensic_log_file);
}

void forensic_logger_destroy(void) {
    if (forensic_log_file) {
        uint64_t timestamp = lum_get_timestamp();
        fprintf(forensic_log_file, "=== FORENSIC LOG ENDED (timestamp: %lu ns) ===\n", timestamp);
        fclose(forensic_log_file);
        forensic_log_file = NULL;
    }
}

void forensic_log(forensic_level_e level, const char* function, const char* format, ...) {
    if (!forensic_log_file) return;
    
    uint64_t timestamp = lum_get_timestamp();
    va_list args;
    va_start(args, format);
    
    fprintf(forensic_log_file, "[%lu] [%d] %s: ", timestamp, level, function);
    vfprintf(forensic_log_file, format, args);
    fprintf(forensic_log_file, "\n");
    fflush(forensic_log_file);
    
    va_end(args);
}
