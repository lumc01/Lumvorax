#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "core/time_ns.h"

/**
 * log_writer_entry
 * Standard Name: log_writer_entry
 * Records a forensic log entry with absolute precision.
 */
void log_writer_entry(const char* module, const char* event, uint64_t value) {
    uint64_t ts = time_ns_get_absolute();
    FILE* f = fopen("trou_noir_sim/logs/raw_data.log", "a");
    if (f) {
        fprintf(f, "[%lu][%s] %s : %lx\n", ts, module, event, value);
        fclose(f);
    }
}

/**
 * checksum_generate
 * Simple XOR checksum for log integrity validation.
 */
uint32_t checksum_generate(const char* data, size_t len) {
    uint32_t hash = 0;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint32_t)data[i];
    }
    return hash;
}
