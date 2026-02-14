#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "core/time_ns.h"

void log_writer_entry(const char* module, const char* event, uint64_t value) {
    uint64_t ts = time_ns_get_absolute();
    
    // Log binaire (raw_data.bin)
    FILE* fb = fopen("trou_noir_sim/logs/raw_data.bin", "ab");
    if (fb) {
        fwrite(&ts, sizeof(uint64_t), 1, fb);
        fwrite(module, 1, 8, fb); // Tronqué à 8 char pour la structure
        fwrite(event, 1, 16, fb); // Tronqué à 16 char
        fwrite(&value, sizeof(uint64_t), 1, fb);
        fclose(fb);
    }

    // Timeline CSV
    FILE* ft = fopen("trou_noir_sim/logs/timeline.csv", "a");
    if (ft) {
        fprintf(ft, "%lu,%s,%s,%lx\n", ts, module, event, value);
        fclose(ft);
    }

    // Index JSON (Append mode is tricky for JSON, but let's simulate entries)
    FILE* fj = fopen("trou_noir_sim/logs/index.json", "a");
    if (fj) {
        fprintf(fj, "{\"ts\":%lu, \"mod\":\"%s\", \"ev\":\"%s\", \"val\":\"%lx\"}\n", ts, module, event, value);
        fclose(fj);
    }
}

uint32_t checksum_generate(const char* data, size_t len) {
    uint32_t hash = 0x811c9dc5;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint32_t)data[i];
        hash *= 0x01000193;
    }
    return hash;
}
