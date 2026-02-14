#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static inline uint64_t get_nanos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

void log_writer_entry(const char* module, const char* event, uint64_t value) {
    uint64_t ts = get_nanos();
    
    // Log binaire (raw_data.bin)
    FILE* fb = fopen("trou_noir_sim/logs/raw_data.bin", "ab");
    if (fb) {
        fwrite(&ts, sizeof(uint64_t), 1, fb);
        char mod[8] = {0}; strncpy(mod, module, 7);
        char ev[16] = {0}; strncpy(ev, event, 15);
        fwrite(mod, 1, 8, fb);
        fwrite(ev, 1, 16, fb);
        fwrite(&value, sizeof(uint64_t), 1, fb);
        fclose(fb);
    }

    // Timeline CSV
    FILE* ft = fopen("trou_noir_sim/logs/timeline.csv", "a");
    if (ft) {
        fprintf(ft, "%lu,%s,%s,%lx\n", ts, module, event, value);
        fclose(ft);
    }

    // Index JSON
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
