
#ifndef FORENSIC_TEST_FRAMEWORK_H
#define FORENSIC_TEST_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Niveaux de validation forensique
typedef enum {
    FORENSIC_LEVEL_BASIC = 1,
    FORENSIC_LEVEL_STANDARD = 2,
    FORENSIC_LEVEL_STRICT = 3,
    FORENSIC_LEVEL_ULTRA_STRICT = 4,
    FORENSIC_LEVEL_PARANOID = 5
} forensic_level_e;

// Structure résultat test forensique
typedef struct {
    bool passed;
    char test_name[128];
    char module_name[64];
    uint64_t execution_time_ns;
    size_t memory_used;
    char failure_reason[512];
    forensic_level_e validation_level;
    uint64_t timestamp;
} forensic_test_result_t;

// Macros test forensique ultra-strict
#define FORENSIC_TEST_ULTRA_STRICT(condition, test_name, module) \
    do { \
        struct timespec _start, _end; \
        clock_gettime(CLOCK_MONOTONIC, &_start); \
        \
        if (!(condition)) { \
            printf("[FORENSIC_FAIL] %s in %s: %s\n", test_name, module, #condition); \
            forensic_log_critical_failure(__FILE__, __LINE__, test_name, module); \
            abort(); \
        } else { \
            clock_gettime(CLOCK_MONOTONIC, &_end); \
            uint64_t _duration = (_end.tv_sec - _start.tv_sec) * 1000000000UL + \
                                (_end.tv_nsec - _start.tv_nsec); \
            printf("[FORENSIC_PASS] %s in %s: %.3f ms\n", test_name, module, _duration / 1000000.0); \
        } \
    } while(0)

#define FORENSIC_ASSERT_MEMORY_SAFE(ptr, size, test_name) \
    FORENSIC_TEST_ULTRA_STRICT((ptr != NULL && size > 0), test_name, "MEMORY_SAFETY")

#define FORENSIC_ASSERT_NO_LEAKS(initial_count, test_name) \
    do { \
        size_t current_count = memory_tracker_get_current_usage(); \
        FORENSIC_TEST_ULTRA_STRICT((current_count <= initial_count), test_name, "LEAK_DETECTION"); \
    } while(0)

// Fonctions framework
void forensic_log_critical_failure(const char* file, int line, const char* test, const char* module);
bool forensic_run_module_tests(const char* module_name, forensic_level_e level);
void forensic_generate_report(const char* output_file);

#endif // FORENSIC_TEST_FRAMEWORK_H
