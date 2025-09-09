
#ifndef FORENSIC_LOGGER_H
#define FORENSIC_LOGGER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../lum/lum_core.h"

bool forensic_logger_init(const char* filename);
void forensic_log_memory_operation(const char* operation, void* ptr, size_t size);
void forensic_log_lum_operation(const char* operation, uint64_t lum_count, double duration_ns);
void forensic_logger_destroy(void);

#endif // FORENSIC_LOGGER_H
