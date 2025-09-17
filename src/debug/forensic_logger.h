#ifndef FORENSIC_LOGGER_H
#define FORENSIC_LOGGER_H

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdarg.h>

// Énumération des niveaux de log forensique
typedef enum {
    FORENSIC_LEVEL_DEBUG = 0,      // Informations de débogage détaillées
    FORENSIC_LEVEL_INFO = 1,       // Informations générales
    FORENSIC_LEVEL_WARNING = 2,    // Avertissements non critiques
    FORENSIC_LEVEL_ERROR = 3,      // Erreurs récupérables
    FORENSIC_LEVEL_CRITICAL = 4    // Erreurs critiques fatales
} forensic_level_e;

#include "../lum/lum_core.h"

bool forensic_logger_init(const char* filename);
void forensic_log_memory_operation(const char* operation, void* ptr, size_t size);
void forensic_log_lum_operation(const char* operation, uint64_t lum_count, double duration_ns);
void forensic_logger_destroy(void);

#endif // FORENSIC_LOGGER_H