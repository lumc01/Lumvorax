
#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

// Constantes communes
#define MAX_STORAGE_PATH_LENGTH 512
#define MAX_ERROR_MESSAGE_LENGTH 256

// Type unifié pour tous les résultats de stockage
typedef struct {
    bool success;
    char filename[MAX_STORAGE_PATH_LENGTH];
    size_t bytes_written;
    size_t bytes_read;
    uint32_t checksum;
    char error_message[MAX_ERROR_MESSAGE_LENGTH];
    void* transaction_ref;
    // Champs WAL extension
    uint64_t wal_sequence_assigned;
    uint64_t wal_transaction_id;
    bool wal_durability_confirmed;
    char wal_error_details[MAX_ERROR_MESSAGE_LENGTH];
    uint32_t magic_number;
} unified_storage_result_t;

// Types forensiques unifiés
typedef enum {
    FORENSIC_LEVEL_DEBUG = 0,
    FORENSIC_LEVEL_INFO = 1,
    FORENSIC_LEVEL_WARNING = 2,
    FORENSIC_LEVEL_ERROR = 3,
    FORENSIC_LEVEL_CRITICAL = 4
} unified_forensic_level_e;

// Interface forensique standardisée
void unified_forensic_log(unified_forensic_level_e level, const char* function, const char* format, ...);

#endif // COMMON_TYPES_H
