
#include "forensic_logger.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>   // Pour mkdir()
#include <unistd.h>     // Pour access()
#include <errno.h>      // Pour errno

static FILE* forensic_log_file = NULL;

bool forensic_logger_init(const char* filename) {
    if (!filename) {
        fprintf(stderr, "[FORENSIC] ERROR: filename is NULL\n");
        return false;
    }
    
    // Créer répertoire si nécessaire avec vérification complète
    char dir_path[256];
    strncpy(dir_path, filename, sizeof(dir_path) - 1);
    dir_path[sizeof(dir_path) - 1] = '\0';
    
    char *last_slash = strrchr(dir_path, '/');
    if (last_slash) {
        *last_slash = '\0';
        
        // Créer récursivement tous les répertoires parents
        char temp_path[256];
        char *token = strtok(dir_path, "/");
        strcpy(temp_path, "");
        
        while (token != NULL) {
            strcat(temp_path, token);
            strcat(temp_path, "/");
            mkdir(temp_path, 0755);
            token = strtok(NULL, "/");
        }
    }
    
    // Tentative d'ouverture avec gestion d'erreur robuste
    forensic_log_file = fopen(filename, "w");
    if (!forensic_log_file) {
        // Fallback vers répertoire courant
        char fallback_name[256];
        snprintf(fallback_name, sizeof(fallback_name), "forensic_fallback_%lu.log", 
                 (unsigned long)time(NULL));
        
        forensic_log_file = fopen(fallback_name, "w");
        if (!forensic_log_file) {
            fprintf(stderr, "[FORENSIC] CRITICAL: Cannot create any log file\n");
            return false;
        }
        
        fprintf(stderr, "[FORENSIC] WARNING: Using fallback log: %s\n", fallback_name);
    }
    
    uint64_t timestamp = lum_get_timestamp();
    fprintf(forensic_log_file, "=== FORENSIC LOG STARTED (timestamp: %lu ns) ===\n", timestamp);
    fprintf(forensic_log_file, "Forensic logging initialized successfully\n");
    fflush(forensic_log_file);
    
    printf("[FORENSIC] Log initialized successfully: %s\n", filename);
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
    
    // NOUVEAU: Log détaillé pour chaque LUM individuel
    printf("[FORENSIC_REALTIME] LUM_%s: count=%lu at timestamp=%lu ns\n", 
           operation, lum_count, timestamp);
}

// NOUVELLE FONCTION: Log systématique pour chaque LUM
void forensic_log_individual_lum(uint32_t lum_id, const char* operation, uint64_t timestamp_ns) {
    if (!forensic_log_file) return;
    
    fprintf(forensic_log_file, "[%lu] [LUM_%u] %s: Individual LUM processing\n",
            timestamp_ns, lum_id, operation);
    fflush(forensic_log_file);
    
    // TEMPS RÉEL: Affichage console obligatoire
    printf("[%lu] [LUM_%u] %s\n", timestamp_ns, lum_id, operation);
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

// Implementation of unified_forensic_log for compatibility
void unified_forensic_log(unified_forensic_level_e level, const char* function, const char* format, ...) {
    if (!forensic_log_file) return;
    
    uint64_t timestamp = lum_get_timestamp();
    va_list args;
    va_start(args, format);
    
    fprintf(forensic_log_file, "[%lu] [UNIFIED_%d] %s: ", timestamp, level, function);
    vfprintf(forensic_log_file, format, args);
    fprintf(forensic_log_file, "\n");
    fflush(forensic_log_file);
    
    va_end(args);
}
