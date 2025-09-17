
#include "log_manager.h"
#include "../debug/memory_tracker.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

static log_manager_t* g_log_manager = NULL;

log_manager_t* log_manager_create(void) {
    if (g_log_manager) {
        return g_log_manager; // Singleton
    }
    
    log_manager_t* manager = TRACKED_MALLOC(sizeof(log_manager_t));
    if (!manager) return NULL;
    
    // Génération session ID
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    snprintf(manager->session_id, sizeof(manager->session_id), "%04d%02d%02d_%02d%02d%02d",
             tm_info->tm_year + 1900, tm_info->tm_mon + 1, tm_info->tm_mday,
             tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec);
    
    manager->module_count = 0;
    manager->auto_archive = true;
    manager->initialized = false;
    
    // Création structure de logs
    log_manager_create_structure(manager);
    
    g_log_manager = manager;
    return manager;
}

bool log_manager_create_structure(log_manager_t* manager) {
    if (!manager) return false;
    
    // Création dossiers principaux
    mkdir("logs", 0755);
    mkdir("logs/current", 0755);
    mkdir("logs/archive", 0755);
    mkdir("logs/modules", 0755);
    
    // Dossiers par module
    const char* modules[] = {
        "lum_core", "vorax_operations", "parser", "binary_converter",
        "logger", "optimization", "parallel", "metrics", "crypto",
        "persistence", "debug", "advanced_calculations", "complex_modules"
    };
    
    for (size_t i = 0; i < sizeof(modules) / sizeof(modules[0]); i++) {
        char path[256];
        snprintf(path, sizeof(path), "logs/modules/%s", modules[i]);
        mkdir(path, 0755);
    }
    
    manager->initialized = true;
    return true;
}

bool log_manager_archive_session(log_manager_t* manager, const char* session_id) {
    if (!manager || !session_id) return false;
    
    char archive_dir[256];
    snprintf(archive_dir, sizeof(archive_dir), "logs/archive/session_%s", session_id);
    mkdir(archive_dir, 0755);
    
    // Archivage logs de la session
    char command[512];
    snprintf(command, sizeof(command), 
             "find logs/current -name '*%s*' -exec mv {} %s/ \\; 2>/dev/null || true", 
             session_id, archive_dir);
    system(command);
    
    snprintf(command, sizeof(command), 
             "find logs/modules -name '*%s*' -exec cp {} %s/ \\; 2>/dev/null || true", 
             session_id, archive_dir);
    system(command);
    
    return true;
}

module_logger_t* log_manager_get_module_logger(log_manager_t* manager, const char* module_name) {
    if (!manager || !module_name) return NULL;
    
    // Recherche logger existant
    for (size_t i = 0; i < manager->module_count; i++) {
        if (strcmp(manager->module_loggers[i].module_name, module_name) == 0) {
            return &manager->module_loggers[i];
        }
    }
    
    // Création nouveau logger
    if (manager->module_count >= MAX_MODULE_LOGGERS) return NULL;
    
    module_logger_t* logger = &manager->module_loggers[manager->module_count];
    strncpy(logger->module_name, module_name, sizeof(logger->module_name) - 1);
    logger->module_name[sizeof(logger->module_name) - 1] = '\0';
    
    // Création fichier log
    snprintf(logger->log_filename, sizeof(logger->log_filename),
             "logs/modules/%s/%s_%s.log", module_name, module_name, manager->session_id);
    
    logger->log_file = fopen(logger->log_filename, "a");
    if (!logger->log_file) return NULL;
    
    logger->enabled = true;
    logger->message_count = 0;
    
    // Log d'initialisation
    fprintf(logger->log_file, "[%s] === MODULE %s LOGGER INITIALIZED ===\n", 
            manager->session_id, module_name);
    fflush(logger->log_file);
    
    manager->module_count++;
    return logger;
}

void log_manager_write_module_log(log_manager_t* manager, const char* module_name, 
                                  const char* level, const char* message) {
    if (!manager || !module_name || !message) return;
    
    module_logger_t* logger = log_manager_get_module_logger(manager, module_name);
    if (!logger || !logger->enabled || !logger->log_file) return;
    
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    
    fprintf(logger->log_file, "[%04d-%02d-%02d %02d:%02d:%02d] [%s] %s\n",
            tm_info->tm_year + 1900, tm_info->tm_mon + 1, tm_info->tm_mday,
            tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec,
            level ? level : "INFO", message);
    fflush(logger->log_file);
    
    logger->message_count++;
}

void log_manager_destroy(log_manager_t* manager) {
    if (!manager) return;
    
    // Archive session actuelle
    if (manager->auto_archive) {
        log_manager_archive_session(manager, manager->session_id);
    }
    
    // Fermeture fichiers
    for (size_t i = 0; i < manager->module_count; i++) {
        if (manager->module_loggers[i].log_file) {
            fclose(manager->module_loggers[i].log_file);
        }
    }
    
    TRACKED_FREE(manager);
    g_log_manager = NULL;
}

log_manager_t* log_manager_get_instance(void) {
    if (!g_log_manager) {
        g_log_manager = log_manager_create();
    }
    return g_log_manager;
}

// Macros pratiques pour logging par module
void LOG_MODULE(const char* module, const char* level, const char* format, ...) {
    log_manager_t* manager = log_manager_get_instance();
    if (!manager) return;
    
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    buffer[sizeof(buffer) - 1] = '\0';
    
    log_manager_write_module_log(manager, module, level, buffer);
}
