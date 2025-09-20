
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

// Système de logs avancé conforme rapport 091
void create_log_directory(const char* base_path) {
    char command[256];
    snprintf(command, sizeof(command), "mkdir -p %s", base_path);
    system(command);
}

void create_enhanced_log(const char* filepath, const char* message) {
    // Créer le répertoire si nécessaire
    char dir_path[256];
    strncpy(dir_path, filepath, sizeof(dir_path) - 1);
    char* last_slash = strrchr(dir_path, '/');
    if (last_slash) {
        *last_slash = '\0';
        create_log_directory(dir_path);
    }
    
    FILE *file = fopen(filepath, "a");
    if (file != NULL) {
        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        char timestamp[32];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
        
        fprintf(file, "[%s] %s\n", timestamp, message);
        fclose(file);
        
        printf("✅ Log créé: %s\n", filepath);
    } else {
        printf("❌ Erreur création log: %s - %s\n", filepath, strerror(errno));
    }
}

void log_module_action_enhanced(const char* module_name, const char* action) {
    char log_filepath[200];
    snprintf(log_filepath, sizeof(log_filepath), "logs/modules/%s.log", module_name);
    
    char message[256];
    snprintf(message, sizeof(message), "MODULE_%s: %s", module_name, action);
    
    create_enhanced_log(log_filepath, message);
}

void log_test_execution(const char* test_name, const char* result, int line_count) {
    char log_filepath[200];
    snprintf(log_filepath, sizeof(log_filepath), "logs/tests/%s.log", test_name);
    
    char message[256];
    snprintf(message, sizeof(message), "TEST: %s | RÉSULTAT: %s | LIGNES: %d", 
             test_name, result, line_count);
    
    create_enhanced_log(log_filepath, message);
    
    printf("📊 Test loggé: %s - %s (%d lignes)\n", test_name, result, line_count);
}
