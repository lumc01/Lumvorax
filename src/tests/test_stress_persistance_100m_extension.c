
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/statvfs.h>
#include "../persistence/data_persistence.h"
#include "../lum/lum_core.h"
#include "../debug/memory_tracker.h"

#define HUNDRED_MILLION_LUMS 100000000UL
#define CHUNK_SIZE 1000000UL
#define MIN_DISK_SPACE_GB 50UL

// Extension conforme STANDARD_NAMES.md
typedef struct {
    uint64_t total_lums_processed;
    uint64_t total_chunks_written;
    uint64_t total_bytes_on_disk;
    uint64_t write_time_nanoseconds;
    uint64_t read_time_nanoseconds;
    uint64_t verification_errors;
    uint64_t io_errors;
    bool stress_test_completed;
    char test_session_id[64];
    uint32_t magic_number;
} stress_100m_extension_result_t;

bool check_available_disk_space(const char* path) {
    struct statvfs stat;
    if (statvfs(path, &stat) != 0) return false;
    
    uint64_t available_gb = (stat.f_bavail * stat.f_frsize) / (1024UL * 1024UL * 1024UL);
    printf("💾 Espace disque disponible: %lu GB\n", available_gb);
    
    return available_gb >= MIN_DISK_SPACE_GB;
}

stress_100m_extension_result_t* execute_100m_lums_stress_extension(void) {
    printf("🚀 === EXTENSION TEST STRESS 100M LUMs ===\n");
    printf("📊 Target: %lu LUMs en chunks de %lu\n", HUNDRED_MILLION_LUMS, CHUNK_SIZE);
    
    stress_100m_extension_result_t* result = TRACKED_MALLOC(sizeof(stress_100m_extension_result_t));
    if (!result) return NULL;
    
    memset(result, 0, sizeof(stress_100m_extension_result_t));
    result->magic_number = 0x53545245; // "STRE"
    
    // Générer ID session unique
    snprintf(result->test_session_id, sizeof(result->test_session_id), 
             "stress100m_%lu", (unsigned long)time(NULL));
    
    // Vérification espace disque
    if (!check_available_disk_space(".")) {
        printf("❌ ÉCHEC: Espace disque insuffisant\n");
        TRACKED_FREE(result);
        return NULL;
    }
    
    // UTILISATION des structures EXISTANTES (conformément STANDARD_NAMES.md)
    persistence_context_t* ctx = persistence_context_create("stress_100m_extension_data");
    if (!ctx) {
        printf("❌ ÉCHEC: Contexte persistance existant indisponible\n");
        TRACKED_FREE(result);
        return NULL;
    }
    
    struct timespec start_total, end_total, start_chunk, end_chunk;
    clock_gettime(CLOCK_MONOTONIC, &start_total);
    
    // Phase 1: ÉCRITURE par chunks avec structures existantes
    printf("\n📝 PHASE 1: ÉCRITURE 100M LUMs utilisant modules existants...\n");
    
    size_t num_chunks = HUNDRED_MILLION_LUMS / CHUNK_SIZE;
    for (size_t chunk = 0; chunk < num_chunks; chunk++) {
        printf("  📦 Chunk %zu/%zu (%.1f%%)...\n", 
               chunk + 1, num_chunks, ((double)(chunk + 1) / num_chunks) * 100.0);
        
        clock_gettime(CLOCK_MONOTONIC, &start_chunk);
        
        // UTILISATION lum_group_t EXISTANT
        lum_group_t* chunk_group = lum_group_create(CHUNK_SIZE);
        if (!chunk_group) {
            printf("❌ ÉCHEC allocation chunk %zu\n", chunk);
            result->io_errors++;
            continue;
        }
        
        // Remplir avec LUMs selon structure existante lum_t
        for (size_t i = 0; i < CHUNK_SIZE; i++) {
            uint64_t global_id = chunk * CHUNK_SIZE + i;
            lum_t* lum = lum_create(
                global_id % 2,  // presence
                (int32_t)(global_id % 100000),  // position_x
                (int32_t)(global_id / 100000),  // position_y
                LUM_STRUCTURE_LINEAR
            );
            
            if (lum) {
                lum->timestamp = time(NULL) * 1000000000UL + global_id;
                lum->checksum = lum_calculate_checksum(lum);
                
                if (!lum_group_add_lum(chunk_group, lum)) {
                    result->io_errors++;
                    lum_destroy(lum);
                }
            } else {
                result->io_errors++;
            }
        }
        
        // UTILISATION fonction EXISTANTE persistence_save_group
        char chunk_filename[256];
        snprintf(chunk_filename, sizeof(chunk_filename), 
                "chunk_%s_%06zu.lum", result->test_session_id, chunk);
        
        storage_result_t* save_result = persistence_save_group(ctx, chunk_group, chunk_filename);
        if (save_result && save_result->success) {
            result->total_lums_processed += chunk_group->count;
            result->total_bytes_on_disk += save_result->bytes_written;
            result->total_chunks_written++;
        } else {
            printf("❌ ÉCHEC sauvegarde chunk %zu\n", chunk);
            result->io_errors++;
        }
        
        if (save_result) storage_result_destroy(save_result);
        lum_group_safe_destroy(chunk_group);
        
        clock_gettime(CLOCK_MONOTONIC, &end_chunk);
        uint64_t chunk_time_ns = (end_chunk.tv_sec - start_chunk.tv_sec) * 1000000000UL +
                                (end_chunk.tv_nsec - start_chunk.tv_nsec);
        result->write_time_nanoseconds += chunk_time_ns;
        
        printf("    ✅ Chunk %zu: %lu LUMs, %.2f ms, %.0f LUMs/sec\n", 
               chunk, chunk_group->count, chunk_time_ns / 1000000.0,
               (double)chunk_group->count / (chunk_time_ns / 1000000000.0));
        
        // Affichage progrès tous les 10 chunks
        if ((chunk + 1) % 10 == 0) {
            printf("🔄 Progression globale: %.1f%% - %lu LUMs traités\n",
                   ((double)(chunk + 1) / num_chunks) * 100.0,
                   result->total_lums_processed);
        }
    }
    
    printf("\n📖 PHASE 2: LECTURE ET VÉRIFICATION...\n");
    
    // Phase 2: LECTURE avec fonctions existantes
    clock_gettime(CLOCK_MONOTONIC, &start_chunk);
    
    for (size_t chunk = 0; chunk < num_chunks; chunk++) {
        char chunk_filename[256];
        snprintf(chunk_filename, sizeof(chunk_filename), 
                "chunk_%s_%06zu.lum", result->test_session_id, chunk);
        
        lum_group_t* loaded_group = NULL;
        storage_result_t* load_result = persistence_load_group(ctx, chunk_filename, &loaded_group);
        
        if (load_result && load_result->success && loaded_group) {
            // Vérification intégrité avec fonctions existantes
            for (size_t i = 0; i < loaded_group->count; i++) {
                lum_t* lum = &loaded_group->lums[i];
                uint32_t calculated_checksum = lum_calculate_checksum(lum);
                
                if (lum->checksum != calculated_checksum) {
                    result->verification_errors++;
                }
                
                uint64_t expected_id = chunk * CHUNK_SIZE + i;
                if (lum->id != expected_id) {
                    result->verification_errors++;
                }
            }
            
            lum_group_safe_destroy(loaded_group);
        } else {
            result->io_errors++;
        }
        
        if (load_result) storage_result_destroy(load_result);
        
        if ((chunk + 1) % 20 == 0) {
            printf("🔍 Vérification: %.1f%% - %lu erreurs détectées\n",
                   ((double)(chunk + 1) / num_chunks) * 100.0,
                   result->verification_errors);
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_chunk);
    result->read_time_nanoseconds = (end_chunk.tv_sec - start_chunk.tv_sec) * 1000000000UL +
                                   (end_chunk.tv_nsec - start_chunk.tv_nsec);
    
    clock_gettime(CLOCK_MONOTONIC, &end_total);
    
    // Phase 3: NETTOYAGE
    printf("\n🧹 PHASE 3: NETTOYAGE...\n");
    for (size_t chunk = 0; chunk < num_chunks; chunk++) {
        char chunk_filename[512];
        snprintf(chunk_filename, sizeof(chunk_filename), 
                "stress_100m_extension_data/chunk_%s_%06zu.lum", 
                result->test_session_id, chunk);
        unlink(chunk_filename);
    }
    rmdir("stress_100m_extension_data");
    
    result->stress_test_completed = (result->total_lums_processed == HUNDRED_MILLION_LUMS &&
                                   result->io_errors == 0 &&
                                   result->verification_errors == 0);
    
    // Résultats finaux
    printf("\n🎉 === RÉSULTATS EXTENSION 100M LUMs ===\n");
    printf("✅ LUMs traitées: %lu / %lu (%.2f%%)\n", 
           result->total_lums_processed, HUNDRED_MILLION_LUMS,
           ((double)result->total_lums_processed / HUNDRED_MILLION_LUMS) * 100.0);
    printf("📊 Chunks écrits: %lu\n", result->total_chunks_written);
    printf("💾 Bytes sur disque: %.2f GB\n", result->total_bytes_on_disk / (1024.0 * 1024.0 * 1024.0));
    printf("⏱️ Temps écriture: %.2f secondes\n", result->write_time_nanoseconds / 1000000000.0);
    printf("⏱️ Temps lecture: %.2f secondes\n", result->read_time_nanoseconds / 1000000000.0);
    printf("🚀 Débit écriture: %.0f LUMs/sec\n", 
           (double)result->total_lums_processed / (result->write_time_nanoseconds / 1000000000.0));
    printf("🚀 Débit lecture: %.0f LUMs/sec\n", 
           (double)result->total_lums_processed / (result->read_time_nanoseconds / 1000000000.0));
    printf("❌ Erreurs I/O: %lu\n", result->io_errors);
    printf("❌ Erreurs vérification: %lu\n", result->verification_errors);
    printf("🆔 Session: %s\n", result->test_session_id);
    
    printf("\n🎯 RÉSULTAT: %s\n", 
           result->stress_test_completed ? "✅ SUCCÈS COMPLET" : "❌ ÉCHEC PARTIEL");
    
    persistence_context_destroy(ctx);
    return result;
}

int main(void) {
    printf("🔥 === EXTENSION TEST 100M+ LUMs AVEC MODULES EXISTANTS ===\n");
    
    memory_tracker_init();
    
    stress_100m_extension_result_t* result = execute_100m_lums_stress_extension();
    
    if (result) {
        printf("\n📄 Extension terminée - Session: %s\n", result->test_session_id);
        if (result->magic_number == 0x53545245) {
            printf("🔒 Intégrité structure validée\n");
        }
        TRACKED_FREE(result);
        
        memory_tracker_report();
        memory_tracker_destroy();
        return 0;
    } else {
        printf("\n❌ Extension échouée\n");
        memory_tracker_report();
        memory_tracker_destroy();
        return 1;
    }
}
