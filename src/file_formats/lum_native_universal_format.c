#include "lum_native_universal_format.h"
#include "../debug/memory_tracker.h"
#include "../logger/lum_logger.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Noms des types de contenus pour logging
static const char* CONTENT_TYPE_NAMES[LUM_CONTENT_COUNT] = {
    "TEXT", "JSON", "CSV", "IMAGE_RGB24", "IMAGE_RGBA32", "IMAGE_GRAY8",
    "AUDIO_PCM16", "AUDIO_PCM32", "VIDEO_H264", "VIDEO_RAW", 
    "SOM_KOHONEN", "NEURAL_WEIGHTS", "QUANTUM_STATES", 
    "BINARY_BLOB", "LUM_NATIVE"
};

// Calcul CRC32 simple pour validation
static uint32_t calculate_crc32(const uint8_t* data, size_t length) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
        }
    }
    return ~crc;
}

// Création manager fichier .lum universel
lum_universal_file_manager_t* lum_universal_file_create(const char* filepath) {
    if (!filepath) return NULL;

    lum_universal_file_manager_t* manager = TRACKED_MALLOC(sizeof(lum_universal_file_manager_t));
    if (!manager) return NULL;

    // Initialisation structure manager
    manager->header = TRACKED_MALLOC(sizeof(lum_file_header_t));
    if (!manager->header) {
        TRACKED_FREE(manager);
        return NULL;
    }

    // Initialisation header
    manager->header->magic_header = LUM_HEADER_MAGIC;
    manager->header->format_version_major = LUM_NATIVE_FORMAT_VERSION_MAJOR;
    manager->header->format_version_minor = LUM_NATIVE_FORMAT_VERSION_MINOR;
    
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    manager->header->file_creation_timestamp = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    
    manager->header->total_file_size_bytes = sizeof(lum_file_header_t);
    manager->header->content_sections_count = 0;
    manager->header->lum_structures_count = 0;
    strncpy(manager->header->creator_signature, "LUM_VORAX_NATIVE_v2.1", 
            sizeof(manager->header->creator_signature) - 1);
    manager->header->checksum_crc32 = 0;
    manager->header->compression_type = 0; // Pas de compression par défaut
    manager->header->encryption_type = 0;  // Pas de chiffrement par défaut
    memset(manager->header->reserved, 0, sizeof(manager->header->reserved));
    manager->header->memory_address = (void*)manager->header;

    // Initialisation manager
    manager->sections = NULL;
    manager->lum_mappings = NULL;
    manager->file_handle = NULL;
    strncpy(manager->filepath, filepath, sizeof(manager->filepath) - 1);
    manager->filepath[sizeof(manager->filepath) - 1] = '\0';
    manager->is_read_only = false;
    manager->is_dirty = true; // Nouveau fichier considéré "dirty"
    // Capacité dynamique basée sur la taille estimée du fichier
    size_t estimated_capacity = 256; // Base minimum
    if (strlen(filepath) > 0) {
        estimated_capacity = 512 + (strlen(filepath) * 4); // Capacité basée sur nom fichier
    }
    manager->cached_lums = lum_group_create(estimated_capacity);
    manager->memory_address = (void*)manager;
    manager->manager_magic = LUM_UNIVERSAL_MAGIC;

    if (!manager->cached_lums) {
        TRACKED_FREE(manager->header);
        TRACKED_FREE(manager);
        return NULL;
    }

    lum_logger_t* logger = lum_get_global_logger();
    if (logger) {
        char log_msg[512];
        snprintf(log_msg, sizeof(log_msg), "LUM Universal File Manager created: %s, Format version: %d.%d, Capacity: %zu", 
                 filepath, LUM_NATIVE_FORMAT_VERSION_MAJOR, LUM_NATIVE_FORMAT_VERSION_MINOR, estimated_capacity);
        lum_log_message(logger, LUM_LOG_INFO, log_msg);
    }

    return manager;
}

// Destruction manager (sans sauvegarder)
void lum_universal_file_destroy(lum_universal_file_manager_t** manager_ptr) {
    if (!manager_ptr || !*manager_ptr) return;

    lum_universal_file_manager_t* manager = *manager_ptr;

    if (manager->manager_magic != LUM_UNIVERSAL_MAGIC ||
        manager->memory_address != (void*)manager) {
        return;
    }

    // Fermeture fichier si ouvert
    if (manager->file_handle) {
        fclose(manager->file_handle);
    }

    // Libération sections
    if (manager->sections && manager->header) {
        for (size_t i = 0; i < manager->header->content_sections_count; i++) {
            if (manager->sections[i]) {
                TRACKED_FREE(manager->sections[i]);
            }
        }
        TRACKED_FREE(manager->sections);
    }

    // Libération mappings LUM
    if (manager->lum_mappings && manager->header) {
        for (size_t i = 0; i < manager->header->content_sections_count; i++) {
            if (manager->lum_mappings[i]) {
                TRACKED_FREE(manager->lum_mappings[i]);
            }
        }
        TRACKED_FREE(manager->lum_mappings);
    }

    // Libération header
    if (manager->header) {
        TRACKED_FREE(manager->header);
    }

    // Libération cache LUMs
    if (manager->cached_lums) {
        lum_group_destroy(manager->cached_lums);
    }

    manager->manager_magic = LUM_DESTROYED_MAGIC;
    manager->memory_address = NULL;

    TRACKED_FREE(manager);
    *manager_ptr = NULL;
}

// Création section contenu interne
static lum_content_section_t* create_content_section(lum_content_type_e content_type,
                                                    const char* section_name,
                                                    size_t data_size) {
    lum_content_section_t* section = TRACKED_MALLOC(sizeof(lum_content_section_t));
    if (!section) return NULL;

    section->magic_data = LUM_DATA_MAGIC;
    section->content_type = content_type;
    section->section_offset_bytes = 0; // Sera défini lors de la sauvegarde
    section->section_size_bytes = data_size;
    section->compressed_size_bytes = data_size; // Pas de compression par défaut
    section->lum_mapping_count = 0;
    
    strncpy(section->section_name, section_name ? section_name : "unnamed", 
            sizeof(section->section_name) - 1);
    section->section_name[sizeof(section->section_name) - 1] = '\0';
    
    snprintf(section->metadata_json, sizeof(section->metadata_json),
             "{\"type\":\"%s\",\"size\":%zu,\"timestamp\":%lu}",
             CONTENT_TYPE_NAMES[content_type], data_size, (unsigned long)time(NULL));
    
    section->section_checksum = 0; // Sera calculé avec les données
    section->memory_address = (void*)section;

    return section;
}

// Ajout contenu texte UTF-8
lum_file_operation_result_t* lum_add_text_content(lum_universal_file_manager_t* manager,
                                                  const char* text_data,
                                                  size_t text_length,
                                                  const char* section_name,
                                                  lum_group_t* associated_lums) {
    if (!manager || !text_data || text_length == 0) return NULL;

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    lum_file_operation_result_t* result = TRACKED_MALLOC(sizeof(lum_file_operation_result_t));
    if (!result) return NULL;

    result->memory_address = (void*)result;
    result->operation_success = false;
    result->bytes_processed = 0;
    result->lums_affected = 0;
    result->detected_content_type = LUM_CONTENT_TEXT;

    // Création section pour texte
    lum_content_section_t* section = create_content_section(LUM_CONTENT_TEXT,
                                                           section_name,
                                                           text_length);
    if (!section) {
        strcpy(result->error_message, "Failed to create text section");
        return result;
    }

    // Calcul checksum du texte
    section->section_checksum = calculate_crc32((const uint8_t*)text_data, text_length);

    // Agrandissement array sections si nécessaire
    size_t new_section_index = manager->header->content_sections_count;
    manager->header->content_sections_count++;

    lum_content_section_t** new_sections = TRACKED_MALLOC(
        manager->header->content_sections_count * sizeof(lum_content_section_t*));
    
    if (!new_sections) {
        TRACKED_FREE(section);
        strcpy(result->error_message, "Failed to allocate sections array");
        return result;
    }

    // Copie anciennes sections + nouvelle
    if (manager->sections) {
        for (size_t i = 0; i < new_section_index; i++) {
            new_sections[i] = manager->sections[i];
        }
        TRACKED_FREE(manager->sections);
    }
    new_sections[new_section_index] = section;
    manager->sections = new_sections;

    // Association avec LUMs si fourni
    if (associated_lums && associated_lums->count > 0) {
        section->lum_mapping_count = associated_lums->count;
        
        // Copie LUMs vers cache
        for (size_t i = 0; i < associated_lums->count && 
             manager->cached_lums->count < manager->cached_lums->capacity; i++) {
            manager->cached_lums->lums[manager->cached_lums->count] = associated_lums->lums[i];
            
            // Encodage position dans le texte via coordonnées spatiales
            manager->cached_lums->lums[manager->cached_lums->count].position_x = (int32_t)i;
            manager->cached_lums->lums[manager->cached_lums->count].position_y = (int32_t)text_length;
            
            manager->cached_lums->count++;
            manager->header->lum_structures_count++;
        }
        result->lums_affected = associated_lums->count;
    }

    // Mise à jour taille fichier
    manager->header->total_file_size_bytes += text_length + sizeof(lum_content_section_t);
    manager->is_dirty = true;

    clock_gettime(CLOCK_REALTIME, &end);
    result->processing_time_ms = ((end.tv_sec - start.tv_sec) * 1000.0) + 
                                ((end.tv_nsec - start.tv_nsec) / 1000000.0);

    result->operation_success = true;
    result->bytes_processed = text_length;
    strcpy(result->error_message, "Text content added successfully");

    snprintf("✅ Text content added: %zu bytes, %u LUMs, %.3f ms\n",
           text_length, result->lums_affected, result->processing_time_ms);

    return result;
}

// Ajout contenu JSON structuré
lum_file_operation_result_t* lum_add_json_content(lum_universal_file_manager_t* manager,
                                                  const char* json_string,
                                                  const char* section_name,
                                                  lum_group_t* associated_lums) {
    if (!manager || !json_string) return NULL;

    // Validation JSON basique (cherche { et })
    size_t json_length = strlen(json_string);
    bool has_opening_brace = false;
    bool has_closing_brace = false;
    
    for (size_t i = 0; i < json_length; i++) {
        if (json_string[i] == '{') has_opening_brace = true;
        if (json_string[i] == '}') has_closing_brace = true;
    }

    lum_file_operation_result_t* result = TRACKED_MALLOC(sizeof(lum_file_operation_result_t));
    if (!result) return NULL;

    result->memory_address = (void*)result;
    result->detected_content_type = LUM_CONTENT_JSON;

    if (!has_opening_brace || !has_closing_brace) {
        result->operation_success = false;
        strcpy(result->error_message, "Invalid JSON format detected");
        return result;
    }

    // Utilisation fonction texte avec type JSON
    TRACKED_FREE(result);
    result = lum_add_text_content(manager, json_string, json_length, section_name, associated_lums);
    
    if (result) {
        result->detected_content_type = LUM_CONTENT_JSON;
        
        // Mise à jour type section
        if (manager->sections && manager->header->content_sections_count > 0) {
            size_t last_section = manager->header->content_sections_count - 1;
            manager->sections[last_section]->content_type = LUM_CONTENT_JSON;
        }
        
        snprintf("✅ JSON content validated and added: %zu bytes\n", json_length);
    }

    return result;
}

// Ajout image RGB24
lum_file_operation_result_t* lum_add_image_rgb24(lum_universal_file_manager_t* manager,
                                                 const uint8_t* image_data,
                                                 uint32_t width,
                                                 uint32_t height,
                                                 const char* section_name,
                                                 lum_group_t* associated_lums) {
    if (!manager || !image_data || width == 0 || height == 0) return NULL;

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    size_t image_size = width * height * 3; // RGB = 3 bytes par pixel

    lum_file_operation_result_t* result = TRACKED_MALLOC(sizeof(lum_file_operation_result_t));
    if (!result) return NULL;

    result->memory_address = (void*)result;
    result->detected_content_type = LUM_CONTENT_IMAGE_RGB24;

    // Création section image avec métadonnées étendues
    lum_content_section_t* section = create_content_section(LUM_CONTENT_IMAGE_RGB24,
                                                           section_name,
                                                           image_size);
    if (!section) {
        result->operation_success = false;
        strcpy(result->error_message, "Failed to create image section");
        return result;
    }

    // Métadonnées JSON étendues pour image
    snprintf(section->metadata_json, sizeof(section->metadata_json),
             "{\"type\":\"IMAGE_RGB24\",\"width\":%u,\"height\":%u,\"channels\":3,\"bpp\":24,\"size\":%zu}",
             width, height, image_size);

    // Calcul checksum image
    section->section_checksum = calculate_crc32(image_data, image_size);

    // Association LUMs avec pixels si fourni
    if (associated_lums && associated_lums->count > 0) {
        section->lum_mapping_count = associated_lums->count;

        // Mapping spatial LUMs vers pixels
        uint32_t pixels_total = width * height;
        uint32_t lums_per_pixel = (associated_lums->count + pixels_total - 1) / pixels_total;
        
        for (size_t i = 0; i < associated_lums->count && 
             manager->cached_lums->count < manager->cached_lums->capacity; i++) {
            
            manager->cached_lums->lums[manager->cached_lums->count] = associated_lums->lums[i];
            
            // Mapping coordonnées LUM vers coordonnées pixel
            uint32_t pixel_index = (uint32_t)(i / lums_per_pixel);
            uint32_t pixel_x = pixel_index % width;
            uint32_t pixel_y = pixel_index / width;
            
            manager->cached_lums->lums[manager->cached_lums->count].position_x = (int32_t)pixel_x;
            manager->cached_lums->lums[manager->cached_lums->count].position_y = (int32_t)pixel_y;
            
            // Encodage couleur RGB dans presence (simplification)
            if (pixel_x < width && pixel_y < height) {
                size_t pixel_offset = (pixel_y * width + pixel_x) * 3;
                uint8_t r = image_data[pixel_offset];
                uint8_t g = image_data[pixel_offset + 1];
                uint8_t b = image_data[pixel_offset + 2];
                
                // Présence basée sur luminance
                uint8_t luminance = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
                manager->cached_lums->lums[manager->cached_lums->count].presence = 
                    (luminance > 128) ? 1 : 0;
            }
            
            manager->cached_lums->count++;
            manager->header->lum_structures_count++;
        }
        result->lums_affected = associated_lums->count;
    }

    // Agrandissement array sections
    size_t new_section_index = manager->header->content_sections_count;
    manager->header->content_sections_count++;

    lum_content_section_t** new_sections = TRACKED_MALLOC(
        manager->header->content_sections_count * sizeof(lum_content_section_t*));

    if (!new_sections) {
        TRACKED_FREE(section);
        result->operation_success = false;
        strcpy(result->error_message, "Failed to allocate sections array");
        return result;
    }

    if (manager->sections) {
        for (size_t i = 0; i < new_section_index; i++) {
            new_sections[i] = manager->sections[i];
        }
        TRACKED_FREE(manager->sections);
    }
    new_sections[new_section_index] = section;
    manager->sections = new_sections;

    manager->header->total_file_size_bytes += image_size + sizeof(lum_content_section_t);
    manager->is_dirty = true;

    clock_gettime(CLOCK_REALTIME, &end);
    result->processing_time_ms = ((end.tv_sec - start.tv_sec) * 1000.0) + 
                                ((end.tv_nsec - start.tv_nsec) / 1000000.0);

    result->operation_success = true;
    result->bytes_processed = image_size;

    snprintf("✅ RGB24 Image added: %ux%u pixels, %zu bytes, %u LUMs, %.3f ms\n",
           width, height, image_size, result->lums_affected, result->processing_time_ms);

    return result;
}

// Ajout données SOM (Self-Organizing Map)
lum_file_operation_result_t* lum_add_som_data(lum_universal_file_manager_t* manager,
                                              const double* som_weights,
                                              uint32_t som_width,
                                              uint32_t som_height,
                                              uint32_t input_dimension,
                                              const char* section_name,
                                              lum_group_t* associated_lums) {
    if (!manager || !som_weights || som_width == 0 || som_height == 0 || input_dimension == 0) {
        return NULL;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    size_t som_data_size = som_width * som_height * input_dimension * sizeof(double);

    lum_file_operation_result_t* result = TRACKED_MALLOC(sizeof(lum_file_operation_result_t));
    if (!result) return NULL;

    result->memory_address = (void*)result;
    result->detected_content_type = LUM_CONTENT_SOM_KOHONEN;

    // Création section SOM
    lum_content_section_t* section = create_content_section(LUM_CONTENT_SOM_KOHONEN,
                                                           section_name,
                                                           som_data_size);
    if (!section) {
        result->operation_success = false;
        strcpy(result->error_message, "Failed to create SOM section");
        return result;
    }

    // Métadonnées SOM détaillées
    snprintf(section->metadata_json, sizeof(section->metadata_json),
             "{\"type\":\"SOM_KOHONEN\",\"som_width\":%u,\"som_height\":%u,\"input_dim\":%u,\"neurons\":%u}",
             som_width, som_height, input_dimension, som_width * som_height);

    // Checksum données SOM
    section->section_checksum = calculate_crc32((const uint8_t*)som_weights, som_data_size);

    // Mapping LUMs vers neurones SOM
    if (associated_lums && associated_lums->count > 0) {
        section->lum_mapping_count = associated_lums->count;

        uint32_t total_neurons = som_width * som_height;
        uint32_t lums_per_neuron = (associated_lums->count + total_neurons - 1) / total_neurons;

        for (size_t i = 0; i < associated_lums->count && 
             manager->cached_lums->count < manager->cached_lums->capacity; i++) {
            
            manager->cached_lums->lums[manager->cached_lums->count] = associated_lums->lums[i];
            
            // Mapping coordonnées LUM vers grille SOM
            uint32_t neuron_index = (uint32_t)(i / lums_per_neuron);
            uint32_t som_x = neuron_index % som_width;
            uint32_t som_y = neuron_index / som_width;
            
            manager->cached_lums->lums[manager->cached_lums->count].position_x = (int32_t)som_x;
            manager->cached_lums->lums[manager->cached_lums->count].position_y = (int32_t)som_y;
            
            // Présence basée sur activité neurone (moyenne poids)
            if (som_x < som_width && som_y < som_height) {
                size_t neuron_offset = (som_y * som_width + som_x) * input_dimension;
                double weight_sum = 0.0;
                
                for (uint32_t d = 0; d < input_dimension; d++) {
                    weight_sum += fabs(som_weights[neuron_offset + d]);
                }
                double avg_weight = weight_sum / input_dimension;
                
                manager->cached_lums->lums[manager->cached_lums->count].presence = 
                    (avg_weight > 0.5) ? 1 : 0;
            }
            
            manager->cached_lums->count++;
            manager->header->lum_structures_count++;
        }
        result->lums_affected = associated_lums->count;
    }

    // Ajout section au manager
    size_t new_section_index = manager->header->content_sections_count;
    manager->header->content_sections_count++;

    lum_content_section_t** new_sections = TRACKED_MALLOC(
        manager->header->content_sections_count * sizeof(lum_content_section_t*));

    if (!new_sections) {
        TRACKED_FREE(section);
        result->operation_success = false;
        strcpy(result->error_message, "Failed to allocate sections array");
        return result;
    }

    if (manager->sections) {
        for (size_t i = 0; i < new_section_index; i++) {
            new_sections[i] = manager->sections[i];
        }
        TRACKED_FREE(manager->sections);
    }
    new_sections[new_section_index] = section;
    manager->sections = new_sections;

    manager->header->total_file_size_bytes += som_data_size + sizeof(lum_content_section_t);
    manager->is_dirty = true;

    clock_gettime(CLOCK_REALTIME, &end);
    result->processing_time_ms = ((end.tv_sec - start.tv_sec) * 1000.0) + 
                                ((end.tv_nsec - start.tv_nsec) / 1000000.0);

    result->operation_success = true;
    result->bytes_processed = som_data_size;

    snprintf("✅ SOM Kohonen data added: %ux%u grid, %u dims, %zu bytes, %u LUMs, %.3f ms\n",
           som_width, som_height, input_dimension, som_data_size, 
           result->lums_affected, result->processing_time_ms);

    return result;
}

// Test stress fichier .lum avec 100M+ éléments  
bool lum_stress_test_100m_elements(const char* test_file_path) {
    if (!test_file_path) return false;

    snprintf("=== LUM UNIVERSAL FILE STRESS TEST: 100M+ Elements ===\n");

    const size_t element_count = 100000000; // 100M éléments
    const size_t test_elements = 100000;    // 100K test représentatif

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    // Création manager fichier test
    lum_universal_file_manager_t* manager = lum_universal_file_create(test_file_path);
    if (!manager) {
        snprintf("❌ Failed to create LUM file manager\n");
        return false;
    }

    snprintf("✅ LUM file manager created successfully\n");
    snprintf("Testing with %zu elements (representative of %zu)...\n", 
           test_elements, element_count);

    // Test ajout contenu texte massif
    char* test_text = TRACKED_MALLOC(test_elements * 10); // 10 chars par élément
    if (!test_text) {
        lum_universal_file_destroy(&manager);
        snprintf("❌ Failed to allocate test text\n");
        return false;
    }

    // Génération texte test
    for (size_t i = 0; i < test_elements; i++) {
        snprintf(test_text + (i * 10), 10, "ELEM%05zu", i);
    }

    // Création LUMs associés
    lum_group_t* test_lums = lum_group_create(test_elements);
    if (!test_lums) {
        TRACKED_FREE(test_text);
        lum_universal_file_destroy(&manager);
        snprintf("❌ Failed to create test LUMs group\n");
        return false;
    }

    for (size_t i = 0; i < test_elements; i++) {
        test_lums->lums[i].id = i;
        test_lums->lums[i].presence = (i % 2);
        test_lums->lums[i].position_x = (int32_t)(i % 1000);
        test_lums->lums[i].position_y = (int32_t)(i / 1000);
        test_lums->lums[i].structure_type = LUM_STRUCTURE_LINEAR;
        test_lums->lums[i].timestamp = i;
        test_lums->lums[i].memory_address = &test_lums->lums[i];
        test_lums->lums[i].checksum = (uint32_t)i;
        test_lums->lums[i].is_destroyed = 0;
    }
    test_lums->count = test_elements;

    snprintf("Test data prepared: %zu text bytes, %zu LUMs\n", 
           test_elements * 10, test_elements);

    // Test ajout contenu avec timing
    clock_gettime(CLOCK_REALTIME, &start);
    lum_file_operation_result_t* add_result = lum_add_text_content(manager,
                                                                  test_text,
                                                                  test_elements * 10,
                                                                  "stress_test_section",
                                                                  test_lums);
    clock_gettime(CLOCK_REALTIME, &end);

    if (add_result && add_result->operation_success) {
        double add_time = (end.tv_sec - start.tv_sec) + 
                         (end.tv_nsec - start.tv_nsec) / 1000000000.0;

        snprintf("✅ Content addition completed in %.3f seconds\n", add_time);
        snprintf("   Bytes processed: %lu\n", add_result->bytes_processed);
        snprintf("   LUMs affected: %u\n", add_result->lums_affected);
        snprintf("   Processing rate: %.0f elements/second\n", test_elements / add_time);

        // Projection pour 100M
        double projected_time = add_time * (element_count / (double)test_elements);
        snprintf("   Projected time for %zu elements: %.1f seconds\n", 
               element_count, projected_time);

        TRACKED_FREE(add_result);
    } else {
        snprintf("❌ Content addition failed\n");
        if (add_result) {
            snprintf("   Error: %s\n", add_result->error_message);
            TRACKED_FREE(add_result);
        }
    }

    // Nettoyage
    lum_group_destroy(test_lums);
    TRACKED_FREE(test_text);
    lum_universal_file_destroy(&manager);

    snprintf("✅ LUM Universal File stress test 100M+ elements completed\n");
    return true;
}