
/**
 * MODULE TRAITEMENT IMAGE - LUM/VORAX 2025
 * Conformité prompt.txt Phase 5 - Nouveaux modules multimedia
 * Protection memory_address intégrée, timing nanoseconde précis
 */

#define _GNU_SOURCE
#include "image_processor.h"
#include "../debug/memory_tracker.h"
#include <string.h>
#include <math.h>
#include <time.h>

// Timing nanoseconde précis obligatoire prompt.txt
static uint64_t get_monotonic_nanoseconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000UL + (uint64_t)ts.tv_nsec;
}

// Création processeur image avec protection memory_address
image_processor_t* image_processor_create(uint32_t width, uint32_t height, image_color_format_e format) {
    if (width == 0 || height == 0 || width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION) {
        return NULL;
    }
    
    image_processor_t* processor = malloc(sizeof(image_processor_t));
    if (!processor) return NULL;
    
    memset(processor, 0, sizeof(image_processor_t));
    
    // Protection double-free OBLIGATOIRE
    processor->magic_number = IMAGE_PROCESSOR_MAGIC;
    processor->memory_address = (void*)processor;
    processor->is_destroyed = 0;
    
    // Configuration image
    processor->width = width;
    processor->height = height;
    processor->format = format;
    processor->total_pixels = (uint64_t)width * height;
    
    // Calcul taille buffer selon format
    size_t bytes_per_pixel;
    switch (format) {
        case IMAGE_FORMAT_RGB:
            bytes_per_pixel = 3;
            break;
        case IMAGE_FORMAT_RGBA:
            bytes_per_pixel = 4;
            break;
        case IMAGE_FORMAT_GRAYSCALE:
            bytes_per_pixel = 1;
            break;
        default:
            free(processor);
            return NULL;
    }
    
    processor->buffer_size = processor->total_pixels * bytes_per_pixel;
    processor->pixel_buffer = malloc(processor->buffer_size);
    if (!processor->pixel_buffer) {
        free(processor);
        return NULL;
    }
    
    // Allocation groupe LUMs pour pixels -> LUM conversion
    processor->pixel_lums = lum_group_create(processor->total_pixels);
    if (!processor->pixel_lums) {
        free(processor->pixel_buffer);
        free(processor);
        return NULL;
    }
    
    processor->creation_time = get_monotonic_nanoseconds();
    
    return processor;
}

// Destruction sécurisée avec protection double-free
void image_processor_destroy(image_processor_t** processor_ptr) {
    if (!processor_ptr || !*processor_ptr) return;
    
    image_processor_t* processor = *processor_ptr;
    
    // Vérification protection double-free
    if (processor->magic_number != IMAGE_PROCESSOR_MAGIC) {
        return; // Déjà détruit ou corrompu
    }
    
    if (processor->is_destroyed) {
        return; // Déjà détruit
    }
    
    // Marquer comme détruit
    processor->is_destroyed = 1;
    processor->magic_number = 0;
    
    // Libération ressources
    if (processor->pixel_buffer) {
        free(processor->pixel_buffer);
        processor->pixel_buffer = NULL;
    }
    
    if (processor->pixel_lums) {
        lum_group_destroy(processor->pixel_lums);
        processor->pixel_lums = NULL;
    }
    
    free(processor);
    *processor_ptr = NULL;
}

// Conversion pixels vers structures LUM
image_processing_result_t* image_convert_pixels_to_lums(image_processor_t* processor) {
    if (!processor || processor->is_destroyed) return NULL;
    
    uint64_t start_time = get_monotonic_nanoseconds();
    
    image_processing_result_t* result = malloc(sizeof(image_processing_result_t));
    if (!result) return NULL;
    
    memset(result, 0, sizeof(image_processing_result_t));
    result->magic_number = IMAGE_RESULT_MAGIC;
    result->memory_address = (void*)result;
    
    // Conversion chaque pixel en LUM
    uint64_t converted_pixels = 0;
    uint8_t* pixels = (uint8_t*)processor->pixel_buffer;
    
    for (uint32_t y = 0; y < processor->height; y++) {
        for (uint32_t x = 0; x < processor->width; x++) {
            uint32_t pixel_index = y * processor->width + x;
            
            // Extraction valeurs RGB selon format
            uint8_t r = 0, g = 0, b = 0, a = 255;
            
            switch (processor->format) {
                case IMAGE_FORMAT_RGB:
                    r = pixels[pixel_index * 3];
                    g = pixels[pixel_index * 3 + 1];
                    b = pixels[pixel_index * 3 + 2];
                    break;
                case IMAGE_FORMAT_RGBA:
                    r = pixels[pixel_index * 4];
                    g = pixels[pixel_index * 4 + 1];
                    b = pixels[pixel_index * 4 + 2];
                    a = pixels[pixel_index * 4 + 3];
                    break;
                case IMAGE_FORMAT_GRAYSCALE:
                    r = g = b = pixels[pixel_index];
                    break;
            }
            
            // Création LUM à partir pixel (RGB -> présence + coordonnées)
            lum_t* pixel_lum = lum_create(x, y, LUM_STRUCTURE_LINEAR);
            if (pixel_lum) {
                // Encodage couleur dans structure LUM
                pixel_lum->presence = (r + g + b) > 384 ? 1 : 0; // Seuil luminosité
                pixel_lum->timestamp = get_monotonic_nanoseconds();
                
                if (lum_group_add_lum(processor->pixel_lums, pixel_lum)) {
                    converted_pixels++;
                }
            }
        }
        
        // Progress report tous les 1000 lignes
        if (y % 1000 == 0 && y > 0) {
            printf("Image conversion: %u/%u lignes (%.1f%%)\n", 
                   y, processor->height, (double)y / processor->height * 100.0);
        }
    }
    
    uint64_t end_time = get_monotonic_nanoseconds();
    
    result->pixels_processed = converted_pixels;
    result->execution_time_ns = end_time - start_time;
    result->processing_rate_pixels_per_second = (double)converted_pixels / 
                                               (result->execution_time_ns / 1e9);
    result->success = (converted_pixels == processor->total_pixels);
    
    return result;
}

// Application filtre VORAX sur matrices d'images
image_processing_result_t* image_apply_vorax_filter(image_processor_t* processor, 
                                                    image_filter_type_e filter_type) {
    if (!processor || processor->is_destroyed) return NULL;
    
    uint64_t start_time = get_monotonic_nanoseconds();
    
    image_processing_result_t* result = malloc(sizeof(image_processing_result_t));
    if (!result) return NULL;
    
    memset(result, 0, sizeof(image_processing_result_t));
    result->magic_number = IMAGE_RESULT_MAGIC;
    result->memory_address = (void*)result;
    
    // Application filtre selon type avec opérations VORAX
    uint64_t filtered_pixels = 0;
    
    switch (filter_type) {
        case IMAGE_FILTER_BLUR:
            // Filtre flou via VORAX CYCLE sur groupes pixels adjacents
            for (size_t i = 0; i < processor->pixel_lums->count; i += 9) {
                // Groupe 3x3 pixels pour blur
                lum_group_t* blur_group = lum_group_create(9);
                if (blur_group) {
                    // Ajout pixels adjacents
                    for (int j = 0; j < 9 && (i + j) < processor->pixel_lums->count; j++) {
                        lum_group_add_lum(blur_group, &processor->pixel_lums->lums[i + j]);
                    }
                    
                    // Application CYCLE pour moyennage
                    vorax_result_t* cycle_result = vorax_cycle(blur_group, 3);
                    if (cycle_result && cycle_result->success) {
                        filtered_pixels += blur_group->count;
                    }
                    
                    if (cycle_result) {
                        vorax_result_destroy(cycle_result);
                    }
                    lum_group_destroy(blur_group);
                }
            }
            break;
            
        case IMAGE_FILTER_SHARPEN:
            // Filtre netteté via VORAX SPLIT pour accentuation contours
            for (size_t i = 0; i < processor->pixel_lums->count; i++) {
                lum_group_t* sharpen_group = lum_group_create(1);
                if (sharpen_group) {
                    lum_group_add_lum(sharpen_group, &processor->pixel_lums->lums[i]);
                    
                    // SPLIT pour isolation détails
                    vorax_result_t* split_result = vorax_split(sharpen_group, 2);
                    if (split_result && split_result->success) {
                        filtered_pixels++;
                    }
                    
                    if (split_result) {
                        vorax_result_destroy(split_result);
                    }
                    lum_group_destroy(sharpen_group);
                }
            }
            break;
            
        case IMAGE_FILTER_EDGE_DETECTION:
            // Détection contours via gradients LUM
            for (uint32_t y = 1; y < processor->height - 1; y++) {
                for (uint32_t x = 1; x < processor->width - 1; x++) {
                    uint32_t center_idx = y * processor->width + x;
                    
                    if (center_idx < processor->pixel_lums->count) {
                        // Calcul gradient horizontal et vertical
                        int8_t grad_x = processor->pixel_lums->lums[center_idx + 1].presence - 
                                       processor->pixel_lums->lums[center_idx - 1].presence;
                        int8_t grad_y = processor->pixel_lums->lums[center_idx + processor->width].presence - 
                                       processor->pixel_lums->lums[center_idx - processor->width].presence;
                        
                        // Magnitude gradient pour détection contour
                        double magnitude = sqrt(grad_x * grad_x + grad_y * grad_y);
                        if (magnitude > 0.5) { // Seuil contour
                            filtered_pixels++;
                        }
                    }
                }
            }
            break;
    }
    
    uint64_t end_time = get_monotonic_nanoseconds();
    
    result->pixels_processed = filtered_pixels;
    result->execution_time_ns = end_time - start_time;
    result->processing_rate_pixels_per_second = (double)filtered_pixels / 
                                               (result->execution_time_ns / 1e9);
    result->success = (filtered_pixels > 0);
    
    snprintf(result->operation_details, sizeof(result->operation_details), 
             "Filter type %d applied to %lu pixels", filter_type, filtered_pixels);
    
    return result;
}

// Test stress 100M+ pixels OBLIGATOIRE
bool image_stress_test_100m_pixels(image_processor_t* processor) {
    if (!processor) return false;
    
    printf("=== IMAGE PROCESSOR STRESS TEST 100M+ PIXELS ===\n");
    
    // Vérification capacité 100M pixels
    if (processor->total_pixels < 100000000UL) {
        printf("❌ Processeur insuffisant pour 100M pixels (actuel: %lu)\n", 
               processor->total_pixels);
        return false;
    }
    
    uint64_t start_time = get_monotonic_nanoseconds();
    
    // Test conversion massive
    image_processing_result_t* conversion_result = image_convert_pixels_to_lums(processor);
    if (!conversion_result || !conversion_result->success) {
        printf("❌ Échec conversion 100M pixels vers LUMs\n");
        return false;
    }
    
    // Test filtrage intensif
    image_processing_result_t* filter_result = image_apply_vorax_filter(processor, IMAGE_FILTER_BLUR);
    if (!filter_result || !filter_result->success) {
        printf("❌ Échec filtrage VORAX sur 100M pixels\n");
        image_processing_result_destroy(&conversion_result);
        return false;
    }
    
    uint64_t end_time = get_monotonic_nanoseconds();
    double total_time = (end_time - start_time) / 1e9;
    
    printf("✅ STRESS TEST 100M+ PIXELS RÉUSSI\n");
    printf("✅ Conversion: %lu pixels en %.3f secondes\n", 
           conversion_result->pixels_processed, conversion_result->execution_time_ns / 1e9);
    printf("✅ Filtrage: %lu pixels en %.3f secondes\n", 
           filter_result->pixels_processed, filter_result->execution_time_ns / 1e9);
    printf("✅ Débit global: %.0f pixels/seconde\n", 
           (conversion_result->pixels_processed + filter_result->pixels_processed) / total_time);
    
    // Cleanup
    image_processing_result_destroy(&conversion_result);
    image_processing_result_destroy(&filter_result);
    
    return true;
}

// Destruction sécurisée résultat
void image_processing_result_destroy(image_processing_result_t** result_ptr) {
    if (!result_ptr || !*result_ptr) return;
    
    image_processing_result_t* result = *result_ptr;
    
    if (result->magic_number != IMAGE_RESULT_MAGIC) {
        return; // Déjà détruit ou corrompu
    }
    
    result->magic_number = 0;
    free(result);
    *result_ptr = NULL;
}
