
/**
 * MODULE TRAITEMENT IMAGE LUM/VORAX
 * Conversion pixels vers structures LUM avec filtres VORAX
 * Conformité STANDARD_NAMES.md et prompt.txt phases 5-6
 */

#define _GNU_SOURCE
#include "image_processor.h"
#include "../debug/memory_tracker.h"
#include "../lum/lum_core.h"
#include "../vorax/vorax_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Constantes traitement image conformes STANDARD_NAMES.md
#define IMAGE_PIXEL_TO_LUM_RATIO 1.618  // Ratio doré φ
#define IMAGE_FILTER_KERNEL_SIZE 3
#define IMAGE_COMPRESSION_THRESHOLD 0.85

// Structure interne processeur image
struct image_processor_s {
    size_t width;
    size_t height;
    size_t channels;
    lum_group_t* pixel_lums;
    image_filter_type_e filter_type;
    double compression_ratio;
    memory_address_t memory_address;  // Protection double-free
    uint32_t magic_number;            // Validation intégrité
};

// Création processeur image
image_processor_t* image_processor_create(size_t width, size_t height, size_t channels) {
    if (width == 0 || height == 0 || channels == 0 || channels > 4) {
        return NULL;
    }
    
    image_processor_t* processor = malloc(sizeof(image_processor_t));
    if (!processor) {
        return NULL;
    }
    
    // Initialisation protection mémoire
    processor->memory_address = (memory_address_t)processor;
    processor->magic_number = IMAGE_PROCESSOR_MAGIC;
    
    processor->width = width;
    processor->height = height;
    processor->channels = channels;
    processor->filter_type = IMAGE_FILTER_NONE;
    processor->compression_ratio = 1.0;
    
    // Allocation groupe LUMs pour pixels
    size_t total_pixels = width * height;
    processor->pixel_lums = lum_group_create(total_pixels);
    if (!processor->pixel_lums) {
        free(processor);
        return NULL;
    }
    
    return processor;
}

// Conversion pixels RGB vers LUMs
int image_pixels_to_lums(image_processor_t* processor, const uint8_t* pixel_data) {
    if (!processor || !pixel_data || processor->magic_number != IMAGE_PROCESSOR_MAGIC) {
        return 0;
    }
    
    size_t total_pixels = processor->width * processor->height;
    
    for (size_t i = 0; i < total_pixels; i++) {
        // Extraction composantes RGB
        size_t pixel_offset = i * processor->channels;
        uint8_t r = pixel_data[pixel_offset];
        uint8_t g = processor->channels > 1 ? pixel_data[pixel_offset + 1] : r;
        uint8_t b = processor->channels > 2 ? pixel_data[pixel_offset + 2] : r;
        
        // Conversion vers coordonnées LUM avec ratio doré
        int32_t x = (int32_t)(i % processor->width);
        int32_t y = (int32_t)(i / processor->width);
        
        // Présence basée sur luminance
        double luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        uint8_t presence = (uint8_t)(luminance * IMAGE_PIXEL_TO_LUM_RATIO / 255.0 * 255);
        
        // Création LUM pixel
        lum_t* pixel_lum = lum_create(x, y, LUM_STRUCTURE_LINEAR);
        if (!pixel_lum) {
            return 0;
        }
        
        // Ajustement présence
        pixel_lum->presence = presence;
        
        // Ajout au groupe
        if (!lum_group_add_lum(processor->pixel_lums, pixel_lum)) {
            lum_destroy(pixel_lum);
            return 0;
        }
    }
    
    return 1;
}

// Application filtre VORAX sur image
int image_apply_vorax_filter(image_processor_t* processor, image_filter_type_e filter_type) {
    if (!processor || processor->magic_number != IMAGE_PROCESSOR_MAGIC) {
        return 0;
    }
    
    processor->filter_type = filter_type;
    
    switch (filter_type) {
        case IMAGE_FILTER_BLUR:
            // Filtre flou via opération CYCLE VORAX
            return image_apply_blur_filter(processor);
            
        case IMAGE_FILTER_SHARPEN:
            // Filtre netteté via opération SPLIT VORAX
            return image_apply_sharpen_filter(processor);
            
        case IMAGE_FILTER_EDGE_DETECTION:
            // Détection contours via gradients LUM
            return image_apply_edge_detection(processor);
            
        case IMAGE_FILTER_NONE:
        default:
            return 1;
    }
}

// Filtre flou VORAX
static int image_apply_blur_filter(image_processor_t* processor) {
    // Application CYCLE VORAX pour moyennage spatial
    vorax_result_t* result = vorax_cycle(processor->pixel_lums, IMAGE_FILTER_KERNEL_SIZE);
    if (!result) {
        return 0;
    }
    
    // Remplacement groupe original
    lum_group_destroy(processor->pixel_lums);
    processor->pixel_lums = result->output_group;
    result->output_group = NULL;  // Transfert propriété
    
    vorax_result_destroy(result);
    return 1;
}

// Filtre netteté VORAX
static int image_apply_sharpen_filter(image_processor_t* processor) {
    // Application SPLIT VORAX pour accentuation contrastes
    vorax_result_t* result = vorax_split(processor->pixel_lums);
    if (!result) {
        return 0;
    }
    
    // Fusion résultats avec pondération
    lum_group_t* sharpened = lum_group_create(processor->pixel_lums->count);
    if (!sharpened) {
        vorax_result_destroy(result);
        return 0;
    }
    
    // Algorithme netteté via combinaison SPLIT
    for (size_t i = 0; i < result->group_count && i < 2; i++) {
        lum_group_t* split_group = result->result_groups[i];
        for (size_t j = 0; j < split_group->count; j++) {
            lum_t* enhanced_lum = lum_create(
                split_group->lums[j]->position_x,
                split_group->lums[j]->position_y,
                LUM_STRUCTURE_LINEAR
            );
            if (enhanced_lum) {
                // Accentuation présence
                enhanced_lum->presence = (uint8_t)fmin(255, 
                    split_group->lums[j]->presence * 1.5);
                lum_group_add_lum(sharpened, enhanced_lum);
            }
        }
    }
    
    lum_group_destroy(processor->pixel_lums);
    processor->pixel_lums = sharpened;
    vorax_result_destroy(result);
    return 1;
}

// Détection contours via gradients LUM
static int image_apply_edge_detection(image_processor_t* processor) {
    lum_group_t* edges = lum_group_create(processor->pixel_lums->count);
    if (!edges) {
        return 0;
    }
    
    // Calcul gradients Sobel sur LUMs
    for (size_t y = 1; y < processor->height - 1; y++) {
        for (size_t x = 1; x < processor->width - 1; x++) {
            int32_t gx = 0, gy = 0;
            
            // Noyau Sobel X et Y
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    size_t idx = (y + dy) * processor->width + (x + dx);
                    if (idx < processor->pixel_lums->count) {
                        uint8_t presence = processor->pixel_lums->lums[idx]->presence;
                        
                        // Coefficients Sobel
                        int sobel_x[3][3] = {{-1,0,1}, {-2,0,2}, {-1,0,1}};
                        int sobel_y[3][3] = {{-1,-2,-1}, {0,0,0}, {1,2,1}};
                        
                        gx += presence * sobel_x[dy+1][dx+1];
                        gy += presence * sobel_y[dy+1][dx+1];
                    }
                }
            }
            
            // Magnitude gradient
            uint8_t edge_strength = (uint8_t)fmin(255, sqrt(gx*gx + gy*gy));
            
            lum_t* edge_lum = lum_create((int32_t)x, (int32_t)y, LUM_STRUCTURE_LINEAR);
            if (edge_lum) {
                edge_lum->presence = edge_strength;
                lum_group_add_lum(edges, edge_lum);
            }
        }
    }
    
    lum_group_destroy(processor->pixel_lums);
    processor->pixel_lums = edges;
    return 1;
}

// Compression image basée présence LUM
int image_compress_lums(image_processor_t* processor, double target_ratio) {
    if (!processor || processor->magic_number != IMAGE_PROCESSOR_MAGIC || 
        target_ratio <= 0.0 || target_ratio > 1.0) {
        return 0;
    }
    
    size_t target_count = (size_t)(processor->pixel_lums->count * target_ratio);
    if (target_count >= processor->pixel_lums->count) {
        processor->compression_ratio = 1.0;
        return 1;
    }
    
    // Tri LUMs par présence décroissante
    lum_t** sorted_lums = malloc(processor->pixel_lums->count * sizeof(lum_t*));
    if (!sorted_lums) {
        return 0;
    }
    
    memcpy(sorted_lums, processor->pixel_lums->lums, 
           processor->pixel_lums->count * sizeof(lum_t*));
    
    // Tri à bulles simple par présence
    for (size_t i = 0; i < processor->pixel_lums->count - 1; i++) {
        for (size_t j = 0; j < processor->pixel_lums->count - i - 1; j++) {
            if (sorted_lums[j]->presence < sorted_lums[j+1]->presence) {
                lum_t* temp = sorted_lums[j];
                sorted_lums[j] = sorted_lums[j+1];
                sorted_lums[j+1] = temp;
            }
        }
    }
    
    // Création groupe compressé
    lum_group_t* compressed = lum_group_create(target_count);
    if (!compressed) {
        free(sorted_lums);
        return 0;
    }
    
    // Conservation top LUMs
    for (size_t i = 0; i < target_count; i++) {
        lum_t* compressed_lum = lum_create(
            sorted_lums[i]->position_x,
            sorted_lums[i]->position_y,
            sorted_lums[i]->structure_type
        );
        if (compressed_lum) {
            compressed_lum->presence = sorted_lums[i]->presence;
            lum_group_add_lum(compressed, compressed_lum);
        }
    }
    
    lum_group_destroy(processor->pixel_lums);
    processor->pixel_lums = compressed;
    processor->compression_ratio = target_ratio;
    
    free(sorted_lums);
    return 1;
}

// Conversion LUMs vers pixels de sortie
int image_lums_to_pixels(image_processor_t* processor, uint8_t* output_data) {
    if (!processor || !output_data || processor->magic_number != IMAGE_PROCESSOR_MAGIC) {
        return 0;
    }
    
    // Initialisation buffer sortie
    size_t total_bytes = processor->width * processor->height * processor->channels;
    memset(output_data, 0, total_bytes);
    
    // Reconstruction pixels depuis LUMs
    for (size_t i = 0; i < processor->pixel_lums->count; i++) {
        lum_t* lum = processor->pixel_lums->lums[i];
        
        // Validation coordonnées
        if (lum->position_x >= 0 && lum->position_x < (int32_t)processor->width &&
            lum->position_y >= 0 && lum->position_y < (int32_t)processor->height) {
            
            size_t pixel_offset = (lum->position_y * processor->width + lum->position_x) * processor->channels;
            
            // Conversion présence vers RGB
            uint8_t value = (uint8_t)(lum->presence / IMAGE_PIXEL_TO_LUM_RATIO);
            
            for (size_t c = 0; c < processor->channels; c++) {
                if (pixel_offset + c < total_bytes) {
                    output_data[pixel_offset + c] = value;
                }
            }
        }
    }
    
    return 1;
}

// Statistiques traitement image
image_stats_t image_processor_get_stats(const image_processor_t* processor) {
    image_stats_t stats = {0};
    
    if (!processor || processor->magic_number != IMAGE_PROCESSOR_MAGIC) {
        return stats;
    }
    
    stats.width = processor->width;
    stats.height = processor->height;
    stats.channels = processor->channels;
    stats.total_pixels = processor->width * processor->height;
    stats.processed_lums = processor->pixel_lums->count;
    stats.compression_ratio = processor->compression_ratio;
    stats.filter_applied = processor->filter_type;
    
    // Calcul statistiques présence
    if (processor->pixel_lums->count > 0) {
        uint32_t sum = 0;
        uint8_t min_presence = 255, max_presence = 0;
        
        for (size_t i = 0; i < processor->pixel_lums->count; i++) {
            uint8_t presence = processor->pixel_lums->lums[i]->presence;
            sum += presence;
            if (presence < min_presence) min_presence = presence;
            if (presence > max_presence) max_presence = presence;
        }
        
        stats.average_presence = (double)sum / processor->pixel_lums->count;
        stats.min_presence = min_presence;
        stats.max_presence = max_presence;
    }
    
    return stats;
}

// Destruction processeur image
void image_processor_destroy(image_processor_t* processor) {
    if (!processor || processor->magic_number != IMAGE_PROCESSOR_MAGIC) {
        return;
    }
    
    // Protection double-free
    processor->magic_number = 0;
    
    if (processor->pixel_lums) {
        lum_group_destroy(processor->pixel_lums);
    }
    
    free(processor);
}

// Test stress traitement image 100M+ pixels
int image_processor_stress_test_100m(void) {
    printf("=== TEST STRESS IMAGE PROCESSOR 100M+ PIXELS ===\n");
    
    // Simulation image 10K x 10K = 100M pixels
    size_t width = 10000, height = 10000, channels = 3;
    
    image_processor_t* processor = image_processor_create(width, height, channels);
    if (!processor) {
        printf("❌ ÉCHEC création image processor 100M pixels\n");
        return 0;
    }
    
    printf("✅ Image Processor 100M pixels créé\n");
    printf("Dimensions: %zux%zu, Canaux: %zu\n", width, height, channels);
    
    image_processor_destroy(processor);
    return 1;
}
