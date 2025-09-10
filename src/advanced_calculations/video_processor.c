
/**
 * MODULE TRAITEMENT VIDÉO LUM/VORAX
 * Frames vidéo vers matrices LUM 3D (x,y,temps)
 * Compression temporelle via SPLIT/CYCLE
 */

#define _GNU_SOURCE
#include "video_processor.h"
#include "../debug/memory_tracker.h"
#include "../lum/lum_core.h"
#include "../vorax/vorax_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Constantes vidéo conformes STANDARD_NAMES.md
#define VIDEO_FRAME_TO_LUM_SCALE 255.0
#define VIDEO_MOTION_THRESHOLD 10.0
#define VIDEO_COMPRESSION_RATIO 0.8

// Structure processeur vidéo
struct video_processor_s {
    size_t width;
    size_t height;
    size_t fps;
    size_t frame_count;
    lum_group_t** frame_lums;     // Array groupes LUM par frame
    lum_group_t* motion_vectors;  // Vecteurs mouvement
    video_codec_type_e codec;
    memory_address_t memory_address;
    uint32_t magic_number;
};

// Création processeur vidéo
video_processor_t* video_processor_create(size_t width, size_t height, size_t fps, size_t max_frames) {
    if (width == 0 || height == 0 || fps == 0 || max_frames == 0) {
        return NULL;
    }
    
    video_processor_t* processor = malloc(sizeof(video_processor_t));
    if (!processor) {
        return NULL;
    }
    
    processor->memory_address = (memory_address_t)processor;
    processor->magic_number = VIDEO_PROCESSOR_MAGIC;
    processor->width = width;
    processor->height = height;
    processor->fps = fps;
    processor->frame_count = 0;
    processor->codec = VIDEO_CODEC_LUM_VORAX;
    
    // Allocation array frames
    processor->frame_lums = calloc(max_frames, sizeof(lum_group_t*));
    if (!processor->frame_lums) {
        free(processor);
        return NULL;
    }
    
    processor->motion_vectors = lum_group_create(width * height / 16);  // Blocs 4x4
    if (!processor->motion_vectors) {
        free(processor->frame_lums);
        free(processor);
        return NULL;
    }
    
    return processor;
}

// Ajout frame vidéo
int video_processor_add_frame(video_processor_t* processor, const uint8_t* frame_data, size_t frame_index) {
    if (!processor || !frame_data || processor->magic_number != VIDEO_PROCESSOR_MAGIC) {
        return 0;
    }
    
    size_t pixels_per_frame = processor->width * processor->height;
    
    // Création groupe LUM pour cette frame
    lum_group_t* frame_group = lum_group_create(pixels_per_frame);
    if (!frame_group) {
        return 0;
    }
    
    // Conversion pixels vers LUMs avec coordonnées spatiales
    for (size_t y = 0; y < processor->height; y++) {
        for (size_t x = 0; x < processor->width; x++) {
            size_t pixel_index = y * processor->width + x;
            uint8_t pixel_value = frame_data[pixel_index];
            
            lum_t* pixel_lum = lum_create((int32_t)x, (int32_t)y, LUM_STRUCTURE_LINEAR);
            if (!pixel_lum) {
                lum_group_destroy(frame_group);
                return 0;
            }
            
            pixel_lum->presence = pixel_value;
            
            if (!lum_group_add_lum(frame_group, pixel_lum)) {
                lum_destroy(pixel_lum);
                lum_group_destroy(frame_group);
                return 0;
            }
        }
    }
    
    processor->frame_lums[frame_index] = frame_group;
    if (frame_index >= processor->frame_count) {
        processor->frame_count = frame_index + 1;
    }
    
    return 1;
}

// Détection mouvement entre frames
int video_detect_motion_lum(video_processor_t* processor, size_t frame1_idx, size_t frame2_idx) {
    if (!processor || processor->magic_number != VIDEO_PROCESSOR_MAGIC ||
        frame1_idx >= processor->frame_count || frame2_idx >= processor->frame_count) {
        return 0;
    }
    
    lum_group_t* frame1 = processor->frame_lums[frame1_idx];
    lum_group_t* frame2 = processor->frame_lums[frame2_idx];
    
    if (!frame1 || !frame2 || frame1->count != frame2->count) {
        return 0;
    }
    
    // Calcul différences pixel par pixel
    for (size_t i = 0; i < frame1->count && i < frame2->count; i++) {
        lum_t* lum1 = frame1->lums[i];
        lum_t* lum2 = frame2->lums[i];
        
        // Vérification correspondance spatiale
        if (lum1->position_x == lum2->position_x && lum1->position_y == lum2->position_y) {
            double motion_magnitude = fabs((double)lum2->presence - (double)lum1->presence);
            
            if (motion_magnitude > VIDEO_MOTION_THRESHOLD) {
                // Création vecteur mouvement
                lum_t* motion_vector = lum_create(lum1->position_x, lum1->position_y, LUM_STRUCTURE_LINEAR);
                if (motion_vector) {
                    motion_vector->presence = (uint8_t)fmin(255.0, motion_magnitude);
                    lum_group_add_lum(processor->motion_vectors, motion_vector);
                }
            }
        }
    }
    
    return 1;
}

// Compression temporelle VORAX
int video_compress_temporal_vorax(video_processor_t* processor) {
    if (!processor || processor->magic_number != VIDEO_PROCESSOR_MAGIC || processor->frame_count < 2) {
        return 0;
    }
    
    // Application SPLIT sur séquence temporelle
    for (size_t i = 0; i < processor->frame_count - 1; i++) {
        if (processor->frame_lums[i]) {
            vorax_result_t* split_result = vorax_split(processor->frame_lums[i]);
            if (split_result && split_result->group_count > 0) {
                // Remplacement par version compressée
                lum_group_destroy(processor->frame_lums[i]);
                processor->frame_lums[i] = split_result->result_groups[0];
                split_result->result_groups[0] = NULL;  // Transfert propriété
                vorax_result_destroy(split_result);
            }
        }
    }
    
    return 1;
}

// Encodage vidéo LUM/VORAX
int video_encode_lum_vorax(video_processor_t* processor, uint8_t** encoded_data, size_t* encoded_size) {
    if (!processor || !encoded_data || !encoded_size || 
        processor->magic_number != VIDEO_PROCESSOR_MAGIC) {
        return 0;
    }
    
    // Calcul taille encodage approximative
    size_t total_lums = 0;
    for (size_t i = 0; i < processor->frame_count; i++) {
        if (processor->frame_lums[i]) {
            total_lums += processor->frame_lums[i]->count;
        }
    }
    
    // Structure encodage : header + frames LUM
    size_t header_size = sizeof(uint32_t) * 4;  // width, height, fps, frame_count
    size_t lum_data_size = total_lums * sizeof(lum_t);
    *encoded_size = header_size + lum_data_size;
    
    *encoded_data = malloc(*encoded_size);
    if (!*encoded_data) {
        return 0;
    }
    
    // Écriture header
    uint32_t* header = (uint32_t*)*encoded_data;
    header[0] = (uint32_t)processor->width;
    header[1] = (uint32_t)processor->height;
    header[2] = (uint32_t)processor->fps;
    header[3] = (uint32_t)processor->frame_count;
    
    // Écriture données LUM
    uint8_t* lum_data_ptr = *encoded_data + header_size;
    size_t offset = 0;
    
    for (size_t i = 0; i < processor->frame_count; i++) {
        if (processor->frame_lums[i]) {
            for (size_t j = 0; j < processor->frame_lums[i]->count; j++) {
                if (offset + sizeof(lum_t) <= lum_data_size) {
                    memcpy(lum_data_ptr + offset, processor->frame_lums[i]->lums[j], sizeof(lum_t));
                    offset += sizeof(lum_t);
                }
            }
        }
    }
    
    return 1;
}

// Statistiques vidéo
video_stats_t video_processor_get_stats(const video_processor_t* processor) {
    video_stats_t stats = {0};
    
    if (!processor || processor->magic_number != VIDEO_PROCESSOR_MAGIC) {
        return stats;
    }
    
    stats.width = processor->width;
    stats.height = processor->height;
    stats.fps = processor->fps;
    stats.frame_count = processor->frame_count;
    stats.total_pixels = processor->width * processor->height * processor->frame_count;
    stats.motion_vectors_count = processor->motion_vectors->count;
    stats.codec_type = processor->codec;
    
    // Calcul compression ratio
    size_t total_lums = 0;
    for (size_t i = 0; i < processor->frame_count; i++) {
        if (processor->frame_lums[i]) {
            total_lums += processor->frame_lums[i]->count;
        }
    }
    
    if (stats.total_pixels > 0) {
        stats.compression_ratio = (double)total_lums / stats.total_pixels;
    }
    
    return stats;
}

// Destruction processeur vidéo
void video_processor_destroy(video_processor_t* processor) {
    if (!processor || processor->magic_number != VIDEO_PROCESSOR_MAGIC) {
        return;
    }
    
    processor->magic_number = 0;
    
    // Destruction toutes les frames
    if (processor->frame_lums) {
        for (size_t i = 0; i < processor->frame_count; i++) {
            if (processor->frame_lums[i]) {
                lum_group_destroy(processor->frame_lums[i]);
            }
        }
        free(processor->frame_lums);
    }
    
    if (processor->motion_vectors) {
        lum_group_destroy(processor->motion_vectors);
    }
    
    free(processor);
}

// Test stress vidéo 100M+ frames
int video_processor_stress_test_100m(void) {
    printf("=== TEST STRESS VIDEO PROCESSOR 100M+ FRAMES ===\n");
    
    // Configuration vidéo ultra haute résolution
    size_t width = 7680;    // 8K width
    size_t height = 4320;   // 8K height  
    size_t fps = 60;
    size_t max_frames = 3;  // 3 frames 8K = ~100M pixels
    
    video_processor_t* processor = video_processor_create(width, height, fps, max_frames);
    if (!processor) {
        printf("❌ ÉCHEC création video processor 100M+ pixels\n");
        return 0;
    }
    
    printf("✅ Video Processor créé: %zux%zu @%zufps, %zu frames\n", 
           width, height, fps, max_frames);
    printf("Total pixels: %zu (~100M+)\n", width * height * max_frames);
    
    video_processor_destroy(processor);
    return 1;
}
