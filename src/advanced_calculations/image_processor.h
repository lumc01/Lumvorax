
#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "../lum/lum_core.h"
#include <stdint.h>

// Structure pixel comme LUM étendu
typedef struct {
    uint32_t memory_address;
    uint8_t red, green, blue, alpha;
    uint16_t x, y;
    double presence_intensity;
    uint64_t timestamp_ns;
} pixel_lum_t;

// Configuration traitement image
typedef struct {
    size_t width, height;
    size_t max_pixels_supported; // Minimum 100M pour conformité
    char filter_type[64];
    double filter_parameters[8];
    bool enable_simd_vectorization;
} image_config_t;

// Fonctions principales
bool image_processor_init(image_config_t* config);
pixel_lum_t* image_to_lum_matrix(uint8_t* image_data, size_t width, size_t height);
uint8_t* lum_matrix_to_image(pixel_lum_t* lum_pixels, size_t width, size_t height);

// Filtres VORAX sur images
bool apply_vorax_filter_split(pixel_lum_t* pixels, size_t count);
bool apply_vorax_filter_cycle(pixel_lum_t* pixels, size_t count);
bool apply_vorax_edge_detection(pixel_lum_t* pixels, size_t width, size_t height);

// Test stress obligatoire 100M+ pixels
bool image_stress_test_100m_pixels(image_config_t* config);

// Métriques performance
typedef struct {
    double pixels_per_second;
    double mbytes_per_second;
    double filter_execution_time_ns;
    size_t memory_usage_bytes;
} image_performance_metrics_t;

image_performance_metrics_t image_get_performance_metrics(void);

#endif // IMAGE_PROCESSOR_H
