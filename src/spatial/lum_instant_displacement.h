
#ifndef LUM_INSTANT_DISPLACEMENT_H
#define LUM_INSTANT_DISPLACEMENT_H

#include "../lum/lum_core.h"
#include <stdbool.h>
#include <stdint.h>

// Types pour déplacement instantané
typedef struct {
    int32_t from_x, from_y;
    int32_t to_x, to_y;
    uint64_t displacement_time_ns;
    bool success;
} lum_displacement_result_t;

typedef struct {
    uint32_t total_displacements;
    uint32_t successful_displacements;
    uint64_t total_time_ns;
    double average_time_ns;
    uint32_t magic_number;
} lum_displacement_metrics_t;

// Constantes
#define LUM_DISPLACEMENT_MAGIC 0xDEADC0DE
#define MAX_DISPLACEMENT_DISTANCE 10000

// Fonctions principales
bool lum_instant_displace(lum_t* lum, int32_t new_x, int32_t new_y, lum_displacement_result_t* result);
bool lum_group_instant_displace_all(lum_group_t* group, int32_t delta_x, int32_t delta_y);
bool lum_validate_displacement_coordinates(int32_t x, int32_t y);

// Métriques et validation
lum_displacement_metrics_t* lum_displacement_metrics_create(void);
void lum_displacement_metrics_destroy(lum_displacement_metrics_t* metrics);
void lum_displacement_metrics_record(lum_displacement_metrics_t* metrics, lum_displacement_result_t* result);
void lum_displacement_metrics_print(const lum_displacement_metrics_t* metrics);

// Tests et validation
bool lum_test_displacement_performance(size_t num_lums);
bool lum_test_displacement_vs_traditional_move(size_t num_operations);

#endif // LUM_INSTANT_DISPLACEMENT_H
