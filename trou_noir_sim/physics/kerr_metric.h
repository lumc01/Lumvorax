#ifndef KERR_METRIC_H
#define KERR_METRIC_H

#include <stdint.h>

typedef struct {
    double mass;          // M
    double spin;          // a = J/M
    double horizon_plus;  // r+
    double horizon_minus; // r-
} kerr_metric_t;

typedef struct {
    double t, r, theta, phi;
    double ut, ur, utheta, uphi;
} geodesic_state_t;

// Kerr metric solver prototypes
void kerr_metric_init(kerr_metric_t* metric, double mass, double spin);
void kerr_geodesic_step(const kerr_metric_t* metric, geodesic_state_t* state, double ds);

// V15 - Kerr-Schild Coordinates
void kerr_schild_geodesic_step(const kerr_metric_t* metric, geodesic_state_t* state, double ds);

#endif
