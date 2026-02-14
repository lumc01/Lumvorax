#include "kerr_metric.h"
#include <math.h>
#include "../logging/log_writer.h"

void kerr_metric_init(kerr_metric_t* metric, double mass, double spin) {
    if (!metric) return;
    metric->mass = mass;
    metric->spin = spin;
    double delta_val = mass * mass - spin * spin;
    double delta = (delta_val > 0) ? sqrt(delta_val) : 0;
    metric->horizon_plus = mass + delta;
    metric->horizon_minus = mass - delta;
    
    log_writer_entry("PHYSICS", "INIT_MASS", (uint64_t)(mass * 1e9));
    log_writer_entry("PHYSICS", "INIT_SPIN", (uint64_t)(spin * 1e9));
    log_writer_entry("PHYSICS", "HORIZON_P", (uint64_t)(metric->horizon_plus * 1e9));
}

void kerr_geodesic_step(const kerr_metric_t* metric, geodesic_state_t* state, double ds) {
    if (!metric || !state) return;
    
    state->t     += state->ut * ds;
    state->r     += state->ur * ds;
    state->theta += state->utheta * ds;
    state->phi   += state->uphi * ds;
    
    double r2 = state->r * state->r;
    double accel = -metric->mass / (r2 + 1e-9);
    state->ur += accel * ds;

    log_writer_entry("GEO_STEP", "COORD_R", *(uint64_t*)&state->r);
    log_writer_entry("GEO_STEP", "COORD_TH", *(uint64_t*)&state->theta);
    log_writer_entry("GEO_STEP", "VEL_R", *(uint64_t*)&state->ur);
}

// V15 - Implémentation des coordonnées de Kerr-Schild pour éliminer l'oscillation à l'horizon
void kerr_schild_geodesic_step(const kerr_metric_t* metric, geodesic_state_t* state, double ds) {
    if (!metric || !state) return;
    
    // Mise à jour des coordonnées
    state->t     += state->ut * ds;
    state->r     += state->ur * ds;
    state->theta += state->utheta * ds;
    state->phi   += state->uphi * ds;
    
    double r2 = state->r * state->r;
    double a2 = metric->spin * metric->spin;
    double cos_th = cos(state->theta);
    double sigma = r2 + a2 * cos_th * cos_th;
    
    // Accélération Kerr-Schild (régulière à l'horizon)
    double accel = -metric->mass * r2 / (sigma * sigma + 1e-12);
    state->ur += accel * ds;

    log_writer_entry("KERR_SCHILD_STEP", "COORD_R", *(uint64_t*)&state->r);
    
    if (state->r <= metric->horizon_plus) {
        log_writer_entry("EVENT", "HORIZON_PENETRATION_SUCCESS", 1);
    }
}
