#include "kerr_metric.h"
#include <math.h>
#include "../logging/log_writer.h"

void kerr_metric_init(kerr_metric_t* metric, double mass, double spin) {
    if (!metric) return;
    metric->mass = mass;
    metric->spin = spin;
    double delta = sqrt(mass * mass - spin * spin);
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

    // Logs ultra-détaillés par pas
    log_writer_entry("GEO_STEP", "COORD_R", *(uint64_t*)&state->r);
    log_writer_entry("GEO_STEP", "COORD_TH", *(uint64_t*)&state->theta);
    log_writer_entry("GEO_STEP", "VEL_R", *(uint64_t*)&state->ur);

    if (state->r <= metric->horizon_plus) {
        log_writer_entry("EVENT", "HORIZON_CROSS", 1);
    }
}
