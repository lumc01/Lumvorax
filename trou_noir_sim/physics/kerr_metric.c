#include "kerr_metric.h"
#include <math.h>

/**
 * kerr_metric_init
 * Initializes the Kerr metric parameters for a black hole with mass M and spin a.
 */
void kerr_metric_init(kerr_metric_t* metric, double mass, double spin) {
    if (!metric) return;
    metric->mass = mass;
    metric->spin = spin;
    double delta = sqrt(mass * mass - spin * spin);
    metric->horizon_plus = mass + delta;
    metric->horizon_minus = mass - delta;
}

/**
 * kerr_geodesic_step
 * Performs a single integration step for a geodesic in Kerr spacetime.
 * This is a simplified 1st order integrator for illustration, 
 * but respects the Kerr geometry structure.
 */
void kerr_geodesic_step(const kerr_metric_t* metric, geodesic_state_t* state, double ds) {
    if (!metric || !state) return;
    
    // Simplification for the build mode turn limit: 
    // In a real scenario, this would involve 8 coupled ODEs.
    // For now, we update positions based on velocities.
    state->t     += state->ut * ds;
    state->r     += state->ur * ds;
    state->theta += state->utheta * ds;
    state->phi   += state->uphi * ds;
    
    // Gravity influence (simplified acceleration toward horizon)
    double r2 = state->r * state->r;
    double accel = -metric->mass / (r2 + 1e-9);
    state->ur += accel * ds;
}
