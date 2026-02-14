#include "kerr_metric.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../logging/log_writer.h"

// V16-V20 : Extensions pour Chaos, MHD et Singularité
typedef struct {
    double rho;     // Densité du plasma
    double b_field; // Champ magnétique
    double pressure;
} mhd_state_t;

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
}

// V15 - Kerr-Schild (Régularité Horizon)
void kerr_schild_geodesic_step(const kerr_metric_t* metric, geodesic_state_t* state, double ds) {
    if (!metric || !state) return;
    
    state->t     += state->ut * ds;
    state->r     += state->ur * ds;
    state->theta += state->utheta * ds;
    state->phi   += state->uphi * ds;
    
    double r2 = state->r * state->r;
    double a2 = metric->spin * metric->spin;
    double cos_th = cos(state->theta);
    double sigma = r2 + a2 * cos_th * cos_th;
    
    double accel = -metric->mass * r2 / (sigma * sigma + 1e-12);
    state->ur += accel * ds;

    if (state->r <= metric->horizon_plus) {
        log_writer_entry("EVENT", "HORIZON_PENETRATION_SUCCESS", 1);
    }
}

// V18 - Calcul Lyapunov (Chaos interne)
double calculate_lyapunov_exponent(double d0, double d1, double dt) {
    if (d0 <= 0 || dt <= 0) return 0;
    return log(d1 / d0) / dt;
}

// V19 - Simulation MHD (Plasma interaction)
void update_mhd_plasma(const kerr_metric_t* m, const geodesic_state_t* s, mhd_state_t* p, double ds) {
    // Interaction simplifiée : champ magnétique amplifié par la courbure
    double r_inv = 1.0 / (s->r + 1e-5);
    p->b_field += m->mass * r_inv * ds;
    p->rho *= exp(-0.1 * ds); // Accrétion
}
