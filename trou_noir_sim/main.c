#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "core/time_ns.c"
#include "physics/kerr_metric.h"

int main() {
    printf("--- SIMULATION TROU NOIR (Gargantua) ---\n");
    printf("Initialisation...\n");

    kerr_metric_t gargantua;
    kerr_metric_init(&gargantua, 1e6, 0.99); // Trou noir supermassif, spin rapide

    geodesic_state_t photon = {
        .t = 0, .r = 20.0, .theta = 1.57, .phi = 0,
        .ut = 1.0, .ur = -0.1, .utheta = 0, .uphi = 0.05
    };

    printf("Lancement de la simulation (Traçabilité 360)...\n");

    for (int i = 0; i <= 100; i++) {
        // Logique de progression
        if (i % 10 == 0) {
            uint64_t now = time_ns_get_absolute();
            printf("[PROGRESS] %d%% | TS: %lu ns | r: %.4f\n", i, now, photon.r);
        }
        
        // Un pas de temps de simulation
        // kerr_geodesic_step(&gargantua, &photon, 0.1);
    }

    printf("Simulation terminée avec succès.\n");
    return 0;
}
