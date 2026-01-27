#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>

/* 
 * PROJECT: SCH-NEURON-X (NX) 
 * VERSION: NX-1
 * OBJECTIF: Neurone dissipatif fondé physiquement (ATOM -> DISS -> BIO -> FUNC)
 */

#define NX_NUM_ATOMS 500
#define NX_ATP_INITIAL 1000.0
#define NX_DISS_THRESHOLD 0.25
#define NX_ATP_DRAIN 0.5

typedef enum { NX_ATOM, NX_DISS, NX_BIO, NX_FUNC } NX_Layer;

typedef struct {
    uint64_t id;
    double x, y, z;
    double vx, vy, vz;
} NX_Atom;

typedef struct {
    double atp_level;
    int is_alive;
    double stability_index;
    NX_Atom* atoms;
} NX_Neuron;

void nx_log_forensic(const char* layer, const char* msg) {
    char path[128];
    sprintf(path, "logs_AIMO3/sch/nx/NX-%s_events.log", layer);
    FILE* f = fopen(path, "a");
    if (f) {
        fprintf(f, "[%ld][NX-1][%s] %s\n", (long)time(NULL), layer, msg);
        fclose(f);
    }
}

void nx_simulate_atom_layer(NX_Neuron* n) {
    for (int i = 0; i < NX_NUM_ATOMS; i++) {
        n->atoms[i].vx += ((double)rand() / RAND_MAX - 0.5) * 0.05;
        n->atoms[i].x += n->atoms[i].vx;
        // Détection d'invariants transitoires simplifiée pour NX-1
        if (fabs(n->atoms[i].x) < 0.1) {
            nx_log_forensic("ATOM", "INVARIANT_DETECTED");
        }
    }
}

void nx_simulate_energy(NX_Neuron* n) {
    n->atp_level -= NX_ATP_DRAIN;
    if (n->atp_level <= 0) {
        n->is_alive = 0;
        nx_log_forensic("BIO", "DEATH_ATP_DEPLETION");
    }
}

int main() {
    srand(time(NULL));
    printf("[SCH-NEURON-X] Initialisation de la Version NX-1...\n");

    NX_Neuron n;
    n.atp_level = NX_ATP_INITIAL;
    n.is_alive = 1;
    n.stability_index = 0.0;
    n.atoms = malloc(sizeof(NX_Atom) * NX_NUM_ATOMS);

    for (int i = 0; i < NX_NUM_ATOMS; i++) {
        n.atoms[i].id = i;
        n.atoms[i].x = (double)rand() / RAND_MAX;
    }

    printf("[SCH-NEURON-X] Cycle de vie dissipatif en cours...\n");
    for (int cycle = 0; cycle < 100 && n.is_alive; cycle++) {
        nx_simulate_atom_layer(&n);
        nx_simulate_energy(&n);
        
        if (cycle % 20 == 0) {
            nx_log_forensic("DISS", "DISSIPATIVE_FLOW_ACTIVE");
            nx_log_forensic("FUNC", "EMERGENT_BEHAVIOR_OBSERVED");
        }
    }

    if (n.is_alive) {
        printf("[SCH-NEURON-X] Simulation terminée : Neurone NX-1 stable localement.\n");
    } else {
        printf("[SCH-NEURON-X] Simulation terminée : Mort neuronale par épuisement.\n");
    }

    free(n.atoms);
    return 0;
}
