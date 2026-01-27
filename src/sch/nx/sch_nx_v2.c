#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>

/*
 * PROJECT: NX
 * PHASES: NX-2 & NX-3
 * OBJECTIF: Science des régimes neuronaux dissipatifs
 */

#define NX_NUM_ATOMS 1000
#define INITIAL_ENERGY 2000.0
#define DT 1.0

typedef struct {
    double x, vx;
    int invariant_type;
} NX_Atom;

typedef struct {
    double atp;
    double noise_level;
    double dissipation_rate;
    NX_Atom* atoms;
    int current_regime; // 0: Inerte, 1: Chaotique, 2: Fonctionnel, 3: Effondrement
} NX_System;

void log_forensic(const char* path, const char* msg) {
    FILE* f = fopen(path, "a");
    if (f) {
        fprintf(f, "[%ld] %s\n", (long)time(NULL), msg);
        fclose(f);
    }
}

void update_physics(NX_System* s) {
    for (int i = 0; i < NX_NUM_ATOMS; i++) {
        // Bruit intrinsèque
        s->atoms[i].vx += ((double)rand() / RAND_MAX - 0.5) * s->noise_level;
        s->atoms[i].x += s->atoms[i].vx * DT;
        
        // Dissipation
        s->atp -= s->dissipation_rate;
    }
}

const char* get_regime_name(int r) {
    switch(r) {
        case 0: return "INERTE";
        case 1: return "CHAOTIQUE";
        case 2: return "FONCTIONNEL_NX";
        case 3: return "EFFONDREMENT";
        default: return "INCONNU";
    }
}

void detect_regime(NX_System* s, int step) {
    int old_regime = s->current_regime;
    
    if (s->atp <= 0) s->current_regime = 3;
    else if (s->noise_level > 0.5) s->current_regime = 1;
    else if (s->noise_level > 0.1 && s->atp > 500) s->current_regime = 2;
    else s->current_regime = 0;
    
    if (old_regime != s->current_regime) {
        char buf[256];
        sprintf(buf, "TRANSITION: %s -> %s (STEP %d)", get_regime_name(old_regime), get_regime_name(s->current_regime), step);
        log_forensic("logs_AIMO3/nx/NX-2_phase_transitions.log", buf);
    }
}

int main() {
    srand(time(NULL));
    NX_System s;
    s.atp = INITIAL_ENERGY;
    s.noise_level = 0.2;
    s.dissipation_rate = 0.5;
    s.current_regime = 0;
    s.atoms = malloc(sizeof(NX_Atom) * NX_NUM_ATOMS);

    for(int i=0; i<NX_NUM_ATOMS; i++) {
        s.atoms[i].x = 0;
        s.atoms[i].vx = 0;
    }

    printf("[NX-2/3] Exploration des régimes et transitions...\n");

    // Phase NX-2: Cartographie
    for (int t = 0; t < 500; t++) {
        update_physics(&s);
        detect_regime(&s, t);
        
        if (t == 100) s.noise_level = 0.6; // Provoque le chaos
        if (t == 200) s.noise_level = 0.2; // Tente la récupération
        if (t == 400) s.dissipation_rate = 5.0; // Accélère l'effondrement
    }

    printf("[NX-2/3] Simulation terminée. Rapports générés.\n");
    free(s.atoms);
    return 0;
}
