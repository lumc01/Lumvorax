#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>

/*
 * PROJECT: NX
 * PHASE: NX-4 - Extensions analytiques avancées
 * OBJECTIF: Quantification, cartographie statistique et diagramme de phase
 */

#define NX_NUM_ATOMS 1000
#define INITIAL_ENERGY 2000.0
#define DT 1.0
#define NUM_STEPS 1000
#define CLUSTER_THRESHOLD 0.3

typedef struct {
    double x, vx;
} NX_Atom;

typedef struct {
    double atp;
    double noise_level;
    double dissipation_rate;
    NX_Atom* atoms;
    int current_regime; // 0: Inerte, 1: Chaotique, 2: Fonctionnel, 3: Effondrement
    uint64_t residence_times[4];
    uint64_t invariant_counts[4];
} NX_System;

void log_append(const char* path, const char* msg) {
    FILE* f = fopen(path, "a");
    if (f) {
        fprintf(f, "%s\n", msg);
        fclose(f);
    }
}

void update_physics(NX_System* s) {
    for (int i = 0; i < NX_NUM_ATOMS; i++) {
        s->atoms[i].vx += ((double)rand() / RAND_MAX - 0.5) * s->noise_level;
        s->atoms[i].x += s->atoms[i].vx * DT;
    }
    s->atp -= s->dissipation_rate * NX_NUM_ATOMS * 0.01;
}

int detect_regime(NX_System* s) {
    if (s->atp <= 0) return 3;
    if (s->noise_level > 0.4) return 1;
    if (s->noise_level > 0.1 && s->atp > 400) return 2;
    return 0;
}

void count_invariants(NX_System* s) {
    int count = 0;
    for (int i = 0; i < NX_NUM_ATOMS; i += 10) { // Sous-échantillonnage pour perf
        for (int j = i + 10; j < NX_NUM_ATOMS; j += 10) {
            if (fabs(s->atoms[i].x - s->atoms[j].x) < CLUSTER_THRESHOLD) {
                count++;
            }
        }
    }
    s->invariant_counts[s->current_regime] += count;
}

int main() {
    srand(time(NULL));
    NX_System s;
    memset(&s, 0, sizeof(NX_System));
    s.atp = INITIAL_ENERGY;
    s.noise_level = 0.2;
    s.dissipation_rate = 0.5;
    s.atoms = malloc(sizeof(NX_Atom) * NX_NUM_ATOMS);

    FILE* csv_phase = fopen("RAPPORT_IAMO3/NX/NX-4_PHASE_DIAGRAM_RAW.csv", "w");
    fprintf(csv_phase, "step,energy,invariant_density,regime\n");

    printf("[NX-4] Analyse analytique avancée en cours...\n");

    for (int t = 0; t < NUM_STEPS; t++) {
        update_physics(&s);
        s.current_regime = detect_regime(&s);
        s.residence_times[s.current_regime]++;
        
        count_invariants(&s);
        
        // Dynamique de test pour explorer l'espace des phases
        if (t == 200) s.noise_level = 0.5; // Vers chaos
        if (t == 400) s.noise_level = 0.15; // Vers fonctionnel
        if (t == 700) s.dissipation_rate = 2.0; // Vers effondrement
        
        double inv_density = (double)s.invariant_counts[s.current_regime] / (s.residence_times[s.current_regime] ? s.residence_times[s.current_regime] : 1);
        fprintf(csv_phase, "%d,%.2f,%.4f,%d\n", t, s.atp, inv_density, s.current_regime);
    }
    fclose(csv_phase);

    // Generation des stats de résidence
    FILE* f_res = fopen("RAPPORT_IAMO3/NX/NX-4_RESIDENCE_STATS.md", "w");
    fprintf(f_res, "# 📊 STATISTIQUES DE TEMPS DE RÉSIDENCE (NX-4)\n\n");
    fprintf(f_res, "| Régime | Temps (cycles) | %% |\n|---|---|---|\n");
    for(int i=0; i<4; i++) {
        const char* names[] = {"INERTE", "CHAOTIQUE", "FONCTIONNEL_NX", "EFFONDREMENT"};
        fprintf(f_res, "| %s | %llu | %.1f%% |\n", names[i], (unsigned long long)s.residence_times[i], (double)s.residence_times[i]/NUM_STEPS*100.0);
    }
    fclose(f_res);

    // Generation des stats d'invariants
    FILE* f_inv = fopen("RAPPORT_IAMO3/NX/NX-4_INVARIANT_DISTRIBUTIONS.md", "w");
    fprintf(f_inv, "# 🧬 DISTRIBUTION DES INVARIANTS PAR RÉGIME (NX-4)\n\n");
    for(int i=0; i<4; i++) {
        const char* names[] = {"INERTE", "CHAOTIQUE", "FONCTIONNEL_NX", "EFFONDREMENT"};
        double avg = (double)s.invariant_counts[i] / (s.residence_times[i] ? s.residence_times[i] : 1);
        fprintf(f_inv, "- **%s** : Densité moyenne d'invariants = %.4f\n", names[i], avg);
    }
    fclose(f_inv);

    printf("[NX-4] Analyse terminée. Rapports et CSV générés.\n");
    free(s.atoms);
    return 0;
}
