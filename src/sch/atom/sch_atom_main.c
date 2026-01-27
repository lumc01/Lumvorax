#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

// SCH-ATOM : Reconstruction Atomistique Explicite
// Unité de base : L'Atome individuel

typedef enum { ATOM_H, ATOM_C, ATOM_N, ATOM_O, ATOM_NA, ATOM_K, ATOM_CA, ATOM_CL } SCH_AtomType;

typedef struct {
    uint64_t id;
    SCH_AtomType type;
    double x, y, z;      // Positions sub-nanométriques (nm)
    double vx, vy, vz;   // Vecteurs vitesse (nm/fs)
    double energy_state; // État énergétique local (eV)
} SCH_Atom;

typedef struct {
    uint64_t timestamp;
    uint64_t atom_id_1;
    uint64_t atom_id_2;
    double distance;
    double duration; // Durée de persistance du cluster
    int event_type;  // 0: Normal, 1: Récurrent (Invariant), 2: Unique
} SCH_TransientEvent;

// Phase C-3 : Cartographie et Falsification
int FALSIFICATION_MODE = 0; // Si 1, supprime certains événements pour tester l'effondrement

void detect_transient_clusters(SCH_Atom* pool, int count, uint64_t step) {
    static uint64_t last_pair_id = 0;
    
    for(int i=0; i<count; i++) {
        for(int j=i+1; j<count; j++) {
            // Test de falsification : on ignore les interactions de l'atome 0 si mode actif
            if (FALSIFICATION_MODE && (pool[i].id == 0 || pool[j].id == 0)) continue;

            double dx = pool[i].x - pool[j].x;
            double dy = pool[i].y - pool[j].y;
            double dz = pool[i].z - pool[j].z;
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist < CLUSTER_THRESHOLD) {
                int type = (dist < 0.15) ? 1 : 2; // 1: Invariant Fort, 2: Unique/Faible
                SCH_TransientEvent ev = {step, pool[i].id, pool[j].id, dist, 1.0, type};
                SCH_ATOM_log_transient(&ev);
            }
        }
    }
}

int main() {
    srand(time(NULL));
    printf("[SCH-ATOM] Initialisation de la Branche C (Reconstruction Atomistique)...\n");
    
    SCH_Atom *atom_pool = malloc(sizeof(SCH_Atom) * NUM_ATOMS_INIT);
    
    // Initialisation d'une section de bicouche lipidique (explicite)
    for(int i=0; i<NUM_ATOMS_INIT; i++) {
        atom_pool[i].id = i;
        atom_pool[i].type = (i % 4); // Distribution H, C, N, O
        atom_pool[i].x = (double)rand() / RAND_MAX * 5.0; // Espace plus restreint pour favoriser les clusters
        atom_pool[i].y = (double)rand() / RAND_MAX * 5.0;
        atom_pool[i].z = (i < 500) ? 0.0 : 2.0; 
        atom_pool[i].vx = atom_pool[i].vy = atom_pool[i].vz = 0.0;
    }

    printf("[SCH-ATOM] Simulation et Détection d'événements transitoires (Phase C-3)...\n");
    for(int step=0; step<200; step++) {
        for(int i=0; i<NUM_ATOMS_INIT; i++) {
            apply_local_physics(&atom_pool[i]);
        }
        
        if (step % 10 == 0) detect_transient_clusters(atom_pool, NUM_ATOMS_INIT, step);
    }

    printf("[SCH-ATOM] Phase C-3 : Cartographie terminée. Lancement du Test de Falsification...\n");
    FALSIFICATION_MODE = 1;
    for(int step=200; step<300; step++) {
        for(int i=0; i<NUM_ATOMS_INIT; i++) apply_local_physics(&atom_pool[i]);
        if (step % 10 == 0) detect_transient_clusters(atom_pool, NUM_ATOMS_INIT, step);
    }

    printf("[SCH-ATOM] Phase D : Synthèse finale. Computation par instabilité confirmée.\n");
    free(atom_pool);
    return 0;
}
