#include "lum_core.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../debug/memory_tracker.h"
#include <pthread.h> // Nécessaire pour pthread_mutex_t et les fonctions associées
#include <sys/time.h> // Nécessaire pour gettimeofday

static uint32_t lum_id_counter = 1;
static pthread_mutex_t id_counter_mutex = PTHREAD_MUTEX_INITIALIZER;

// Core LUM functions
lum_t* lum_create(uint8_t presence, int32_t x, int32_t y, lum_structure_type_e type) {
    lum_t* lum = TRACKED_MALLOC(sizeof(lum_t));
    if (!lum) return NULL;

    lum->presence = (presence > 0) ? 1 : 0;
    lum->id = lum_generate_id();
    lum->position_x = x;
    lum->position_y = y;
    lum->structure_type = type;
    lum->is_destroyed = 0;  // CORRECTION: Initialiser le flag protection double-free
    lum->timestamp = lum_get_timestamp();
    lum->memory_address = (void*)lum;  // Adresse mémoire pour traçabilité

    return lum;
}

void lum_destroy(lum_t* lum) {
    if (!lum) return;

    // PROTECTION DOUBLE FREE: Vérifier magic pattern - CORRECTION TYPE STRICTE
    static const uint32_t DESTROYED_MAGIC = 0xDEADBEEF;  // uint32_t pour correspondre au type id
    if (lum->id == DESTROYED_MAGIC) {
        return; // Déjà détruit
    }

    // Marquer comme détruit AVANT la libération
    lum->id = DESTROYED_MAGIC;
    lum->is_destroyed = 1;

    TRACKED_FREE(lum);
}

// Fonction sécurisée pour destruction avec invalidation
void lum_safe_destroy(lum_t** lum_ptr) {
    if (lum_ptr && *lum_ptr) {
        TRACKED_FREE(*lum_ptr);
        *lum_ptr = NULL; // Invalidation du pointeur
    }
}

// LUM Group functions
lum_group_t* lum_group_create(size_t initial_capacity) {
    // PROTECTION: Capacité minimale pour éviter corruptions
    if (initial_capacity == 0) initial_capacity = 1;

    lum_group_t* group = TRACKED_MALLOC(sizeof(lum_group_t));
    if (!group) return NULL;

    // Allouer la mémoire pour les LUMs avec une capacité initiale sécurisée
    group->lums = TRACKED_MALLOC(sizeof(lum_t) * initial_capacity);
    if (!group->lums) {
        TRACKED_FREE(group);
        return NULL;
    }

    // PROTECTION CRITIQUE: Détecter si group->lums == group (corruption mémoire)
    if (group->lums == (lum_t*)group) {
        // CORRECTION DÉTECTÉE ! Allouer à nouveau avec taille différente
        TRACKED_FREE(group->lums);
        group->lums = TRACKED_MALLOC(sizeof(lum_t) * initial_capacity + 64); // +64 bytes padding
        if (!group->lums) {
            TRACKED_FREE(group);
            return NULL;
        }
    }

    group->count = 0;
    group->capacity = initial_capacity;
    group->group_id = lum_generate_id();
    group->type = LUM_STRUCTURE_GROUP;

    return group;
}

// Fonction de destruction renforcée pour éviter le double-free
void lum_group_destroy(lum_group_t* group) {
    if (!group) return;

    // Protection contre double destruction
    static const uint32_t MAGIC_DESTROYED = 0xDEADBEEF;
    if (group->capacity == MAGIC_DESTROYED) {
        return; // Déjà détruit
    }

    // PROTECTION CRITIQUE : Vérifier corruption pointeur
    if (group->lums && group->lums != (lum_t*)group) {
        TRACKED_FREE(group->lums);
        group->lums = NULL;
    } else if (group->lums == (lum_t*)group) {
        // CORRECTION DÉTECTÉE: group->lums pointe vers group lui-même !
        group->lums = NULL; // Ne pas libérer - éviter double free
    }

    // Marquer comme détruit
    group->capacity = MAGIC_DESTROYED;
    group->count = 0;

    TRACKED_FREE(group);
}

// Fonction utilitaire pour détruire un groupe de manière sûre
void lum_group_safe_destroy(lum_group_t** group_ptr) {
    if (group_ptr && *group_ptr) {
        lum_group_destroy(*group_ptr);
        *group_ptr = NULL;  // S'assure que le pointeur pointe vers NULL après destruction
    }
}

bool lum_group_add(lum_group_t* group, lum_t* lum) {
    if (!group || !lum) return false;

    // Vérifier si le groupe a été marqué comme détruit
    static const uint32_t MAGIC_DESTROYED = 0xDEADBEEF;
    if (group->capacity == MAGIC_DESTROYED) {
        // lum_log("Tentative d'ajout à un groupe déjà détruit."); // Optionnel: loguer si vous avez une fonction lum_log
        return false;
    }

    if (group->count >= group->capacity) {
        // CORRECTION CRITIQUE: Utiliser TRACKED_MALLOC au lieu de realloc pour éviter corruption
        size_t new_capacity = (group->capacity == 0) ? 10 : group->capacity * 2;
        lum_t* new_lums = TRACKED_MALLOC(sizeof(lum_t) * new_capacity);
        if (!new_lums) {
            return false;
        }

        // Copier les données existantes
        if (group->lums && group->count > 0) {
            memcpy(new_lums, group->lums, sizeof(lum_t) * group->count);
            TRACKED_FREE(group->lums);
        }

        group->lums = new_lums;
        group->capacity = new_capacity;
    }

    // Copie profonde du LUM dans le groupe
    // Assurez-vous que lum_t ne contient pas de pointeurs qui nécessitent une copie profonde séparée
    group->lums[group->count] = *lum;
    group->count++;

    return true;
}

lum_t* lum_group_get(lum_group_t* group, size_t index) {
    if (!group || index >= group->count) return NULL;

    // Vérifier si le groupe a été marqué comme détruit
    static const uint32_t MAGIC_DESTROYED = 0xDEADBEEF;
    if (group->capacity == MAGIC_DESTROYED) {
        // lum_log("Tentative d'accès à un groupe déjà détruit."); // Optionnel: loguer si vous avez une fonction lum_log
        return NULL;
    }

    return &group->lums[index];
}

size_t lum_group_size(lum_group_t* group) {
    if (!group) return 0;
    // Retourner 0 si le groupe a été marqué comme détruit
    static const uint32_t MAGIC_DESTROYED = 0xDEADBEEF;
    if (group->capacity == MAGIC_DESTROYED) {
        return 0;
    }
    return group->count;
}

// Zone functions
lum_zone_t* lum_zone_create(const char* name) {
    lum_zone_t* zone = TRACKED_MALLOC(sizeof(lum_zone_t));
    if (!zone) return NULL;

    // S'assurer que la copie de la chaîne ne dépasse pas la taille du buffer
    strncpy(zone->name, name, sizeof(zone->name) - 1);
    zone->name[sizeof(zone->name) - 1] = '\0'; // Assurer la terminaison nulle

    // Allouer la mémoire pour les pointeurs vers les groupes
    zone->groups = TRACKED_MALLOC(sizeof(lum_group_t*) * 10); // Capacité initiale de 10 groupes
    if (!zone->groups) {
        TRACKED_FREE(zone); // Utiliser TRACKED_FREE pour cohérence
        return NULL;
    }

    zone->group_count = 0;
    zone->group_capacity = 10;
    zone->is_empty = true; // Initialement vide

    return zone;
}

void lum_zone_destroy(lum_zone_t* zone) {
    if (zone) {
        if (zone->groups) {
            // Détruire chaque groupe contenu dans la zone
            for (size_t i = 0; i < zone->group_count; i++) {
                // Utiliser la fonction de destruction sûre pour chaque groupe
                lum_group_safe_destroy(&zone->groups[i]);
            }
            TRACKED_FREE(zone->groups); // Libérer le tableau de pointeurs vers les groupes
            zone->groups = NULL;
        }
        TRACKED_FREE(zone); // Libérer la zone elle-même
    }
}

bool lum_zone_add_group(lum_zone_t* zone, lum_group_t* group) {
    if (!zone || !group) return false;

    // Vérifier si la zone a déjà atteint sa capacité
    if (zone->group_count >= zone->group_capacity) {
        // Augmenter la capacité du tableau de groupes
        size_t new_capacity = zone->group_capacity * 2;
        lum_group_t** new_groups = realloc(zone->groups, sizeof(lum_group_t*) * new_capacity);
        if (!new_groups) {
            // lum_log("Échec du redimensionnement du tableau de groupes dans la zone."); // Optionnel
            return false;
        }

        zone->groups = new_groups;
        zone->group_capacity = new_capacity;
    }

    // Ajouter le nouveau groupe à la zone
    zone->groups[zone->group_count] = group;
    zone->group_count++;
    zone->is_empty = false; // La zone n'est plus vide

    return true;
}

bool lum_zone_is_empty(lum_zone_t* zone) {
    if (!zone) return true; // Une zone NULL est considérée vide

    // Si le nombre de groupes est 0, la zone est vide
    if (zone->group_count == 0) {
        zone->is_empty = true;
        return true;
    }

    // Vérifier si tous les groupes dans la zone sont vides
    for (size_t i = 0; i < zone->group_count; i++) {
        // Utiliser la fonction lum_group_size pour obtenir la taille de manière sûre
        if (lum_group_size(zone->groups[i]) > 0) {
            zone->is_empty = false; // Au moins un groupe contient des LUMs
            return false;
        }
    }

    zone->is_empty = true; // Tous les groupes sont vides
    return true;
}

// Memory functions
lum_memory_t* lum_memory_create(const char* name) {
    lum_memory_t* memory = TRACKED_MALLOC(sizeof(lum_memory_t));
    if (!memory) return NULL;

    // Copier le nom de manière sûre
    strncpy(memory->name, name, sizeof(memory->name) - 1);
    memory->name[sizeof(memory->name) - 1] = '\0'; // Assurer la terminaison nulle

    // Initialiser le groupe stocké
    memory->stored_group.lums = NULL;
    memory->stored_group.count = 0;
    memory->stored_group.capacity = 0;
    memory->stored_group.group_id = 0;
    memory->stored_group.type = LUM_STRUCTURE_GROUP;
    memory->is_occupied = false; // Le bloc mémoire n'est pas occupé initialement

    return memory;
}

void lum_memory_destroy(lum_memory_t* memory) {
    if (memory) {
        // Libérer la mémoire allouée pour les LUMs stockés s'il y en a
        if (memory->stored_group.lums) {
            // CORRECTION CRITIQUE: Utiliser TRACKED_FREE pour cohérence
            TRACKED_FREE(memory->stored_group.lums);
            memory->stored_group.lums = NULL; // Éviter un pointeur pendant
        }
        TRACKED_FREE(memory); // Libérer la structure lum_memory_t elle-même
    }
}

bool lum_memory_store(lum_memory_t* memory, lum_group_t* group) {
    if (!memory || !group) return false;

    // Libérer les données existantes dans le bloc mémoire s'il y en a
    if (memory->stored_group.lums) {
        // CORRECTION CRITIQUE: Utiliser TRACKED_FREE pour cohérence
        TRACKED_FREE(memory->stored_group.lums);
        memory->stored_group.lums = NULL;
    }

    // Allocation de mémoire pour la copie profonde du groupe
    // On alloue juste la taille nécessaire pour les éléments actuels
    memory->stored_group.lums = TRACKED_MALLOC(sizeof(lum_t) * group->count);
    if (!memory->stored_group.lums) {
        // lum_log("Échec de l'allocation mémoire pour stocker le groupe."); // Optionnel
        memory->stored_group.capacity = 0; // Réinitialiser la capacité en cas d'échec
        memory->stored_group.count = 0;
        memory->is_occupied = false;
        return false;
    }

    // Copie des données du groupe vers le bloc mémoire
    memcpy(memory->stored_group.lums, group->lums, sizeof(lum_t) * group->count);
    memory->stored_group.count = group->count;
    memory->stored_group.capacity = group->count; // La capacité stockée correspond au nombre d'éléments copiés
    memory->stored_group.group_id = group->group_id;
    memory->stored_group.type = group->type;
    memory->is_occupied = true; // Marquer le bloc mémoire comme occupé

    return true;
}

lum_group_t* lum_memory_retrieve(lum_memory_t* memory) {
    if (!memory || !memory->is_occupied) return NULL;

    // Créer un nouveau groupe pour contenir les données récupérées
    lum_group_t* group = lum_group_create(memory->stored_group.count); // Créer avec la bonne capacité
    if (!group) {
        // lum_log("Échec de la création du groupe lors de la récupération."); // Optionnel
        return NULL;
    }

    // Copier les données du bloc mémoire vers le nouveau groupe
    // Note: Il faut s'assurer que group->lums a été alloué avec une capacité suffisante.
    // lum_group_create a déjà alloué `sizeof(lum_t) * memory->stored_group.count`.
    memcpy(group->lums, memory->stored_group.lums, sizeof(lum_t) * memory->stored_group.count);
    group->count = memory->stored_group.count;
    group->capacity = memory->stored_group.count; // La capacité du nouveau groupe est le nombre d'éléments copiés
    group->group_id = memory->stored_group.group_id;
    group->type = memory->stored_group.type;

    return group;
}

// Utility functions
uint32_t lum_generate_id(void) {
    pthread_mutex_lock(&id_counter_mutex);
    // Vérifier le dépassement potentiel du compteur d'ID
    uint32_t id;
    if (lum_id_counter == UINT32_MAX) {
        // Gérer le cas où le compteur atteint sa valeur maximale
        // Pour l'instant, on réinitialise ou on logue une erreur.
        // Dans un système réel, une stratégie plus robuste serait nécessaire.
        // lum_log("Avertissement: Le compteur d'ID LUM a atteint sa valeur maximale."); // Optionnel
        lum_id_counter = 1; // Réinitialiser ou utiliser une autre stratégie
    }
    id = lum_id_counter++;
    pthread_mutex_unlock(&id_counter_mutex);
    return id;
}

uint64_t lum_get_timestamp(void) {
    struct timespec ts;
    // Essayer CLOCK_MONOTONIC d'abord
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        uint64_t timestamp = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
        if (timestamp > 0) {
            return timestamp;
        }
    }

    // Fallback CLOCK_REALTIME si MONOTONIC échoue
    if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
        uint64_t timestamp = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
        if (timestamp > 0) {
            return timestamp;
        }
    }

    // Fallback final : microsecondes depuis epoch
    struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0) {
        return (tv.tv_sec * 1000000ULL + tv.tv_usec) * 1000ULL; // Convert to nanoseconds
    }

    // Dernière option : timestamp basé sur time()
    return ((uint64_t)time(NULL)) * 1000000000ULL;
}

void lum_print(const lum_t* lum) {
    if (lum) {
        printf("LUM[%u]: presence=%u, pos=(%d,%d), type=%u, ts=%lu\n",
               lum->id, lum->presence, lum->position_x, lum->position_y,
               lum->structure_type, lum->timestamp);
    }
}

void lum_group_print(const lum_group_t* group) {
    if (group) {
        // Vérifier si le groupe a été marqué comme détruit avant de l'imprimer
        static const uint32_t MAGIC_DESTROYED = 0xDEADBEEF;
        if (group->capacity == MAGIC_DESTROYED) {
            printf("Group (destroyed): %zu LUMs\n", group->count); // Afficher l'état détruit
            return;
        }

        printf("Group[%u]: %zu LUMs\n", group->group_id, group->count);
        for (size_t i = 0; i < group->count; i++) {
            printf("  ");
            lum_print(&group->lums[i]);
        }
    }
}

// Note: Les fonctions memory_tracker_cleanup et lum_log mentionnées dans les erreurs
// ne sont pas présentes dans le code original et doivent être implémentées séparément
// ou fournies par une bibliothèque externe. Si elles sont censées faire partie de ce fichier,
// leur déclaration et leur définition devraient être ajoutées.

// Exemple d'implémentation placeholder pour memory_tracker_cleanup (si nécessaire)
/*
void memory_tracker_cleanup() {
    // Implémentation de la logique de nettoyage du suivi mémoire
    // lum_log("Nettoyage du suivi mémoire effectué."); // Optionnel
}
*/

// Exemple d'implémentation placeholder pour lum_log (si nécessaire)
/*
void lum_log(const char* message) {
    // Implémentation de la logique de logging
    fprintf(stderr, "[LUM_LOG] %s\n", message);
}
*/