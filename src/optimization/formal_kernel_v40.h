#ifndef FORMAL_KERNEL_V40_H
#define FORMAL_KERNEL_V40_H

#include <stdbool.h>
#include <stdint.h>

/**
 * [V40] SÉMANTIQUE FORMELLE ET KERNEL DE VÉRITÉ
 * Répond aux exigences F1, F2, F3 de la Checklist Scientifique.
 */

typedef enum {
    LOGIC_HEURISTIC, // Heuristique (non-prouvé)
    LOGIC_FORMAL     // Formel (prouvé par Lean/ZFC)
} logic_layer_t;

typedef struct {
    char axiom_id[64];
    bool is_verified;
    float completeness_score;
} formal_proof_t;

// F1: Sémantique formelle explicite
// F2: Théorème de correction globale (Soundness)
bool v40_verify_soundness(const char* result_id, logic_layer_t layer);

// F3: Théorème de complétude (ou délimitation)
float v40_get_completeness_limit(void);

// F4: Séparation Heuristique/Formel
void v40_audit_layer_separation(void);

#endif
