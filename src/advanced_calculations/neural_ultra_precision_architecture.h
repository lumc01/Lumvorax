// NOUVEAU FICHIER : src/advanced_calculations/neural_ultra_precision_architecture.h

#ifndef NEURAL_ULTRA_PRECISION_ARCHITECTURE_H
#define NEURAL_ULTRA_PRECISION_ARCHITECTURE_H

#include "neural_blackbox_computer.h"

// Configuration architecture ultra-précise
typedef struct {
    size_t precision_target_digits;    // Nombre de digits précision requis (ex: 15)
    size_t base_depth;                // Profondeur de base du réseau
    size_t precision_layers;          // Couches supplémentaires pour précision
    size_t neurons_per_precision_digit; // Neurones par digit de précision
    double memory_scaling_factor;     // Facteur échelle mémoire
    // Champs ajoutés pour compatibilité complète avec tests
    double precision_target;          // Cible de précision (ex: 1e-15)
    size_t input_dimensions;          // Dimensions d'entrée
    size_t output_dimensions;         // Dimensions de sortie
    double computation_scaling_factor; // Facteur échelle computation
    bool enable_adaptive_precision;   // Précision adaptative
    bool enable_error_correction;     // Correction d'erreur
    uint32_t magic_number;           // Protection double-free
} neural_ultra_precision_config_t;

// EXPLICATION TECHNIQUE :
// Cette structure définit comment adapter l'architecture neuronale selon
// la précision requise. Plus de digits = plus de couches + plus de neurones.

// Calcul architecture selon précision requise
neural_architecture_config_t* neural_calculate_ultra_precision_architecture(
    size_t input_dim, 
    size_t output_dim, 
    size_t precision_digits
);

// Calcul profondeur optimale selon complexité
size_t neural_calculate_optimal_depth(
    size_t input_dim,
    size_t output_dim,
    neural_complexity_target_e complexity
);

// Calcul largeur optimale selon dimensions
size_t neural_calculate_optimal_width(
    size_t input_dim,
    size_t output_dim,
    size_t depth
);

// Fonctions d'activation ultra-précises (sans perte numérique)
double activation_ultra_precise_tanh(double x);
double activation_ultra_precise_sigmoid(double x);  
double activation_ultra_precise_piecewise(double x);

// Constantes associées
#define NEURAL_ULTRA_PRECISION_MAGIC 0xFEEDFACE
#define MAX_PRECISION_DIGITS 50
#define DEFAULT_PRECISION_LAYERS 10
#define DEFAULT_NEURONS_PER_DIGIT 100

// Configuration architecture adaptative ultra-précise
neural_ultra_precision_config_t* neural_create_ultra_precision_config(
    size_t precision_digits
);

neural_ultra_precision_config_t* neural_ultra_precision_config_create(
    size_t precision_digits, 
    size_t input_dims, 
    size_t output_dims
);

void neural_destroy_ultra_precision_config(neural_ultra_precision_config_t** config);

void neural_ultra_precision_config_destroy(neural_ultra_precision_config_t* config);

bool neural_ultra_precision_config_validate(const neural_ultra_precision_config_t* config);

// Validation architecture ultra-précise
bool neural_validate_ultra_precision_architecture(
    neural_architecture_config_t* config,
    size_t precision_target_digits
);

#endif // NEURAL_ULTRA_PRECISION_ARCHITECTURE_H