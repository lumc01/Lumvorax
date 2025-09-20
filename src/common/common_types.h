
#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

// Constantes communes
#define MAX_STORAGE_PATH_LENGTH 512
#define MAX_ERROR_MESSAGE_LENGTH 256

// Type unifié pour tous les résultats de stockage
typedef struct {
    bool success;
    char filename[MAX_STORAGE_PATH_LENGTH];
    size_t bytes_written;
    size_t bytes_read;
    uint32_t checksum;
    char error_message[MAX_ERROR_MESSAGE_LENGTH];
    void* transaction_ref;
    // Champs WAL extension
    uint64_t wal_sequence_assigned;
    uint64_t wal_transaction_id;
    bool wal_durability_confirmed;
    char wal_error_details[MAX_ERROR_MESSAGE_LENGTH];
    uint32_t magic_number;
} unified_storage_result_t;

// Types forensiques unifiés
typedef enum {
    FORENSIC_LEVEL_DEBUG = 0,
    FORENSIC_LEVEL_INFO = 1,
    FORENSIC_LEVEL_WARNING = 2,
    FORENSIC_LEVEL_ERROR = 3,
    FORENSIC_LEVEL_CRITICAL = 4
} unified_forensic_level_e;

// Interface forensique standardisée
void unified_forensic_log(unified_forensic_level_e level, const char* function, const char* format, ...);

// === SHARED NEURAL NETWORK TYPES ===
// These types are shared across neural modules to avoid conflicts

// Activation function types (shared)
#ifndef ACTIVATION_FUNCTION_E_DEFINED
#define ACTIVATION_FUNCTION_E_DEFINED
typedef enum {
    ACTIVATION_TANH = 0,
    ACTIVATION_SIGMOID = 1,
    ACTIVATION_RELU = 2,
    ACTIVATION_GELU = 3,
    ACTIVATION_SWISH = 4,
    ACTIVATION_LEAKY_RELU = 5,
    ACTIVATION_SOFTMAX = 6,
    ACTIVATION_LINEAR = 7
} activation_function_e;
#endif

// Neural plasticity rules (shared)
#ifndef NEURAL_PLASTICITY_RULES_E_DEFINED
#define NEURAL_PLASTICITY_RULES_E_DEFINED
typedef enum {
    PLASTICITY_HEBBIAN = 0,         // Apprentissage Hebbien
    PLASTICITY_ANTI_HEBBIAN = 1,    // Anti-Hebbien
    PLASTICITY_STDP = 2,            // Spike-Timing Dependent Plasticity
    PLASTICITY_HOMEOSTATIC = 3      // Plasticité homéostatique
} neural_plasticity_rules_e;
#endif

// Neural layer structure (shared)
#ifndef NEURAL_LAYER_T_DEFINED
#define NEURAL_LAYER_T_DEFINED
typedef struct neural_layer_t {
    size_t neuron_count;        // Nombre de neurones dans cette couche
    size_t input_size;          // Nombre d'entrées par neurone
    size_t output_size;         // Nombre de sorties (= neuron_count)
    double* weights;            // Poids synaptiques [neuron_count * input_size]
    double* biases;             // Biais pour chaque neurone [neuron_count]
    double* outputs;            // Sorties calculées [neuron_count]
    double* layer_error;        // Erreurs pour backpropagation [neuron_count]
    activation_function_e activation; // Type de fonction d'activation
    uint32_t layer_id;          // Identifiant unique de couche
    uint32_t magic_number;      // Protection intégrité (0xABCDEF01)
    void* memory_address;       // Protection double-free OBLIGATOIRE
} neural_layer_t;
#endif

#endif // COMMON_TYPES_H
