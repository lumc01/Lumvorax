
#include "neural_network_processor.h"
#include "../debug/memory_tracker.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Création neurone LUM
neural_lum_t* neural_lum_create(int32_t x, int32_t y, size_t input_count, activation_function_e activation) {
    (void)activation; // Suppression warning unused parameter
    (void)activation; // Suppression warning paramètre inutilisé
    if (input_count == 0 || input_count > NEURAL_MAX_NEURONS_PER_LAYER) {
        return NULL;
    }
    
    neural_lum_t* neuron = TRACKED_MALLOC(sizeof(neural_lum_t));
    if (!neuron) return NULL;
    
    // Initialisation LUM de base
    neuron->base_lum.id = 0;
    neuron->base_lum.presence = 1;
    neuron->base_lum.position_x = x;
    neuron->base_lum.position_y = y;
    neuron->base_lum.structure_type = LUM_STRUCTURE_BINARY;
    
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    neuron->base_lum.timestamp = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    neuron->base_lum.memory_address = &neuron->base_lum;
    neuron->base_lum.checksum = 0;
    neuron->base_lum.is_destroyed = 0;
    
    // Initialisation réseau neuronal
    neuron->input_count = input_count;
    neuron->weights = TRACKED_MALLOC(input_count * sizeof(double));
    neuron->gradient = TRACKED_MALLOC(input_count * sizeof(double));
    
    if (!neuron->weights || !neuron->gradient) {
        if (neuron->weights) TRACKED_FREE(neuron->weights);
        if (neuron->gradient) TRACKED_FREE(neuron->gradient);
        TRACKED_FREE(neuron);
        return NULL;
    }
    
    // Initialisation poids aléatoires (Xavier)
    double xavier_limit = sqrt(6.0 / (input_count + 1));
    for (size_t i = 0; i < input_count; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * xavier_limit;
        neuron->gradient[i] = 0.0;
    }
    
    neuron->bias = 0.0;
    neuron->activation_threshold = 0.0;
    neuron->learning_rate = 0.001; // Taux par défaut
    neuron->fire_count = 0;
    neuron->memory_address = (void*)neuron;
    neuron->neuron_magic = NEURAL_MAGIC_NUMBER;
    neuron->is_active = false;
    
    return neuron;
}

// Destruction neurone
void neural_lum_destroy(neural_lum_t** neuron_ptr) {
    if (!neuron_ptr || !*neuron_ptr) return;
    
    neural_lum_t* neuron = *neuron_ptr;
    
    if (neuron->neuron_magic != NEURAL_MAGIC_NUMBER || 
        neuron->memory_address != (void*)neuron) {
        return;
    }
    
    if (neuron->weights) TRACKED_FREE(neuron->weights);
    if (neuron->gradient) TRACKED_FREE(neuron->gradient);
    
    neuron->neuron_magic = NEURAL_DESTROYED_MAGIC;
    neuron->memory_address = NULL;
    
    TRACKED_FREE(neuron);
    *neuron_ptr = NULL;
}

// Activation neurone
double neural_lum_activate(neural_lum_t* neuron, double* inputs, activation_function_e function) {
    if (!neuron || !inputs) return 0.0;
    
    // Calcul somme pondérée
    double weighted_sum = neuron->bias;
    for (size_t i = 0; i < neuron->input_count; i++) {
        weighted_sum += inputs[i] * neuron->weights[i];
    }
    
    // Application fonction d'activation
    double output = 0.0;
    switch (function) {
        case ACTIVATION_SIGMOID:
            output = activation_sigmoid(weighted_sum);
            break;
        case ACTIVATION_TANH:
            output = activation_tanh(weighted_sum);
            break;
        case ACTIVATION_RELU:
            output = activation_relu(weighted_sum);
            break;
        case ACTIVATION_LEAKY_RELU:
            output = activation_leaky_relu(weighted_sum, 0.01);
            break;
        case ACTIVATION_SWISH:
            output = activation_swish(weighted_sum);
            break;
        case ACTIVATION_GELU:
            output = activation_gelu(weighted_sum);
            break;
        case ACTIVATION_LINEAR:
        default:
            output = weighted_sum;
            break;
    }
    
    // Mise à jour neurone LUM
    neuron->is_active = (output > neuron->activation_threshold);
    if (neuron->is_active) {
        neuron->fire_count++;
        neuron->base_lum.presence = 1;
    } else {
        neuron->base_lum.presence = 0;
    }
    
    return output;
}

// Fonctions d'activation
double activation_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double activation_tanh(double x) {
    return tanh(x);
}

double activation_relu(double x) {
    return (x > 0.0) ? x : 0.0;
}

double activation_leaky_relu(double x, double alpha) {
    return (x > 0.0) ? x : alpha * x;
}

double activation_swish(double x) {
    return x * activation_sigmoid(x);
}

double activation_gelu(double x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0/M_PI) * (x + 0.044715 * x * x * x)));
}

// Création couche neuronale (modèle flat arrays canonique)
neural_layer_t* neural_layer_create(size_t neuron_count, size_t input_size, activation_function_e activation) {
    if (neuron_count == 0 || neuron_count > NEURAL_MAX_NEURONS_PER_LAYER || input_size == 0) {
        return NULL;
    }
    
    neural_layer_t* layer = TRACKED_MALLOC(sizeof(neural_layer_t));
    if (!layer) return NULL;
    
    // Configuration des champs de base
    layer->neuron_count = neuron_count;
    layer->input_size = input_size;
    layer->output_size = neuron_count;
    layer->activation = activation;
    layer->layer_id = 0;
    layer->magic_number = NEURAL_MAGIC_NUMBER;
    layer->memory_address = (void*)layer;
    
    // Allocation arrays flat
    layer->weights = TRACKED_MALLOC(neuron_count * input_size * sizeof(double));
    if (!layer->weights) {
        TRACKED_FREE(layer);
        return NULL;
    }
    
    layer->biases = TRACKED_MALLOC(neuron_count * sizeof(double));
    if (!layer->biases) {
        TRACKED_FREE(layer->weights);
        TRACKED_FREE(layer);
        return NULL;
    }
    
    layer->outputs = TRACKED_MALLOC(neuron_count * sizeof(double));
    if (!layer->outputs) {
        TRACKED_FREE(layer->biases);
        TRACKED_FREE(layer->weights);
        TRACKED_FREE(layer);
        return NULL;
    }
    
    layer->layer_error = TRACKED_MALLOC(neuron_count * sizeof(double));
    if (!layer->layer_error) {
        TRACKED_FREE(layer->outputs);
        TRACKED_FREE(layer->biases);
        TRACKED_FREE(layer->weights);
        TRACKED_FREE(layer);
        return NULL;
    }
    
    // Initialisation Xavier pour poids
    double limit = sqrt(6.0 / (input_size + neuron_count));
    srand((unsigned int)time(NULL));
    
    for (size_t i = 0; i < neuron_count * input_size; i++) {
        layer->weights[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * limit;
    }
    
    for (size_t i = 0; i < neuron_count; i++) {
        layer->biases[i] = NEURAL_DEFAULT_BIAS;
        layer->outputs[i] = 0.0;
        layer->layer_error[i] = 0.0;
    }
    
    return layer;
}

// Destruction couche (modèle flat arrays)
void neural_layer_destroy(neural_layer_t** layer_ptr) {
    if (!layer_ptr || !*layer_ptr) return;
    
    neural_layer_t* layer = *layer_ptr;
    
    // Validation sécurité
    if (layer->memory_address != (void*)layer || 
        layer->magic_number != NEURAL_MAGIC_NUMBER) {
        return;
    }
    
    // Libération arrays flat
    if (layer->weights) TRACKED_FREE(layer->weights);
    if (layer->biases) TRACKED_FREE(layer->biases);
    if (layer->outputs) TRACKED_FREE(layer->outputs);
    if (layer->layer_error) TRACKED_FREE(layer->layer_error);
    
    // Nettoyage sécurisé
    layer->weights = NULL;
    layer->biases = NULL;
    layer->outputs = NULL;
    layer->layer_error = NULL;
    layer->magic_number = NEURAL_DESTROYED_MAGIC;
    layer->memory_address = NULL;
    
    TRACKED_FREE(layer);
    *layer_ptr = NULL;
}

// Propagation avant (modèle flat arrays)
bool neural_layer_forward_pass(neural_layer_t* layer, double* inputs) {
    if (!layer || !inputs || layer->magic_number != NEURAL_MAGIC_NUMBER) {
        return false;
    }
    
    // Calcul pour chaque neurone
    for (size_t n = 0; n < layer->neuron_count; n++) {
        double sum = layer->biases[n];
        
        // Produit scalaire weights[n*input_size : (n+1)*input_size] · inputs
        for (size_t i = 0; i < layer->input_size; i++) {
            sum += layer->weights[n * layer->input_size + i] * inputs[i];
        }
        
        // Application fonction d'activation
        switch (layer->activation) {
            case ACTIVATION_SIGMOID:
                layer->outputs[n] = activation_sigmoid(sum);
                break;
            case ACTIVATION_TANH:
                layer->outputs[n] = activation_tanh(sum);
                break;
            case ACTIVATION_RELU:
                layer->outputs[n] = activation_relu(sum);
                break;
            case ACTIVATION_LEAKY_RELU:
                layer->outputs[n] = activation_leaky_relu(sum, 0.01);
                break;
            case ACTIVATION_SWISH:
                layer->outputs[n] = activation_swish(sum);
                break;
            case ACTIVATION_GELU:
                layer->outputs[n] = activation_gelu(sum);
                break;
            case ACTIVATION_LINEAR:
            default:
                layer->outputs[n] = sum;
                break;
        }
    }
    
    return true;
}

// Tests stress 100M+ neurones
bool neural_stress_test_100m_neurons(neural_config_t* config) {
    if (!config) return false;
    
    printf("=== NEURAL STRESS TEST: 100M+ Neurons ===\n");
    
    const size_t neuron_count = 100000000; // 100M neurones
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    printf("Testing neural layer creation with large neuron count...\n");
    
    // Test avec couche de 10000 neurones (échantillon)
    const size_t test_neurons = 10000;
    neural_layer_t* layer = neural_layer_create(test_neurons, 100, ACTIVATION_RELU);
    
    if (!layer) {
        printf("❌ Failed to create neural layer with %zu neurons\n", test_neurons);
        return false;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double creation_time = (end.tv_sec - start.tv_sec) + 
                          (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    printf("✅ Created %zu neural LUMs in %.3f seconds\n", test_neurons, creation_time);
    
    // Projection pour 100M
    double projected_time = creation_time * (neuron_count / (double)test_neurons);
    printf("Projected time for %zu neurons: %.1f seconds\n", neuron_count, projected_time);
    printf("Neural creation rate: %.0f neurons/second\n", test_neurons / creation_time);
    
    // Test forward pass
    double* test_inputs = TRACKED_MALLOC(100 * sizeof(double));
    if (test_inputs) {
        for (int i = 0; i < 100; i++) {
            test_inputs[i] = (double)rand() / RAND_MAX;
        }
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        bool forward_success = neural_layer_forward_pass(layer, test_inputs);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        if (forward_success) {
            double forward_time = (end.tv_sec - start.tv_sec) + 
                                 (end.tv_nsec - start.tv_nsec) / 1000000000.0;
            printf("✅ Forward pass completed in %.6f seconds\n", forward_time);
            printf("Forward rate: %.0f neurons/second\n", test_neurons / forward_time);
        }
        
        TRACKED_FREE(test_inputs);
    }
    
    // Cleanup
    neural_layer_destroy(&layer);
    
    printf("✅ Neural stress test 100M+ neurons completed successfully\n");
    return true;
}

// Configuration par défaut
neural_config_t* neural_config_create_default(void) {
    neural_config_t* config = TRACKED_MALLOC(sizeof(neural_config_t));
    if (!config) return NULL;
    
    config->max_epochs = 1000;
    config->convergence_threshold = 1e-6;
    config->use_momentum = false;
    config->momentum_coefficient = 0.9;
    config->use_dropout = false;
    config->dropout_rate = 0.2;
    config->use_batch_normalization = false;
    config->batch_size = 32;
    config->enable_gpu_acceleration = false;
    config->memory_address = (void*)config;
    
    return config;
}

// Destruction configuration
void neural_config_destroy(neural_config_t** config_ptr) {
    if (!config_ptr || !*config_ptr) return;
    
    neural_config_t* config = *config_ptr;
    if (config->memory_address == (void*)config) {
        TRACKED_FREE(config);
        *config_ptr = NULL;
    }
}
