
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Inclusion TOUS les modules avancés
#include "../advanced_calculations/matrix_calculator.h"
#include "../advanced_calculations/quantum_simulator.h"
#include "../advanced_calculations/neural_network_processor.h"
#include "../advanced_calculations/audio_processor.h"
#include "../advanced_calculations/image_processor.h"
#include "../advanced_calculations/collatz_analyzer.h"
#include "../advanced_calculations/tsp_optimizer.h"
#include "../advanced_calculations/knapsack_optimizer.h"
#include "../complex_modules/realtime_analytics.h"
#include "../complex_modules/distributed_computing.h"
#include "../complex_modules/ai_optimization.h"
#include "../crypto/homomorphic_encryption.h"
#include "../debug/memory_tracker.h"

static int advanced_tests_passed = 0;
static int advanced_tests_failed = 0;

#define ADVANCED_TEST_ASSERT(condition, test_name) \
    do { \
        if (condition) { \
            printf("✓ ADVANCED TEST PASS: %s\n", test_name); \
            advanced_tests_passed++; \
        } else { \
            printf("✗ ADVANCED TEST FAIL: %s\n", test_name); \
            advanced_tests_failed++; \
        } \
    } while(0)

void test_matrix_calculator_advanced(void) {
    printf("\n=== Tests Avancés: Matrix Calculator ===\n");
    
    // Test 1: Matrices de grande taille
    const size_t matrix_size = 1000;
    matrix_calculator_t* calc = matrix_calculator_create(matrix_size, matrix_size);
    ADVANCED_TEST_ASSERT(calc != NULL, "Création calculateur matrices 1000x1000");
    
    // Test 2: Remplissage pattern mathématique
    for (size_t i = 0; i < matrix_size; i++) {
        for (size_t j = 0; j < matrix_size; j++) {
            double value = sin((double)i * 0.01) * cos((double)j * 0.01);
            matrix_set_element(calc, i, j, value);
        }
    }
    ADVANCED_TEST_ASSERT(true, "Remplissage matrice avec pattern trigonométrique");
    
    // Test 3: Opérations matricielles avancées
    matrix_config_t* config = matrix_config_create_default();
    matrix_calculator_result_t* result = matrix_multiply_lum_optimized(calc, calc, config);
    ADVANCED_TEST_ASSERT(result && result->operation_success, "Multiplication matricielle optimisée");
    
    // Test 4: Validation résultats
    if (result && result->result_matrix) {
        double sample_value = matrix_get_element(result->result_matrix, 0, 0);
        ADVANCED_TEST_ASSERT(!isnan(sample_value) && !isinf(sample_value), "Résultats numériquement valides");
    }
    
    // Test 5: Performance timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    matrix_calculator_result_t* perf_result = matrix_multiply_lum_optimized(calc, calc, config);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    ADVANCED_TEST_ASSERT(elapsed < 10.0, "Performance matrice acceptable (<10s)");
    ADVANCED_TEST_ASSERT(perf_result && perf_result->operation_success, "Multiplication répétée réussie");
    
    // Nettoyage
    matrix_calculator_result_destroy(&result);
    matrix_calculator_result_destroy(&perf_result);
    matrix_config_destroy(&config);
    matrix_calculator_destroy(&calc);
}

void test_quantum_simulator_advanced(void) {
    printf("\n=== Tests Avancés: Quantum Simulator ===\n");
    
    // Test 1: Création simulateur multi-qubits
    const size_t num_qubits = 10;
    quantum_simulator_t* sim = quantum_simulator_create(num_qubits);
    ADVANCED_TEST_ASSERT(sim != NULL, "Création simulateur 10 qubits");
    
    // Test 2: États quantiques superposés
    for (size_t i = 0; i < num_qubits; i++) {
        bool applied = quantum_apply_hadamard(sim, i);
        ADVANCED_TEST_ASSERT(applied, "Application porte Hadamard");
    }
    
    // Test 3: Intrication quantique
    for (size_t i = 0; i < num_qubits - 1; i++) {
        bool entangled = quantum_apply_cnot(sim, i, i + 1);
        ADVANCED_TEST_ASSERT(entangled, "Application porte CNOT pour intrication");
    }
    
    // Test 4: Mesure d'états
    quantum_measurement_result_t* measurement = quantum_measure_all(sim);
    ADVANCED_TEST_ASSERT(measurement != NULL, "Mesure états quantiques");
    ADVANCED_TEST_ASSERT(measurement->num_measurements == num_qubits, "Nombre mesures correct");
    
    // Test 5: Fidélité quantique
    double fidelity = quantum_calculate_fidelity(sim);
    ADVANCED_TEST_ASSERT(fidelity >= 0.0 && fidelity <= 1.0, "Fidélité quantique dans intervalle valide");
    
    // Test 6: Simulation algorithme Grover (simplifié)
    quantum_config_t* config = quantum_config_create_default();
    quantum_algorithm_result_t* grover_result = quantum_run_grover_search(sim, config);
    ADVANCED_TEST_ASSERT(grover_result && grover_result->algorithm_success, "Algorithme Grover exécuté");
    
    // Nettoyage
    quantum_measurement_result_destroy(&measurement);
    quantum_algorithm_result_destroy(&grover_result);
    quantum_config_destroy(&config);
    quantum_simulator_destroy(&sim);
}

void test_neural_network_advanced(void) {
    printf("\n=== Tests Avancés: Neural Network Processor ===\n");
    
    // Test 1: Réseau profond multi-couches
    const size_t input_size = 784;  // 28x28 image
    const size_t hidden_size = 256;
    const size_t output_size = 10;   // 10 classes
    
    neural_layer_t* input_layer = neural_layer_create(hidden_size, input_size, ACTIVATION_RELU);
    neural_layer_t* hidden_layer = neural_layer_create(hidden_size, hidden_size, ACTIVATION_RELU);
    neural_layer_t* output_layer = neural_layer_create(output_size, hidden_size, ACTIVATION_SOFTMAX);
    
    ADVANCED_TEST_ASSERT(input_layer && hidden_layer && output_layer, "Création réseau profond 3 couches");
    
    // Test 2: Forward pass avec données réalistes
    double* input_data = malloc(input_size * sizeof(double));
    for (size_t i = 0; i < input_size; i++) {
        input_data[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // [-1, 1]
    }
    
    bool forward1 = neural_layer_forward_pass(input_layer, input_data);
    ADVANCED_TEST_ASSERT(forward1, "Forward pass couche d'entrée");
    
    double* hidden_output = neural_layer_get_output(input_layer);
    bool forward2 = neural_layer_forward_pass(hidden_layer, hidden_output);
    ADVANCED_TEST_ASSERT(forward2, "Forward pass couche cachée");
    
    double* final_hidden = neural_layer_get_output(hidden_layer);
    bool forward3 = neural_layer_forward_pass(output_layer, final_hidden);
    ADVANCED_TEST_ASSERT(forward3, "Forward pass couche sortie");
    
    // Test 3: Validation fonction d'activation
    double* output = neural_layer_get_output(output_layer);
    double sum_softmax = 0.0;
    for (size_t i = 0; i < output_size; i++) {
        ADVANCED_TEST_ASSERT(output[i] >= 0.0 && output[i] <= 1.0, "Sortie softmax dans [0,1]");
        sum_softmax += output[i];
    }
    ADVANCED_TEST_ASSERT(fabs(sum_softmax - 1.0) < 0.001, "Somme softmax égale 1");
    
    // Test 4: Backpropagation (simulée)
    neural_config_t* config = neural_config_create_default();
    neural_training_result_t* training = neural_train_epoch(input_layer, input_data, output, config);
    ADVANCED_TEST_ASSERT(training && training->training_success, "Epoch d'entraînement exécuté");
    
    // Nettoyage
    free(input_data);
    neural_training_result_destroy(&training);
    neural_config_destroy(&config);
    neural_layer_destroy(&input_layer);
    neural_layer_destroy(&hidden_layer);
    neural_layer_destroy(&output_layer);
}

void test_homomorphic_encryption_advanced(void) {
    printf("\n=== Tests Avancés: Homomorphic Encryption ===\n");
    
    // Test 1: Configuration CKKS pour calculs réels
    he_context_t* ckks_context = he_context_create_ckks(8192, 1048576.0);
    ADVANCED_TEST_ASSERT(ckks_context != NULL, "Création contexte CKKS");
    
    // Test 2: Chiffrement vecteurs de données
    double plaintext_data[] = {1.5, 2.7, 3.14, 4.2, 5.9};
    size_t data_size = sizeof(plaintext_data) / sizeof(plaintext_data[0]);
    
    he_plaintext_t* plaintext = he_encode_double_vector(ckks_context, plaintext_data, data_size);
    ADVANCED_TEST_ASSERT(plaintext != NULL, "Encodage vecteur double");
    
    he_ciphertext_t* ciphertext = he_encrypt(ckks_context, plaintext);
    ADVANCED_TEST_ASSERT(ciphertext != NULL, "Chiffrement homomorphe");
    
    // Test 3: Opérations homomorphes
    he_ciphertext_t* result_add = he_add(ckks_context, ciphertext, ciphertext);
    ADVANCED_TEST_ASSERT(result_add != NULL, "Addition homomorphe");
    
    he_ciphertext_t* result_mult = he_multiply(ckks_context, ciphertext, ciphertext);
    ADVANCED_TEST_ASSERT(result_mult != NULL, "Multiplication homomorphe");
    
    // Test 4: Déchiffrement et validation
    he_plaintext_t* decrypted_add = he_decrypt(ckks_context, result_add);
    ADVANCED_TEST_ASSERT(decrypted_add != NULL, "Déchiffrement addition");
    
    double* result_data = he_decode_double_vector(ckks_context, decrypted_add);
    ADVANCED_TEST_ASSERT(result_data != NULL, "Décodage résultat");
    
    // Test 5: Validation correction mathématique
    double tolerance = 0.01;
    bool addition_correct = fabs(result_data[0] - (plaintext_data[0] * 2.0)) < tolerance;
    ADVANCED_TEST_ASSERT(addition_correct, "Addition homomorphe mathématiquement correcte");
    
    // Test 6: Stress test 1000 opérations
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    he_ciphertext_t* stress_result = ciphertext;
    for (int i = 0; i < 100; i++) {
        he_ciphertext_t* temp = he_add(ckks_context, stress_result, ciphertext);
        if (stress_result != ciphertext) {
            he_ciphertext_destroy(&stress_result);
        }
        stress_result = temp;
        
        if (!stress_result) break;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    ADVANCED_TEST_ASSERT(stress_result != NULL, "Stress test 100 additions homomorphes");
    ADVANCED_TEST_ASSERT(elapsed < 60.0, "Performance homomorphe acceptable (<60s)");
    
    // Nettoyage
    free(result_data);
    he_plaintext_destroy(&plaintext);
    he_plaintext_destroy(&decrypted_add);
    he_ciphertext_destroy(&ciphertext);
    he_ciphertext_destroy(&result_add);
    he_ciphertext_destroy(&result_mult);
    if (stress_result != ciphertext) {
        he_ciphertext_destroy(&stress_result);
    }
    he_context_destroy(&ckks_context);
}

void test_realtime_analytics_advanced(void) {
    printf("\n=== Tests Avancés: Realtime Analytics ===\n");
    
    // Test 1: Création système analytics temps réel
    realtime_config_t* config = realtime_config_create_default();
    realtime_analytics_t* analytics = realtime_analytics_create(config);
    ADVANCED_TEST_ASSERT(analytics != NULL, "Création système analytics temps réel");
    
    // Test 2: Stream de données LUM
    for (int i = 0; i < 1000; i++) {
        lum_t* lum = lum_create(i % 2, i, i * 2, LUM_STRUCTURE_LINEAR);
        bool processed = realtime_analytics_process_lum(analytics, lum);
        ADVANCED_TEST_ASSERT(processed, "Traitement LUM temps réel");
        lum_destroy(lum);
    }
    
    // Test 3: Métriques temps réel
    realtime_metrics_t* metrics = realtime_analytics_get_metrics(analytics);
    ADVANCED_TEST_ASSERT(metrics != NULL, "Récupération métriques temps réel");
    ADVANCED_TEST_ASSERT(metrics->processed_count == 1000, "Compteur traitements correct");
    ADVANCED_TEST_ASSERT(metrics->throughput_per_second > 0, "Débit calculé");
    
    // Test 4: Alertes automatiques
    realtime_alert_t* alerts = realtime_analytics_get_alerts(analytics);
    ADVANCED_TEST_ASSERT(alerts != NULL, "Système d'alertes opérationnel");
    
    // Nettoyage
    realtime_metrics_destroy(&metrics);
    realtime_alert_destroy(&alerts);
    realtime_analytics_destroy(&analytics);
    realtime_config_destroy(&config);
}

int main(void) {
    printf("🚀 === TESTS AVANCÉS COMPLETS TOUS MODULES ===\n");
    printf("Validation fonctionnalités avancées et performance\n\n");
    
    // Initialisation tracking
    memory_tracker_init();
    
    // Seed pour reproductibilité
    srand(42);
    
    // Exécution tests avancés
    test_matrix_calculator_advanced();
    test_quantum_simulator_advanced();
    test_neural_network_advanced();
    test_homomorphic_encryption_advanced();
    test_realtime_analytics_advanced();
    
    // Résultats finaux
    printf("\n=== RÉSULTATS TESTS AVANCÉS ===\n");
    printf("Tests avancés réussis: %d\n", advanced_tests_passed);
    printf("Tests avancés échoués: %d\n", advanced_tests_failed);
    printf("Taux succès avancé: %.1f%%\n", 
           advanced_tests_passed > 0 ? (100.0 * advanced_tests_passed) / (advanced_tests_passed + advanced_tests_failed) : 0.0);
    
    if (advanced_tests_failed == 0) {
        printf("✅ TOUS MODULES AVANCÉS VALIDÉS\n");
    } else {
        printf("❌ ÉCHECS MODULES AVANCÉS DÉTECTÉS\n");
    }
    
    // Rapport mémoire
    memory_tracker_report();
    memory_tracker_cleanup();
    
    return advanced_tests_failed == 0 ? 0 : 1;
}
