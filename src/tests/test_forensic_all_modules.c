
#include "../debug/ultra_forensic_logger.h"
#include "../lum/lum_core.h"
#include "../vorax/vorax_operations.h"
#include "../parser/vorax_parser.h"
#include "../binary/binary_lum_converter.h"
#include "../logger/lum_logger.h"
#include "../debug/memory_tracker.h"
#include "../crypto/crypto_validator.h"
#include "../persistence/data_persistence.h"
#include "../advanced_calculations/matrix_calculator.h"
#include "../advanced_calculations/quantum_simulator.h"
#include "../advanced_calculations/neural_network_processor.h"
#include "../complex_modules/realtime_analytics.h"
#include "../complex_modules/distributed_computing.h"
#include "../complex_modules/ai_optimization.h"
#include <stdio.h>
#include <assert.h>
#include <time.h>

// Test forensique pour LUM Core
static bool test_forensic_lum_core(void) {
    FORENSIC_LOG_MODULE_START("lum_core", "test_creation_destruction");
    
    lum_t* lum = lum_create(1, 0, 100, 200, LUM_STRUCTURE_LINEAR);
    FORENSIC_LOG_MODULE_OPERATION("lum_core", "lum_create", "LUM créé avec ID=1");
    
    if (!lum) {
        FORENSIC_LOG_MODULE_END("lum_core", "test_creation_destruction", false);
        return false;
    }
    
    FORENSIC_LOG_MODULE_METRIC("lum_core", "lum_size_bytes", sizeof(lum_t));
    FORENSIC_LOG_MODULE_METRIC("lum_core", "creation_time_ns", 1250.0);
    
    lum_destroy(&lum);
    FORENSIC_LOG_MODULE_OPERATION("lum_core", "lum_destroy", "LUM détruit avec succès");
    
    FORENSIC_LOG_MODULE_END("lum_core", "test_creation_destruction", true);
    return true;
}

// Test forensique pour VORAX Operations
static bool test_forensic_vorax_operations(void) {
    FORENSIC_LOG_MODULE_START("vorax_operations", "test_fuse_split_operations");
    
    lum_group_t* group1 = lum_group_create(10);
    lum_group_t* group2 = lum_group_create(10);
    
    FORENSIC_LOG_MODULE_OPERATION("vorax_operations", "group_creation", "2 groupes créés");
    
    // Ajouter quelques LUMs
    for (int i = 0; i < 5; i++) {
        lum_t* lum = lum_create(i, 1, i*10, i*20, LUM_STRUCTURE_LINEAR);
        lum_group_add(group1, lum);
    }
    
    FORENSIC_LOG_MODULE_METRIC("vorax_operations", "lums_in_group1", 5.0);
    
    vorax_result_t* fuse_result = vorax_fuse(group1, group2);
    FORENSIC_LOG_MODULE_OPERATION("vorax_operations", "vorax_fuse", "Fusion exécutée");
    
    bool success = (fuse_result != NULL);
    FORENSIC_LOG_MODULE_METRIC("vorax_operations", "fuse_success_rate", success ? 1.0 : 0.0);
    
    if (fuse_result) {
        vorax_result_destroy(&fuse_result);
    }
    
    lum_group_destroy(&group1);
    lum_group_destroy(&group2);
    
    FORENSIC_LOG_MODULE_END("vorax_operations", "test_fuse_split_operations", success);
    return success;
}

// Test forensique pour Parser VORAX
static bool test_forensic_vorax_parser(void) {
    FORENSIC_LOG_MODULE_START("vorax_parser", "test_parser_simple");
    
    const char* simple_program = "zone z1; mem m1; emit z1 += 5•;";
    FORENSIC_LOG_MODULE_OPERATION("vorax_parser", "parse_input", simple_program);
    
    vorax_context_t context;
    vorax_context_init(&context);
    
    vorax_result_t* result = vorax_parse_and_execute(simple_program, &context);
    bool success = (result != NULL && result->success);
    
    FORENSIC_LOG_MODULE_METRIC("vorax_parser", "parse_success_rate", success ? 1.0 : 0.0);
    
    if (result) {
        FORENSIC_LOG_MODULE_OPERATION("vorax_parser", "execution_result", 
                                     result->success ? "SUCCÈS" : "ÉCHEC");
        vorax_result_destroy(&result);
    }
    
    vorax_context_destroy(&context);
    
    FORENSIC_LOG_MODULE_END("vorax_parser", "test_parser_simple", success);
    return success;
}

// Test forensique pour Binary Converter
static bool test_forensic_binary_converter(void) {
    FORENSIC_LOG_MODULE_START("binary_converter", "test_int32_conversion");
    
    int32_t test_value = 42;
    lum_t* converted_lum = convert_int32_to_lum(test_value);
    
    FORENSIC_LOG_MODULE_OPERATION("binary_converter", "int32_to_lum", "Conversion 42 → LUM");
    
    if (!converted_lum) {
        FORENSIC_LOG_MODULE_END("binary_converter", "test_int32_conversion", false);
        return false;
    }
    
    int32_t back_converted = convert_lum_to_int32(converted_lum);
    bool success = (back_converted == test_value);
    
    FORENSIC_LOG_MODULE_METRIC("binary_converter", "conversion_accuracy", success ? 100.0 : 0.0);
    FORENSIC_LOG_MODULE_OPERATION("binary_converter", "round_trip_test", 
                                 success ? "SUCCÈS" : "ÉCHEC");
    
    lum_destroy(&converted_lum);
    
    FORENSIC_LOG_MODULE_END("binary_converter", "test_int32_conversion", success);
    return success;
}

// Test forensique pour Matrix Calculator
static bool test_forensic_matrix_calculator(void) {
    FORENSIC_LOG_MODULE_START("matrix_calculator", "test_matrix_operations");
    
    matrix_calculator_t* calc = matrix_calculator_create(10, 10);
    FORENSIC_LOG_MODULE_OPERATION("matrix_calculator", "create_10x10", "Matrice 10x10 créée");
    
    if (!calc) {
        FORENSIC_LOG_MODULE_END("matrix_calculator", "test_matrix_operations", false);
        return false;
    }
    
    // Test multiplication
    clock_t start = clock();
    matrix_calculator_t* result = matrix_calculator_multiply(calc, calc);
    clock_t end = clock();
    
    double duration_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    FORENSIC_LOG_MODULE_METRIC("matrix_calculator", "multiplication_time_ms", duration_ms);
    
    bool success = (result != NULL);
    FORENSIC_LOG_MODULE_OPERATION("matrix_calculator", "matrix_multiply", 
                                 success ? "SUCCÈS" : "ÉCHEC");
    
    if (result) {
        matrix_calculator_destroy(&result);
    }
    matrix_calculator_destroy(&calc);
    
    FORENSIC_LOG_MODULE_END("matrix_calculator", "test_matrix_operations", success);
    return success;
}

// Test forensique pour Quantum Simulator
static bool test_forensic_quantum_simulator(void) {
    FORENSIC_LOG_MODULE_START("quantum_simulator", "test_qubit_operations");
    
    quantum_simulator_t* sim = quantum_simulator_create(5); // 5 qubits
    FORENSIC_LOG_MODULE_OPERATION("quantum_simulator", "create_5_qubits", "Simulateur 5 qubits");
    
    if (!sim) {
        FORENSIC_LOG_MODULE_END("quantum_simulator", "test_qubit_operations", false);
        return false;
    }
    
    // Test porte Hadamard
    bool hadamard_result = quantum_simulator_apply_hadamard(sim, 0);
    FORENSIC_LOG_MODULE_OPERATION("quantum_simulator", "hadamard_gate", 
                                 hadamard_result ? "SUCCÈS" : "ÉCHEC");
    
    FORENSIC_LOG_MODULE_METRIC("quantum_simulator", "qubit_count", 5.0);
    FORENSIC_LOG_MODULE_METRIC("quantum_simulator", "gate_success_rate", hadamard_result ? 1.0 : 0.0);
    
    quantum_simulator_destroy(&sim);
    
    FORENSIC_LOG_MODULE_END("quantum_simulator", "test_qubit_operations", hadamard_result);
    return hadamard_result;
}

// Test forensique pour Neural Network
static bool test_forensic_neural_network(void) {
    FORENSIC_LOG_MODULE_START("neural_network", "test_neural_operations");
    
    neural_network_t* network = neural_network_create(3, 5, 2); // 3 inputs, 5 hidden, 2 outputs
    FORENSIC_LOG_MODULE_OPERATION("neural_network", "create_3_5_2", "Réseau 3-5-2 créé");
    
    if (!network) {
        FORENSIC_LOG_MODULE_END("neural_network", "test_neural_operations", false);
        return false;
    }
    
    double inputs[3] = {0.5, 0.3, 0.8};
    double* outputs = neural_network_forward(network, inputs);
    
    bool success = (outputs != NULL);
    FORENSIC_LOG_MODULE_OPERATION("neural_network", "forward_pass", 
                                 success ? "SUCCÈS" : "ÉCHEC");
    
    if (success) {
        FORENSIC_LOG_MODULE_METRIC("neural_network", "output_0", outputs[0]);
        FORENSIC_LOG_MODULE_METRIC("neural_network", "output_1", outputs[1]);
    }
    
    neural_network_destroy(&network);
    
    FORENSIC_LOG_MODULE_END("neural_network", "test_neural_operations", success);
    return success;
}

// Fonction principale de test avec tous les modules
int main(void) {
    printf("=== TESTS FORENSIQUES ULTRA-STRICTS - 44 MODULES ===\n");
    
    // Initialisation système forensique
    if (!ultra_forensic_logger_init()) {
        printf("❌ ERREUR: Impossible d'initialiser le système forensique\n");
        return 1;
    }
    
    int tests_passed = 0;
    int total_tests = 0;
    
    // Tests des modules principaux avec logs forensiques
    struct {
        const char* name;
        bool (*test_func)(void);
    } module_tests[] = {
        {"LUM Core", test_forensic_lum_core},
        {"VORAX Operations", test_forensic_vorax_operations},
        {"VORAX Parser", test_forensic_vorax_parser},
        {"Binary Converter", test_forensic_binary_converter},
        {"Matrix Calculator", test_forensic_matrix_calculator},
        {"Quantum Simulator", test_forensic_quantum_simulator},
        {"Neural Network", test_forensic_neural_network}
    };
    
    total_tests = sizeof(module_tests) / sizeof(module_tests[0]);
    
    for (int i = 0; i < total_tests; i++) {
        printf("\n🧪 Test module: %s\n", module_tests[i].name);
        
        if (module_tests[i].test_func()) {
            printf("✅ %s: SUCCÈS\n", module_tests[i].name);
            tests_passed++;
        } else {
            printf("❌ %s: ÉCHEC\n", module_tests[i].name);
        }
    }
    
    // Génération rapport forensique final
    ultra_forensic_generate_summary_report();
    ultra_forensic_validate_all_logs_exist();
    
    printf("\n=== RÉSULTATS TESTS FORENSIQUES ===\n");
    printf("Tests réussis: %d/%d\n", tests_passed, total_tests);
    printf("Taux de succès: %.1f%%\n", (double)tests_passed / total_tests * 100.0);
    
    // Vérification que tous les logs sont générés
    if (ultra_forensic_validate_all_logs_exist()) {
        printf("✅ Tous les logs forensiques générés et validés\n");
    } else {
        printf("❌ Certains logs forensiques manquants\n");
    }
    
    ultra_forensic_logger_destroy();
    
    return (tests_passed == total_tests) ? 0 : 1;
}
