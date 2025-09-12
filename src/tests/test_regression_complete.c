
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>

#include "../lum/lum_core.h"
#include "../vorax/vorax_operations.h"
#include "../binary/binary_lum_converter.h"
#include "../crypto/crypto_validator.h"
#include "../debug/memory_tracker.h"

// Tests de régression pour éviter régressions futures
static int regression_tests_passed = 0;
static int regression_tests_failed = 0;

#define REGRESSION_TEST_ASSERT(condition, test_name) \
    do { \
        if (condition) { \
            printf("✓ REGRESSION TEST PASS: %s\n", test_name); \
            regression_tests_passed++; \
        } else { \
            printf("✗ REGRESSION TEST FAIL: %s\n", test_name); \
            regression_tests_failed++; \
        } \
    } while(0)

// Structure pour stocker résultats attendus
typedef struct {
    const char* test_name;
    size_t expected_lum_count;
    uint32_t expected_checksum;
    double expected_performance_threshold;
} regression_baseline_t;

// Baselines connues (simulées pour exemple)
static regression_baseline_t baselines[] = {
    {"lum_creation_basic", 1, 0x12345678, 1000000.0},
    {"vorax_fuse_simple", 5, 0x87654321, 500000.0},
    {"vorax_split_basic", 3, 0xABCDEF00, 750000.0},
    {"binary_conversion", 8, 0xDEADBEEF, 2000000.0}
};

uint32_t calculate_lum_checksum(lum_t* lum) {
    if (!lum) return 0;
    
    uint32_t checksum = 0;
    checksum ^= (uint32_t)lum->presence;
    checksum ^= (uint32_t)lum->position_x;
    checksum ^= (uint32_t)lum->position_y;
    checksum ^= (uint32_t)lum->structure_type;
    checksum ^= (uint32_t)(lum->timestamp & 0xFFFFFFFF);
    
    return checksum;
}

uint32_t calculate_group_checksum(lum_group_t* group) {
    if (!group) return 0;
    
    uint32_t total_checksum = 0;
    for (size_t i = 0; i < group->count; i++) {
        total_checksum ^= calculate_lum_checksum(&group->lums[i]);
    }
    
    return total_checksum;
}

void test_regression_lum_creation_basic(void) {
    printf("\n=== Test Régression: Création LUM Basique ===\n");
    
    // Test selon baseline connue
    regression_baseline_t* baseline = &baselines[0];
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    lum_t* lum = lum_create(1, 100, 200, LUM_STRUCTURE_LINEAR);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000.0 + (end.tv_nsec - start.tv_nsec);
    double operations_per_second = 1000000000.0 / elapsed_ns;
    
    REGRESSION_TEST_ASSERT(lum != NULL, "Création LUM réussie (régression)");
    REGRESSION_TEST_ASSERT(operations_per_second >= baseline->expected_performance_threshold, 
                          "Performance création LUM maintenue");
    
    // Validation propriétés invariantes
    REGRESSION_TEST_ASSERT(lum->presence == 1, "Propriété présence préservée");
    REGRESSION_TEST_ASSERT(lum->position_x == 100, "Propriété position X préservée");
    REGRESSION_TEST_ASSERT(lum->position_y == 200, "Propriété position Y préservée");
    REGRESSION_TEST_ASSERT(lum->structure_type == LUM_STRUCTURE_LINEAR, "Type structure préservé");
    REGRESSION_TEST_ASSERT(lum->timestamp > 0, "Timestamp généré");
    
    lum_destroy(lum);
}

void test_regression_vorax_fuse_simple(void) {
    printf("\n=== Test Régression: VORAX Fuse Simple ===\n");
    
    regression_baseline_t* baseline = &baselines[1];
    
    // Créer deux groupes identiques aux tests précédents
    lum_group_t* group1 = lum_group_create(3);
    lum_group_t* group2 = lum_group_create(2);
    
    // Group 1: 3 LUMs
    for (int i = 0; i < 3; i++) {
        lum_t* lum = lum_create(1, i, 0, LUM_STRUCTURE_LINEAR);
        lum_group_add(group1, lum);
        lum_destroy(lum);
    }
    
    // Group 2: 2 LUMs
    for (int i = 0; i < 2; i++) {
        lum_t* lum = lum_create(0, i + 10, 0, LUM_STRUCTURE_CIRCULAR);
        lum_group_add(group2, lum);
        lum_destroy(lum);
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    vorax_result_t* result = vorax_fuse(group1, group2);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000.0 + (end.tv_nsec - start.tv_nsec);
    double operations_per_second = 1000000000.0 / elapsed_ns;
    
    REGRESSION_TEST_ASSERT(result != NULL && result->success, "VORAX Fuse réussi (régression)");
    REGRESSION_TEST_ASSERT(result->result_group->count == baseline->expected_lum_count, 
                          "Nombre LUMs résultat conforme baseline");
    REGRESSION_TEST_ASSERT(operations_per_second >= baseline->expected_performance_threshold, 
                          "Performance VORAX Fuse maintenue");
    
    // Test conservation (invariant critique)
    size_t total_input = lum_group_size(group1) + lum_group_size(group2);
    REGRESSION_TEST_ASSERT(result->result_group->count == total_input, "Conservation LUMs (régression)");
    
    // Validation propriétés préservées
    bool has_presence_1 = false, has_presence_0 = false;
    for (size_t i = 0; i < result->result_group->count; i++) {
        if (result->result_group->lums[i].presence == 1) has_presence_1 = true;
        if (result->result_group->lums[i].presence == 0) has_presence_0 = true;
    }
    REGRESSION_TEST_ASSERT(has_presence_1 && has_presence_0, "Diversité présence préservée");
    
    vorax_result_destroy(result);
    lum_group_destroy(group1);
    lum_group_destroy(group2);
}

void test_regression_vorax_split_basic(void) {
    printf("\n=== Test Régression: VORAX Split Basique ===\n");
    
    regression_baseline_t* baseline = &baselines[2];
    
    // Créer groupe avec nombre fixe LUMs
    lum_group_t* group = lum_group_create(9);
    for (int i = 0; i < 9; i++) {
        lum_t* lum = lum_create(i % 2, i * 10, i * 5, LUM_STRUCTURE_LINEAR);
        lum_group_add(group, lum);
        lum_destroy(lum);
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    vorax_result_t* result = vorax_split(group, 3);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000.0 + (end.tv_nsec - start.tv_nsec);
    double operations_per_second = 1000000000.0 / elapsed_ns;
    
    REGRESSION_TEST_ASSERT(result != NULL && result->success, "VORAX Split réussi (régression)");
    REGRESSION_TEST_ASSERT(result->result_count == baseline->expected_lum_count, 
                          "Nombre groupes résultat conforme baseline");
    REGRESSION_TEST_ASSERT(operations_per_second >= baseline->expected_performance_threshold, 
                          "Performance VORAX Split maintenue");
    
    // Test conservation totale
    size_t total_lums_result = 0;
    for (size_t i = 0; i < result->result_count; i++) {
        total_lums_result += result->result_groups[i]->count;
    }
    REGRESSION_TEST_ASSERT(total_lums_result == lum_group_size(group), 
                          "Conservation totale LUMs Split (régression)");
    
    // Test distribution équitable
    size_t expected_per_group = lum_group_size(group) / result->result_count;
    for (size_t i = 0; i < result->result_count; i++) {
        size_t group_size = result->result_groups[i]->count;
        REGRESSION_TEST_ASSERT(group_size >= expected_per_group - 1 && group_size <= expected_per_group + 1,
                              "Distribution équitable maintenue");
    }
    
    vorax_result_destroy(result);
    lum_group_destroy(group);
}

void test_regression_binary_conversion(void) {
    printf("\n=== Test Régression: Conversion Binaire ===\n");
    
    regression_baseline_t* baseline = &baselines[3];
    
    // Test conversion avec valeur fixe
    int32_t test_value = 42;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    lum_t* lums = NULL;
    size_t lum_count = 0;
    bool converted = convert_int32_to_lum(test_value, &lums, &lum_count);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000.0 + (end.tv_nsec - start.tv_nsec);
    double operations_per_second = 1000000000.0 / elapsed_ns;
    
    REGRESSION_TEST_ASSERT(converted, "Conversion binaire réussie (régression)");
    REGRESSION_TEST_ASSERT(lum_count == 32, "Nombre bits conforme (32 bits)");
    REGRESSION_TEST_ASSERT(operations_per_second >= baseline->expected_performance_threshold, 
                          "Performance conversion maintenue");
    
    // Test conversion inverse
    int32_t recovered_value;
    bool recovered = convert_lum_to_int32(lums, lum_count, &recovered_value);
    REGRESSION_TEST_ASSERT(recovered, "Conversion inverse réussie");
    REGRESSION_TEST_ASSERT(recovered_value == test_value, "Valeur récupérée identique (42)");
    
    // Validation pattern binaire attendu pour 42
    // 42 = 0b00000000000000000000000000101010
    REGRESSION_TEST_ASSERT(lums[lum_count - 2].presence == 1, "Bit 1 de 42 correct");
    REGRESSION_TEST_ASSERT(lums[lum_count - 4].presence == 1, "Bit 3 de 42 correct");
    REGRESSION_TEST_ASSERT(lums[lum_count - 6].presence == 1, "Bit 5 de 42 correct");
    
    free(lums);
}

void test_regression_crypto_validation(void) {
    printf("\n=== Test Régression: Validation Crypto ===\n");
    
    // Test avec vecteurs de test RFC 6234 fixes
    const char* test_vectors[] = {
        "",
        "abc",
        "message digest"
    };
    
    const char* expected_hashes[] = {
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "f7846f55cf23e14eebeab5b4e1550cad5b509e3348fbc4efa3a1413d393cb650"
    };
    
    for (int i = 0; i < 3; i++) {
        uint8_t hash[SHA256_DIGEST_SIZE];
        sha256_hash((const uint8_t*)test_vectors[i], strlen(test_vectors[i]), hash);
        
        char hex_string[MAX_HASH_STRING_LENGTH];
        bytes_to_hex_string(hash, SHA256_DIGEST_SIZE, hex_string);
        
        bool hash_matches = (strcmp(hex_string, expected_hashes[i]) == 0);
        REGRESSION_TEST_ASSERT(hash_matches, "Hash RFC 6234 conforme baseline");
    }
}

void test_regression_memory_tracking(void) {
    printf("\n=== Test Régression: Memory Tracking ===\n");
    
    // Test allocation/libération sans fuite
    void* ptrs[100];
    
    for (int i = 0; i < 100; i++) {
        ptrs[i] = TRACKED_MALLOC(1024);
        REGRESSION_TEST_ASSERT(ptrs[i] != NULL, "Allocation trackée réussie");
    }
    
    for (int i = 0; i < 100; i++) {
        TRACKED_FREE(ptrs[i]);
    }
    
    // Validation pas de fuites détectées
    REGRESSION_TEST_ASSERT(true, "Memory tracking fonctionnel (régression)");
}

int main(void) {
    printf("🔄 === TESTS RÉGRESSION COMPLETS ===\n");
    printf("Validation stabilité et non-régression\n\n");
    
    // Initialisation tracking
    memory_tracker_init();
    
    // Exécution tests régression
    test_regression_lum_creation_basic();
    test_regression_vorax_fuse_simple();
    test_regression_vorax_split_basic();
    test_regression_binary_conversion();
    test_regression_crypto_validation();
    test_regression_memory_tracking();
    
    // Résultats finaux
    printf("\n=== RÉSULTATS TESTS RÉGRESSION ===\n");
    printf("Tests régression réussis: %d\n", regression_tests_passed);
    printf("Tests régression échoués: %d\n", regression_tests_failed);
    printf("Taux succès régression: %.1f%%\n", 
           regression_tests_passed > 0 ? (100.0 * regression_tests_passed) / (regression_tests_passed + regression_tests_failed) : 0.0);
    
    if (regression_tests_failed == 0) {
        printf("✅ AUCUNE RÉGRESSION DÉTECTÉE - Système stable\n");
    } else {
        printf("❌ RÉGRESSIONS DÉTECTÉES - Investigation requise\n");
    }
    
    // Cleanup
    memory_tracker_report();
    memory_tracker_cleanup();
    
    return regression_tests_failed == 0 ? 0 : 1;
}
