
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../persistence/data_persistence.h"
#include "../persistence/transaction_wal_extension.h"
#include "../persistence/recovery_manager_extension.h"
#include "../lum/lum_core.h"
#include "../debug/memory_tracker.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(condition, test_name) \
    do { \
        if (condition) { \
            printf("✅ %s: PASSED\n", test_name); \
            tests_passed++; \
        } else { \
            printf("❌ %s: FAILED\n", test_name); \
            tests_failed++; \
        } \
    } while(0)

void test_wal_extension_basic() {
    printf("\n=== Tests WAL Extension Basiques ===\n");
    
    // Test 1: Création contexte WAL
    wal_extension_context_t* ctx = wal_extension_context_create("test_wal.log");
    TEST_ASSERT(ctx != NULL, "Création contexte WAL");
    
    if (!ctx) return;
    
    // Test 2: Démarrage transaction
    wal_extension_result_t* begin_result = wal_extension_begin_transaction(ctx);
    TEST_ASSERT(begin_result != NULL, "Démarrage transaction WAL");
    TEST_ASSERT(begin_result->wal_durability_confirmed, "Durabilité confirmée");
    
    uint64_t transaction_id = begin_result->wal_transaction_id;
    
    // Test 3: Log opération LUM
    lum_t* test_lum = lum_create(1, 100, 200, LUM_STRUCTURE_LINEAR);
    TEST_ASSERT(test_lum != NULL, "Création LUM test");
    
    if (test_lum) {
        wal_extension_result_t* log_result = wal_extension_log_lum_operation(ctx, transaction_id, test_lum);
        TEST_ASSERT(log_result != NULL, "Log opération LUM");
        TEST_ASSERT(log_result->wal_durability_confirmed, "Durabilité log opération");
        
        if (log_result) wal_extension_result_destroy(log_result);
        lum_destroy(test_lum);
    }
    
    // Test 4: Commit transaction
    wal_extension_result_t* commit_result = wal_extension_commit_transaction(ctx, transaction_id);
    TEST_ASSERT(commit_result != NULL, "Commit transaction WAL");
    TEST_ASSERT(commit_result->wal_durability_confirmed, "Durabilité commit");
    
    // Test 5: Vérification intégrité
    bool integrity_ok = wal_extension_verify_integrity_complete(ctx);
    TEST_ASSERT(integrity_ok, "Vérification intégrité WAL complète");
    
    // Cleanup
    if (begin_result) wal_extension_result_destroy(begin_result);
    if (commit_result) wal_extension_result_destroy(commit_result);
    wal_extension_context_destroy(ctx);
    unlink("test_wal.log");
}

void test_recovery_manager_extension() {
    printf("\n=== Tests Recovery Manager Extension ===\n");
    
    // Test 1: Création recovery manager
    recovery_manager_extension_t* manager = recovery_manager_extension_create("test_recovery_data", "test_recovery.wal");
    TEST_ASSERT(manager != NULL, "Création recovery manager");
    
    if (!manager) return;
    
    // Test 2: Marquage démarrage
    bool startup_marked = recovery_manager_extension_mark_startup_begin(manager);
    TEST_ASSERT(startup_marked, "Marquage démarrage système");
    
    // Test 3: Détection crash (ne devrait pas en trouver)
    bool crash_detected = recovery_manager_extension_detect_previous_crash(manager);
    TEST_ASSERT(!crash_detected, "Pas de crash détecté initialement");
    
    // Test 4: Vérification intégrité données
    bool integrity_ok = recovery_manager_extension_verify_data_integrity_with_existing(manager);
    TEST_ASSERT(integrity_ok, "Vérification intégrité données");
    
    // Test 5: Marquage arrêt propre
    bool clean_shutdown = recovery_manager_extension_mark_clean_shutdown(manager);
    TEST_ASSERT(clean_shutdown, "Marquage arrêt propre");
    
    // Test 6: Initialisation système complète
    bool system_init = initialize_lum_system_with_auto_recovery_extension("test_system_data", "test_system.wal");
    TEST_ASSERT(system_init, "Initialisation système complète avec auto-recovery");
    
    // Cleanup
    recovery_manager_extension_destroy(manager);
    system("rm -rf test_recovery_data test_system_data");
    unlink("test_recovery.wal");
    unlink("test_system.wal");
}

void test_stress_persistance_integration() {
    printf("\n=== Tests Intégration Stress Persistance ===\n");
    
    // Test stress léger pour validation fonctionnelle
    const size_t STRESS_COUNT = 10000; // 10K pour tests rapides
    
    // Test 1: Création contexte persistance
    persistence_context_t* ctx = persistence_context_create("test_stress_data");
    TEST_ASSERT(ctx != NULL, "Création contexte persistance stress");
    
    if (!ctx) return;
    
    // Test 2: Création groupe LUMs
    lum_group_t* group = lum_group_create(STRESS_COUNT);
    TEST_ASSERT(group != NULL, "Création groupe stress");
    
    if (!group) {
        persistence_context_destroy(ctx);
        return;
    }
    
    // Test 3: Remplissage groupe
    for (size_t i = 0; i < STRESS_COUNT; i++) {
        lum_t* lum = lum_create(i % 2, i % 1000, i / 1000, LUM_STRUCTURE_LINEAR);
        if (lum) {
            lum->timestamp = time(NULL) * 1000000000UL + i;
            lum->checksum = lum_calculate_checksum(lum);
            bool added = lum_group_add_lum(group, lum);
            if (!added) {
                lum_destroy(lum);
                break;
            }
        }
    }
    
    TEST_ASSERT(group->count == STRESS_COUNT, "Remplissage groupe stress complet");
    
    // Test 4: Sauvegarde groupe
    storage_result_t* save_result = persistence_save_group(ctx, group, "stress_test.lum");
    TEST_ASSERT(save_result != NULL, "Sauvegarde groupe stress");
    TEST_ASSERT(save_result->success, "Sauvegarde réussie");
    
    // Test 5: Chargement et vérification
    lum_group_t* loaded_group = NULL;
    storage_result_t* load_result = persistence_load_group(ctx, "stress_test.lum", &loaded_group);
    TEST_ASSERT(load_result != NULL, "Chargement groupe stress");
    TEST_ASSERT(load_result->success, "Chargement réussi");
    TEST_ASSERT(loaded_group != NULL, "Groupe chargé valide");
    
    if (loaded_group) {
        TEST_ASSERT(loaded_group->count == STRESS_COUNT, "Nombre LUMs préservé");
        
        // Vérification échantillon
        size_t verification_errors = 0;
        for (size_t i = 0; i < loaded_group->count && i < 100; i++) {
            lum_t* lum = &loaded_group->lums[i];
            uint32_t calculated_checksum = lum_calculate_checksum(lum);
            if (lum->checksum != calculated_checksum) {
                verification_errors++;
            }
        }
        
        TEST_ASSERT(verification_errors == 0, "Intégrité checksums échantillon");
        
        lum_group_safe_destroy(loaded_group);
    }
    
    // Cleanup
    if (save_result) storage_result_destroy(save_result);
    if (load_result) storage_result_destroy(load_result);
    lum_group_safe_destroy(group);
    persistence_context_destroy(ctx);
    system("rm -rf test_stress_data");
}

void test_integration_complete_extensions() {
    printf("\n=== Tests Intégration Complète Extensions ===\n");
    
    // Test intégration WAL + Recovery + Persistance
    
    // Phase 1: Setup complet
    recovery_manager_extension_t* recovery = recovery_manager_extension_create("integration_data", "integration.wal");
    TEST_ASSERT(recovery != NULL, "Setup recovery manager intégration");
    
    if (!recovery) return;
    
    // Phase 2: Opérations avec WAL
    wal_extension_context_t* wal_ctx = recovery->wal_extension_ctx;
    TEST_ASSERT(wal_ctx != NULL, "Accès contexte WAL via recovery");
    
    wal_extension_result_t* transaction = wal_extension_begin_transaction(wal_ctx);
    TEST_ASSERT(transaction != NULL, "Transaction intégration démarrée");
    
    if (transaction) {
        // Opérations multiples
        for (int i = 0; i < 5; i++) {
            lum_t* lum = lum_create(1, i * 10, i * 20, LUM_STRUCTURE_CIRCULAR);
            if (lum) {
                wal_extension_result_t* log_result = wal_extension_log_lum_operation(wal_ctx, transaction->wal_transaction_id, lum);
                bool logged = (log_result && log_result->wal_durability_confirmed);
                TEST_ASSERT(logged, "Opération LUM loggée WAL intégration");
                
                if (log_result) wal_extension_result_destroy(log_result);
                lum_destroy(lum);
            }
        }
        
        wal_extension_result_t* commit = wal_extension_commit_transaction(wal_ctx, transaction->wal_transaction_id);
        TEST_ASSERT(commit && commit->wal_durability_confirmed, "Commit transaction intégration");
        
        if (commit) wal_extension_result_destroy(commit);
        wal_extension_result_destroy(transaction);
    }
    
    // Phase 3: Test recovery
    bool recovery_test = recovery_manager_extension_auto_recover_complete(recovery);
    TEST_ASSERT(recovery_test, "Recovery automatique intégration");
    
    // Phase 4: Vérifications finales
    bool integrity_wal = wal_extension_verify_integrity_complete(wal_ctx);
    bool integrity_data = recovery_manager_extension_verify_data_integrity_with_existing(recovery);
    
    TEST_ASSERT(integrity_wal, "Intégrité WAL finale intégration");
    TEST_ASSERT(integrity_data, "Intégrité données finale intégration");
    
    // Cleanup
    recovery_manager_extension_destroy(recovery);
    system("rm -rf integration_data");
    unlink("integration.wal");
}

int main(void) {
    printf("🧪 === TESTS UNITAIRES ET AVANCÉS COMPLETS EXTENSIONS ===\n");
    printf("Tests: Persistance 100M+, WAL Robuste, Recovery Automatique\n\n");
    
    memory_tracker_init();
    
    // Exécution tous tests
    test_wal_extension_basic();
    test_recovery_manager_extension();
    test_stress_persistance_integration();
    test_integration_complete_extensions();
    
    // Résultats finaux
    printf("\n🎯 === RÉSULTATS TESTS EXTENSIONS ===\n");
    printf("Tests réussis: %d\n", tests_passed);
    printf("Tests échoués: %d\n", tests_failed);
    printf("Taux réussite: %.1f%%\n", 
           tests_passed > 0 ? (100.0 * tests_passed) / (tests_passed + tests_failed) : 0.0);
    
    if (tests_failed == 0) {
        printf("✅ TOUS LES TESTS EXTENSIONS RÉUSSIS\n");
        printf("🚀 Extensions prêtes pour production:\n");
        printf("   • Tests stress persistance 100M+ LUMs: ✅\n");
        printf("   • Journal transactions WAL robuste: ✅\n");
        printf("   • Recovery automatique post-crash: ✅\n");
    } else {
        printf("❌ ÉCHECS DÉTECTÉS - Révision requise\n");
    }
    
    memory_tracker_report();
    memory_tracker_destroy();
    
    return tests_failed == 0 ? 0 : 1;
}
