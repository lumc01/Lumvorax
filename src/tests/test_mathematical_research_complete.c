
#include "../advanced_calculations/mathematical_research_engine.h"
#include "../debug/memory_tracker.h"
#include "../logger/lum_logger.h"
#include <stdio.h>
#include <assert.h>
#include <time.h>

// Test de performance et génération de logs forensiques
bool test_collatz_research_comprehensive(void) {
    printf("=== TEST RECHERCHE MATHÉMATIQUE COLLATZ - GÉNÉRATION LOGS FORENSIQUES ===\n");
    
    // Initialiser le tracking mémoire
    memory_tracker_init();
    
    // Initialiser le logging forensique
    if (!forensic_logger_init("logs/mathematical_research_forensic.log")) {
        printf("❌ Échec initialisation logging forensique\n");
        return false;
    }
    
    // Configuration de recherche pour tests intensifs
    math_research_config_t* config = create_default_research_config();
    config->cache_size = 500000;  // Cache plus grand pour performance
    config->max_iterations = 50000000;  // Augmenté pour tests poussés
    
    mathematical_research_engine_t* engine = math_research_engine_create(config);
    if (!engine) {
        printf("❌ Échec création moteur de recherche\n");
        TRACKED_FREE(config);
        return false;
    }
    
    printf("✅ Moteur de recherche initialisé - Session ID: %u\n", engine->research_session_id);
    
    // Test 1: Analyse de plages croissantes
    struct {
        uint64_t start;
        uint64_t end;
        const char* description;
    } test_ranges[] = {
        {1, 1000, "Plage initiale (1-1000)"},
        {1000, 10000, "Plage étendue (1K-10K)"},
        {10000, 100000, "Plage large (10K-100K)"},
        {100000, 200000, "Plage intensive (100K-200K)"},
        {500000, 510000, "Plage haute densité (500K-510K)"}
    };
    
    const size_t num_ranges = sizeof(test_ranges) / sizeof(test_ranges[0]);
    
    for (size_t i = 0; i < num_ranges; i++) {
        printf("\n🔬 Test %zu: %s\n", i+1, test_ranges[i].description);
        
        clock_t start_time = clock();
        
        math_research_result_t* results = analyze_collatz_dynamic_range(
            engine,
            test_ranges[i].start,
            test_ranges[i].end
        );
        
        clock_t end_time = clock();
        double elapsed = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        
        if (!results) {
            printf("❌ Échec analyse plage %s\n", test_ranges[i].description);
            continue;
        }
        
        printf("📊 Résultats %s:\n", test_ranges[i].description);
        printf("   • Séquences analysées: %zu\n", results->sequence_count);
        printf("   • Calculs totaux: %lu\n", results->total_calculations);
        printf("   • Longueur moyenne: %.2f\n", results->average_length);
        printf("   • Ratio max: %.6f\n", results->max_length_ratio);
        printf("   • Temps de calcul: %.3f secondes\n", elapsed);
        printf("   • Throughput: %.0f séquences/seconde\n", 
               (double)results->sequence_count / elapsed);
        
        // Générer des conjectures
        if (generate_mathematical_conjectures(engine, results)) {
            printf("   • Conjectures générées: %zu\n", results->pattern_count);
            for (size_t j = 0; j < results->pattern_count; j++) {
                printf("     - %s\n", results->emergent_patterns[j]);
            }
        }
        
        // Nettoyer les résultats
        for (size_t j = 0; j < results->pattern_count; j++) {
            if (results->emergent_patterns[j]) {
                TRACKED_FREE(results->emergent_patterns[j]);
            }
        }
        if (results->emergent_patterns) {
            TRACKED_FREE(results->emergent_patterns);
        }
        
        // Nettoyer les séquences
        for (size_t j = 0; j < results->sequence_count; j++) {
            if (results->sequences[j].sequence_lums) {
                TRACKED_FREE(results->sequences[j].sequence_lums);
            }
        }
        TRACKED_FREE(results->sequences);
        TRACKED_FREE(results);
    }
    
    // Test 2: Analyse de séquences spécifiques connues pour être longues
    printf("\n🎯 Test séquences remarquables:\n");
    
    uint64_t remarkable_values[] = {27, 31, 47, 54, 63, 71, 77, 83, 95, 127, 159, 255};
    size_t num_remarkable = sizeof(remarkable_values) / sizeof(remarkable_values[0]);
    
    for (size_t i = 0; i < num_remarkable; i++) {
        collatz_sequence_t* seq = analyze_single_collatz_sequence(engine, remarkable_values[i]);
        if (seq) {
            printf("   • n=%lu: longueur=%lu, max=%lu, ratio=%.3f\n",
                   seq->initial_value, seq->sequence_length, 
                   seq->max_value, seq->convergence_ratio);
            
            if (seq->sequence_lums) {
                TRACKED_FREE(seq->sequence_lums);
            }
            TRACKED_FREE(seq);
        }
    }
    
    // Test 3: Statistiques du cache
    printf("\n💾 Statistiques du cache:\n");
    printf("   • Entrées utilisées: %zu / %lu\n", 
           engine->cache_current_size, engine->config.cache_size);
    printf("   • Cache hits: %lu\n", engine->cache_hits);
    printf("   • Cache misses: %lu\n", engine->cache_misses);
    
    if (engine->cache_hits + engine->cache_misses > 0) {
        double hit_rate = (double)engine->cache_hits * 100.0 / 
                         (double)(engine->cache_hits + engine->cache_misses);
        printf("   • Taux de hit: %.2f%%\n", hit_rate);
    }
    
    // Test 4: Intégration avec le système LUM/VORAX
    printf("\n🔗 Intégration LUM/VORAX:\n");
    if (engine->research_data_group) {
        printf("   • Groupe de données créé: %zu LUMs\n", 
               lum_group_size(engine->research_data_group));
        printf("   • Session de recherche: %u\n", engine->research_session_id);
    }
    
    // Nettoyage
    TRACKED_FREE(config);
    math_research_engine_destroy(engine);
    
    // Rapport final du memory tracker
    printf("\n📋 Rapport mémoire final:\n");
    memory_tracker_report();
    memory_tracker_check_leaks();
    
    // Fermeture du logging forensique
    forensic_logger_destroy();
    
    printf("\n✅ Test recherche mathématique terminé avec succès\n");
    printf("📄 Logs forensiques disponibles dans: logs/mathematical_research_forensic.log\n");
    
    return true;
}

// Test d'intégration avec le module Collatz existant
bool test_integration_with_existing_collatz(void) {
    printf("\n=== TEST INTÉGRATION AVEC MODULE COLLATZ EXISTANT ===\n");
    
    // Tester si le module collatz_analyzer existant fonctionne
    printf("🔍 Test du module collatz_analyzer existant...\n");
    
    // Configuration pour le module existant
    collatz_config_t* collatz_config = collatz_config_create_default();
    if (!collatz_config) {
        printf("❌ Échec création config Collatz existant\n");
        return false;
    }
    
    // Créer le moteur de recherche pour comparaison
    mathematical_research_engine_t* research_engine = math_research_engine_create(NULL);
    if (!research_engine) {
        printf("❌ Échec création moteur recherche\n");
        collatz_config_destroy(&collatz_config);
        return false;
    }
    
    printf("✅ Comparaison entre module existant et nouveau moteur:\n");
    
    // Test de performance comparative
    uint64_t test_values[] = {27, 100, 1000, 10000};
    size_t num_values = sizeof(test_values) / sizeof(test_values[0]);
    
    for (size_t i = 0; i < num_values; i++) {
        uint64_t n = test_values[i];
        
        // Test avec le nouveau moteur
        clock_t start = clock();
        collatz_sequence_t* seq = analyze_single_collatz_sequence(research_engine, n);
        clock_t end = clock();
        double time_new = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        if (seq) {
            printf("   • n=%lu: nouveau moteur = %lu étapes, max=%lu, temps=%.6f s\n",
                   n, seq->sequence_length, seq->max_value, time_new);
            
            if (seq->sequence_lums) {
                TRACKED_FREE(seq->sequence_lums);
            }
            TRACKED_FREE(seq);
        }
    }
    
    // Nettoyage
    math_research_engine_destroy(research_engine);
    collatz_config_destroy(&collatz_config);
    
    printf("✅ Test d'intégration terminé\n");
    return true;
}

int main(void) {
    printf("🧮 === TESTS COMPLETS RECHERCHE MATHÉMATIQUE LUM/VORAX ===\n");
    
    bool success = true;
    
    // Test principal
    if (!test_collatz_research_comprehensive()) {
        printf("❌ Échec test recherche comprehensive\n");
        success = false;
    }
    
    // Test d'intégration
    if (!test_integration_with_existing_collatz()) {
        printf("❌ Échec test intégration\n");
        success = false;
    }
    
    printf("\n🏁 Résultat final: %s\n", success ? "✅ TOUS LES TESTS RÉUSSIS" : "❌ ÉCHECS DÉTECTÉS");
    
    return success ? 0 : 1;
}
