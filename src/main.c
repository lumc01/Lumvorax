#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "lum/lum_core.h"
#include "vorax/vorax_operations.h"
#include "parser/vorax_parser.h"
#include "binary/binary_lum_converter.h"
#include "logger/lum_logger.h"
#include "logger/log_manager.h"
#include "persistence/storage_backend.h" // Added for persistence tests

// Demo functions
void demo_basic_lum_operations(void);
void demo_vorax_operations(void);
void demo_binary_conversion(void);
void demo_parser(void);
void demo_complete_scenario(void);
void test_persistence_integration(void); // Added declaration for persistence test

int main(int argc __attribute__((unused)), char* argv[] __attribute__((unused))) {
    printf("=== LUM/VORAX System Demo ===\n");
    printf("Implementation complete du concept LUM/VORAX en C\n\n");

    // Initialize automatic log management system
    log_manager_t* log_manager = log_manager_create();
    if (!log_manager) {
        printf("Erreur: Impossible de créer le gestionnaire de logs\n");
        return 1;
    }

    LOG_MODULE("system", "INFO", "LUM/VORAX System Demo Started");
    LOG_MODULE("system", "INFO", "Log Manager Session: %s", log_manager->session_id);

    // Initialize main logger
    lum_logger_t* logger = lum_logger_create("logs/lum_vorax.log", true, true);
    if (!logger) {
        printf("Erreur: Impossible de créer le logger\n");
        return 1;
    }

    lum_logger_set_level(logger, LUM_LOG_INFO);
    lum_logger_enable_tracing(logger, true);

    // Set as global logger for system-wide usage
    lum_set_global_logger(logger);

    // Log system startup
    lum_log_message(logger, LUM_LOG_INFO, "LUM/VORAX System Demo Started");

    printf("1. Test des opérations de base LUM...\n");
    demo_basic_lum_operations();

    printf("\n2. Test des opérations VORAX...\n");
    demo_vorax_operations();

    printf("\n3. Test de conversion binaire <-> LUM...\n");
    demo_binary_conversion();

    printf("\n4. Test du parser VORAX...\n");
    demo_parser();

    printf("\n5. Scénario complet...\n");
    demo_complete_scenario();

    printf("\n6. Test persistance complète\n"); // Added menu option for persistence test
    test_persistence_integration(); // Added call to persistence test

    printf("\nDémo terminée avec succès!\n");
    printf("Consultez le fichier lum_vorax.log pour les détails.\n");

    lum_logger_destroy(logger);
    return 0;
}

void demo_basic_lum_operations(void) {
    LOG_LUM_CORE("INFO", "Starting basic LUM operations demo");

    // Créer des LUMs individuelles
    lum_t* lum1 = lum_create(1, 0, 0, LUM_STRUCTURE_LINEAR);
    lum_t* lum2 = lum_create(1, 1, 0, LUM_STRUCTURE_LINEAR);
    lum_t* lum3 = lum_create(0, 2, 0, LUM_STRUCTURE_LINEAR);

    LOG_LUM_CORE("INFO", "Created 3 LUMs: lum1=%p, lum2=%p, lum3=%p", lum1, lum2, lum3);

    if (lum1 && lum2 && lum3) {
        printf("  ✓ Création de 3 LUMs: ");
        lum_print(lum1);
        lum_print(lum2);
        lum_print(lum3);

        // Créer un groupe
        lum_group_t* group = lum_group_create(10);
        if (group) {
            lum_group_add(group, lum1);
            lum_group_add(group, lum2);
            lum_group_add(group, lum3);

            printf("  ✓ Groupe créé avec %zu LUMs\n", lum_group_size(group));
            lum_group_print(group);

            lum_group_destroy(group);
        }

        lum_destroy(lum1);
        lum_destroy(lum2);
        lum_destroy(lum3);
    }
}

void demo_vorax_operations(void) {
    // Créer deux groupes pour la fusion
    lum_group_t* group1 = lum_group_create(5);
    lum_group_t* group2 = lum_group_create(5);

    if (!group1 || !group2) {
        printf("  ✗ Erreur création des groupes\n");
        return;
    }

    // Ajouter des LUMs aux groupes
    for (int i = 0; i < 3; i++) {
        lum_t* lum = lum_create(1, i, 0, LUM_STRUCTURE_LINEAR);
        if (lum) {
            lum_group_add(group1, lum);
            lum_destroy(lum);
        }
    }

    for (int i = 0; i < 2; i++) {
        lum_t* lum = lum_create(1, i, 1, LUM_STRUCTURE_LINEAR);
        if (lum) {
            lum_group_add(group2, lum);
            lum_destroy(lum);
        }
    }

    printf("  Groupe 1: %zu LUMs, Groupe 2: %zu LUMs\n", group1->count, group2->count);

    // Test fusion (⧉)
    vorax_result_t* fuse_result = vorax_fuse(group1, group2);
    if (fuse_result && fuse_result->success) {
        printf("  ✓ Fusion réussie: %zu LUMs -> %zu LUMs\n", 
               group1->count + group2->count, fuse_result->result_group->count);

        // Test split (⇅)
        vorax_result_t* split_result = vorax_split(fuse_result->result_group, 2);
        if (split_result && split_result->success) {
            printf("  ✓ Split réussi: %zu LUMs -> %zu groupes\n",
                   fuse_result->result_group->count, split_result->result_count);

            // Test cycle (⟲)
            if (split_result->result_count > 0) {
                vorax_result_t* cycle_result = vorax_cycle(split_result->result_groups[0], 3);
                if (cycle_result && cycle_result->success) {
                    printf("  ✓ Cycle réussi: %s\n", cycle_result->message);
                }
                vorax_result_destroy(cycle_result);
            }
        }
        vorax_result_destroy(split_result);
    }
    vorax_result_destroy(fuse_result);

    lum_group_destroy(group1);
    lum_group_destroy(group2);
}

void demo_binary_conversion(void) {
    // Test conversion entier -> LUM
    int32_t test_value = 42;
    printf("  Conversion de l'entier %d en LUMs...\n", test_value);

    binary_lum_result_t* result = convert_int32_to_lum(test_value);
    if (result && result->success) {
        printf("  ✓ Conversion réussie: %zu bits traités\n", result->bits_processed);

        // Afficher la représentation binaire
        char* binary_str = lum_group_to_binary_string(result->lum_group);
        if (binary_str) {
            printf("  Binaire: %s\n", binary_str);
            free(binary_str);
        }

        // Test conversion inverse
        int32_t converted_back = convert_lum_to_int32(result->lum_group);
        printf("  ✓ Conversion inverse: %d -> %d %s\n", 
               test_value, converted_back, 
               (test_value == converted_back) ? "(OK)" : "(ERREUR)");
    }
    binary_lum_result_destroy(result);

    // Test conversion chaîne binaire -> LUM
    const char* bit_string = "11010110";
    printf("\n  Conversion de la chaîne binaire '%s' en LUMs...\n", bit_string);

    binary_lum_result_t* bit_result = convert_bits_to_lum(bit_string);
    if (bit_result && bit_result->success) {
        printf("  ✓ Conversion réussie: %zu LUMs créées\n", bit_result->lum_group->count);
        lum_group_print(bit_result->lum_group);
    }
    binary_lum_result_destroy(bit_result);
}

void demo_parser(void) {
    const char* vorax_code = 
        "zone A, B, C;\n"
        "mem buf;\n"
        "emit A += 3•;\n"
        "split A -> [B, C];\n"
        "move B -> C, 1•;\n";

    printf("  Parsing du code VORAX:\n%s\n", vorax_code);

    vorax_ast_node_t* ast = vorax_parse(vorax_code);
    if (ast) {
        printf("  ✓ Parsing réussi, AST créé:\n");
        vorax_ast_print(ast, 2);

        // Test d'exécution
        vorax_execution_context_t* ctx = vorax_execution_context_create();
        if (ctx) {
            bool exec_result = vorax_execute(ctx, ast);
            printf("  ✓ Exécution: %s\n", exec_result ? "Succès" : "Échec");
            printf("  Zones créées: %zu\n", ctx->zone_count);
            printf("  Mémoires créées: %zu\n", ctx->memory_count);

            vorax_execution_context_destroy(ctx);
        }

        vorax_ast_destroy(ast);
    } else {
        printf("  ✗ Erreur de parsing\n");
    }
}

void demo_complete_scenario(void) {
    printf("  Scénario: Pipeline de traitement LUM avec logging complet\n");

    // Créer le contexte d'exécution
    vorax_execution_context_t* ctx = vorax_execution_context_create();
    if (!ctx) {
        printf("  ✗ Erreur création contexte\n");
        return;
    }

    // Créer zones et mémoire
    vorax_context_add_zone(ctx, "Input");
    vorax_context_add_zone(ctx, "Process");
    vorax_context_add_zone(ctx, "Output");
    vorax_context_add_memory(ctx, "buffer");

    // Récupérer les zones
    lum_zone_t* input_zone = vorax_context_find_zone(ctx, "Input");
    lum_zone_t* process_zone = vorax_context_find_zone(ctx, "Process");
    lum_zone_t* output_zone = vorax_context_find_zone(ctx, "Output");
    lum_memory_t* buffer_mem = vorax_context_find_memory(ctx, "buffer");

    if (input_zone && process_zone && output_zone && buffer_mem) {
        // Émettre des LUMs dans la zone d'entrée
        vorax_result_t* emit_result = vorax_emit_lums(input_zone, 7);
        if (emit_result && emit_result->success) {
            printf("  ✓ Émission de 7 LUMs dans Input\n");

            // Déplacer vers Process
            vorax_result_t* move_result = vorax_move(input_zone, process_zone, 7);
            if (move_result && move_result->success) {
                printf("  ✓ Déplacement vers Process: %s\n", move_result->message);

                // Stocker quelques LUMs en mémoire
                vorax_result_t* store_result = vorax_store(buffer_mem, process_zone, 2);
                if (store_result && store_result->success) {
                    printf("  ✓ Stockage en mémoire: %s\n", store_result->message);

                    // Récupérer depuis la mémoire vers Output
                    vorax_result_t* retrieve_result = vorax_retrieve(buffer_mem, output_zone);
                    if (retrieve_result && retrieve_result->success) {
                        printf("  ✓ Récupération vers Output: %s\n", retrieve_result->message);
                    }
                    vorax_result_destroy(retrieve_result);
                }
                vorax_result_destroy(store_result);
            }
            vorax_result_destroy(move_result);
        }
        vorax_result_destroy(emit_result);

        printf("  État final:\n");
        printf("    Input: %s\n", lum_zone_is_empty(input_zone) ? "vide" : "non-vide");
        printf("    Process: %s\n", lum_zone_is_empty(process_zone) ? "vide" : "non-vide");
        printf("    Output: %s\n", lum_zone_is_empty(output_zone) ? "vide" : "non-vide");
        printf("    Buffer: %s\n", buffer_mem->is_occupied ? "occupé" : "vide");
    }

    vorax_execution_context_destroy(ctx);
    printf("  ✓ Scénario complet terminé\n");
}

// Added function for persistence integration testing
void test_persistence_integration(void) {
    printf("🔄 Test persistance avec extensions WAL/Recovery...\n");

    // Test 1: Persistance de base
    storage_backend_t* backend = storage_backend_create("test_persistence.db");
    if (!backend) {
        printf("❌ Échec création backend persistance\n");
        return;
    }

    // Test 2: Création et stockage 1000 LUMs
    printf("📝 Stockage 1000 LUMs...\n");
    for (int i = 0; i < 1000; i++) {
        lum_t* lum = lum_create(i % 2, i * 10, i * 5, LUM_STRUCTURE_LINEAR);
        char key[64];
        snprintf(key, sizeof(key), "test_lum_%d", i);

        bool stored = store_lum(backend, key, lum);
        if (!stored) {
            printf("❌ Échec stockage LUM %d\n", i);
        }
        lum_destroy(lum);
    }

    // Test 3: Récupération échantillon
    printf("📖 Récupération échantillon...\n");
    for (int i = 0; i < 10; i++) {
        char key[64];
        snprintf(key, sizeof(key), "test_lum_%d", i * 100);

        lum_t* loaded = load_lum(backend, key);
        if (loaded) {
            printf("✅ LUM %d récupéré: pos=(%d,%d)\n", i, loaded->position_x, loaded->position_y);
            lum_destroy(loaded);
        }
    }

    storage_backend_destroy(backend);
    printf("✅ Test persistance terminé\n");
}