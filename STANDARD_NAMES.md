
# FICHIER DES NOMS STANDARDISÉS - SYSTÈME LUM/VORAX

## STRUCTURES DE DONNÉES

### Types de base LUM
- `lum_t` : Structure principale LUM (presence, position_x, position_y, structure_type)
- `lum_group_t` : Groupe de LUMs (lums[], count, capacity)
- `lum_zone_t` : Zone spatiale contenant des LUMs
- `lum_memory_t` : Mémoire pour stockage LUMs
- `lum_structure_e` : Énumération des types de structure (LINEAR, CIRCULAR, BINARY)

### Types VORAX Operations
- `vorax_operation_e` : Types d'opérations (FUSE, SPLIT, CYCLE, MOVE, etc.)
- `vorax_result_t` : Résultat d'opération VORAX
- `vorax_ast_node_t` : Nœud AST du parser
- `vorax_execution_context_t` : Contexte d'exécution

### Types Conversion Binaire
- `binary_lum_result_t` : Résultat de conversion binaire
- `conversion_config_t` : Configuration de conversion

### Types Logging
- `lum_logger_t` : Logger principal
- `lum_log_level_e` : Niveaux de log (DEBUG, INFO, WARNING, ERROR)

### Types Processing Parallèle
- `parallel_processor_t` : Processeur parallèle principal
- `parallel_task_t` : Tâche parallèle
- `parallel_task_type_e` : Types de tâches parallèles
- `task_queue_t` : Queue de tâches
- `worker_thread_t` : Information thread worker
- `thread_pool_t` : Pool de threads (compatibilité)
- `parallel_process_result_t` : Résultat de traitement parallèle
- `work_distribution_t` : Distribution de travail

### Types Modules Avancés
- `memory_pool_t` : Pool mémoire optimisé
- `memory_stats_t` : Statistiques mémoire
- `metrics_collector_t` : Collecteur de métriques
- `perf_timer_t` : Timer de performance
- `memory_usage_t` : Usage mémoire
- `cpu_stats_t` : Statistiques CPU
- `throughput_stats_t` : Statistiques débit
- `performance_profile_t` : Profil de performance
- `hash_calculator_t` : Calculateur de hash
- `hash_result_t` : Résultat de hash
- `integrity_result_t` : Résultat intégrité
- `signature_result_t` : Résultat signature
- `storage_backend_t` : Backend de stockage
- `serialized_data_t` : Données sérialisées
- `transaction_t` : Transaction de données

## CONSTANTES ET ENUMS

### Constantes système
- `MAX_WORKER_THREADS` : 16
- `DEFAULT_WORKER_COUNT` : 4
- `LUM_MAX_GROUPS` : 1024
- `VORAX_MAX_ZONES` : 256
- `VORAX_MAX_MEMORIES` : 128

### Types de hachage
- `HASH_SHA256` : Algorithme SHA-256

## FONCTIONS PRINCIPALES

### LUM Core
- `lum_create()`, `lum_destroy()`, `lum_print()`
- `lum_group_*()` : Gestion groupes
- `lum_zone_*()` : Gestion zones
- `lum_memory_*()` : Gestion mémoire

### VORAX Operations
- `vorax_fuse()`, `vorax_split()`, `vorax_cycle()`
- `vorax_move()`, `vorax_store()`, `vorax_retrieve()`
- `vorax_compress()`, `vorax_expand()`

### Processing Parallèle
- `parallel_processor_*()` : Gestion processeur
- `thread_pool_*()` : Gestion pool threads
- `parallel_process_lums()` : Traitement haut niveau

### Modules Avancés
- `memory_pool_*()` : Optimisation mémoire
- `metrics_collector_*()` : Collecte métriques
- `hash_calculator_*()` : Validation crypto
- `storage_backend_*()` : Persistance données

## CONVENTIONS DE NOMMAGE

- Structures : `nom_t`
- Énumérations : `nom_e`  
- Fonctions : `module_action()`
- Constantes : `MODULE_CONSTANT`
- Variables : `snake_case`
