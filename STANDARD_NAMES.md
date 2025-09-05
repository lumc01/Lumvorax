
# STANDARD_NAMES.md - Registre des Noms Standards LUM/VORAX

**Date de création initiale:** 2025-01-05

## 1. STRUCTURES DE DONNÉES (2025-01-05)

### Structures Core LUM
- `lum_t` - Structure LUM de base
- `lum_group_t` - Groupe de LUMs
- `lum_zone_t` - Zone de stockage LUM
- `lum_memory_t` - Mémoire LUM

### Structures Crypto
- `sha256_context_t` - Contexte SHA-256
- `crypto_validation_result_t` - Résultat validation crypto
- `file_integrity_t` - Intégrité fichier
- `integrity_database_t` - Base de données intégrité
- `custody_record_t` - Enregistrement chaîne de custody
- `custody_chain_t` - Chaîne de custody complète

### Structures Parser
- `ast_node_t` - Nœud AST
- `parser_context_t` - Contexte parser
- `token_t` - Token de parsing

### Structures Performance
- `performance_metrics_t` - Métriques de performance
- `memory_stats_t` - Statistiques mémoire
- `cpu_usage_t` - Utilisation CPU

## 2. FONCTIONS (2025-01-05)

### Fonctions LUM Core
- `lum_create()` - Création LUM
- `lum_destroy()` - Destruction LUM
- `lum_group_create()` - Création groupe
- `lum_group_destroy()` - Destruction groupe
- `lum_zone_create()` - Création zone
- `lum_zone_destroy()` - Destruction zone

### Fonctions Crypto
- `sha256_init()` - Initialisation SHA-256
- `sha256_update()` - Mise à jour SHA-256
- `sha256_final()` - Finalisation SHA-256
- `sha256_hash()` - Hash direct SHA-256
- `bytes_to_hex_string()` - Conversion bytes vers hex
- `hex_string_to_bytes()` - Conversion hex vers bytes
- `compute_file_hash()` - Calcul hash fichier
- `compute_data_hash()` - Calcul hash données
- `crypto_validate_data()` - Validation données crypto
- `crypto_validate_file()` - Validation fichier crypto
- `crypto_validate_lum_data()` - Validation données LUM
- `crypto_validate_execution_log()` - Validation log exécution
- `crypto_validate_source_files()` - Validation fichiers source
- `crypto_validate_sha256_implementation()` - **FONCTION MANQUANTE**

### Fonctions Parser
- `parse_vorax_code()` - Parse code VORAX
- `ast_create_node()` - Création nœud AST
- `ast_destroy_tree()` - Destruction arbre AST

### Fonctions Optimisation
- `memory_optimizer_create()` - Création optimiseur mémoire
- `memory_optimizer_destroy()` - Destruction optimiseur
- `parallel_processor_create()` - Création processeur parallèle
- `parallel_processor_destroy()` - Destruction processeur

## 3. CONSTANTES (2025-01-05)

### Constantes Crypto
- `SHA256_DIGEST_SIZE` = 32
- `SHA256_BLOCK_SIZE` = 64
- `MAX_HASH_STRING_LENGTH` = 65

### Constantes LUM
- `LUM_STRUCTURE_LINEAR` - Structure linéaire
- `LUM_STRUCTURE_TREE` - Structure arbre
- `LUM_STRUCTURE_GRAPH` - Structure graphe

## 4. ERREURS DÉTECTÉES (2025-01-05)

### Erreur Critique 1: Fonction Manquante
- **Nom:** `crypto_validate_sha256_implementation`
- **Localisation:** Appelée dans `src/main.c:54`
- **Status:** NON DÉFINIE
- **Impact:** Échec de compilation

## 5. NOUVEAUX NOMS REQUIS (2025-01-05)

### À Créer
- `crypto_validate_sha256_implementation()` - Validation implémentation SHA-256
