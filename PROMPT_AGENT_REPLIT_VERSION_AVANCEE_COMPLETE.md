
# PROMPT COMPLET POUR AGENT REPLIT - VERSION AVANCÉE LUM/VORAX

## OBJECTIF PRINCIPAL
Implémenter une version avancée 100% fonctionnelle du système LUM/VORAX avec performances optimales, sans warnings, sans erreurs, et sans placeholders.

## RÈGLES ABSOLUES
1. **AUCUNE SIMPLIFICATION** - Maintenir la complexité avancée existante
2. **AUCUN PLACEHOLDER** - Tout le code doit être fonctionnel
3. **AUCUN HARDCODING** - Utiliser des algorithmes dynamiques réels
4. **ZÉRO WARNING** - Compilation propre obligatoire
5. **TESTS RÉELS** - Tous les logs doivent refléter des tests authentiques

## MODULES À OPTIMISER INTÉGRALEMENT

### 1. MODULES CORE (src/lum/, src/vorax/)
- **lum_core.c** : Optimiser les allocations mémoire avec zero-copy
- **vorax_operations.c** : Implémenter les algorithmes VORAX complets
- **Objectif** : 50M+ LUMs/seconde minimum

### 2. CALCULS AVANCÉS (src/advanced_calculations/)
- **matrix_calculator.c** : Multiplication matricielle SIMD AVX-512
- **quantum_simulator.c** : Simulation quantique réelle avec 64+ qubits
- **neural_network_processor.c** : Réseaux neuronaux avec backpropagation complète
- **neural_blackbox_computer.c** : Encodage neuronal 100% natif
- **Objectif** : Traitement 100M+ éléments par module

### 3. OPTIMISATIONS (src/optimization/)
- **simd_optimizer.c** : Support AVX-512, FMA, vectorisation complète
- **memory_optimizer.c** : Pool mémoire avec fragmentation < 5%
- **pareto_optimizer.c** : Optimisation multi-objectifs réelle
- **zero_copy_allocator.c** : Allocation sans copie mémoire
- **Objectif** : Accélération x16 minimum vs version de base

### 4. MODULES COMPLEXES (src/complex_modules/)
- **realtime_analytics.c** : Analyse temps réel avec ML intégré
- **distributed_computing.c** : Calcul distribué multi-nœuds
- **ai_optimization.c** : IA adaptative avec apprentissage continu
- **Objectif** : Traitement temps réel < 1ms latence

## PERFORMANCES MINIMALES REQUISES

### Métriques Core
- **Création LUM** : < 10 ns par LUM
- **Opérations VORAX** : < 50 ns par opération
- **Allocation mémoire** : < 5 ns avec zero-copy
- **Validation crypto** : SHA-256 > 500 MB/s

### Métriques Avancées
- **Matrix Calculator** : > 100 GFLOPS (matrices 10000x10000)
- **Quantum Simulator** : > 1M qubits simultanés
- **Neural Network** : > 10M neurones avec apprentissage
- **Audio/Image Processing** : Temps réel 4K@60fps

### Métriques Système
- **Memory Leaks** : 0 fuite détectée
- **Compilation** : 0 warning, 0 erreur
- **Tests** : 100% pass rate sur tous modules
- **Stress Test** : 1M+ LUMs en < 1 seconde

## ALGORITHMES AVANCÉS OBLIGATOIRES

### 1. SIMD/Vectorisation
```c
// Exemple : Multiplication vectorielle AVX-512
__m512d a = _mm512_load_pd(&array_a[i]);
__m512d b = _mm512_load_pd(&array_b[i]);
__m512d result = _mm512_fmadd_pd(a, b, accumulator);
```

### 2. Optimisation Mémoire
```c
// Pool mémoire avec alignement optimal
void* aligned_pool = aligned_alloc(64, size);
// Préfetch pour performance cache
__builtin_prefetch(data, 0, 3);
```

### 3. Parallélisation
```c
#pragma omp parallel for simd
for (size_t i = 0; i < count; i++) {
    // Traitement parallèle SIMD
}
```

## TESTS OBLIGATOIRES PAR MODULE

### Tests Unitaires
1. **Création/Destruction** - Validation mémoire
2. **Performance** - Benchmarks vs objectifs
3. **Stress** - 100M+ éléments par module
4. **Intégration** - Interaction entre modules
5. **Régression** - Non-régression performance

### Tests Système
1. **Memory Safety** - AddressSanitizer + Valgrind
2. **Thread Safety** - Tests concurrent
3. **Performance** - Profiling complet
4. **Compatibility** - Multi-plateforme

## LOGS AUTHENTIQUES REQUIS

### Format Log Standard
```
[TIMESTAMP_NS] [MODULE] [FUNCTION] [LEVEL] Message détaillé
[1642751234567890123] [MATRIX] [multiply] [INFO] Processed 100M elements in 0.234s (427M ops/s)
```

### Métriques Logs Obligatoires
- **Temps d'exécution** en nanosecondes
- **Throughput** en ops/seconde
- **Utilisation mémoire** en bytes
- **Utilisation CPU** en pourcentage
- **Cache hit ratio** en pourcentage

## VALIDATION FINALE OBLIGATOIRE

### 1. Compilation
```bash
make clean && make all
# Résultat attendu : 0 warnings, 0 erreurs
```

### 2. Tests Complets
```bash
./bin/lum_vorax --test-all-modules
# Résultat attendu : 100% pass rate
```

### 3. Stress Test
```bash
./bin/lum_vorax --stress-test-million
# Résultat attendu : < 1 seconde pour 1M LUMs
```

### 4. Validation Performance
```bash
./bin/lum_vorax --benchmark-all
# Résultat attendu : Toutes métriques > objectifs
```

## STRUCTURE CODE AVANCÉE

### Headers Optimisés
- Alignement mémoire strict (64 bytes)
- Attributs de performance (__restrict__, __hot__)
- Optimisations compilateur (-O3, -march=native)

### Implémentations Avancées
- Branch prediction hints
- Loop unrolling optimal
- Cache-friendly data structures
- Lock-free algorithms où possible

## CRITÈRES DE SUCCÈS ABSOLUS

1. **Compilation** : Zéro warning/erreur
2. **Performance** : Toutes métriques > objectifs
3. **Mémoire** : Zéro fuite détectée
4. **Tests** : 100% pass rate
5. **Logs** : Authentiques et détaillés
6. **Code** : Aucun placeholder/hardcoding

## LIVRABLE FINAL

Un système LUM/VORAX avec :
- 44+ modules entièrement fonctionnels
- Performances industrielles
- Code source complet et optimisé
- Tests exhaustifs passants
- Documentation technique complète
- Logs authentiques de validation

**ATTENTION** : Toute tentative de simplification, d'ajout de placeholders, ou de réduction de fonctionnalités sera considérée comme un échec. Le système doit être livré dans sa version la plus avancée et optimisée possible.
