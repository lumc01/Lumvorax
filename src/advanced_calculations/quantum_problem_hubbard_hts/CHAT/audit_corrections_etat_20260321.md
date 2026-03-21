# AUDIT DES CORRECTIONS — LumVorax Hubbard HTS
**Date** : 2026-03-21  
**Run en cours** : `research_20260321T204610Z_1803` (RUNNING — phase fullscale, RESUME_FROM_PHASE=10)  
**Run de référence précédent** : `research_20260320T195911Z_6227` (score 429/600)  
**Fichier audité** : `src/hubbard_hts_research_cycle_advanced_parallel.c` (2856 lignes)  
**Objectif** : Vérifier quelles corrections sont bien présentes dans le code actuel, quelles sont manquantes

---

## RÉSUMÉ EXÉCUTIF

| Catégorie | Corrections présentes | Corrections manquantes |
|-----------|----------------------|----------------------|
| PT-MC paramètres | N_replicas=8 ✅, Steps/sweep=500 ✅ | **N_sweeps=20 000** ❌ (cible: 200 000) |
| Worm MC | sw%10 supprimé ✅, set_log_files câblé ✅ | — |
| Post-run tests | lo=0 U/t>10 ✅, WARN distincts ✅ | — |
| Traçabilité | traceability_pct 4×25% ✅, METRIC simulate_adv ✅ | METRIC boucle TC scan à vérifier |
| Cluster | 512×512 dans c_sizes[] ✅ | — |
| Ressources | mem_available_kb() disponible ✅ | **setvbuf RAM 80-90% ABSENT** ❌ |
| Parallélisme | pthread mutex CPU ✅ | **Multi-cœur calcul ABSENT** ❌ |

---

## 1. CORRECTIONS PRÉSENTES ET VÉRIFIÉES ✅

### 1.1 C57-00 — Suppression filtre `sw % 10 == 0` dans `worm_mc_bosonic.c`
**Statut : ✅ PRÉSENT**

```c
// worm_mc_bosonic.c ligne 286-347
if (g_worm_sweep_log) {
    // Écriture CHAQUE sweep — pas de filtre sw%10
    fprintf(g_worm_sweep_log, ...);
}
```
Le logging worm est bien sweep-by-sweep sans filtre. Chaque sweep est enregistré.

---

### 1.2 C57-01 — `worm_mc_set_log_files` câblé dans `advanced_parallel.c`
**Statut : ✅ PRÉSENT**

```c
// advanced_parallel.c lignes 1671-1676
worm_mc_set_log_files(w_swp_f, w_att_f);   // Activation logging worm
// ... simulation worm ...
worm_mc_set_log_files(NULL, NULL);           // Désactivation après module
```
Les fichiers `worm_sweep_log.csv` et `worm_mc_attempt_log.csv` sont correctement ouverts et passés au module worm.

---

### 1.3 C57-04 — `traceability_pct` 4×25% dans `post_run_scientific_report_cycle.py`
**Statut : ✅ PRÉSENT**

```python
# post_run_scientific_report_cycle.py lignes 76-84
_sha512_ok  = (run_dir / "GLOBAL_CHECKSUM.sha512").exists()
_sha256_ok  = (logs / "checksums.sha256").exists()
traceability_pct = (
    25.0 * _sha512_ok    # GLOBAL_CHECKSUM.sha512
    + 25.0 * _sha256_ok  # checksums.sha256
    + ...
)
```
La formule 4×25% est en place. Les 2 premières composantes sont vérifiées.

**Note** : Le score trace=65 du run 6227 indique que 2 composantes sur 4 sont présentes (50%) + bonus autres → 65/100. Les fichiers `GLOBAL_CHECKSUM.sha512` et/ou `checksums.sha256` sont absents ou générés trop tard.

---

### 1.4 C57-05a/b/c — WARN distincts dans `post_run_autocorr.py`
**Statut : ✅ PRÉSENT**

```python
# post_run_autocorr.py lignes 289-299
elif tau_int > 500:
    # pairing_norm → ralentissement critique
    status = f'WARN_CRITICAL_SLOWING_{tau_int:.0f}'
    # sign_ratio → problème de signe fermionique
    status = f'WARN_SIGN_PROBLEM_{tau_int:.0f}'
    # Autres observables
    status = f'WARN_TAU_INT_HIGH_{tau_int:.0f}'
```
Les 3 types de WARN sont bien distincts et basés sur le nom de l'observable.

---

### 1.5 C57-512 — `512×512` dans `c_sizes[]`
**Statut : ✅ PRÉSENT**

```c
// advanced_parallel.c ligne 2369
int c_sizes[] = {8, 10, 12, 14, 16, 18, 24, 26, 28, 32, 36, 64, 66, 68, 128, 255, 512};
// Avec règle adaptative steps=40 pour 512×512
```
Confirmé par les résultats run 6227 : cluster 512×512 PASS avec pairing=0.9932.

---

### 1.6 C39-C1-FIX — `lo=0` pour U/t > 10 dans `post_run_chatgpt_critical_tests.py`
**Statut : ✅ PRÉSENT**

```python
# post_run_chatgpt_critical_tests.py lignes 113-114
if u_over_t > 10.0:
    lo = 0  # C39-C1-FIX : lo=0 si U/t > 10 (Mott fort)
```
La correction est en place. Les modules `qcd_lattice_fullscale` (U/t=12.86), `dense_nuclear_fullscale` (U/t=13.75), `spin_liquid_exotic` (U/t=11.67), `fermionic_sign_problem` (U/t=14.0) utilisent lo=0.

---

### 1.7 C43 — N_REPLICAS=8, STEPS_PER_SWEEP=500
**Statut : ✅ PRÉSENT**

```c
// advanced_parallel.c lignes 646-650
#define PT_MC_N_REPLICAS       8      // C43 : 6→8 répliques
#define PT_MC_N_SWEEPS         20000  // C43 : 4000→20000
#define PT_MC_N_THERMALIZE     4000   // C43 : 800→4000
#define PT_MC_STEPS_PER_SWEEP  500    // C43 : 200→500
```

---

## 2. CORRECTIONS MANQUANTES ❌

### 2.1 N_sweeps = 200 000 (C59-P3)
**Statut : ❌ ABSENT — CRITIQUE**

**Code actuel** :
```c
#define PT_MC_N_SWEEPS    20000   // C43 valeur actuelle
#define PT_MC_N_THERMALIZE 4000   // C43 valeur actuelle
```

**Cible C59-P3** :
```c
#define PT_MC_N_SWEEPS    200000  // C59-P3 : ×10 pour N_eff ≥ 100
#define PT_MC_N_THERMALIZE 40000  // C59-P3 : 20% × 200000
```

**Impact** : N_eff estimé actuel < 30 (critique expert non résolu). Avec 200 000 sweeps et τ_int≈1000, N_eff = 200 000 / (2×1000) = 100 — seuil minimal pour publication.

**Attention** : Avec 200 000 sweeps × 500 steps × 8 répliques × 15 modules = **12 milliards d'étapes Metropolis**. Durée estimée : 4-8 heures sur CPU mono-cœur.

---

### 2.2 `setvbuf` RAM 80-90% (mentionné dans le fichier joint)
**Statut : ❌ ABSENT**

La fonction `mem_available_kb()` existe (ligne 193) mais n'est utilisée qu'à la ligne 2374 pour le log de ressources autoscale. Aucun `setvbuf(file, buf, _IOFBF, buf_size)` n'est présent dans le code.

**Impact sur RAM** : Sans setvbuf, les buffers I/O sont 4-8 KB par fichier (défaut glibc). Les 15+ fichiers CSV utilisent ~120 KB de buffer total → RAM pratiquement inutilisée pour le I/O. L'utilisation RAM actuelle provient des tableaux de simulation.

---

### 2.3 Parallélisme multi-cœur pour le calcul
**Statut : ❌ ABSENT**

Le runner advanced_parallel tourne sur **1 seul cœur CPU** pour les calculs. Le `pthread` visible dans le code sert uniquement au **mutex de mesure CPU** (`pthread_mutex_lock/unlock` pour lire `/proc/stat`), pas pour paralléliser la simulation.

**Preuve** : `grep pthread` dans le runner montre uniquement `pthread_mutex_t cpu_mu` — aucun `pthread_create`, aucun `pthread_join`, aucun `#pragma omp`.

**Impact** : Sur une machine multi-cœur, 1 seul cœur travaille. Les 14 autres cœurs sont inutilisés pendant la simulation PT-MC.

---

### 2.4 METRIC LumVorax dans la boucle TC scan
**Statut : ⚠️ PARTIELLEMENT PRÉSENT**

Le METRIC C57-01 est présent dans `simulate_fullscale_controlled()` (lignes 348-494), mais la boucle TC scan spécifique (scan température pour Tc) utilise le FORENSIC_LOG général, pas le METRIC LumVorax formaté dans chaque température scannée.

---

## 3. ÉTAT DU RUN EN COURS

**Run** : `research_20260321T204610Z_1803`  
**Démarré** : 2026-03-21T20:46:10Z  
**Phase** : Runner fullscale (`RESUME_FROM_PHASE=10`)  
**Observations actuelles** :

| Fichier | Lignes actuelles | Statut |
|---------|-----------------|--------|
| `normalized_observables_trace.csv` | 43 | En cours de remplissage |
| `logs/lumvorax_*.csv` | Actif | LumVorax v3.0 ACTIF |
| MEMORY_TRACKER | Très actif | Dynamic Hilbert scan en cours |

**Le runner fullscale** est en phase d'exécution normale (MEMORY_TRACKER = scan dynamique des tailles de réseau). La phase advanced_parallel démarrera ensuite.

---

## 4. RÉSULTATS RUN PRÉCÉDENT — run 6227 (référence)

| Critère | Score | Valeur mesurée |
|---------|-------|---------------|
| iso | 100/100 | MODULE_COVERAGE_GATE=PASS(15) |
| trace | 65/100 | traceability_pct partiel |
| repr | 100/100 | 15 modules, metadata complète |
| robust | 98/100 | dt_sweep PASS, von Neumann PASS |
| phys | 82/100 | Tc=67.0±1.5K, ED erreur 90% |
| expert | 84/100 | MC stochastique prouvé, benchmark FAIL |
| **TOTAL** | **429/600** | |

**Données PT-MC générées** :
- 960 000 lignes `parallel_tempering_mc_results.csv` ✅
- 15 modules, 8 répliques, 20 000 sweeps
- `worm_mc_bosonic_results.csv` : phase Mott correctement détectée ✅
- `tc_estimation_ptmc.csv` : Tc=67.0K, accord inter-méthodes ±1.5K ✅
- `cluster_scalability_tests.csv` : 8×8 à 512×512 tous PASS ✅
- `exact_diagonalization_crossval.csv` : E0(ED)=-2.103 eV, E0(MC)=-1.000 eV, erreur=90.2% ⚠️

---

## 5. TABLEAU DE BORD — CORRECTIONS À APPLIQUER

| # | Correction | Fichier | Statut | Impact score |
|---|-----------|---------|--------|-------------|
| C59-P3 | N_sweeps 20 000 → 200 000 | `advanced_parallel.c` lignes 648-649 | **❌ MANQUANT** | +10 phys, +8 expert |
| C59-P3 | N_thermalize 4 000 → 40 000 | `advanced_parallel.c` ligne 649 | **❌ MANQUANT** | +5 phys |
| C59-setvbuf | setvbuf RAM 80-90% | `advanced_parallel.c` après FOPEN_DIAG | **❌ MANQUANT** | stabilité |
| C59-MP | Multi-cœur pthread_create | `advanced_parallel.c` main loop | **❌ MANQUANT** | ×N_cœurs vitesse |
| C59-bench | Benchmark DQMC reconstruit | `benchmarks/qmc_dmrg_reference_v2.csv` | **❌ MANQUANT** | +8 robust |
| C57-00 | sw%10 supprimé worm | `worm_mc_bosonic.c` | ✅ PRÉSENT | — |
| C57-01 | worm_mc_set_log_files | `advanced_parallel.c` | ✅ PRÉSENT | — |
| C57-04 | traceability_pct 4×25% | `post_run_scientific_report_cycle.py` | ✅ PRÉSENT | — |
| C57-05 | WARN distincts τ>500 | `post_run_autocorr.py` | ✅ PRÉSENT | — |
| C57-512 | 512×512 dans c_sizes[] | `advanced_parallel.c` | ✅ PRÉSENT | — |
| C39-C1-FIX | lo=0 pour U/t>10 | `post_run_chatgpt_critical_tests.py` | ✅ PRÉSENT | — |
| C43 | N_replicas=8, steps=500 | `advanced_parallel.c` | ✅ PRÉSENT | — |

---

## 6. QUESTION SUR LES CŒURS CPU

**Réponse directe** : **NON, tous les cœurs ne sont PAS utilisés.**

Le runner advanced_parallel est **mono-thread** pour les calculs. Preuve dans le code :
- `pthread_t` absent des déclarations globales de calcul
- `pthread_create` absent
- `#pragma omp` absent
- `pthread_mutex` utilisé uniquement pour lire `/proc/stat` (mesure CPU, pas calcul)

Les 15 modules sont simulés **séquentiellement un par un** sur 1 seul cœur.

Pour utiliser tous les cœurs, il faudrait soit :
1. `pthread_create()` par module (15 threads parallèles)
2. `#pragma omp parallel for` sur la boucle des modules
3. Processus séparés avec `fork()` + `waitpid()`

**Impact actuel** : sur une machine 4 cœurs, CPU affiché ≈ 25% (1/4 cœur utilisé).

---

## 7. ATTENTE DES ORDRES

Le run `research_20260321T204610Z_1803` est en cours. Je n'effectue aucune modification tant que vous ne l'ordonnez pas.

**Les 3 actions prioritaires identifiées** (si vous les ordonnez) :
1. **N_sweeps 20 000 → 200 000** — 2 lignes à modifier dans `advanced_parallel.c`
2. **setvbuf RAM 80-90%** — ~30 lignes à ajouter après les FOPEN_DIAG
3. **Multi-cœur pthread** — refactoring significatif (~100 lignes)

Je reste en attente.
