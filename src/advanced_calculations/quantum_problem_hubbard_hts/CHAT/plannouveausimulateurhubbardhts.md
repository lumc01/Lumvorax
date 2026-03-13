# PLAN PROTOCOLE — NOUVEAU SIMULATEUR HUBBARD_HTS FULLSCALE INDÉPENDANT
## Version définitive consolidée — Session 2026-03-13T19:35Z
## Intégrant toutes les corrections des rapports précédents + trous identifiés dans `analysechatgpt8.md`

**Auteur** : Agent Replit (session autonome — inspection totale src/)
**Basé sur** : ANALYSECHATGPT.txt, src(2).zip, src/ courant, runs 6084/6260/7163, analysechatgpt1-8.md
**Répertoire cible** : `src/advanced_calculations/quantum_problem_hubbard_hts/Hubbard_HTS/`

---

## VALIDATION DE LECTURE DES SOURCES

J'ai inspecté personnellement et manuellement :
- `src/advanced_calculations/quantum_problem_hubbard_hts/src/hubbard_hts_module.c` (339 lignes)
- `src/advanced_calculations/quantum_problem_hubbard_hts/src/hubbard_hts_research_cycle.c` (1268 lignes)
- `src/advanced_calculations/quantum_problem_hubbard_hts/src/hubbard_hts_research_cycle_advanced_parallel.c` (1343 lignes)
- `src/advanced_calculations/quantum_problem_hubbard_hts/independent_modules/` (qmc_module.py, dmrg_module.py, arpes_module.py, stm_module.py)
- `src/advanced_calculations/quantum_problem_hubbard_hts/tools/` (run_independent_physics_modules.py, post_run_physics_readiness_pack.py, post_run_hfbl360_forensic_logger.py)
- `src/advanced_calculations/quantum_problem_hubbard_hts/benchmarks/` (qmc_dmrg_reference_v2.csv, external_module_benchmarks_v1.csv)
- `src/advanced_calculations/quantum_problem_hubbard_hts/results/` (3 derniers runs : 6084, 6260, 7163)
- Tous les fichiers CHAT/ (analysechatgpt.md à analysechatgpt8.md, AUTO_PROMPT_*, RAPPORT_*)

Noms exacts vérifiés dans les sources — rien inventé.

---

## 1. BUGS ET ANOMALIES CRITIQUES À CORRIGER (LISTE EXHAUSTIVE CONSOLIDÉE)

### Depuis ANALYSECHATGPT.txt et src(2).zip (anciens rapports) :

| ID | Fichier source exact | Lignes | Nature | Statut actuel |
|---|---|---|---|---|
| BC-A1 | `hubbard_hts_module.c` | 194-200 | Invariant E∼P∼n : même `fluct` pour energy+pairing+sign | **NON CORRIGÉ** |
| BC-A2 | `hubbard_hts_module.c` | 198 | `step_energy += U*density²- t*fabs(fluct)` : hopping via fluct, pas via voisins | **NON CORRIGÉ** |
| BC-A3 | Tous runners | Toutes | Euler explicite → instabilité cumulative | **CORRIGÉ (RK2)** |
| BC-A4 | Tous runners | Toutes | Normalisation énergie par N sites au lieu de 2N liens (cinétique) | **PARTIELLEMENT CORRIGÉ** |
| BC-A5 | Tous runners | Toutes | Pipeline CSV auto-référentiel | **PARTIELLEMENT CORRIGÉ** |
| BC-A6 | `independent_modules/` | Tout | qmc/dmrg/arpes/stm reçoivent CSV, pas états quantiques | **NON CORRIGÉ** |
| BC-A7 | Tous runners | Toutes | Lattice trop petit (max 10×10 par défaut) | **PARTIELLEMENT CORRIGÉ** (cluster scaling ajouté) |

### Découverts dans `analysechatgpt8.md` (inspection ligne par ligne, session 2026-03-13) :

| ID | Fichier source exact | Lignes | Nature | Priorité |
|---|---|---|---|---|
| BC-01 | `hubbard_hts_module.c` | 194-200 | Sources stochastiques non séparées | P1 BLOQUANT |
| BC-02 | `hubbard_hts_research_cycle_advanced_parallel.c` | 314-321 | Feedback atomique sur énergie du pas précédent (stale) | P5 |
| BC-03 | `hubbard_hts_research_cycle_advanced_parallel.c` | 326-328 | d_left/d_right post-tanh au lieu de pré-RK2 | P2 |
| BC-04 | Deux runners | 280 / 343 | Pairing /= N sites (doit être /= 2*Lx*Ly) | P3 |
| BC-05 | `hubbard_hts_research_cycle.c` | 522-531 | Shift non adaptatif dans solveur exact 2x2 | P6 |
| BC-06 | Deux runners | 276 / 335 | Sign_ratio = signe nombre aléatoire, pas proxy fermionique | P4 |
| OC-02 | Tous | Toutes | Score 88.95% = plafond structurel permanent (Q12 plasma, Q15 ARPES/STM non connectés) | STRUCTUREL |
| OC-03 | Tous | Tests | Seuil `within_error_bar` = 40% trop permissif (standard = 80%+) | SEUIL |

---

## 2. ARCHITECTURE DE LA NOUVELLE VERSION FULLSCALE (NOMS EXACTS)

```
src/advanced_calculations/quantum_problem_hubbard_hts/Hubbard_HTS/
├── core/
│   ├── hamiltonian.c          # Hamiltonien Hubbard exact : -t Σ(c†c+h.c.) + U Σ n↑n↓
│   ├── hamiltonian.h
│   ├── fock_space.c           # Construction espace de Fock : états |n1↑,n1↓,...,nN↑,nN↓⟩
│   ├── fock_space.h
│   ├── green_function.c       # Matrice de Green fermionique G(k,ω), G(i,τ)
│   ├── green_function.h
│   └── observables.c          # Énergie, pairing, densité, corrélations — sources séparées
│   └── observables.h
├── modules/
│   ├── dqmc_module.c          # DQMC : découplage Hubbard-Stratonovich, det(M), balayage Monte Carlo
│   ├── dqmc_module.h
│   ├── exactdiag_module.c     # Exact diagonalization (Lanczos) pour L ≤ 12
│   ├── exactdiag_module.h
│   ├── dmrg_module.c          # DMRG (MPS) pour lattices L > 12 — ou interface Python
│   ├── dmrg_module.h
│   ├── arpes_module.c         # ARPES : spectre A(k,ω) depuis G(k,ω) — connecté à états quantiques
│   ├── arpes_module.h
│   ├── stm_module.c           # STM : LDOS, carte de densité locale — connecté à états quantiques
│   └── stm_module.h
├── stability/
│   ├── integrator_tests.c     # Tests Euler vs RK2 vs symplectique — conservation énergie
│   ├── von_neumann_analysis.c # Analyse Von Neumann exacte (pas approximation)
│   ├── dt_sweep.c             # Sweep Δt = 0.001, 0.005, 0.010 — drift < 1e-6
│   └── spectral_analysis.c    # FFT avec vérification que dominant_freq ∝ U/t
├── tests/
│   ├── test_pearson_invariant.c   # NOUVEAU : Pearson(E,P) < 0.5 — gate FAIL obligatoire
│   ├── test_lattice_scaling.c     # L = 8, 16, 32, 64 — fit loi de puissance, limit L→∞
│   ├── test_u_t_dependence.c      # U/t = 0→12 — phases Mott, Mott insulator, pseudogap
│   ├── test_doping.c              # Dopage δ = 0→0.3 — pairing d-wave vs s-wave
│   ├── test_temperature.c         # β = 0.1→10 — pairing vs T, signe moyen vs β
│   ├── test_sign_problem.c        # sign_ratio vs (U, T, δ) — décroissance exponentielle attendue
│   ├── test_bc_comparison.c       # PBC vs OBC — impact sur pairing d-wave
│   ├── test_scrambling.c          # Permutation pipeline E,P,n — invariant doit disparaître
│   ├── test_noise_robustness.c    # Bruit η(t) ajouté à ψ(t) — robustesse
│   └── test_basis_independence.c  # Site basis ↔ momentum basis — invariance
├── benchmarks/
│   ├── qmc_reference_hubbard.csv  # Références QMC publiées (Hirsch, White) — unités eV/site
│   ├── dmrg_reference_hubbard.csv # Références DMRG publiées — unités eV/site
│   └── exact_diag_reference.csv   # Références exact diagonalization — unités eV/site
├── logs/
│   ├── hfbl360_logger.c           # HFBL360 : logger forensique persistant (intégration LUM VORAX)
│   ├── hfbl360_logger.h
│   ├── HFBAL_360/                 # Logs persistants horodatés UTC, bit-à-bit
│   └── checksums/                 # SHA512 par fichier de résultat
├── results/                       # Sorties isolées par run (même convention research_YYYYMMDDTHHMMSSZ_PID)
├── scripts/
│   ├── run_hubbard_hts.sh         # Script principal du nouveau simulateur
│   ├── generate_report.py         # Rapport scientifique automatique
│   ├── plot_observables.py        # Graphiques E(U/t), P(T), sign(β)
│   └── validate_physics.py        # Validation automatique des contraintes physiques
└── docs/
    ├── PROTOCOL.md                # Ce document + checklist complète
    ├── PHYSICS_REFERENCE.md       # Équations exactes, unités, conventions
    └── CHANGELOG.md               # Historique de toutes les corrections
```

---

## 3. HAMILTONIEN EXACT — IMPLÉMENTATION REQUISE

```c
/* Hamiltonien de Hubbard sur réseau 2D :
   H = -t Σ_{<i,j>,σ} (c†_{iσ} c_{jσ} + h.c.) + U Σ_i n_{i↑} n_{i↓}
       - μ Σ_{i,σ} n_{iσ}

   Représentation : espace de Fock de dimension 4^N (pour N sites)
   Base : produit tensoriel |↑⟩, |↓⟩, |↑↓⟩, |0⟩ par site

   Pour N ≤ 12 : Lanczos exact diagonalization
   Pour N > 12  : DQMC (découplage Hubbard-Stratonovich sur champ auxiliaire s_i(τ))
*/

/* Structure de l'état de Fock */
typedef struct {
    uint32_t up;  /* Bitmask occupation spin-up  : bit i = site i */
    uint32_t dn;  /* Bitmask occupation spin-down : bit i = site i */
} fock_state_t;

/* Application H|ψ⟩ = |φ⟩ */
void apply_hubbard_hamiltonian(
    const fock_state_t* basis, int dim,
    double t, double u, double mu,
    int lx, int ly,
    const double* psi,   /* Vecteur d'état d'entrée  [dim] */
    double* phi          /* Vecteur résultat         [dim] */
);
```

---

## 4. CORRECTIONS NORMALISATIONS (OBLIGATOIRES)

```c
/* ÉNERGIE CINÉTIQUE — normalisée par nombre de liens */
int n_bonds_horizontal = lx * ly;        /* PBC horizontal */
int n_bonds_vertical   = lx * ly;        /* PBC vertical   */
int n_bonds_total      = n_bonds_horizontal + n_bonds_vertical; /* = 2*lx*ly */
double E_kinetic_per_bond = E_kinetic / (double)n_bonds_total;

/* ÉNERGIE D'INTERACTION — normalisée par nombre de sites */
double E_interaction_per_site = E_interaction / (double)(lx * ly);

/* PAIRING — normalisé par nombre de paires de liens (d-wave) */
/* Pour pairing d-wave : Δ_d = (1/N_bonds) Σ_{<i,j>} φ_{ij} * (cos(ki-kj)) */
int n_pairs = n_bonds_total;  /* 1 paire par lien pour nearest-neighbor */
double pairing_norm = pairing_raw / (double)n_pairs;

/* SIGN — calculé depuis déterminant matrice de Green */
/* <sign> = (1/N_MC) Σ_C sign(det M_up(C) * det M_dn(C)) */
/* où C est une configuration du champ auxiliaire DQMC */
```

---

## 5. CHECKLIST SCIENTIFIQUE COMPLÈTE (AU BIT PRÈS)

### Catégorie A — Physique Hubbard fondamentale

| ✓ | Test | Fichier | Critère exact |
|---|---|---|---|
| [ ] | Taille lattice L=8 | `test_lattice_scaling.c` | E(L=8)/site convergée à < 1% de E(L=∞) |
| [ ] | Taille lattice L=16 | `test_lattice_scaling.c` | Fit loi de puissance E(L) = E(∞) + a/L^α |
| [ ] | Taille lattice L=32 | `test_lattice_scaling.c` | α confirmé physique (~2 pour Heisenberg 2D) |
| [ ] | Taille lattice L=64 | `test_lattice_scaling.c` | Extrapolation L→∞ < 0.5% d'erreur |
| [ ] | U/t = 0.1 (métallique) | `test_u_t_dependence.c` | E ≈ E_free_fermion (analytique), pairing élevé |
| [ ] | U/t = 4 (modéré) | `test_u_t_dependence.c` | Accord QMC Hirsch±5% |
| [ ] | U/t = 8 (Mott) | `test_u_t_dependence.c` | Accord QMC Hirsch±5%, gap de Mott détecté |
| [ ] | U/t = 12 (fort couplage) | `test_u_t_dependence.c` | Limite demi-remplissage → Heisenberg J=4t²/U |
| [ ] | Dopage δ=0 (demi-remplissage) | `test_doping.c` | Isolant de Mott pour U/t > 8 |
| [ ] | Dopage δ=0.1 | `test_doping.c` | Pseudogap détecté dans A(k,ω) |
| [ ] | Dopage δ=0.3 | `test_doping.c` | Métal de Fermi-liquide |
| [ ] | β=0.5 (haute T) | `test_temperature.c` | Pairing ≈ 0 (état normal) |
| [ ] | β=5 (basse T) | `test_temperature.c` | Pairing croissant — onset SC |
| [ ] | β=10 (T→0) | `test_temperature.c` | Convergence vers état fondamental |
| [ ] | Pairing d-wave vs s-wave | `test_doping.c` | d-wave dominant pour dopage optimal |

### Catégorie B — Validation numérique

| ✓ | Test | Fichier | Critère exact |
|---|---|---|---|
| [ ] | Drift énergie Δt=0.001 | `dt_sweep.c` | < 1e-6 par step |
| [ ] | Drift énergie Δt=0.005 | `dt_sweep.c` | < 1e-6 par step |
| [ ] | Drift énergie Δt=0.010 | `dt_sweep.c` | < 1e-6 par step (si instable → FAIL obligatoire) |
| [ ] | Conservation énergie Euler | `integrator_tests.c` | Drift > 1e-4 → confirme nécessité RK2 |
| [ ] | Conservation énergie RK2 | `integrator_tests.c` | Drift < 1e-6 |
| [ ] | Conservation énergie Symplectique | `integrator_tests.c` | Drift < 1e-8 (référence) |
| [ ] | Rayon spectral Von Neumann ≤ 1 | `von_neumann_analysis.c` | Pour tous U/t et Δt testés |
| [ ] | Convergence Lanczos N=4 (2x2) | `exactdiag_module.c` | |E_Lanczos - E_exact| < 1e-10 |
| [ ] | Convergence Lanczos N=8 (4×2) | `exactdiag_module.c` | |E_Lanczos - E_ref| < 1e-8 |
| [ ] | DQMC convergence N_warmup | `dqmc_module.c` | Observables stables après N_warmup = 500 sweeps |
| [ ] | DQMC statistiques N_measure | `dqmc_module.c` | Erreur barre < 0.1% après 5000 mesures |

### Catégorie C — Détection invariants artificiels

| ✓ | Test | Fichier | Critère exact (NOUVEAU — gate FAIL) |
|---|---|---|---|
| [ ] | **Pearson(E, P) < 0.5** | `test_pearson_invariant.c` | **GATE FAIL obligatoire si ≥ 0.5** |
| [ ] | Pearson(E, sign_ratio) non-trivial | `test_pearson_invariant.c` | Dépendance physique attendue (pas ~0 constant) |
| [ ] | Sign_ratio vs U/t | `test_sign_problem.c` | Doit décroître avec U/t pour β fixé |
| [ ] | Sign_ratio vs β | `test_sign_problem.c` | Doit décroître avec β pour U fixé |
| [ ] | Scrambling pipeline | `test_scrambling.c` | Permuter E,P,n → résultats incohérents (pas PASS) |
| [ ] | Bruit η(t) sur ψ(t) | `test_noise_robustness.c` | Observables stables à ±0.5% avec η_amp < 0.01 |
| [ ] | Invariance base site ↔ momentum | `test_basis_independence.c` | E(site) = E(k-space) à < 0.01% |
| [ ] | FFT dominant_freq ∝ U/t | `spectral_analysis.c` | dominant_freq doit varier avec U/t (pas constante) |

### Catégorie D — Benchmarks scientifiques

| ✓ | Test | Fichier | Critère exact |
|---|---|---|---|
| [ ] | Accord QMC Hirsch (U=4, β=5, L=8) | `test_u_t_dependence.c` | RMSE ≤ 0.05 eV/site |
| [ ] | Accord QMC Blankenbecler (U=8) | `test_u_t_dependence.c` | RMSE ≤ 0.05 eV/site |
| [ ] | Accord DMRG White (chaîne 1D) | `dmrg_module.c` | RMSE ≤ 0.01 eV/site |
| [ ] | Accord exact diag 2x2 (toutes U/t) | `exactdiag_module.c` | |E_DQMC - E_exact| ≤ 0.005 eV |
| [ ] | **within_error_bar ≥ 80%** | Tous benchmarks | **GATE FAIL obligatoire si < 80%** |
| [ ] | CI95 halfwidth ≤ 0.05 eV/site | Tous benchmarks | Standard publication |
| [ ] | ARPES : A(k,ω) — dispersion correcte | `arpes_module.c` | Band bottom à -2t (U=0, analytique) |
| [ ] | STM : LDOS — gap de Mott détecté | `stm_module.c` | Gap ≈ U pour U/t >> 1 |

### Catégorie E — Traçabilité et forensique (HFBL360 / LUM VORAX)

| ✓ | Test | Fichier | Critère exact |
|---|---|---|---|
| [ ] | Log HFBAL_360 présent | `hfbl360_logger.c` | `logs/HFBAL_360/hfbl360_realtime_persistent.log` généré |
| [ ] | Horodatage UTC sur chaque événement | `hfbl360_logger.c` | Format `ts_ns=XXXXXXXXXX event=...` |
| [ ] | State hash par step | `hfbl360_logger.c` | `state_hash=XXXXXXXXXXXXXXXX` (64-bit FNV) |
| [ ] | Checksum SHA512 par fichier résultat | `logs/checksums/` | Fichier `.sha512` pour chaque CSV output |
| [ ] | **Checksum global du run** | Script `run_hubbard_hts.sh` | **Absent dans moteur actuel — obligatoire** |
| [ ] | Progression % affiché toutes les 100 steps | `run_hubbard_hts.sh` | `[XX.XX%] step YYYY/ZZZZ energy=...` |
| [ ] | Certification de complétude | `run_hubbard_hts.sh` | `CERTIFICATION: run complete, 0 FAIL, checksum_verified` |
| [ ] | Backup sources au démarrage | `run_hubbard_hts.sh` | Copie `src/` et `benchmarks/` dans `backups/run_XXX/` |
| [ ] | Log hardware snapshot | `hfbl360_logger.c` | `/proc/cpuinfo`, `/proc/meminfo`, `/proc/loadavg` |
| [ ] | Provenance log | `run_hubbard_hts.sh` | Git hash, date, hostname, user, version |

### Catégorie F — Isolation pipeline (ANTI-AUTO-RÉFÉRENTIALITÉ)

| ✓ | Test | Fichier | Critère exact |
|---|---|---|---|
| [ ] | Aucun CSV précédent en entrée | Tout code | Grep `baseline_reanalysis_metrics.csv` en entrée → 0 résultat |
| [ ] | Observables depuis états quantiques uniquement | `observables.c` | E, P, sign calculés depuis `fock_state_t`, pas depuis CSV |
| [ ] | Modules ARPES/STM connectés à G(k,ω) | `arpes_module.c`, `stm_module.c` | Reçoivent Green function, pas séries temporelles CSV |
| [ ] | Benchmarks depuis fichiers de référence publiés | `benchmarks/` | `qmc_reference_hubbard.csv` = données Hirsch 1985 |
| [ ] | Aucune réinjection de résultats précédents | Tout code | Chaque run repart de l'état initial ψ(0) |

### Catégorie G — Conditions aux bords et lattice

| ✓ | Test | Fichier | Critère exact |
|---|---|---|---|
| [ ] | PBC (conditions périodiques) | `hamiltonian.c` | Hopping wrap-around sur tous les bords |
| [ ] | OBC (conditions ouvertes) | `test_bc_comparison.c` | Disponible pour comparaison |
| [ ] | APBC (anti-périodiques) | `hamiltonian.c` | Option pour étude flux magnétique |
| [ ] | Grille 2D carrée L×L | `fock_space.c` | L = 8, 16, 32, 64 supportés |
| [ ] | Grille rectangulaire Lx×Ly | `fock_space.c` | Lx ≠ Ly pour étude anisotropie |
| [ ] | Grille 1D L×1 | `fock_space.c` | Chaîne 1D pour validation DMRG |

---

## 6. MODULES À RECONSTRUIRE (NOMS EXACTS — VÉRIFIÉS DANS src/)

### Modules existants à connecter correctement (ne pas réinventer) :

| Module existant | Chemin actuel | Problème | Solution nouveau simulateur |
|---|---|---|---|
| `post_run_hfbl360_forensic_logger.py` | `tools/` | Appelé post-run seulement | Intégrer en temps réel dans HFBL360 |
| `post_run_physics_readiness_pack.py` | `tools/` | Post-run, pas de lien quantique | Connecter à observables.c |
| `run_independent_physics_modules.py` | `tools/` | Appelle modules sur CSV | Remplacer par appel sur Green function |
| `qmc_module.py` | `independent_modules/` | Reçoit CSV, pas Hamiltonien | Reconstruire avec DQMC complet |
| `dmrg_module.py` | `independent_modules/` | Reçoit CSV, pas MPS | Reconstruire avec tenseur network |
| `arpes_module.py` | `independent_modules/` | Reçoit CSV, pas G(k,ω) | Connecter à `green_function.c` |
| `stm_module.py` | `independent_modules/` | Reçoit CSV, pas LDOS | Connecter à `observables.c` |

### Nouveaux modules à créer (absents dans src/ actuel) :

| Module | Chemin nouveau | Nature |
|---|---|---|
| `fock_space.c/h` | `Hubbard_HTS/core/` | Construction base de Fock — ABSENT dans moteur actuel |
| `green_function.c/h` | `Hubbard_HTS/core/` | Matrice de Green — ABSENT |
| `dqmc_module.c/h` | `Hubbard_HTS/modules/` | DQMC complet avec découplage HS — ABSENT |
| `exactdiag_module.c/h` | `Hubbard_HTS/modules/` | Lanczos pour L≤12 — partiellement dans solveur 2x2 actuel |
| `test_pearson_invariant.c` | `Hubbard_HTS/tests/` | Gate FAIL Pearson(E,P) — ABSENT |
| `test_sign_problem.c` | `Hubbard_HTS/tests/` | Vérification physique sign problem — ABSENT |
| `PHYSICS_REFERENCE.md` | `Hubbard_HTS/docs/` | Documentation physique exacte — ABSENT |

---

## 7. TROUS AJOUTÉS PAR LE PRÉSENT RAPPORT (CE QUI ÉTAIT OUBLIÉ)

Ces éléments manquaient dans toutes les versions précédentes du plan :

### Trou T01 — Gate Pearson obligatoire (NOUVEAU CRITIQUE)
**Ce qui manquait** : Aucun test ne vérifie que les observables sont physiquement indépendantes. Sans cela, l'invariant E∼P∼n peut subsister sans être détecté.
**Solution** : `test_pearson_invariant.c` — `Pearson(energy_series, pairing_series) < 0.5` est une **gate FAIL obligatoire**. Si ≥ 0.5, le run est invalidé automatiquement.

### Trou T02 — Checksum global du run (OBLIGATOIRE MANQUANT)
**Ce qui manquait** : Le rapport 7163 l'indique explicitement : "Fichier checksum global présent: non". Tous les fichiers individuels ont des checksums mais pas le run dans son ensemble.
**Solution** : À la fin de chaque run, calculer SHA512 de la concaténation de tous les CSVs de résultats et l'écrire dans `results/run_XXX/GLOBAL_CHECKSUM.sha512`.

### Trou T03 — Sign_ratio physique vs placeholder (CORRECTION ESSENTIELLE)
**Ce qui manquait** : Le `sign_ratio` actuel est le signe d'un nombre aléatoire. Le plan précédent ne mentionnait pas explicitement la correction.
**Solution** : `sign_ratio` = `<sign>` DQMC = `(1/N_MC) Σ sign(det M_up * det M_dn)`. Ce signe doit décroître exponentiellement avec le volume et U/t (sign problem physique réel). Test de cette décroissance dans `test_sign_problem.c`.

### Trou T04 — Cohérence temporelle d_left/d_right (BUG RUNNER ADVANCED)
**Ce qui manquait** : BC-03 — dans le runner advanced_parallel, `d_left` et `d_right` sont lus APRÈS `tanh(d[i])`, mélangeant des instants différents. Le plan précédent ne mentionnait pas cette asymétrie entre runners.
**Solution** : Sauvegarder `d_left_t0`, `d_right_t0` avant le bloc RK2. Documenter dans `CHANGELOG.md`.

### Trou T05 — Test FFT multi-U/t (VALIDATION FRÉQUENCE PHYSIQUE)
**Ce qui manquait** : La fréquence FFT dominante est identique sur tous les runs (~0.003886 Hz), ce qui indique qu'elle est artificielle. Le plan précédent ne prévoyait pas de la faire varier avec U/t.
**Solution** : Dans `spectral_analysis.c`, exécuter la FFT pour U/t = 2, 4, 8, 12 et vérifier que `dominant_freq` change. Si constante, c'est une anomalie à reporter en FAIL.

### Trou T06 — Conditions aux bords PBC/OBC/APBC (LATTICE COMPLET)
**Ce qui manquait** : Le plan précédent mentionnait L×L mais pas les types de conditions aux bords. Le pairing d-wave dépend fortement des BC.
**Solution** : `test_bc_comparison.c` exécute le simulateur avec PBC, OBC et APBC, et compare l'énergie fondamentale et le pairing. L'écart PBC-OBC doit être quantifié et inférieur à 5% pour L=16.

### Trou T07 — Extrapolation thermodynamique L→∞ (SCALING COMPLET)
**Ce qui manquait** : Le plan prévoyait L=8,16,32,64 mais pas le fit de loi de puissance ni l'extrapolation vers L=∞.
**Solution** : Dans `test_lattice_scaling.c`, fitter `E(L)/site = E(∞)/site + a/L^α + b/L^(2α)` (correction de taille finie). L'exposant α doit être physiquement cohérent (α≈2 pour système 2D).

### Trou T08 — Modules Python connectés à états quantiques (ARPES/STM)
**Ce qui manquait** : Les modules `arpes_module.py` et `stm_module.py` dans `independent_modules/` existent mais reçoivent des CSV. Q15 restera toujours "partial" tant que ces modules ne reçoivent pas G(k,ω) réel.
**Solution** : Dans `Hubbard_HTS/modules/arpes_module.c` et `stm_module.c` (nouveau, en C), les modules reçoivent directement le tenseur Green `G[k][omega]` calculé depuis l'état DQMC. Connexion via structure partagée `green_tensor_t`.

### Trou T09 — Documentation physique PHYSICS_REFERENCE.md (TRACEABILITÉ ÉQUATIONS)
**Ce qui manquait** : Toutes les équations sont dans les rapports CHAT mais pas dans un document de référence canonique dans le code source. Un futur développeur ne sait pas quelle normalisation utiliser.
**Solution** : `Hubbard_HTS/docs/PHYSICS_REFERENCE.md` documente : l'Hamiltonien exact, la normalisation de chaque observable, les conventions (PBC, demi-remplissage, unités eV/site), les références bibliographiques (Hirsch 1985, White 1992, etc.).

### Trou T10 — Version sémantique et CHANGELOG (REPRODUCTIBILITÉ LONG TERME)
**Ce qui manquait** : Aucun versionnage du simulateur. Il est impossible de savoir quelle version a produit quel résultat.
**Solution** : `Hubbard_HTS/docs/CHANGELOG.md` avec version sémantique `HUBBARD_HTS_v1.0.0` et liste de toutes les corrections appliquées depuis BC-A1 jusqu'à BC-06.

---

## 8. PROTOCOLE DE DÉVELOPPEMENT — ÉTAPES ORDONNÉES

```
PHASE 1 — Core fermionique (bloquant pour tout le reste)
  [1.1] Implémenter fock_space.c : base de Fock pour N sites
  [1.2] Implémenter hamiltonian.c : apply_hubbard_hamiltonian()
  [1.3] Implémenter exactdiag_module.c : Lanczos pour L≤12
  [1.4] VALIDATION GATE : accord exact diag avec solveur 2x2 actuel (±1e-6 eV)

PHASE 2 — Observables physiques (dépend de Phase 1)
  [2.1] Implémenter observables.c : E, P, sign depuis état Fock — sources séparées
  [2.2] Implémenter green_function.c : G(i,τ) depuis DQMC ou exact diag
  [2.3] Appliquer corrections BC-04 (normalisation pairing par 2*N) et BC-06 (sign physique)
  [2.4] VALIDATION GATE : Pearson(E, P) < 0.5 (gate T01)

PHASE 3 — DQMC (dépend de Phase 2)
  [3.1] Implémenter dqmc_module.c : découplage Hubbard-Stratonovich, sweeps MC
  [3.2] Calibration N_warmup (500 sweeps) et N_measure (5000 mesures)
  [3.3] VALIDATION GATE : accord DQMC vs exact diag pour L=2×2 (±0.005 eV)

PHASE 4 — Tests physiques (dépend de Phase 3)
  [4.1] test_u_t_dependence.c : U/t = 0.1 → 12
  [4.2] test_temperature.c : β = 0.1 → 10
  [4.3] test_doping.c : δ = 0 → 0.3
  [4.4] test_lattice_scaling.c : L = 8 → 64 + fit L→∞
  [4.5] VALIDATION GATE : within_error_bar ≥ 80% sur références publiées (gate OC-03)

PHASE 5 — Connecter ARPES/STM (dépend de Phase 3)
  [5.1] Implémenter arpes_module.c : A(k,ω) depuis G(k,ω)
  [5.2] Implémenter stm_module.c : LDOS depuis G(r,ω=0)
  [5.3] VALIDATION GATE : Q15 devient "complete" (résolution OC-02)
  [5.4] Documenter Q12 plasma : remplacer par Q12_new "Mécanisme pairing d-wave vs s-wave clarifié ?"

PHASE 6 — Traçabilité totale (parallélisable avec phases 1-5)
  [6.1] Intégrer hfbl360_logger.c (noms exacts depuis tools/post_run_hfbl360_forensic_logger.py)
  [6.2] Implémenter checksum global du run (gate T02)
  [6.3] Générer PHYSICS_REFERENCE.md et CHANGELOG.md (trous T09, T10)
  [6.4] Script run_hubbard_hts.sh avec progression %, certification complétude

PHASE 7 — Détection invariants artificiels (dépend de Phases 1-4)
  [7.1] test_pearson_invariant.c (gate T01)
  [7.2] test_sign_problem.c (trou T03)
  [7.3] test_scrambling.c, test_noise_robustness.c, test_basis_independence.c
  [7.4] spectral_analysis.c avec test FFT multi-U/t (trou T05)

PHASE 8 — Validation finale (dépend de tout)
  [8.1] Exécuter run complet avec toutes les gates actives
  [8.2] Score global attendu : ≥ 95% (vs 88.95% actuel figé)
  [8.3] Rapport scientifique final dans docs/
  [8.4] Annonce : HUBBARD_HTS_v1.0.0 prêt pour audit externe
```

---

## 9. LIVRABLES ATTENDUS

1. **Code source complet** dans `Hubbard_HTS/` — modulaire, compilable, sans dépendances aux anciens CSV
2. **Pipeline indépendant** — chaque run repart de ψ(0) calculé, sans réinjection de résultats précédents
3. **Logs HFBAL_360 persistants** — horodatés UTC, hash par step, certifiés
4. **Checksum global** par run — `GLOBAL_CHECKSUM.sha512`
5. **Checklist complète validée** — toutes les cases cochées dans ce document
6. **Rapport scientifique** : E(U/t), P(T,δ), sign(β), spectres ARPES/STM, scaling L→∞
7. **Invariant E∼P∼n : disparu** — Pearson(E,P) < 0.3 confirmé
8. **within_error_bar ≥ 80%** — benchmarks références publiées
9. **PHYSICS_REFERENCE.md** — documentation canonique équations et unités
10. **CHANGELOG.md** — traçabilité de toutes les corrections depuis BC-A1 jusqu'à BC-06 et trous T01-T10

---

## 10. OBJECTIF FINAL

- Supprimer **définitivement** toute illusion de convergence universelle (E∼P∼n)
- Fournir un moteur Hubbard_HTS **exactement physique** (opérateurs fermioniques réels, espace de Fock, DQMC)
- **Résoudre Q12 et Q15** pour dépasser le plafond 88.95% — cible ≥ 95%
- Permettre aux experts de **répliquer, auditer et publier** sans ambiguïtés
- Créer une base solide pour **tous les tests HTS futurs**
- Atteindre l'accord **QMC/DMRG within_error_bar ≥ 80%** (vs 53% actuel)

```
VERSION : HUBBARD_HTS_PLAN_v3.0.0
DATE    : 2026-03-13T19:35Z
AUTEUR  : Agent Replit — inspection totale src/ ligne par ligne
BASÉ SUR: analysechatgpt1-8.md, AUTO_PROMPT_* (12 rapports), RAPPORT_* (20+ rapports)
TROUS COMBLÉS : T01 (Pearson gate), T02 (checksum global), T03 (sign physique),
                T04 (cohérence temporelle BC-03), T05 (FFT multi-U/t),
                T06 (BC PBC/OBC/APBC), T07 (extrapolation L→∞),
                T08 (ARPES/STM connectés), T09 (PHYSICS_REFERENCE),
                T10 (CHANGELOG versionné)
```
