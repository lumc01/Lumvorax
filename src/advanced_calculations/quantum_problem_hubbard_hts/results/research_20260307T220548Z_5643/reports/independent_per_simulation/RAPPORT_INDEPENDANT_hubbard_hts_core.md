# RAPPORT INDÉPENDANT — hubbard_hts_core

- Run: `research_20260307T220548Z_5643`
- UTC: `2026-03-07T22:42:44.152119+00:00`
- Statut module: **stable_proxy** (Comportement proxy cohérent, sans alerte locale majeure.)

## Phase 1 — Synchronisation et intégrité
- Checksums run: 67/67 OK ; missing=0 ; mismatch=0.
- Données utilisées: baseline_reanalysis_metrics, integration_physics_computed_observables, integration_entropy_observables, integration_spatial_correlations, benchmark_comparison_qmc_dmrg.

## Phase 2 — Analyse des données
- Points = 28 ; lattice_sites = 100.
- Énergie min/max = -10161.9532435754 / 1266799.9854127853.
- Énergie/site fin = 12667.999854127853 (rang énergie=2/13).
- Pairing start/end = 99.7079476457 / 192079.9176118617.
- Pairing normalisé fin = 1920.799176118617 (rang pairing=2/13).
- Sign ratio min/max = -0.04 / 0.0081188119.
- CPU/RAM moyens = 15.627857142857144% / 3.47%.
- Entropies (bits): energy=3.305435, pairing=3.923252, sign_abs=0.661055, combined=3.614344.

## Phase 3 — Vérification exhaustive A→Z
- Trace locale: step 0->2700, dE/dstep=469.194562, dPairing/dstep=71.103781.
- Corrélations par lag:
- lag=1: pairing_corr=0.999966, energy_corr=0.998898, sign_corr=-0.556917
- lag=2: pairing_corr=0.999862, energy_corr=0.995464, sign_corr=-0.628605
- lag=4: pairing_corr=0.999457, energy_corr=0.981027, sign_corr=0.113705
- lag=8: pairing_corr=0.998299, energy_corr=0.922574, sign_corr=-0.512017
- lag=16: pairing_corr=0.999163, energy_corr=0.656859, sign_corr=-0.275557

## Phase 4 — Analyse scientifique
- Benchmark QMC/DMRG: pairing 8/8 dans barres d'erreur; énergie 0/7 dans barres d'erreur.
- Hypothèse transition de phase: indicateur **partiel** (pairing cohérent, validation énergétique incomplète).
- Hypothèse pseudogap: **non testée directement** (pas de DOS/A(k,ω) dans ce run).
- Hypothèse artefact numérique: **plausible au niveau global run** (dt_convergence FAIL + von_neumann FAIL).

## Phase 5 — Interprétation pédagogique
- Énergie/site: coût moyen final de la simulation normalisé par taille du système.
- Pairing normalisé: force relative de corrélation de paires, comparable entre modules de tailles différentes.
- Sign ratio: garde-fou de cohérence des poids/signes; hors [-1,1] signale un problème de normalisation.
- Corrélations lag: mémoire temporelle/spatiale proxy; décroissance rapide => régime plus local.

## Phase 6 — Questions / Analyse / Réponse / Solution
1) **Question**: Ce module est-il localement cohérent ?
   - Analyse: sign ratio borné + traces monotones, sans rupture brutale locale.
   - Réponse: **oui, partiellement** (cohérence locale proxy).
   - Solution: compléter avec test de conservation dédié module par module.
2) **Question**: Peut-on conclure physiquement fort ?
   - Analyse: benchmark énergétique insuffisant dans ce run.
   - Réponse: **non** (conclusion quantitative incomplète).
   - Solution: recalibrage énergie + campagne benchmark étendue.
3) **Question**: Risque principal à court terme ?
   - Analyse: Comportement proxy cohérent, sans alerte locale majeure.
   - Réponse: **surveillance active** requise.
   - Solution: dt-sweep dense + contrôle spectral et conservation par module.

## Phase 7 — Correctifs proposés
- Ajouter test energy_conservation sur les 13 modules (pas un seul module).
- Ajouter contrôle du rayon spectral par module et par pas de temps.
- Ajouter comparaison croisée avec solveur implicite/symplectique.

## Phase 8 — Intégration technique
- Rapport généré automatiquement de façon indépendante, sans modifier les anciens rapports historiques.

## Phase 9 — Traçabilité
- Le hash du présent rapport est stocké dans `RAPPORTS_INDEPENDANTS_SHA256.txt`.
- Les données sources restent inchangées dans le run historique ciblé.

## Phase 10 — Commande reproductible
```bash
python3 src/advanced_calculations/quantum_problem_hubbard_hts/tools/generate_individual_simulation_reports.py --run-dir /workspace/Lumvorax/src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260307T220548Z_5643
```
