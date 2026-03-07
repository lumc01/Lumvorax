# RAPPORT INDÉPENDANT — quantum_field_noneq

- Run: `research_20260307T220548Z_5643`
- UTC: `2026-03-07T22:42:44.152119+00:00`
- Statut module: **surveillance** (Pairing normalisé plus faible que le groupe principal.)

## Phase 1 — Synchronisation et intégrité
- Checksums run: 67/67 OK ; missing=0 ; mismatch=0.
- Données utilisées: baseline_reanalysis_metrics, integration_physics_computed_observables, integration_entropy_observables, integration_spatial_correlations, benchmark_comparison_qmc_dmrg.

## Phase 2 — Analyse des données
- Points = 21 ; lattice_sites = 64.
- Énergie min/max = -9064.6413370327 / 425579.0607619419.
- Énergie/site fin = 6649.672824405342 (rang énergie=11/13).
- Pairing start/end = 63.6972204345 / 75452.169692348.
- Pairing normalisé fin = 1178.9401514429376 (rang pairing=13/13).
- Sign ratio min/max = -0.125 / 0.0006871564.
- CPU/RAM moyens = 15.82952380952381% / 3.4242857142857144%.
- Entropies (bits): energy=2.810447, pairing=3.784942, sign_abs=0.548954, combined=3.297694.

## Phase 3 — Vérification exhaustive A→Z
- Trace locale: step 0->2000, dE/dstep=212.798560, dPairing/dstep=37.694236.
- Corrélations par lag:
- lag=1: pairing_corr=0.999862, energy_corr=0.997335, sign_corr=0.752534
- lag=2: pairing_corr=0.999480, energy_corr=0.988829, sign_corr=-0.081623
- lag=4: pairing_corr=0.998377, energy_corr=0.951677, sign_corr=-0.229350
- lag=8: pairing_corr=0.997998, energy_corr=0.771269, sign_corr=0.109694
- lag=16: pairing_corr=0.999956, energy_corr=-0.999266, sign_corr=0.704808

## Phase 4 — Analyse scientifique
- Aucun benchmark QMC/DMRG spécifique à ce module dans ce run.
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
   - Analyse: Pairing normalisé plus faible que le groupe principal.
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
