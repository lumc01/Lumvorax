# RAPPORT INDÉPENDANT — far_from_equilibrium_kinetic_lattices

- Run: `research_20260307T220548Z_5643`
- UTC: `2026-03-07T22:42:44.152119+00:00`
- Statut module: **stable_proxy** (Comportement proxy cohérent, sans alerte locale majeure.)

## Phase 1 — Synchronisation et intégrité
- Checksums run: 67/67 OK ; missing=0 ; mismatch=0.
- Données utilisées: baseline_reanalysis_metrics, integration_physics_computed_observables, integration_entropy_observables, integration_spatial_correlations, benchmark_comparison_qmc_dmrg.

## Phase 2 — Analyse des données
- Points = 24 ; lattice_sites = 99.
- Énergie min/max = -9453.0738776762 / 893209.724111225.
- Énergie/site fin = 9022.32044556793 (rang énergie=9/13).
- Pairing start/end = 98.5435972367 / 143292.0101919628.
- Pairing normalisé fin = 1447.3940423430586 (rang pairing=11/13).
- Sign ratio min/max = -0.0002349072 / 0.0101010101.
- CPU/RAM moyens = 16.632083333333334% / 3.32%.
- Entropies (bits): energy=2.999390, pairing=3.886842, sign_abs=3.006867, combined=3.443116.

## Phase 3 — Vérification exhaustive A→Z
- Trace locale: step 0->2300, dE/dstep=388.362973, dPairing/dstep=62.258029.
- Corrélations par lag:
- lag=1: pairing_corr=0.999938, energy_corr=0.998413, sign_corr=0.499213
- lag=2: pairing_corr=0.999760, energy_corr=0.993446, sign_corr=0.464167
- lag=4: pairing_corr=0.999194, energy_corr=0.972434, sign_corr=0.339493
- lag=8: pairing_corr=0.998453, energy_corr=0.885614, sign_corr=-0.549505
- lag=16: pairing_corr=0.999752, energy_corr=-0.805572, sign_corr=0.554384

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
