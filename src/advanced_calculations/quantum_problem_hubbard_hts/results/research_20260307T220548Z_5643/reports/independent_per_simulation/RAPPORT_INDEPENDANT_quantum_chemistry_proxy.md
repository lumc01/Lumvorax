# RAPPORT INDÉPENDANT — quantum_chemistry_proxy

- Run: `research_20260307T220548Z_5643`
- UTC: `2026-03-07T22:42:44.152119+00:00`
- Statut module: **stable_proxy** (Comportement proxy cohérent, sans alerte locale majeure.)

## Phase 1 — Synchronisation et intégrité
- Checksums run: 67/67 OK ; missing=0 ; mismatch=0.
- Données utilisées: baseline_reanalysis_metrics, integration_physics_computed_observables, integration_entropy_observables, integration_spatial_correlations, benchmark_comparison_qmc_dmrg.

## Phase 2 — Analyse des données
- Points = 22 ; lattice_sites = 56.
- Énergie min/max = -9285.6031890364 / 365840.9699513367.
- Énergie/site fin = 6532.874463416726 (rang énergie=12/13).
- Pairing start/end = 55.8853476137 / 96286.8675503502.
- Pairing normalisé fin = 1719.4083491133965 (rang pairing=5/13).
- Sign ratio min/max = -0.1428571429 / 0.0222772277.
- CPU/RAM moyens = 15.997272727272728% / 3.368181818181818%.
- Entropies (bits): energy=2.949464, pairing=3.913977, sign_abs=0.790767, combined=3.431721.

## Phase 3 — Vérification exhaustive A→Z
- Trace locale: step 0->2100, dE/dstep=174.221978, dPairing/dstep=45.824277.
- Corrélations par lag:
- lag=1: pairing_corr=0.999983, energy_corr=0.997711, sign_corr=-0.651106
- lag=2: pairing_corr=0.999932, energy_corr=0.990472, sign_corr=-0.779287
- lag=4: pairing_corr=0.999748, energy_corr=0.959467, sign_corr=-0.356880
- lag=8: pairing_corr=0.999448, energy_corr=0.825907, sign_corr=0.151029
- lag=16: pairing_corr=0.999965, energy_corr=-0.995236, sign_corr=-0.415693

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
