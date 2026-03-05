# Rapport technique itératif — cycle 15 (intégrations du cycle 14 appliquées immédiatement)

**Nom exact du rapport :** `RAPPORT_RECHERCHE_CYCLE_15_INTEGRATIONS_APPLIQUEES_20260305T003000Z.md`

## 1) Ce qui a été intégré maintenant (sans attendre)
- Intégration d’un **guard post-run** exécuté automatiquement après chaque `run_research_cycle.sh`.
- Génération automatique des artefacts suivants à chaque nouveau run:
  - `tests/integration_terms_glossary.csv`
  - `tests/integration_claim_confidence_tags.csv`
  - `tests/integration_absent_metadata_fields.csv`
  - `tests/integration_manifest_check.csv`
  - `tests/integration_run_drift_monitor.csv`
  - `tests/integration_gate_summary.csv`

## 2) Problèmes rencontrés en cours de route et solutions appliquées
1. **Problème**: les recommandations cycle 14 étaient documentées mais non exécutées automatiquement.
   - **Solution**: ajout de `tools/post_run_cycle_guard.py` et appel direct depuis `run_research_cycle.sh`.
   - **Résultat**: les nouveaux artefacts d’intégration sont générés à chaque run.

2. **Problème**: compréhension non-expert insuffisante.
   - **Solution**: glossaire automatique des termes/valeurs (`integration_terms_glossary.csv`).
   - **Résultat**: explication standardisée intégrée au pipeline.

3. **Problème**: confusion possible entre certitude et hypothèse.
   - **Solution**: tags de confiance (`certain/probable/unknown`).
   - **Résultat**: claims explicitement qualifiés dans `integration_claim_confidence_tags.csv`.

4. **Problème**: manque de visibilité des métadonnées physiques absentes.
   - **Solution**: extracteur ABSENT automatique.
   - **Résultat**: `integration_absent_metadata_fields.csv` liste les champs manquants à intégrer avant claims physiques forts.

5. **Problème**: besoin de distinguer dérive infra et signal modèle.
   - **Solution**: run drift monitor vs run précédent.
   - **Résultat**: dérive `elapsed_ns` suivie explicitement, observables physiques comparées point-à-point.

## 3) Résultats de validation des intégrations sur le run `research_20260305T002238Z_5054`
- Gate summary:
  - `CSV_INTEGRITY_GATE = PASS`
  - `MODULE_COVERAGE_GATE = PASS`
  - `GLOSSARY_GATE = PASS`
  - `CONFIDENCE_TAG_GATE = PASS`
  - `ABSENT_METADATA_EXTRACTOR_GATE = PASS`
- Drift monitor:
  - `max_abs_diff_energy = 0.0`
  - `max_abs_diff_pairing = 0.0`
  - `max_abs_diff_sign_ratio = 0.0`
  - `max_abs_diff_elapsed_ns > 0` (variabilité performance/infrastructure)

## 4) Commande Replit à exécuter
```bash
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
```

## 5) Ce que cette commande fera désormais automatiquement
1. Build + exécution du cycle de recherche.
2. Génération du run horodaté.
3. Exécution automatique du guard d’intégration cycle 14.
4. Génération des checksums finaux de tous les artefacts du run.

## 6) Étape suivante
Après ton exécution Replit, envoie le nouveau chemin `Research cycle terminé: ...` et je lance l’analyse cycle suivant avec la même méthode.
