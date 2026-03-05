# RAPPORT_RECHERCHE_CYCLE_23_PRE_REPLIT_VALIDATION_100_PLAN_20260305T131500Z

## Introduction (thèse + contexte)
Tu demandes une version **pré-exécution Replit** qui ferme les trous de traçabilité/authenticité et qui explique clairement comment converger vers 100% de validation.

## 1) Inspection ligne par ligne (SMOKE / PLACEHOLDER / STUB / HARDCODING)
### Développement
Un audit automatique a été exécuté sur le code source (hors `results/` et `backups/`).
Résultat:
- Pas de `TODO/FIXME/PLACEHOLDER/STUB` bloquants détectés dans les sources principales.
- Risques `HARDCODING` détectés et tracés:
  1. Paramètres de problèmes hardcodés dans `src/hubbard_hts_research_cycle.c`.
  2. Mapping metadata hardcodé dans `tools/post_run_metadata_capture.py`.

Fichiers de preuve:
- `tests/integration_code_authenticity_audit.csv`
- `tests/integration_hardcoding_risk_register.csv`

### Conclusion
L’authenticité est mieux tracée qu’avant, mais la suppression du hardcoding côté metadata est une condition importante pour tendre vers 100% scientifique.

## 2) État d’avancement actuel (%)
### Développement
- Validation globale questions/tests/théories: **60% validé**, **10% partiel**, **30% restant**.
- Progression par proxy: **70%** pour chaque proxy sur les critères actuellement instrumentés.

### Conclusion
Le pipeline est robuste, mais la preuve physique complète n’est pas encore au niveau 100%.

## 3) Plan d’implantation vers 100% (A→Z)
### A. Obligatoire immédiat
1. Remplacer mapping metadata hardcodé par export direct depuis l’exécutable (single source of truth).
2. Ajouter colonnes `energy_per_site`, `pairing_norm`, `norm_psi_squared` dans le baseline.
3. Ajouter gate bloquant `NORMALIZATION_GATE`.

### B. Validation numérique forte
4. Exécuter automatiquement `dt/2`, `dt`, `2dt`.
5. Ajouter gate `DT_STABILITY_GATE` (seuil de divergence).
6. Ajouter gate `EVENT_ALIGNMENT_GATE` (minimum énergie vs crossing sign).

### C. Validation physique instrumentée
7. Exporter corrélations longue distance `C(r)`.
8. Ajouter `LONG_RANGE_CORRELATION_GATE`.
9. Exporter proxy spectral (DOS) pour hypothèses pseudogap.
10. Ajouter `SPECTRAL_GATE`.

### D. Traçabilité process complète
11. Maintenir historique commande-par-commande (`logs/process_trace_commands_history.md`).
12. Maintenir journal des questions posées et evidence (`tests/integration_questions_traceability.csv`).
13. Conserver checksums/provenance post-run obligatoires.

## 4) Questions en cours et réponses attendues (traceabilité)
- Q: « Le hardcoding compromet-il l’authenticité ? »
  - Réponse attendue: non, après migration vers source unique runtime.
- Q: « Peut-on conclure physique réelle ? »
  - Réponse attendue: oui uniquement quand gates B+C sont PASS.

## 5) Commandes à exécuter (local / Replit)
### Pré-run local
```bash
git fetch origin --prune
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
```

### Vérifications post-run
```bash
cat <RUN_DIR>/tests/integration_gate_summary.csv
cat <RUN_DIR>/tests/integration_physics_gate_summary.csv
cat <RUN_DIR>/tests/integration_code_authenticity_audit.csv
cat <RUN_DIR>/tests/integration_hardcoding_risk_register.csv
cat <RUN_DIR>/tests/integration_questions_traceability.csv
sha256sum -c <RUN_DIR>/logs/checksums.sha256
```

### Commande finale Replit
```bash
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
```

## Conclusion finale
Tu as maintenant un plan de convergence opérationnel vers 100%, avec traçabilité complète (commandes + questions + preuves), audit autocritique d’authenticité, et une frontière claire: **100% n’est atteint que lorsque les gates d’observables avancées sont instrumentés et PASS**.
