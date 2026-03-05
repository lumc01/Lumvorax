# Rapport technique itératif — cycle 12 (copie travaillée depuis cycle 11 + exécution Replit guidée + notifications problème/solution)

**Nom exact du rapport :** `RAPPORT_RECHERCHE_CYCLE_12_PLAN_EXECUTION_REPLIT_ET_CORRECTIONS_20260304T221024Z.md`

## 0) Origine et méthode de mise à jour
- Ce rapport est construit **à partir de la version cycle 11** puis enrichi pour couvrir: exécution Replit, suivi problème/solution en cours de route, revue anti-oubli, et ajustements.
- Fichier de base copié/travaillé: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260304T185929Z_1030/reports/RAPPORT_RECHERCHE_CYCLE_11_PLAN_CORRECTIF_VALIDATION_100_20260304T205012Z.md`

## 1) Commande exacte à exécuter sur Replit (immédiat)
Exécute exactement:

```bash
bash src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
```

Puis, pour récupérer le dernier dossier de résultats:

```bash
LATEST=$(ls -dt src/advanced_calculations/quantum_problem_hubbard_hts/results/research_* | head -n 1) && echo "Nouveau dossier: $LATEST" && find "$LATEST" -maxdepth 3 -type f | sort
```

Et envoie-moi ensuite:
- le chemin `Research cycle terminé: ...`,
- le contenu de `logs/baseline_reanalysis_metrics.csv`,
- les nouveaux `tests/*.csv` et `logs/*.log`.

## 2) Notifications obligatoires problème → solution (en cours de route)
Chaque fois qu’un problème est rencontré, consigner: **Symptôme**, **Cause racine probable**, **Solution appliquée**, **Preuve**, **Statut**.
- Registre CSV prêt: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260304T185929Z_1030/tests/cycle12_problem_solution_register_20260304T221024Z.csv`

## 3) Revue complète anti-oubli (ce qui doit être revérifié à chaque cycle)
- Checklist CSV prête: `src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260304T185929Z_1030/tests/cycle12_review_checklist_20260304T221024Z.csv`
- Aucun rapport ne doit partir si un item P0 reste FAIL.

## 4) Corrections à intégrer simultanément (P0 immédiat)
1. Writer CSV transactionnel (tmp->fsync->rename) + checksum footer + row count.
2. Validation de complétude modules/steps (manifest) avant publication.
3. Métadonnées physiques obligatoires (`U/t`, lattice, BC, dt, Hamiltonian, seed, version algo).
4. Observables normalisées (pairing_per_site, susceptibilité, corrélateur connecté).
5. Statistiques robustes (CI95/bootstrap/autocorr/ESS).
6. Garde-fous CI bloquants (intégrité, couverture, métadonnées, tests).

## 5) Réponses expertes ciblées sur 1.B, 1.C, 1.D, 2, 3, 4 (version actionnable)
### 1.B Pairing
- Problème: non interprétable physiquement en brut.
- Solution: définition normalisée + unités + benchmark analytique.
- Critère PASS: erreur <2% sur cas jouet + cohérence finite-size.

### 1.C sign_ratio
- Problème: diagnostic statistique incomplet.
- Solution: ESS + iat + bootstrap + courbe sign(beta,L).
- Critère PASS: conclusion autorisée seulement avec CI et puissance statistique suffisante.

### 1.D Runtime
- Problème: complexité extrapolée depuis échantillon local.
- Solution: sweep de tailles + fit exposant + R².
- Critère PASS: exponent stable multi-run.

### 2 Validation technique vs physique
- Problème: claims physiques trop tôt.
- Solution: gate Physics-Claim bloquant sans prérequis.
- Critère PASS: tous prérequis renseignés et validés.

### 3 Diagnostic probable
- Problème: cause instabilité non isolée.
- Solution: ablation factorielle (dt/intégrateur/forcing/normalisation).
- Critère PASS: facteurs dominants quantifiés.

### 4 Clarification engineering vs science
- Problème: confusion possible du niveau de preuve.
- Solution: double badge TechGate/PhysicsGate dans chaque rapport.
- Critère PASS: badges explicites + raisons documentées.

## 6) Ajustements ajoutés (ce qui avait pu être oublié)
- Plan de notification en continu problème/solution.
- Revue systématique anti-oubli avant livraison.
- Commandes Replit prêtes + protocole de retour pour réanalyse immédiate.

## 7) Limite explicitée (honnêteté scientifique)
- On peut viser une validation **très robuste**; on ne promet pas une vérité physique “100% absolue” sans nouvelles preuves expérimentales/numériques après intégration des correctifs.

## 8) Prochaine étape opératoire
1. Tu exécutes la commande Replit ci-dessus.
2. Tu m’envoies le nouveau dossier généré.
3. Je lance cycle 13: réanalyse complète + notification détaillée problème/solution + nouveaux correctifs si besoin.
