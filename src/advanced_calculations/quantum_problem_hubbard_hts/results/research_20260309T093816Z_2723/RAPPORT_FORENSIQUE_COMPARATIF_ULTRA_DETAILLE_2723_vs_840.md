# Rapport forensique comparatif ultra-détaillé — 2723 vs 840

> Contraintes respectées : lecture directe CSV/logs locaux, aucune relance de simulation, nouveau fichier uniquement, anciens `.md` non modifiés.

## A. Mise à jour du dépôt distant

- Remote cible : `https://github.com/lumc01/Lumvorax.git`.
- Synchronisation faite via `git fetch origin --prune`.
- Écart actuel `HEAD...origin/main` : `1/0` (branche locale avec 1 commit d'analyse en avance).

## B. Validation / invalidation des nouveaux tests (comparaison en %)

| Suite | PASS% run 2723 | PASS% run 840 | Différence points | Différence relative % | Verdict |
|---|---:|---:|---:|---:|---|
| new_tests_results.csv | 23.7500 | 26.2500 | -2.5000 | -9.52 | Régression |
| integration_chatgpt_critical_tests.csv | 66.6667 | 66.6667 | 0.0000 | 0.00 | Stable |
| integration_forensic_extension_tests.csv | 3.5714 | 17.8571 | -14.2857 | -80.00 | Régression |
| GLOBAL | 19.5946 | 26.3514 | -6.7568 | -25.64 | Régression |

### Interprétation pédagogique
- De plus, la suite `GLOBAL` passe de 26.3514% à 19.5946% (régression de -6.7568 points).
- Cependant, `integration_chatgpt_critical_tests.csv` reste stable à 66.6667%.
- Néanmoins, la suite forensique (`integration_forensic_extension_tests.csv`) chute fortement en PASS% (beaucoup plus de résultats classés `OBSERVED`).

## C. État d'avancement par simulation (module par module) + différence vs run 840

| Module | Progression 2723 % | Progression 840 % | Δ points | Δ relatif % | Reste à valider 2723 % | Alignement benchmark 2723 % | Alignement benchmark 840 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| bosonic_multimode_systems | 47.00 | 57.35 | -10.35 | -18.05 | 53.00 | 0.00 | 0.00 |
| correlated_fermions_non_hubbard | 47.00 | 59.75 | -12.75 | -21.34 | 53.00 | 0.00 | 0.00 |
| dense_nuclear_proxy | 47.00 | 58.60 | -11.60 | -19.80 | 53.00 | 0.00 | 0.00 |
| far_from_equilibrium_kinetic_lattices | 47.00 | 56.47 | -9.47 | -16.77 | 53.00 | 0.00 | 0.00 |
| hubbard_hts_core | 47.00 | 70.03 | -23.03 | -32.89 | 53.00 | 0.00 | 53.33 |
| multi_correlated_fermion_boson_networks | 47.00 | 59.82 | -12.82 | -21.43 | 53.00 | 0.00 | 0.00 |
| multi_state_excited_chemistry | 47.00 | 69.44 | -22.44 | -32.32 | 53.00 | 0.00 | 0.00 |
| multiscale_nonlinear_field_models | 47.00 | 65.76 | -18.76 | -28.53 | 53.00 | 0.00 | 0.00 |
| qcd_lattice_proxy | 47.00 | 56.41 | -9.41 | -16.68 | 53.00 | 0.00 | 0.00 |
| quantum_chemistry_proxy | 47.00 | 69.91 | -22.91 | -32.77 | 53.00 | 0.00 | 0.00 |
| quantum_field_noneq | 47.00 | 63.43 | -16.43 | -25.90 | 53.00 | 0.00 | 0.00 |
| spin_liquid_exotic | 47.00 | 62.33 | -15.33 | -24.59 | 53.00 | 0.00 | 0.00 |
| topological_correlated_materials | 47.00 | 61.93 | -14.93 | -24.11 | 53.00 | 0.00 | 0.00 |

## D. Cours pédagogique complet — problème par problème (Introduction / Développement / Conclusion)

### bosonic_multimode_systems

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 57.35% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 10.35 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 0.9560 (vs 21.8763). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### correlated_fermions_non_hubbard

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 59.75% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 12.75 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 0.8804 (vs 12.1826). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### dense_nuclear_proxy

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 58.60% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 11.60 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 0.0000 (vs 0.0000). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### far_from_equilibrium_kinetic_lattices

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 56.47% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 9.47 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 1.7112 (vs 14.6048). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### hubbard_hts_core

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 70.03% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 23.03 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 53.33% avant), avec une erreur relative moyenne 0.7193 (vs 0.0208). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### multi_correlated_fermion_boson_networks

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 59.82% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 12.82 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 1.0146 (vs 8.0869). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### multi_state_excited_chemistry

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 69.44% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 22.44 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 0.6736 (vs 8.7918). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### multiscale_nonlinear_field_models

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 65.76% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 18.76 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 1.1722 (vs 16.2795). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### qcd_lattice_proxy

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 56.41% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 9.41 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 0.0000 (vs 0.0000). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### quantum_chemistry_proxy

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 69.91% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 22.91 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 0.0000 (vs 0.0000). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### quantum_field_noneq

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 63.43% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 16.43 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 0.0000 (vs 0.0000). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### spin_liquid_exotic

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 62.33% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 15.33 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 0.7290 (vs 15.5449). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

### topological_correlated_materials

**Introduction (thèse + contexte).** Ce module est à 47.00% de progression dans le run 2723, contre 61.93% dans le run 840. De plus, le besoin utilisateur est de savoir si la correction est scientifiquement acceptable à 100%.

**Développement (argumentation).** Cependant, on observe une régression de 14.93 points. Également, il reste 53.00% à compléter pour atteindre 100%. Le taux d'alignement benchmark est 0.00% (vs 0.00% avant), avec une erreur relative moyenne 0.7826 (vs 11.2441). Néanmoins, quand cet alignement est nul, cela veut dire que le modèle ne rentre pas dans les barres d'erreur de référence : la fidélité physique est insuffisante même si la pipeline technique tourne correctement.

**Conclusion (solution + clôture).** Donc, pour débloquer ce problème : (1) normaliser les observables avec unités explicites, (2) ajouter un solveur indépendant non-proxy pour cross-check, (3) transformer les `OBSERVED` critiques en critères PASS/FAIL avec seuils. Ainsi, la validation scientifique pourra progresser de manière traçable.

**Points forts observés.** Reproductibilité inter-run monitorée; métadonnées présentes; chaîne d'intégration active.

**Points critiques observés.** Alignement benchmark insuffisant; progression partielle; dépendances proxy.

## E. Classement complet par risque numérique + score de confiance par module

| Rang risque | Module | Score risque (0 bas, + élevé = plus risqué) | Score confiance % | Erreur relative moyenne (2723) | Reste à compléter % |
|---:|---|---:|---:|---:|---:|
| 1 | far_from_equilibrium_kinetic_lattices | 127.37 | 23.58 | 1.7112 | 53.00 |
| 2 | multiscale_nonlinear_field_models | 97.72 | 41.37 | 1.1722 | 53.00 |
| 3 | multi_correlated_fermion_boson_networks | 89.06 | 46.57 | 1.0146 | 53.00 |
| 4 | bosonic_multimode_systems | 85.83 | 48.50 | 0.9560 | 53.00 |
| 5 | correlated_fermions_non_hubbard | 81.67 | 51.00 | 0.8804 | 53.00 |
| 6 | topological_correlated_materials | 76.29 | 54.22 | 0.7826 | 53.00 |
| 7 | spin_liquid_exotic | 73.34 | 55.99 | 0.7290 | 53.00 |
| 8 | hubbard_hts_core | 72.81 | 56.31 | 0.7193 | 53.00 |
| 9 | multi_state_excited_chemistry | 70.30 | 57.82 | 0.6736 | 53.00 |
| 10 | quantum_field_noneq | 33.25 | 80.05 | 0.0000 | 53.00 |
| 11 | quantum_chemistry_proxy | 33.25 | 80.05 | 0.0000 | 53.00 |
| 12 | qcd_lattice_proxy | 33.25 | 80.05 | 0.0000 | 53.00 |
| 13 | dense_nuclear_proxy | 33.25 | 80.05 | 0.0000 | 53.00 |

## F. Inspection authenticité : SMOKE / PLACEHOLDER / STUB / HARDCODING (ligne par ligne)

- Comptage patterns run 2723: {'HARDCODING': 20, 'TODO': 2, 'FIXME': 2, 'PLACEHOLDER': 4, 'STUB': 3, 'MOCK': 1}
- Comptage patterns run 840 : {'HARDCODING': 19, 'TODO': 2, 'FIXME': 2, 'PLACEHOLDER': 4, 'STUB': 3, 'MOCK': 1}

| Pattern | Sévérité | Fichier | Ligne | Snippet | Risque pour validité réelle |
|---|---|---|---:|---|---|
| HARDCODING | review | src/hubbard_hts_research_cycle (copy).c | 561 | problem_t probs[] = { | Élevé/à vérifier |
| HARDCODING | review | src/hubbard_hts_research_cycle.c | 561 | problem_t probs[] = { | Élevé/à vérifier |
| HARDCODING | review | src/hubbard_hts_research_cycle_advanced_parallel.c | 561 | problem_t probs[] = { | Élevé/à vérifier |
| TODO | warning | tools/inspect_quantum_simulator_stacks.py | 17 | 'stub_or_placeholder': re.compile(r'\b(stub\|placeholder\|todo\|fixme\|mock)\b', re.IGNORECASE), | Moyen/à confirmer |
| FIXME | warning | tools/inspect_quantum_simulator_stacks.py | 17 | 'stub_or_placeholder': re.compile(r'\b(stub\|placeholder\|todo\|fixme\|mock)\b', re.IGNORECASE), | Moyen/à confirmer |
| PLACEHOLDER | risk | tools/inspect_quantum_simulator_stacks.py | 17 | 'stub_or_placeholder': re.compile(r'\b(stub\|placeholder\|todo\|fixme\|mock)\b', re.IGNORECASE), | Élevé/à vérifier |
| STUB | risk | tools/inspect_quantum_simulator_stacks.py | 17 | 'stub_or_placeholder': re.compile(r'\b(stub\|placeholder\|todo\|fixme\|mock)\b', re.IGNORECASE), | Élevé/à vérifier |
| MOCK | risk | tools/inspect_quantum_simulator_stacks.py | 17 | 'stub_or_placeholder': re.compile(r'\b(stub\|placeholder\|todo\|fixme\|mock)\b', re.IGNORECASE), | Élevé/à vérifier |
| HARDCODING | review | tools/inspect_quantum_simulator_stacks.py | 18 | 'hardcoding_risk': re.compile(r'\b(hardcod\|magic number\|temporary\|test-only)\b', re.IGNORECASE), | Élevé/à vérifier |
| HARDCODING | review | tools/inspect_quantum_simulator_stacks.py | 75 | lines.append(f"- hardcoding_risk_hits: {counts['hardcoding_risk']}") | Élevé/à vérifier |
| TODO | warning | tools/inspect_quantum_simulator_stacks.py | 99 | cmd = ['rg', '-n', '-i', 'stub\|placeholder\|todo\|fixme\|hardcod', *TARGETS] | Moyen/à confirmer |
| FIXME | warning | tools/inspect_quantum_simulator_stacks.py | 99 | cmd = ['rg', '-n', '-i', 'stub\|placeholder\|todo\|fixme\|hardcod', *TARGETS] | Moyen/à confirmer |
| PLACEHOLDER | risk | tools/inspect_quantum_simulator_stacks.py | 99 | cmd = ['rg', '-n', '-i', 'stub\|placeholder\|todo\|fixme\|hardcod', *TARGETS] | Élevé/à vérifier |
| STUB | risk | tools/inspect_quantum_simulator_stacks.py | 99 | cmd = ['rg', '-n', '-i', 'stub\|placeholder\|todo\|fixme\|hardcod', *TARGETS] | Élevé/à vérifier |
| HARDCODING | review | tools/inspect_quantum_simulator_stacks.py | 99 | cmd = ['rg', '-n', '-i', 'stub\|placeholder\|todo\|fixme\|hardcod', *TARGETS] | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 9 | "hubbard_hts_core": {"lattice_size": "10x10", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.0, "U": 8.0, "mu": 0.2, "T": 95.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_hubbard_ | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 10 | "qcd_lattice_proxy": {"lattice_size": "9x9", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 0.7, "U": 9.0, "mu": 0.1, "T": 140.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_qcd_v4ne | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 11 | "quantum_field_noneq": {"lattice_size": "8x8", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.3, "U": 7.0, "mu": 0.05, "T": 180.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_qft_v | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 12 | "dense_nuclear_proxy": {"lattice_size": "9x8", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 0.8, "U": 11.0, "mu": 0.3, "T": 80.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_n | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 13 | "quantum_chemistry_proxy": {"lattice_size": "8x7", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.6, "U": 6.5, "mu": 0.4, "T": 60.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_prox | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 14 | "spin_liquid_exotic": {"lattice_size": "12x10", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 0.9, "U": 10.5, "mu": 0.12, "T": 55.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 15 | "topological_correlated_materials": {"lattice_size": "11x11", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.1, "U": 7.8, "mu": 0.15, "T": 70.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvor | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 16 | "correlated_fermions_non_hubbard": {"lattice_size": "10x9", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.2, "U": 8.6, "mu": 0.18, "T": 85.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lum | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 17 | "multi_state_excited_chemistry": {"lattice_size": "9x9", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.5, "U": 6.8, "mu": 0.22, "T": 48.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_pr | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 18 | "bosonic_multimode_systems": {"lattice_size": "10x8", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 0.6, "U": 5.2, "mu": 0.06, "T": 110.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 19 | "multiscale_nonlinear_field_models": {"lattice_size": "12x8", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.4, "U": 9.2, "mu": 0.10, "T": 125.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": " | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 20 | "far_from_equilibrium_kinetic_lattices": {"lattice_size": "11x9", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.0, "U": 8.0, "mu": 0.09, "T": 150.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id | Élevé/à vérifier |
| HARDCODING | review | tools/post_run_metadata_capture.py | 21 | "multi_correlated_fermion_boson_networks": {"lattice_size": "10x10", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.05, "U": 7.4, "mu": 0.14, "T": 100.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id" | Élevé/à vérifier |
| PLACEHOLDER | risk | tools/post_run_hfbl360_forensic_logger.py | 8 | Purpose: verify (without placeholder data) whether requested forensic naming/ | Élevé/à vérifier |
| PLACEHOLDER | risk | tools/generate_cycle06_benchmark_course_report.py | 53 | '- Règle qualité: aucun placeholder/stub/hardcoding ajouté; uniquement exploitation des données brutes existantes et génération de nouveaux artefacts sidecar.', | Élevé/à vérifier |
| STUB | risk | tools/generate_cycle06_benchmark_course_report.py | 53 | '- Règle qualité: aucun placeholder/stub/hardcoding ajouté; uniquement exploitation des données brutes existantes et génération de nouveaux artefacts sidecar.', | Élevé/à vérifier |
| HARDCODING | review | tools/generate_cycle06_benchmark_course_report.py | 53 | '- Règle qualité: aucun placeholder/stub/hardcoding ajouté; uniquement exploitation des données brutes existantes et génération de nouveaux artefacts sidecar.', | Élevé/à vérifier |

## G. Tableau “question ouverte → test ciblé à lancer” (sans exécution)

| Question ouverte | Pourquoi c'est bloquant | Test ciblé recommandé | Critère de validation |
|---|---|---|---|
| Are physical units explicit and consistent for all observables? | Invalide la conclusion scientifique si non traité | Add units schema and unit-consistency gate | PASS si seuil chiffré + preuve versionnée |
| Do proxy results match at least one independent non-proxy solver on larger lattice? | Invalide la conclusion scientifique si non traité | Maintain benchmark_comparison_qmc_dmrg.csv and extend lattice coverage | PASS si seuil chiffré + preuve versionnée |
| Is dt stability validated by true multi-run dt/2,dt,2dt (not proxy only)? | Invalide la conclusion scientifique si non traité | Schedule 3-run sweep in CI night job | PASS si seuil chiffré + preuve versionnée |
| Are phase-transition criteria explicit (order parameter + finite-size scaling)? | Invalide la conclusion scientifique si non traité | Add formal criteria and thresholds | PASS si seuil chiffré + preuve versionnée |
| Can V4 NEXT rollback instantly on degraded metrics? | Invalide la conclusion scientifique si non traité | Keep rollout controller and rollback contract active | PASS si seuil chiffré + preuve versionnée |

## H. Dictionnaire pédagogique des termes techniques (niveau débutant)

| Terme | Explication simple |
|---|---|
| problem | Nom du module simulé |
| step | Indice d'avancement numérique |
| energy | Observable énergétique interne |
| pairing | Observable de corrélation |
| sign_ratio | Indicateur statistique de signe |
| cpu_percent | Charge CPU système |
| mem_percent | Usage RAM système |
| elapsed_ns | Temps écoulé en nanosecondes |

## I. Questions d'experts et réponses observables dans les artefacts

1. **La correction est-elle scientifiquement acceptable à 100% ?** Non. Le run 2723 conserve un alignement benchmark nul sur les tableaux de comparaison.
2. **Que révèlent les anomalies ?** Une forte dépendance proxy et des écarts d'échelle des observables.
3. **Découvertes inconnues au sens littérature ?** Ce dépôt local ne fournit pas de preuve robuste d'une découverte nouvelle; il montre surtout des zones d'incertitude méthodologique à lever.
4. **Comparaison aux simulations existantes en ligne ?** Ici la comparaison est strictement limitée aux références locales QMC/DMRG et modules externes versionnés dans le dépôt, sans scraping web supplémentaire.

## J. Algorithmes utilisés et rôle exact pour chaque simulation (lecture artefacts)

- **Euler explicite (proxy)** : intègre pas-à-pas les dynamiques (rapide, mais potentiellement biaisé).
- **Autocorrélation lag** : mesure la persistance temporelle des signaux.
- **Entropie de Shannon proxy** : quantifie la dispersion des observables.
- **Bootstrap CI95** : estime l'incertitude des exposants de scaling.
- **Drift monitor inter-run** : compare deux runs pour vérifier stabilité numérique/reproductibilité.

## K. Verdict final

La correction n'est **pas validée scientifiquement à 100%**. De cette manière, il faut prioriser les tests de cross-solveur, l'explicitation des unités et la contractualisation des seuils PASS/FAIL avant toute revendication scientifique forte.
