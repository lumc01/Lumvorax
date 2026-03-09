# Rapport forensique pédagogique complet — Run research_20260309T093816Z_2723

> **Méthode imposée respectée**: analyse statique uniquement (lecture directe CSV/logs locaux), sans ré-exécution de simulation et sans modification des anciens rapports.

## 1) Mise à jour dépôt distant

- Dépôt synchronisé avec `origin` (`https://github.com/lumc01/Lumvorax.git`) via `git fetch origin --prune`.
- Écart `HEAD...origin/main`: `0 / 0` (local déjà à jour).

## 2) Validation globale des nouveaux tests (dernière correction)

- **new_tests_results.csv**: total=80, PASS=19, OBSERVED=49, FAIL=12, PASS%=23.75, statut=OK.
- **integration_chatgpt_critical_tests.csv**: total=12, PASS=8, OBSERVED=1, FAIL=3, PASS%=66.6667, statut=OK.
- **integration_forensic_extension_tests.csv**: total=56, PASS=2, OBSERVED=53, FAIL=1, PASS%=3.5714, statut=OK.
- **GLOBAL**: total=148, PASS=29, OBSERVED=103, FAIL=16, PASS%=19.5946, statut=OK.

### Interprétation
- Le statut `OBSERVED` indique une mesure collectée mais non contractualisée en PASS/FAIL strict.
- Le pipeline est utilisable pour surveillance, mais **pas scientifiquement validé à 100%** pour revendication de fidélité physique (forts écarts benchmark).

## 3) État d'avancement par simulation (module par module)

| Module | Progression solution % | PASS % local | FAIL % local | Alignement benchmark % | Confiance % (heuristique) | Reste à compléter % |
|---|---:|---:|---:|---:|---:|---:|
| bosonic_multimode_systems | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| correlated_fermions_non_hubbard | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| dense_nuclear_proxy | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| far_from_equilibrium_kinetic_lattices | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| hubbard_hts_core | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| multi_correlated_fermion_boson_networks | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| multi_state_excited_chemistry | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| multiscale_nonlinear_field_models | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| qcd_lattice_proxy | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| quantum_chemistry_proxy | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| quantum_field_noneq | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| spin_liquid_exotic | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |
| topological_correlated_materials | 47.00 | 0.00 | 0.00 | 0.00 | 36.15 | 53.00 |

### Lecture pédagogique
- **Progression solution (47%)**: capacité du pipeline à couvrir des prérequis (traces, métadonnées, contrôles), pas preuve de vérité physique.
- **Reste à compléter (53%)**: blocage structurel explicite (`-8 common_missing_tests`) commun à tous les modules.
- **Alignement benchmark (0%)** sur les comparaisons QMC/DMRG et externes (toutes lignes `within_error_bar=0`).

## 4) Classement forensique par risque numérique (du plus risqué au moins risqué)

| Rang | Module | Score risque | Erreur relative benchmark moyenne | FAIL % local | Progression % | Confiance % |
|---:|---|---:|---:|---:|---:|---:|
| 1 | far_from_equilibrium_kinetic_lattices | 101.46 | 1.7112 | 0.00 | 47.00 | 36.15 |
| 2 | multiscale_nonlinear_field_models | 74.51 | 1.1722 | 0.00 | 47.00 | 36.15 |
| 3 | multi_correlated_fermion_boson_networks | 66.63 | 1.0146 | 0.00 | 47.00 | 36.15 |
| 4 | bosonic_multimode_systems | 63.70 | 0.9560 | 0.00 | 47.00 | 36.15 |
| 5 | correlated_fermions_non_hubbard | 59.92 | 0.8804 | 0.00 | 47.00 | 36.15 |
| 6 | topological_correlated_materials | 55.03 | 0.7826 | 0.00 | 47.00 | 36.15 |
| 7 | spin_liquid_exotic | 52.35 | 0.7290 | 0.00 | 47.00 | 36.15 |
| 8 | hubbard_hts_core | 51.87 | 0.7193 | 0.00 | 47.00 | 36.15 |
| 9 | multi_state_excited_chemistry | 49.58 | 0.6736 | 0.00 | 47.00 | 36.15 |
| 10 | quantum_field_noneq | 15.90 | 0.0000 | 0.00 | 47.00 | 36.15 |
| 11 | quantum_chemistry_proxy | 15.90 | 0.0000 | 0.00 | 47.00 | 36.15 |
| 12 | qcd_lattice_proxy | 15.90 | 0.0000 | 0.00 | 47.00 | 36.15 |
| 13 | dense_nuclear_proxy | 15.90 | 0.0000 | 0.00 | 47.00 | 36.15 |

## 5) Problèmes non passants — cours pédagogique (un par un)

### bosonic_multimode_systems

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 0.9560 (max 1.0285), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### correlated_fermions_non_hubbard

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 0.8804 (max 1.0163), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### dense_nuclear_proxy

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 0.0000 (max 0.0000), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### far_from_equilibrium_kinetic_lattices

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 1.7112 (max 2.4059), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### hubbard_hts_core

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 0.7193 (max 1.0011), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### multi_correlated_fermion_boson_networks

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 1.0146 (max 1.0186), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### multi_state_excited_chemistry

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 0.6736 (max 1.0229), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### multiscale_nonlinear_field_models

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 1.1722 (max 1.3185), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### qcd_lattice_proxy

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 0.0000 (max 0.0000), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### quantum_chemistry_proxy

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 0.0000 (max 0.0000), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### quantum_field_noneq

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 0.0000 (max 0.0000), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### spin_liquid_exotic

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 0.7290 (max 1.0100), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

### topological_correlated_materials

**Introduction (thèse + contexte)**. Ce module présente une progression opérationnelle à 47.00% mais un alignement benchmark à 0.00%, ce qui signifie que la chaîne de calcul fonctionne, cependant la validité physique revendiquée reste insuffisante.

**Développement (argumentation)**. De plus, le taux PASS local est 0.00% sur 0 tests inférés module; néanmoins le taux FAIL est 0.00%. Également, l'erreur relative benchmark moyenne est 0.7826 (max 1.0118), ce qui traduit un écart important à la référence. Cependant, les traces de reproductibilité inter-run sont stables (drift nul sur métriques communes), donc le problème principal est la **fidélité du modèle**, pas l'instabilité numérique brute.

**Conclusion (solution + clôture)**. Donc il faut: (1) recalibrer l'échelle des observables (normalisation/units), (2) connecter un solveur non-proxy pour contre-vérification, (3) convertir les tests OBSERVED critiques en PASS/FAIL contractuels. Ainsi, ce module pourra passer du statut “pipeline fonctionnel” à “résultat physiquement défendable”.

**Forces observées**: reproductibilité, métadonnées présentes, gate physique vert.

**Critiques majeures**: mismatch benchmark, progression plafonnée à 47%, dépendance proxy.

## 6) Détail ligne par ligne: SMOKE / PLACEHOLDER / STUB / HARDCODING

- Comptage patterns: {'HARDCODING': 20, 'TODO': 2, 'FIXME': 2, 'PLACEHOLDER': 4, 'STUB': 3, 'MOCK': 1}
- Comptage sévérités: {'review': 20, 'warning': 4, 'risk': 8}

| pattern | severity | file | line | snippet | interprétation |
|---|---|---|---:|---|---|
| HARDCODING | review | src/hubbard_hts_research_cycle (copy).c | 561 | problem_t probs[] = { | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | src/hubbard_hts_research_cycle.c | 561 | problem_t probs[] = { | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | src/hubbard_hts_research_cycle_advanced_parallel.c | 561 | problem_t probs[] = { | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| TODO | warning | tools/inspect_quantum_simulator_stacks.py | 17 | 'stub_or_placeholder': re.compile(r'\b(stub\|placeholder\|todo\|fixme\|mock)\b', re.IGNORECASE), | Mot-clé de maintenance; à confirmer contextuellement. |
| FIXME | warning | tools/inspect_quantum_simulator_stacks.py | 17 | 'stub_or_placeholder': re.compile(r'\b(stub\|placeholder\|todo\|fixme\|mock)\b', re.IGNORECASE), | Mot-clé de maintenance; à confirmer contextuellement. |
| PLACEHOLDER | risk | tools/inspect_quantum_simulator_stacks.py | 17 | 'stub_or_placeholder': re.compile(r'\b(stub\|placeholder\|todo\|fixme\|mock)\b', re.IGNORECASE), | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| STUB | risk | tools/inspect_quantum_simulator_stacks.py | 17 | 'stub_or_placeholder': re.compile(r'\b(stub\|placeholder\|todo\|fixme\|mock)\b', re.IGNORECASE), | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| MOCK | risk | tools/inspect_quantum_simulator_stacks.py | 17 | 'stub_or_placeholder': re.compile(r'\b(stub\|placeholder\|todo\|fixme\|mock)\b', re.IGNORECASE), | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/inspect_quantum_simulator_stacks.py | 18 | 'hardcoding_risk': re.compile(r'\b(hardcod\|magic number\|temporary\|test-only)\b', re.IGNORECASE), | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/inspect_quantum_simulator_stacks.py | 75 | lines.append(f"- hardcoding_risk_hits: {counts['hardcoding_risk']}") | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| TODO | warning | tools/inspect_quantum_simulator_stacks.py | 99 | cmd = ['rg', '-n', '-i', 'stub\|placeholder\|todo\|fixme\|hardcod', *TARGETS] | Mot-clé de maintenance; à confirmer contextuellement. |
| FIXME | warning | tools/inspect_quantum_simulator_stacks.py | 99 | cmd = ['rg', '-n', '-i', 'stub\|placeholder\|todo\|fixme\|hardcod', *TARGETS] | Mot-clé de maintenance; à confirmer contextuellement. |
| PLACEHOLDER | risk | tools/inspect_quantum_simulator_stacks.py | 99 | cmd = ['rg', '-n', '-i', 'stub\|placeholder\|todo\|fixme\|hardcod', *TARGETS] | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| STUB | risk | tools/inspect_quantum_simulator_stacks.py | 99 | cmd = ['rg', '-n', '-i', 'stub\|placeholder\|todo\|fixme\|hardcod', *TARGETS] | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/inspect_quantum_simulator_stacks.py | 99 | cmd = ['rg', '-n', '-i', 'stub\|placeholder\|todo\|fixme\|hardcod', *TARGETS] | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 9 | "hubbard_hts_core": {"lattice_size": "10x10", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.0, "U": 8.0, "mu": 0.2, "T": 95.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_hubbard_ | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 10 | "qcd_lattice_proxy": {"lattice_size": "9x9", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 0.7, "U": 9.0, "mu": 0.1, "T": 140.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_qcd_v4ne | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 11 | "quantum_field_noneq": {"lattice_size": "8x8", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.3, "U": 7.0, "mu": 0.05, "T": 180.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_qft_v | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 12 | "dense_nuclear_proxy": {"lattice_size": "9x8", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 0.8, "U": 11.0, "mu": 0.3, "T": 80.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy_n | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 13 | "quantum_chemistry_proxy": {"lattice_size": "8x7", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.6, "U": 6.5, "mu": 0.4, "T": 60.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_prox | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 14 | "spin_liquid_exotic": {"lattice_size": "12x10", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 0.9, "U": 10.5, "mu": 0.12, "T": 55.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_proxy | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 15 | "topological_correlated_materials": {"lattice_size": "11x11", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.1, "U": 7.8, "mu": 0.15, "T": 70.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvor | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 16 | "correlated_fermions_non_hubbard": {"lattice_size": "10x9", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.2, "U": 8.6, "mu": 0.18, "T": 85.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lum | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 17 | "multi_state_excited_chemistry": {"lattice_size": "9x9", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.5, "U": 6.8, "mu": 0.22, "T": 48.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax_pr | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 18 | "bosonic_multimode_systems": {"lattice_size": "10x8", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 0.6, "U": 5.2, "mu": 0.06, "T": 110.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": "lumvorax | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 19 | "multiscale_nonlinear_field_models": {"lattice_size": "12x8", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.4, "U": 9.2, "mu": 0.10, "T": 125.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id": " | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 20 | "far_from_equilibrium_kinetic_lattices": {"lattice_size": "11x9", "geometry": "rectangular", "boundary_conditions": "periodic_proxy", "t": 1.0, "U": 8.0, "mu": 0.09, "T": 150.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/post_run_metadata_capture.py | 21 | "multi_correlated_fermion_boson_networks": {"lattice_size": "10x10", "geometry": "square", "boundary_conditions": "periodic_proxy", "t": 1.05, "U": 7.4, "mu": 0.14, "T": 100.0, "dt": 1.0, "method": "advanced_proxy_deterministic", "model_id" | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| PLACEHOLDER | risk | tools/post_run_hfbl360_forensic_logger.py | 8 | Purpose: verify (without placeholder data) whether requested forensic naming/ | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| PLACEHOLDER | risk | tools/generate_cycle06_benchmark_course_report.py | 53 | '- Règle qualité: aucun placeholder/stub/hardcoding ajouté; uniquement exploitation des données brutes existantes et génération de nouveaux artefacts sidecar.', | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| STUB | risk | tools/generate_cycle06_benchmark_course_report.py | 53 | '- Règle qualité: aucun placeholder/stub/hardcoding ajouté; uniquement exploitation des données brutes existantes et génération de nouveaux artefacts sidecar.', | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |
| HARDCODING | review | tools/generate_cycle06_benchmark_course_report.py | 53 | '- Règle qualité: aucun placeholder/stub/hardcoding ajouté; uniquement exploitation des données brutes existantes et génération de nouveaux artefacts sidecar.', | Risque de biais/artefact à vérifier manuellement avant validation scientifique. |

## 7) Questions d'experts + réponses révélées par les résultats

1. **Les résultats sont-ils scientifiquement corrects à 100% ?** → Non. Les benchmarks de référence sont tous hors barre d'erreur (`within_error_bar=0`).
2. **La correction agent est-elle acceptable scientifiquement à 100% ?** → Non, acceptable en ingénierie de pipeline, insuffisant en fidélité physique.
3. **Le système est-il reproductible ?** → Oui sur les métriques comparées (drift max=0).
4. **Quels blocages empêchent la validation finale ?** → open questions: unités, cross-check solveur, vrai sweep dt, critères de phase explicites, guardrails production.

## 8) Nouvelles choses non intégrées (gaps)

- **Q_missing_units**: Are physical units explicit and consistent for all observables? (statut=Open) → action: Add units schema and unit-consistency gate.
- **Q_solver_crosscheck**: Do proxy results match at least one independent non-proxy solver on larger lattice? (statut=Open) → action: Maintain benchmark_comparison_qmc_dmrg.csv and extend lattice coverage.
- **Q_dt_real_sweep**: Is dt stability validated by true multi-run dt/2,dt,2dt (not proxy only)? (statut=Open) → action: Schedule 3-run sweep in CI night job.
- **Q_phase_criteria**: Are phase-transition criteria explicit (order parameter + finite-size scaling)? (statut=Open) → action: Add formal criteria and thresholds.
- **Q_production_guardrails**: Can V4 NEXT rollback instantly on degraded metrics? (statut=Open) → action: Keep rollout controller and rollback contract active.

## 9) Questions ouvertes → test ciblé à lancer (sans exécution ici)

| Question ouverte | Test ciblé recommandé | Critère de succès |
|---|---|---|
| Are physical units explicit and consistent for all observables? | Add units schema and unit-consistency gate | PASS si métrique contractualisée et seuil documenté. |
| Do proxy results match at least one independent non-proxy solver on larger lattice? | Maintain benchmark_comparison_qmc_dmrg.csv and extend lattice coverage | PASS si métrique contractualisée et seuil documenté. |
| Is dt stability validated by true multi-run dt/2,dt,2dt (not proxy only)? | Schedule 3-run sweep in CI night job | PASS si métrique contractualisée et seuil documenté. |
| Are phase-transition criteria explicit (order parameter + finite-size scaling)? | Add formal criteria and thresholds | PASS si métrique contractualisée et seuil documenté. |
| Can V4 NEXT rollback instantly on degraded metrics? | Keep rollout controller and rollback contract active | PASS si métrique contractualisée et seuil documenté. |

## 10) Algorithmes utilisés et rôle (d'après artefacts)

- **Euler explicite proxy**: intégration temporelle déterministe (`integration_scheme=euler_explicit`) pour générer trajectoires rapides, mais biais de modélisation possible.
- **Autocorrélation par lag**: mesure mémoire temporelle des séries (`integration_spatial_correlations.csv`).
- **Entropy proxy (Shannon)**: quantifie dispersion observables (`integration_entropy_observables.csv`).
- **Bootstrap CI95 exposant alpha**: incertitude sur loi d'échelle (`integration_forensic_extension_tests.csv`).
- **Drift monitor inter-run**: contrôle reproductibilité point-à-point (`integration_run_drift_monitor.csv`).

## 11) Comparaison littérature/simulations existantes (cadre disponible localement)

- La base locale compare déjà à des références `qmc_dmrg_reference` et modules externes.
- Résultat: écart systématique important (aucune ligne dans barre d'erreur), suggérant désalignement d'échelle ou de définition d'observable.
- Une comparaison “en ligne” exhaustive (collecte externe de nouveaux papiers/runs) n'est pas réalisée dans ce rapport pour respecter la contrainte d'inspection locale sans nouvelles exécutions expérimentales.

## 12) Conclusion exécutive

Le run est **opérationnel mais non validé scientifiquement à 100%**. De cette manière, la prochaine étape n'est pas de produire plus de résumés, mais de fermer les 5 questions ouvertes, réduire les écarts benchmark sous barres d'erreur, et convertir les mesures OBSERVED en critères PASS/FAIL robustes.
