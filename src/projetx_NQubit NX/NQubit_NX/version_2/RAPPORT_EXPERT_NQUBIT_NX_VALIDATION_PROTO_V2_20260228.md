# RAPPORT EXPERT COMPLEMENTAIRE — NQubit NX (VERSION 2)

> **Portée V2** : itération du rapport précédent en conservant strictement la V1 intacte.
>
> **Contrainte respectée** : travail réalisé uniquement dans `src/projetx_NQubit NX/NQubit_NX` via une copie en dossier versionné.
>
> **Source V1 copiée depuis** : `RAPPORT_IAMO3/RAPPORT_EXPERT_NQUBIT_NX_VALIDATION_PROTO_20260228.md`.

---

## 1) Réponse directe à la demande

Oui, la prochaine version est produite.
- La V1 n’est pas modifiée.
- La V2 est placée dans un nouveau dossier : `src/projetx_NQubit NX/NQubit_NX/version_2/`.
- Cette V2 renforce :
  1) l’explication du **comportement réel**,
  2) la lecture **processus par processus**,
  3) l’évaluation de validation proto avec niveaux de preuve,
  4) les **tests logiques suivants** et les questions d’experts.

---

## 2) Ce que le prototype fait réellement (sans extrapolation)

Le prototype actuel implémente un simulateur C comparatif entre deux dynamiques :
- **NX** : bruit stochastique + rappel de stabilité via terme Lyapunov saturé (`tanh`).
- **Baseline classique** : relaxation linéaire + bruit simulé réduit.

La boucle d’évaluation exécute 360 scénarios, journalise les scores dans le log forensic, exporte un CSV benchmark, puis génère un rapport synthèse. Le résultat observé est une supériorité de `nx_score` sur `classical_score` dans 360/360 cas pour ce protocole.

### Interprétation stricte

Ce résultat valide fortement la **cohérence interne du modèle logiciel** et du pipeline d’exécution, mais ne suffit pas à démontrer un avantage hardware réel tant que le bruit physique n’est pas mesuré et injecté expérimentalement.

---

## 3) Pipeline réel détaillé (étapes de calcul)

1. Chargement des paramètres par défaut (`nqbit_default_config`).
2. Détection CPU/RAM hôte (`sysconf`) et écriture forensic.
3. Pour chaque scénario `i` :
   - construction des paramètres (`seed`, `junction_noise_sigma`, `thermal_factor`),
   - exécution `nqbit_run_test_case`,
   - log de `nx_score` et `classical_score`,
   - écriture ligne CSV.
4. Agrégation finale : calcul `win_rate`.
5. Écriture du rapport final local du runner.

### Ce que cela prouve

- Reproductibilité par seed.
- Traçabilité temporelle fine.
- Exécution déterministe du protocole de benchmark défini.

### Ce que cela ne prouve pas encore

- Bénéfice physique en présence de bruit réel transistor/memristor.
- Stabilité cross-hardware (plateformes et capteurs différents).
- Supériorité universelle vis-à-vis de baselines fortes.

---

## 4) Analyse de validité scientifique (niveau de preuve)

### Niveau A — Démonstrateur logiciel (VALIDÉ)

Critères satisfaits :
- protocole exécutable,
- résultats cohérents,
- traces forensic présentes,
- KPI final explicite (win rate).

### Niveau B — Hypothèse physique (PARTIEL)

Critères partiellement satisfaits :
- l’hypothèse thermodynamique est plausible,
- mais la source de bruit reste pseudo-aléatoire (PRNG), pas une acquisition matérielle directe.

### Niveau C — Validation hardware (NON VALIDÉ)

Critères non satisfaits actuellement :
- banc de mesure physique,
- calibration capteur,
- profil spectral bruit,
- campagne de répétabilité inter-machines.

---

## 5) Forces, faiblesses, biais possibles

### Forces

- Implémentation C claire et compacte.
- Processus auditable de bout en bout (log + CSV + rapport).
- Paramètres explicites, facilement ablatables.

### Faiblesses

- Baseline possiblement trop faible (dynamique non adaptative).
- Fonction de score asymétrique (pénalité énergie différente entre NX et baseline).
- Pas de test d’incertitude formelle (IC, bootstrap, tests non paramétriques).

### Biais méthodologiques à corriger en V2 expérimentale

1. Harmoniser les pénalités énergétiques entre méthodes.
2. Ajouter baselines plus fortes (optimiseurs stochastiques adaptatifs).
3. Évaluer les effets de distribution (bruit gaussien vs non gaussien).
4. Produire des intervalles de confiance et tailles d’effet.

---

## 6) Validation du rapport `ANALYSE_POTENTIEL_HARDWARE_BRUIT_NX.md`

### Verdict

Le rapport prospectif est **cohérent comme vision de recherche**, mais sa validation doit être recadrée :
- validé pour la partie **concept logiciel simulé**,
- non validé pour la partie **preuve hardware réelle**.

En termes experts : c’est une bonne thèse de conception, pas encore une démonstration expérimentale de rupture en conditions matérielles contrôlées.

---

## 7) Prochaine suite logique (plan d’action concret)

## 7.1 Expériences logicielles immédiates (dans ce module)

- **Ablation A** : supprimer `lyapunov_gain`, conserver le bruit.
- **Ablation B** : supprimer le bruit, conserver Lyapunov.
- **Ablation C** : bruit impulsionnel et heavy-tail.
- **Équité de scoring** : même pénalité énergie pour NX et baseline.
- **Répétitions** : 30+ exécutions indépendantes par scénario.

## 7.2 Passage au réel (phase hardware)

- Acquisition de signal de bruit réel (ADC).
- Injection directe de ce signal dans la dynamique NX.
- Comparaison PRNG vs bruit réel à budget calcul identique.
- Étude stabilité thermique et vieillissement.

## 7.3 Critère décisionnel go/no-go

- **GO** si gain robuste (statistiquement significatif) après symétrisation des métriques.
- **NO-GO** si gain disparaît ou dépend d’un réglage non transférable.

---

## 8) Questions répondues et nouvelles questions d’experts

### Déjà répondu (partiellement)

1. Le bruit est-il exploitable sans chaos destructeur ?
   - Oui, dans le cadre paramétrique testé.
2. NX dépasse-t-il la baseline implémentée ?
   - Oui, systématiquement sur ce benchmark.

### Nouvelles questions critiques

1. Quelle part du gain vient du modèle vs de la fonction de score ?
2. Le résultat tient-il sur baselines fortes et équitables ?
3. Que devient la performance avec bruit mesuré réel ?
4. Existe-t-il une zone d’instabilité hors distribution de test ?
5. Quel coût énergétique réel par point de gain en précision ?

---

## 9) Compléments experts ajoutés (ce qui manquait)

1. **Métrologie** : définir protocole de calibration du bruit avant toute revendication hardware.
2. **Sûreté** : ajouter seuils de coupure runtime si dérive entropique.
3. **Reproductibilité** : signer les artefacts (hash, manifest) pour audit tiers.
4. **Transfert industriel** : dissocier roadmap scientifique vs roadmap produit.
5. **Publication** : exiger une section "Threats to validity" explicite.

---

## 10) Conclusion V2

- Le prototype NQubit NX est un **bon démonstrateur algorithmique** de calcul bruité guidé.
- La promesse hardware dissipative est **encore à démontrer expérimentalement**.
- La suite logique est une campagne mixte :
  1) correction des biais de benchmark,
  2) validation statistique stricte,
  3) transition vers bruit physique instrumenté.

---

**Date** : 2026-02-28  
**Version** : V2 (copie améliorée, V1 conservée)  
**Emplacement** : `src/projetx_NQubit NX/NQubit_NX/version_2/`
