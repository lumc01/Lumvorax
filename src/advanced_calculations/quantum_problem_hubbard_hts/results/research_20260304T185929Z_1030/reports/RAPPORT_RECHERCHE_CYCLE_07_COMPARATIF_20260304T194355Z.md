# Rapport technique itératif — cycle 07 comparatif (exécutions Replit x2)

**Nom exact du rapport :** `RAPPORT_RECHERCHE_CYCLE_07_COMPARATIF_20260304T194355Z.md`

- Run A analysé : `research_20260304T185430Z_173`
- Run B analysé : `research_20260304T185929Z_1030`
- Fichier demandé explicitement : `research_20260304T185430Z_173/logs/baseline_reanalysis_metrics.csv`
- Principe de non-écrasement respecté : aucun ancien log modifié/supprimé.

---

## 0) Mise à jour dépôt distant (traçabilité)
Tentative de synchronisation distante effectuée avant analyse, mais l'environnement a refusé l'accès HTTPS GitHub (`CONNECT tunnel failed, response 403`).

Conséquence : analyse effectuée rigoureusement sur les données locales présentes dans le dépôt courant.

---

## 1) Analyse pédagogique structurée (niveau débutant)

### 1.1 Contexte
Le fichier `baseline_reanalysis_metrics.csv` contient des mesures séquentielles par module (`problem`) à des pas de calcul (`step`). Chaque ligne représente un instant de simulation pour un proxy physique (Hubbard, QCD proxy, champ hors équilibre, etc.).

### 1.2 Définitions des colonnes
- `problem` : module simulé (sous-système numérique).
- `step` : index de temps/discrétisation.
- `energy` : observable d'énergie interne du proxy (sans unité explicitée dans ce dataset).
- `pairing` : observable de couplage/corrélation (croît ici de façon cumulative).
- `sign_ratio` : ratio signé utile pour diagnostiquer bruit/symétrie/statistique de signe.
- `cpu_percent`, `mem_percent` : charge système pendant l'exécution.
- `elapsed_ns` : temps écoulé (nanosecondes) depuis le début du module.

### 1.3 Hypothèses opérationnelles
- Les séries sont comparables entre run A et run B si `(problem, step)` est identique.
- Une validation physique forte exigerait des unités, normalisations, taille lattice, Hamiltonien explicite, incertitudes et protocoles de convergence formels.

### 1.4 Méthode utilisée dans ce rapport
1. Vérification d'intégrité CSV (lignes bien formées vs malformées).
2. Comparaison point à point run A vs run B sur les couples communs `(problem, step)`.
3. Diagnostics sur le baseline demandé (run A 185430): min/max, monotonie, dispersion `sign_ratio`.
4. Consolidation dans des artefacts horodatés UTC + checksums.

### 1.5 Résultats pédagogiques clés
- Run A contient **114** lignes valides.
- Run B contient **87** lignes valides + **1 ligne malformée** (`dense_nuclear_proxy,1600,453950`).
- Sur les **87 points communs**, les valeurs `energy`, `pairing`, `sign_ratio` sont **strictement identiques** (diff max = 0).
- Donc le moteur est **reproductible** sur l'intersection des points, mais run B est **incomplet/corrompu** côté journalisation.

### 1.6 Interprétation
- Vous validez une **stabilité logicielle de calcul** sur les points communs.
- Vous ne validez pas une convergence physique vers un état fondamental : l'énergie traverse négatif→positif puis croît fortement.
- La variable `pairing` est monotone croissante dans tous les modules du run A (signature cumulative, pas un ordre paramétrique normalisé à elle seule).

---

## 2) Questions d'expert + statut de réponse

| Question experte | Réponse issue des logs | Statut |
|---|---|---|
| Reproductibilité inter-run ? | Oui sur 87 couples `(problem, step)` communs : diff max = 0 sur `energy/pairing/sign_ratio`. | **Complète** |
| Intégrité des logs ? | Run B contient 1 ligne malformée, donc intégrité partielle. | **Complète** |
| Convergence énergétique ? | Non observée : croissance non bornée après transition de signe. | **Complète** |
| Observable pairing interprétable physiquement directement ? | Non, comportement cumulatif monotone sans normalisation explicite. | **Partielle** |
| Sign problem sévère présent ? | Pas d'effondrement exponentiel visible; faibles fluctuations autour de 0 (ordre ~1e-3 à 1e-2). | **Partielle** |
| Paramètres physiques (U/t, lattice, bords, action, β, volume) documentés dans ce CSV ? | Non présents dans ce fichier. | **Absente** |
| Barres d'erreur/statistiques d'ensemble ? | Non incluses dans ce fichier baseline. | **Absente** |
| Validation contre solution analytique interne ? | Non dans ce CSV (existe potentiellement ailleurs, non inférable ici). | **Partielle** |

---

## 3) Détection d'anomalies / incohérences / découvertes potentielles

### 3.1 Anomalies factuelles
1. **Ligne CSV malformée dans run B** : perte de colonnes à la fin (`NF=3` au lieu de 8).
2. **Module `quantum_chemistry_proxy` absent de run B** (présent en run A avec 22 points).
3. **Troncature `dense_nuclear_proxy` run B** : s'arrête à step 1500 valide; ligne 1600 cassée.

### 3.2 Incohérences physiques potentielles
- Croissance énergétique massive et prolongée (ex: Hubbard jusqu'à ~1.27e6) sans critère explicite de stabilisation dans ce log.
- `pairing` monotone partout : plus compatible avec accumulation numérique qu'avec observable d'ordre déjà normalisée.

### 3.3 Hypothèses explicatives
- Artefact de pipeline de sortie/logging (écriture interrompue, flush incomplet) pour run B.
- Schéma d'intégration explicite/cumulatif non renormalisé pour la dynamique des observables.

### 3.4 Statut découverte vs artefact
- **Pas de preuve de découverte physique** à ce stade.
- **Preuve solide d'un artefact de traçage run B** + preuve de reproductibilité partielle des valeurs calculées.

---

## 4) Comparaison état de l'art / littérature (dans le cadre des données disponibles)
- Une simulation Hubbard/QCD physiquement exploitable exige généralement : paramètres dimensionnés, volumes, erreurs, finite-size scaling, convergence statistique.
- Le CSV baseline seul ne contient pas ces éléments de validation externe.
- Donc la cohérence à la littérature ne peut être que **qualitative** ici : architecture numérique correcte, validation physique **non concluante** sur ce seul fichier.

---

## 5) Nouveaux tests définis et exécutés (sans toucher aux anciens logs)

### Tests exécutés maintenant
1. **Test d'intégrité CSV** (format 8 colonnes par ligne).
2. **Test de reproductibilité point à point** entre les 2 runs (intersection des points).
3. **Test de couverture de modules et steps** (détection manquants).
4. **Test de monotonie et bornes** sur run A (énergie/pairing).
5. **Traçabilité cryptographique** (SHA256 des nouveaux artefacts).

### Critères et verdicts
- Intégrité run A : PASS.
- Intégrité run B : FAIL (1 ligne malformée).
- Reproductibilité intersection : PASS (diff=0).
- Complétude des modules entre runs : FAIL (module absent run B).
- Pairing monotone (run A): PASS (5/5 modules).
- Énergie monotone croissante (run A): FAIL (0/5 modules), avec bascule puis forte hausse.

---

## 6) Exécution complète + nouveaux logs distincts
Nouveaux artefacts créés (UTC horodatés, indépendants) :

- `tests/execution_comparison_20260304T194355Z.csv`
- `tests/baseline_185430_diagnostics_20260304T194355Z.csv`
- `logs/post_analysis_provenance_20260304T194355Z.log`
- `logs/post_analysis_checksums_20260304T194355Z.sha256`

Aucun ancien fichier n'a été écrasé.

---

## 7) Réponse explicite à votre bloc “analyse structurée”

Votre critique initiale est **globalement confirmée** par le fichier demandé (`185430/.../baseline_reanalysis_metrics.csv`) :
- énergie instationnaire avec croissance forte;
- `pairing` cumulatif monotone;
- `sign_ratio` faible amplitude autour de 0;
- validation surtout logicielle/infra plutôt que preuve physique.

Nuance ajoutée par comparaison 2 runs :
- les valeurs sont parfaitement reproductibles sur la zone commune;
- le second run introduit un problème de traçage (ligne malformée + module manquant), ce qui limite l'auditabilité globale.

---

## 8) Plan de cycle itératif suivant (obligatoire)
1. Corriger le writer CSV (flush/atomic write + validation post-run).
2. Rejouer un cycle complet pour récupérer 5/5 modules intègres.
3. Ajouter normalisations physiques (par site/degré de liberté).
4. Ajouter barres d'erreur + répétitions seeds + test pas de temps.
5. Refaire le même rapport comparatif horodaté, sans écraser l'historique.

