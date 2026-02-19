# Prompt agent Replit — Validation complète LUM/VORAX (tests unitaires + intégration)

Tu dois exécuter **exactement** les tests préparés dans ce dossier, puis produire un rapport final lisible pour validation.

## Contexte
Le système LUM/VORAX doit être validé avec preuves machine (JSON + logs) et résumé humain.

## Étapes obligatoires
1. Ouvre et lis le script:
   - `RAPPORT-VESUVIUS/validation_lumvorax/run_lumvorax_validation.py`
2. Exécute:
   - `python3 RAPPORT-VESUVIUS/validation_lumvorax/run_lumvorax_validation.py`
3. Vérifie la présence des artefacts:
   - `RAPPORT-VESUVIUS/validation_lumvorax/validation_results.json`
   - `RAPPORT-VESUVIUS/validation_lumvorax/VALIDATION_LUMVORAX_SYSTEME_COMPLET_20260219.md`
4. Relis ces deux fichiers et produis un **rapport d'analyse** dans:
   - `RAPPORT-VESUVIUS/validation_lumvorax/RAPPORT_AGENT_REPLIT_POST_TESTS.md`

## Exigences du rapport post-tests
- Résumer test par test (source indentation, roundtrip `.lum`, intégration 3D Python, tentative compile native C).
- Indiquer clairement **confirmé** vs **en attente**.
- Lister les blocages et les actions suivantes prioritaires.
- Conserver un style pédagogique (termes techniques expliqués brièvement).

## Interdictions
- Ne pas supprimer/modifier les anciens rapports.
- Ne pas utiliser placeholders/stubs dans le rapport final.
- Ne pas masquer les erreurs: si un test échoue, explique la cause réelle.
