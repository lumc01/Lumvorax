# NQubit_v5 — Correctifs d’intégrité forensic et clarification audit final

Cette version V5 ajoute une chaîne d’intégrité **canonique et vérifiable** sans modifier les anciens artefacts V4.

## Objectif V5
- Définir quel artefact fait foi pour l’audit final.
- Ajouter un mécanisme de signature au-delà du SHA256.
- Fournir un protocole vérifiable de génération/validation de manifest.

## Décision d’audit (V5)
- Artefact qui fait foi : **`manifest_forensic_v5.json` signé**.
- `sha256_replit_v4.txt` reste utile comme trace opérationnelle, mais n’est plus l’autorité finale.

## Outils V5
- `tools/build_manifest_v5.py` : génère le manifest canonique.
- `tools/verify_manifest_v5.py` : vérifie tous les hash du manifest.
- `tools/sign_manifest_v5.sh` : signe le manifest (`openssl`) et peut vérifier la signature.

## Protocole recommandé
1. Geler les artefacts de run.
2. Générer le manifest V5.
3. Signer le manifest avec une clé privée d’audit.
4. Vérifier hash + signature en CI avant publication.
