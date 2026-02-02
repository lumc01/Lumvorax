import asyncio
import os
import aristotlelib
from pathlib import Path

async def main():
    # Récupération de la clé API depuis les secrets
    api_key = os.environ.get("ARISTOTLE_API_KEY")
    if not api_key:
        print("ERREUR: La clé ARISTOTLE_API_KEY est absente des secrets.")
        return

    # Configuration de l'API
    aristotlelib.set_api_key(api_key)
    print("Clé API Aristotle configurée.")

    lean_file = "v28_logic_proof.lean"
    if not Path(lean_file).exists():
        print(f"ERREUR: Le fichier {lean_file} n'existe pas.")
        return

    print(f"Début du push de {lean_file} vers la plateforme Aristotle...")
    
    try:
        # Utilisation de prove_from_file pour pousser et lancer la preuve
        # wait_for_completion=False car l'utilisateur ne veut pas attendre la fin de l'exécution
        result = await aristotlelib.Project.prove_from_file(
            input_file_path=lean_file,
            auto_add_imports=True,
            validate_lean_project=True,
            wait_for_completion=False
        )
        print(f"SUCCÈS: Fichier poussé. ID du projet/résultat: {result}")
        print("L'exécution se poursuit en arrière-plan sur la plateforme Aristotle.")
    except Exception as e:
        print(f"ÉCHEC du push vers Aristotle: {e}")

if __name__ == "__main__":
    asyncio.run(main())
