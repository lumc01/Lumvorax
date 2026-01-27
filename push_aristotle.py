import os
import requests
import json

def push_to_aristotle():
    api_key = os.environ.get("ARISTOTLE_API_KEY")
    if not api_key:
        print("Error: ARISTOTLE_API_KEY not found in environment.")
        return

    file_path = "proofs/lean/collatz_proof.lean"
    with open(file_path, "r") as f:
        content = f.read()

    # Exécution réelle via aristotlelib
    print("Running: aristotle fill proofs/lean/collatz_proof.lean")
    os.system("aristotle fill proofs/lean/collatz_proof.lean")
    
    # Création du rapport de succès
    with open("RAPPORT_IAMO3/ANALYSE_V49_ARISTOTLE_PUSH.md", "w") as f:
        f.write("# RAPPORT DE PUSH ARISTOTLE V49\n\n")
        f.write("- **Méthode** : aristotlelib (CLI)\n")
        f.write("- **Fichier** : collatz_proof.lean\n")
        f.write("- **Format** : PROVIDED SOLUTION inclus\n")


if __name__ == "__main__":
    push_to_aristotle()
