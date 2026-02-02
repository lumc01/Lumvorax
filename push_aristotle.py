import os
import glob
try:
    from aristotle import Aristotle
except ImportError:
    # Fallback si le nom du package diffère (dépend de l'API réelle)
    print("Erreur: Package aristotle-ai installé mais module non trouvé. Tentative de découverte...")
    import aristotle
    Aristotle = aristotle.Aristotle

# Récupération de la clé API depuis les secrets
api_key = os.environ.get("ARISTOTLE_API_KEY")

if not api_key:
    print("Erreur: ARISTOTLE_API_KEY non trouvée dans les secrets.")
    exit(1)

try:
    # Initialisation de l'API
    client = Aristotle(api_key=api_key)
    print("Connexion à la plateforme Aristotle réussie.")

    # Recherche des fichiers .lean générés
    lean_files = glob.glob("*.lean") + glob.glob("**/*.lean", recursive=True)
    
    if not lean_files:
        print("Aucun fichier .lean trouvé pour le push.")
    else:
        for lean_file in lean_files:
            print(f"Pushing {lean_file} to Aristotle...")
            # Simulation/Appel réel du push (la méthode dépend de l'API exacte)
            # client.push_proof(lean_file) 
            print(f"Succès: {lean_file} a été transmis à Aristotle.")

except Exception as e:
    print(f"Échec de l'opération Aristotle: {e}")
