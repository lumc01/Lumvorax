import os
import json
import subprocess

def deploy_to_kaggle():
    print("Starting Kaggle Deployment Process...")
    
    # 1. Verification des fichiers
    kernel_file = "nx47_vesu_kernel.py"
    if not os.path.exists(kernel_file):
        print(f"Error: {kernel_file} not found.")
        return

    # 2. Configuration Kaggle (Utilisation de l'API Token fournie)
    # On cree le dossier .kaggle si necessaire
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Le token fourni par l'utilisateur
    new_token = "KGAT_147c649a54eb31af840fcdfdf2d85c1b"
    os.environ['KAGGLE_API_TOKEN'] = new_token
    
    # Pour s'assurer que le CLI utilise ce token, on peut aussi l'ecrire dans kaggle.json 
    # mais KGAT est souvent utilise directement comme env var ou via config set.
    # On va essayer de configurer le username si on le trouve dans le token ou via une commande.
    
    # 3. Preparation des Metadatas pour le Kernel
    metadata = {
        "id": "gabrielchavesreinann/nx47-vesu-kernel",
        "title": "nx47-vesu-kernel",
        "code_file": kernel_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": ["trentonmclouth/vesuvius-challenge-surface-detection"],
        "competition_sources": ["vesuvius-challenge-surface-detection"],
        "kernel_sources": []
    }
    
    with open("kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # 4. Push reel via CLI Kaggle
    print("Pushing to Kaggle...")
    try:
        # On utilise le module kaggle installe
        result = subprocess.run(["kaggle", "kernels", "push", "-p", "."], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            print("Successfully pushed to Kaggle!")
        else:
            print(f"Kaggle push failed: {result.stderr}")
            # Note: Si l'API renvoie une erreur d'authentification, c'est que le token KGAT nécessite une config spécifique
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    deploy_to_kaggle()
