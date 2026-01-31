"""
Script pour réorganiser et fusionner les données des dossiers Test_Set et Training_Set
vers une structure unique avec images renommées et CSV fusionné.
"""

import os
import pandas as pd
import shutil
from pathlib import Path
from typing import Tuple, Dict

# Configuration des chemins
DATASETS_PATH = "./source"
DATA_OUTPUT_PATH = "./Training_set"

TEST_SET_PATH = os.path.join(DATASETS_PATH, "Test_Set")
TRAINING_SET_PATH = os.path.join(DATASETS_PATH, "Training_Set")
IMAGES_OUTPUT_PATH = os.path.join(DATA_OUTPUT_PATH, "Training")
CSV_OUTPUT_PATH = os.path.join(DATA_OUTPUT_PATH, "Training_Labels.csv")
CSV_OUTPUT_PATH_BACKUP = os.path.join(DATA_OUTPUT_PATH, "Training_Labels_Backup.csv")


def validate_directories() -> bool:
    """Vérifie l'existence des dossiers source."""
    if not os.path.exists(TEST_SET_PATH):
        print(f"Dossier Test_Set non trouvé: {TEST_SET_PATH}")
        return False
    if not os.path.exists(TRAINING_SET_PATH):
        print(f"Dossier Training_Set non trouvé: {TRAINING_SET_PATH}")
        return False
    if not os.path.exists(DATA_OUTPUT_PATH):
        print(f"Dossier destination non trouvé: {DATA_OUTPUT_PATH}")
        return False
    
    print("✅ Tous les dossiers source existent")
    return True


def read_dataset_info(dataset_name: str, dataset_path: str) -> Tuple[pd.DataFrame, list]:
    """
    Lit le CSV et liste les images d'un dataset.
    
    Returns:
        Tuple[DataFrame, list]: DataFrame du CSV et liste des noms d'images
    """
    # Chercher le fichier CSV
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {dataset_path}")
    
    csv_file = os.path.join(dataset_path, csv_files[0])
    print(f"Lecture CSV pour {dataset_name}: {csv_files[0]}")
    
    df = pd.read_csv(csv_file)
    
    # Chercher le dossier d'images
    image_folders = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d)) and d != '__pycache__']
    if not image_folders:
        raise FileNotFoundError(f"Aucun dossier d'images trouvé dans {dataset_path}")
    
    image_folder = os.path.join(dataset_path, image_folders[0])
    print(f"Dossier d'images trouvé: {image_folders[0]}")
    
    images = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    
    print(f"   - CSV: {len(df)} lignes")
    print(f"   - Images: {len(images)} fichiers PNG\n")
    
    return df, images, image_folder


def create_output_structure():
    """Crée la structure du dossier de destination."""
    if os.path.exists(IMAGES_OUTPUT_PATH):
        print(f"Dossier {IMAGES_OUTPUT_PATH} existe déjà")
        response = input("Voulez-vous le supprimer et recommencer? (o/n): ").lower()
        if response == 'o':
            shutil.rmtree(IMAGES_OUTPUT_PATH)
            os.makedirs(IMAGES_OUTPUT_PATH)
            print(f"Dossier recréé: {IMAGES_OUTPUT_PATH}\n")
        else:
            print("Opération annulée.")
            return False
    else:
        os.makedirs(IMAGES_OUTPUT_PATH)
        print(f"Dossier créé: {IMAGES_OUTPUT_PATH}\n")
    
    return True


def reorganize_datasets():
    """Réorganise les datasets avec renommage continu des images."""
    print("=" * 70)
    print("DÉMARRAGE DE LA RÉORGANISATION DES DONNÉES")
    print("=" * 70 + "\n")
    
    # Validation
    if not validate_directories():
        return False
    
    print("\n" + "=" * 70)
    print("LECTURE DES DATASETS")
    print("=" * 70 + "\n")
    
    try:
        # Lire les données
        test_df, test_images, test_image_folder = read_dataset_info("Test_Set", TEST_SET_PATH)
        training_df, training_images, training_image_folder = read_dataset_info("Training_Set", TRAINING_SET_PATH)
    except Exception as e:
        print(f"Erreur lors de la lecture: {e}")
        return False
    
    # Créer la structure de sortie
    if not create_output_structure():
        return False
    
    print("=" * 70)
    print("RENOMMAGE ET COPIE DES IMAGES")
    print("=" * 70 + "\n")
    
    # Dictionnaires de correspondance ancien -> nouveau nom
    test_mapping = {}
    training_mapping = {}
    
    # Traiter les images Test_Set
    print(f"Test_Set: {len(test_images)} images")
    for idx, old_name in enumerate(test_images, start=1):
        new_name = f"{idx}.png"
        old_path = os.path.join(test_image_folder, old_name)
        new_path = os.path.join(IMAGES_OUTPUT_PATH, new_name)
        
        shutil.copy2(old_path, new_path)
        test_mapping[old_name] = new_name
        
        if (idx) % 100 == 0 or idx == len(test_images):
            print(f"   ✓ {idx}/{len(test_images)} images copiées")
    
    print(f"   {len(test_images)} images copiées\n")
    
    # Traiter les images Training_Set
    start_id = len(test_images) + 1
    print(f"Training_Set: {len(training_images)} images (commençant à ID {start_id})")
    
    for idx, old_name in enumerate(training_images, start=start_id):
        new_name = f"{idx}.png"
        old_path = os.path.join(training_image_folder, old_name)
        new_path = os.path.join(IMAGES_OUTPUT_PATH, new_name)
        
        shutil.copy2(old_path, new_path)
        training_mapping[old_name] = new_name
        
        if (idx - start_id + 1) % 100 == 0 or idx == start_id + len(training_images) - 1:
            print(f"   ✓ {idx - start_id + 1}/{len(training_images)} images copiées")
    
    print(f"   {len(training_images)} images copiées\n")
    
    print("=" * 70)
    print("FUSION DES FICHIERS CSV")
    print("=" * 70 + "\n")
    
    # Préparer les DataFrames
    test_df['source'] = 'Test'
    test_df['ancien_nom'] = test_df['ID'].astype(str) + '.png'
    test_df['ID'] = range(1, len(test_df) + 1)
    
    training_df['source'] = 'Training'
    training_df['ancien_nom'] = training_df['ID'].astype(str) + '.png'
    training_df['ID'] = range(len(test_df) + 1, len(test_df) + len(training_df) + 1)
    
    # Fusionner les DataFrames
    combined_df = pd.concat([test_df, training_df], ignore_index=False)
    
    # Réordonner les colonnes pour mettre ID, source, ancien_nom au début
    cols = ['ID'] + [col for col in combined_df.columns 
                                               if col not in ['ID', 'source', 'ancien_nom']]
    cols_backup = ['ID'] + [col for col in combined_df.columns 
                                               if col not in ['ID', 'source', 'ancien_nom']] + ['source', 'ancien_nom']
    combined_df_data = combined_df[cols]
    combined_df_backup = combined_df[cols_backup]
    
    # Sauvegarder le CSV
    combined_df_data.to_csv(CSV_OUTPUT_PATH, index=False)
    combined_df_backup.to_csv(CSV_OUTPUT_PATH_BACKUP, index=False)
    print(f"CSV fusionné sauvegardé: {CSV_OUTPUT_PATH}\n")
    
    print("=" * 70)
    print("VÉRIFICATION ET RÉSUMÉ")
    print("=" * 70 + "\n")
     
    # Vérifications
    nb_images_output = len(os.listdir(IMAGES_OUTPUT_PATH))
    nb_rows_csv = len(combined_df)
    
    print(f"RÉSUMÉ FINAL:")
    print(f"   - Images Test_Set: {len(test_images)}")
    print(f"   - Images Training_Set: {len(training_images)}")
    print(f"   - Total images: {len(test_images) + len(training_images)}")
    print(f"   - Images dans le dossier de sortie: {nb_images_output}")
    print(f"   - Lignes dans le CSV: {nb_rows_csv}\n")
    
    if nb_images_output == len(test_images) + len(training_images) and nb_rows_csv == nb_images_output:
        print("VALIDATION: Tous les nombres correspondent! ✅\n")
        success = True
    else:
        print("ALERTE: Les nombres ne correspondent pas!")
        success = False
    
    # Aperçu du CSV
    print(f"APERÇU DU CSV FUSIONNÉ (premières et dernières lignes):\n")
    print(combined_df[['ID', 'source', 'ancien_nom', 'Disease_Risk', 'DR', 'ARMD']].head(3).to_string(index=False))
    print("...")
    print(combined_df[['ID', 'source', 'ancien_nom', 'Disease_Risk', 'DR', 'ARMD']].tail(3).to_string(index=False))
    print(f"\nRÉORGANISATION TERMINÉE AVEC SUCCÈS!\n")
    
    return success


if __name__ == "__main__":
    success = reorganize_datasets()
    if not success:
        print("❌ La réorganisation a échoué.")
        exit(1)
