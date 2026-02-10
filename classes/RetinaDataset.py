"""
Docstring for Classes.RetinaDataset
Classe personnalisée pour le chargement des données
"""

from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch

class RetinaDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, type_of_classification="binary", label_for_multilabel_classification=["DR", "MH", "ODC"]):
        """
        Args:
            root_dir (string): Directory containing the images.
            csv_file (string): Path to the CSV file containing the labels.
            transform (callable, optional): Transformations to apply on the images.
            type_of_classification (string): Type of classification ("binary" or "multilabel").
            label_for_multilabel_classification (list): List of column names to use as labels for multilabel classification (ignored if type_of_classification is "binary").

        Necessary data structure:
            - Directory : Evaluation-Set/Validation/ & Evaluation-Set/Validation_Labels.csv, Training-Set/Training/ & Training-Set/Training_Labels.csv
            - Images : Each image has an ID corresponding to its name (e.g., 1.png → ID 1). The image Validation/1.png is different from Training/1.png
            - CSV : Each row in the CSV corresponds to an image, with the ID in the first column and labels in subsequent columns.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.labels_df = pd.read_csv(csv_file)
        # On suppose que l'ID est la première colonne et que les labels commencent à la 3ème colonne (ex: DR, ARMD, etc.)
        if type_of_classification == "binary":
            self.labels = self.labels_df.iloc[:, 1:2].values.copy()  # On ignore la 2ème colonne (Disease_Risk) car elle n'est pas un label
        elif type_of_classification == "multilabel":
            self.labels = self.labels_df[label_for_multilabel_classification].values.copy()  # On sélectionne uniquement les colonnes correspondant aux labels de classification multilabel
        else:
            raise ValueError("type_of_classification must be 'binary' or 'multilabel'")
        self.image_ids = self.labels_df.iloc[:, 0].values  # ID des images

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{self.image_ids[idx]}.png")
        image = Image.open(img_name).convert('RGB')  # Charge l'image en RGB
        label = self.labels[idx]
        label = torch.FloatTensor(label)  # Convertit le label en tenseur

        if self.transform:
            image = self.transform(image)

        return image, label