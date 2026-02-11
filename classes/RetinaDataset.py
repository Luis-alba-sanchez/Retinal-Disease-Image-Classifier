"""
Docstring for Classes.RetinaDataset
Classe personnalisée pour le chargement des données
"""

from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
import numpy as np

class RetinaDataset(Dataset):
    def __init__(
            self, 
            root_dir, 
            csv_file, 
            transform=None, 
            type_of_classification="binary", 
            label_for_multilabel_classification=["DR", "MH", "ODC"], 
            add_normal_label=False):
        """
        Args:
            root_dir (string): Directory containing the images.
            csv_file (string): Path to the CSV file containing the labels.
            transform (callable, optional): Transformations to apply on the images.
            type_of_classification (string): Type of classification ("binary" or "multilabel").
            label_for_multilabel_classification (list): List of column names to use as labels for multilabel classification (ignored if type_of_classification is "binary").
            add_normal_label (bool): Whether to add a "normal" label for samples that have no positive labels (default: False).

        Necessary data structure:
            - Directory : Evaluation-Set/Validation/ & Evaluation-Set/Validation_Labels.csv, Training-Set/Training/ & Training-Set/Training_Labels.csv
            - Images : Each image has an ID corresponding to its name (e.g., 1.png → ID 1). The image Validation/1.png is different from Training/1.png
            - CSV : Each row in the CSV corresponds to an image, with the ID in the first column and labels in subsequent columns.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.labels_df = pd.read_csv(csv_file)

        # The ID is the first column, desease_risk the second and the others are the labels of the pathologies
        if type_of_classification == "binary":
            self.labels = self.labels_df.iloc[:, 1:2].values.copy() 
            self.image_ids = self.labels_df.iloc[:, 0].values

        elif type_of_classification == "multilabel" and add_normal_label:
            # On ajoute une colonne "Normal" basée sur la colonne Disease_Risk
            # Supprimer les lignes où tous les labels sélectionnés sont 0
            # (incluant la colonne "Normal" inversée) pour n'entraîner que
            # sur les images contenant au moins une des pathologies choisies.
            normal_label = np.array(self.labels_df["Disease_Risk"].replace([0, 1], [1, 0])).reshape(-1, 1)
            disease_labels = self.labels_df[label_for_multilabel_classification].values
            all_labels = np.concatenate((disease_labels, normal_label), axis=1)
            # Garder uniquement les lignes où au moins un label (disease ou normal) est positif
            mask = all_labels.sum(axis=1) > 0
            self.labels_df = self.labels_df[mask].reset_index(drop=True)
            self.labels = all_labels[mask]
            self.image_ids = self.labels_df.iloc[:, 0].values

        elif type_of_classification == "multilabel" and not add_normal_label:
            self.labels = self.labels_df[label_for_multilabel_classification].values.copy()
            labels_df = self.labels_df[label_for_multilabel_classification].copy()
            mask = labels_df.sum(axis=1) > 0
            self.labels_df = self.labels_df[mask].reset_index(drop=True)
            self.labels = self.labels_df[label_for_multilabel_classification].values.copy()
            self.image_ids = self.labels_df.iloc[:, 0].values
        else:
            raise ValueError("type_of_classification must be 'binary' or 'multilabel'")

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