from tqdm import tqdm
import pandas as pd
import os

import torch
from torchvision import transforms

# Local imports
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
tool_path = os.path.abspath(os.path.normpath(parent_dir))
if tool_path not in sys.path:
    sys.path.insert(0, tool_path)

from tools.data_tools import crop_image_tensor
from classes.RetinaDataset import RetinaDataset


def get_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images = 0
    for _, pack in tqdm(enumerate(loader), total=len(loader)):
        images = pack[0]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    mean /= total_images
    std /= total_images
    return mean, std


# Appliquer les transformations
transform_get_mean_std = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(crop_image_tensor),
    transforms.Resize((256, 256)),  # Redimensionne l'image après rognage
])

train_dataset_get_mean_std = RetinaDataset(
    root_dir='../../data/Training-Set/Training',
    csv_file='../../data/Training-Set/Training_Labels.csv',
    transform=transform_get_mean_std
)

train_loader_get_mean_std = torch.utils.data.DataLoader(
    train_dataset_get_mean_std,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

mean, std = get_mean_std(train_loader_get_mean_std) # ~3 minutes
print(f"Mean: {mean}")
print(f"Std: {std}")

df_mean_std = pd.DataFrame({
    'Mean': mean.numpy(),
    'Std': std.numpy()
})
df_mean_std.to_csv(os.path.join(parent_dir, 'data', 'mean-std', 'mean_std_256_256.csv'), index=False)