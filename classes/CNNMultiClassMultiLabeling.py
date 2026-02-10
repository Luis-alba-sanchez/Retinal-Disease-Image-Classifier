import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMultiClassMultiLabeling(nn.Module):
    def __init__(self, number_of_classes=3):
        super(CNNMultiClassMultiLabeling, self).__init__()
        # Couches convolutives
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # (16, 512, 512)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # (16, 256, 256)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (32, 256, 256)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # (32, 128, 128)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (64, 128, 128)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # (64, 64, 64)

        # Couches fully connected
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, number_of_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 64 * 64 * 64)  # Aplatir
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        return torch.sigmoid(self.fc3(x)) 