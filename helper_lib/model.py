import torch
import torch.nn as nn
import torch.nn.functional as F


# From Module 4 Practical 1 (FCNN notebook)
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x


# From Module 4 Practical 3 (CNN notebook)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# From Module 4 Practical 3 (CNN notebook)
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Assignment 2 architecture (input: 64x64 RGB image)
class AssignmentCNN(nn.Module):
    def __init__(self):
        super(AssignmentCNN, self).__init__()
        # 64x64x3 -> 64x64x16 (padding 1 keeps the size)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # halves the size: 64x64 -> 32x32, then 32x32 -> 16x16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32x32x16 -> 32x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # after two poolings: 32 channels x 16 x 16 = 8192 values
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(model_name):
    if model_name == "FCNN":
        model = MLP()
    elif model_name == "CNN":
        model = SimpleCNN()
    elif model_name == "EnhancedCNN":
        model = EnhancedCNN()
    elif model_name == "AssignmentCNN":
        model = AssignmentCNN()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model