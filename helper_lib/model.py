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


# Assignment 3: GAN for MNIST (architecture from the assignment)
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Noise vector (100) -> 7 x 7 x 128
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        # 7x7x128 -> 14x14x64
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        # 14x14x64 -> 28x28x1
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)  # Reshape
        x = self.act1(self.bn1(self.deconv1(x)))
        x = self.tanh(self.deconv2(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 28x28x1 -> 14x14x64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        # 14x14x64 -> 7x7x128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)
        # 7x7x128 -> single real/fake probability
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.sigmoid(self.fc(x))
        return x


class GAN(nn.Module):
    # Holds both networks so get_model("GAN") returns one object
    def __init__(self, z_dim=100):
        super(GAN, self).__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()

    def forward(self, z):
        return self.generator(z)


def get_model(model_name):
    if model_name == "FCNN":
        model = MLP()
    elif model_name == "CNN":
        model = SimpleCNN()
    elif model_name == "EnhancedCNN":
        model = EnhancedCNN()
    elif model_name == "AssignmentCNN":
        model = AssignmentCNN()
    elif model_name == "GAN":
        model = GAN()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model